import os
import subprocess
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from wandb import AlertLevel

from moment.common import PATHS
from moment.models.momentwithTFC import MOMENT_re, MOMENTDecoder, TFC
from moment.utils.utils import MetricsStore, dtype_map, make_dir_if_not_exists
from moment.data.datatransform import dataconversion
from .base import Tasks
from moment.models.loss import NTXentLoss_poly
warnings.filterwarnings("ignore")


class Pretraining(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self.encoder_re = MOMENT_re(args)
        self.decoder = MOMENTDecoder(args)
        self.tfc = TFC(args)
        self.nt_xent_poly = NTXentLoss_poly(args)

    def validation(self, data_loader, return_preds: bool = False):
        trues, preds, masks, losses = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            for batch_x in tqdm(data_loader, total=len(data_loader)):
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=None
                    )

                recon_loss = self.criterion(outputs.reconstruction, timeseries)
                observed_mask = input_mask * (1 - outputs.pretrain_mask)
                n_channels = outputs.reconstruction.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                losses.append(loss.item())

                if return_preds:
                    trues.append(timeseries.detach().cpu().numpy())
                    preds.append(outputs.reconstruction.detach().cpu().numpy())
                    masks.append(outputs.pretrain_mask.detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)
        self.model.train()

        if return_preds:
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            masks = np.concatenate(masks, axis=0)
            return average_loss, losses, (trues, preds, masks)
        else:
            return average_loss

    def train(self):
        self.run_name = self.logger.name
        path = os.path.join(self.args.checkpoint_path, self.run_name)
        make_dir_if_not_exists(path, verbose=True)

        all_params = list(self.encoder_re.parameters()) + \
             list(self.decoder.parameters()) + \
             list(self.tfc.parameters())

        # 중복 제거
        seen = set()
        unique_params = []
        for p in all_params:
            if id(p) not in seen:
                unique_params.append(p)
                seen.add(id(p))

        self.optimizer = torch.optim.AdamW(
            unique_params,
            lr=self.args.init_lr,
            weight_decay=self.args.weight_decay,
        )


        self.criterion = self._select_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler()

        self.encoder_re.to(self.device)
        self.decoder.to(self.device)
        self.tfc.to(self.device)
        # self.evaluate_and_log()

        opt_steps = 0
        cur_epoch = 0
        while opt_steps < self.args.max_opt_steps or cur_epoch < self.args.max_epoch:
            self.encoder_re.train()
            self.decoder.train()
            self.tfc.train()

            for batch_x in tqdm(
                self.train_dataloader, total=len(self.train_dataloader)
            ):
                self.optimizer.zero_grad(set_to_none=True)
                timeseries = batch_x.timeseries.float().to(self.device)
                input_mask = batch_x.input_mask.long().to(self.device)

                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                t2, f1, f2, mask_t2, dummy_mask_f1, dummy_mask_f2 = dataconversion(timeseries, self.args, input_mask=input_mask, mask=None)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    out_t1, out_t2, out_f1, out_f2, input_mask, pretrain_mask, mean , stdev = self.encoder_re(
                        x_enc_re=timeseries, x_enc_t2=t2, x_enc_f1=f1, x_enc_f2=f2, input_mask=input_mask, mask=None, mask_t2=mask_t2, mask_f1=dummy_mask_f1, mask_f2=dummy_mask_f2)
                
                    
                    
                dec_out =  self.decoder(
                    enc_out=out_t1,
                    input_mask=input_mask,
                    pretrain_mask=pretrain_mask,
                    mean=mean,
                    stdev=stdev
                )


                h_t, z_t, h_f, z_f = self.tfc(out_t1, out_f1)
                h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.tfc(out_t2, out_f2)


                recon_loss = self.criterion(dec_out.reconstruction, timeseries)
                observed_mask = input_mask * (1 - dec_out.pretrain_mask)
                n_channels = dec_out.reconstruction.shape[1]
                observed_mask = observed_mask.unsqueeze(1).repeat((1, n_channels, 1))
                masked_loss = observed_mask * recon_loss
                loss_moment = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)



                loss_t = self.nt_xent_poly(h_t, h_t_aug)
                loss_f = self.nt_xent_poly(h_f, h_f_aug)
                l_TF = self.nt_xent_poly(z_t, z_f) # this is the initial version of TF loss

                l_1, l_2, l_3 = self.nt_xent_poly(z_t, z_f_aug), self.nt_xent_poly(z_t_aug, z_f), self.nt_xent_poly(z_t_aug, z_f_aug)
                loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

                lam = 0.2
                loss_tfc = lam*(loss_t + loss_f) + l_TF

                lam2 = 0.1
                total_loss = loss_moment + lam2*loss_tfc

                self.logger.log(
                    {
                        "loss_tfc": loss_tfc.item(),
                        "loss_moment": loss_moment.item(),
                        "step_train_loss": total_loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

                if self.args.debug and opt_steps >= 1:
                    self.debug_model_outputs(loss_moment, dec_out, batch_x)

                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                # Gradient Clipping
                nn.utils.clip_grad_norm_(
                    list(self.encoder_re.parameters())
                    + list(self.decoder.parameters())
                    + list(self.tfc.parameters()),
                    self.args.max_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                opt_steps = opt_steps + 1

                # if opt_steps % self.args.log_interval == 0:
                #     self.evaluate_and_log()

                if opt_steps % self.args.checkpoint_interval == 0:
                    self.logger.alert(
                        title="Saving model",
                        text=f"Saving model after {opt_steps} steps",
                        level=AlertLevel.INFO,
                    )
                    self.save_model(
                        models={
                            "encoder_re": self.encoder_re,
                            "decoder": self.decoder,
                            "tfc": self.tfc,
                        },
                        path=path,
                        opt_steps=opt_steps,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                    )

                    self.evaluate_model_external(path, opt_steps, self.device)
                self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)

            cur_epoch = cur_epoch + 1

        return {
            "encoder_re": self.encoder_re,
            "decoder": self.decoder,
            "tfc": self.tfc,
        }

