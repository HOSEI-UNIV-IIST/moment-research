
import os
import subprocess
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import autocast
from tqdm import tqdm
from wandb import AlertLevel

from moment.common import PATHS
from moment.models.moment import MOMENT
from moment.utils.utils import MetricsStore, dtype_map, make_dir_if_not_exists

from .base import Tasks
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

class TSNE_(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        print(f"[DEBUG TSNE] Train dataset size: {len(self.train_dataloader.dataset)}")


    def train(self):
        self.run_name = self.logger.name
        
        self.checkpoint_path = os.path.join(self.args.checkpoint_path, self.run_name)
        make_dir_if_not_exists(self.checkpoint_path, verbose=True)

        self.results_dir = self._create_results_dir(
            experiment_name="tsne"
        )

        # Load pre-trained MOMENT model before fine-tuning
        if self.args.model_name == "MOMENT" and self.args.pretraining_run_name != "null":
            self.load_pretrained_moment()
        self.model.to(self.device)

        for batch_x in tqdm(
            self.train_dataloader, total=len(self.train_dataloader)
        ):
            timeseries = batch_x.timeseries.float().to(self.device)

            scaler = self.train_dataloader.dataset.scaler
            ts_scaled = batch_x.timeseries.cpu().numpy()
            
            B, C, T = ts_scaled.shape
            ts_flat = ts_scaled.transpose(0, 2, 1).reshape(-1, C)   # [B*T, C]

            ts_unscaled_flat = scaler.inverse_transform(ts_flat)
            ts_unscaled = ts_unscaled_flat.reshape(B, T, C).transpose(0, 2, 1)  # [B, C, T]
            ts_unscaled = torch.from_numpy(ts_unscaled).to(self.device).float()

            self.original_seq_plot(ts_unscaled, batch_x.metadata)

            outputs = self.model(
            x_enc=timeseries, input_mask=None, mask=None
            )

            self.tsne_plot(outputs.embeddings, batch_x.metadata)

    
    def original_seq_plot(self, ts_unscaled, metadata):
        # [sb] original time series plot
        # label:0 front time series distribution
        # label:1 back time series distribution

        ts_unscaled = ts_unscaled.detach().cpu().numpy()  # [B, C, T]

        # 분할
        group_0 = [(ts_unscaled[i], meta) for i, meta in enumerate(metadata) if meta["group_label"] == 0]
        group_1 = [(ts_unscaled[i], meta) for i, meta in enumerate(metadata) if meta["group_label"] == 1]

        ncols = max(len(group_0), len(group_1))
        fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(2 * ncols, 6), sharex=True, sharey=True)

        for col, (ts, meta) in enumerate(group_0):
            ax = axes[0, col]
            ax.plot(ts[0])
            ax.set_title(f"Group 0 - Start {meta['orig_start']}")

        for col, (ts, meta) in enumerate(group_1):
            ax = axes[1, col]
            ax.plot(ts[0])
            ax.set_title(f"Group 1 - Start {meta['orig_start']}")

        plt.suptitle("Original Unscaled Time Series by Group", fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(self.checkpoint_path,self.args.pretraining_run_name, "original_seq_plot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[Saved] Original Sequence Plot saved to {save_path}")
    
    def tsne_plot(self, embeddings: torch.Tensor, metadata: list[dict]):
        """
        embeddings: torch.Tensor [B, D] - latent representation
        metadata: list of dicts with "group_label" and optionally "orig_start"
        """
       
        embeddings_np = embeddings.detach().cpu().numpy()  # [B, D]
        labels = np.array([meta["group_label"] for meta in metadata])
        starts = np.array([meta.get("orig_start", -1) for meta in metadata])


        tsne = TSNE(n_components=2, random_state=42, perplexity=5)
        emb_2d = tsne.fit_transform(embeddings_np)  # [B, 2]


        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="coolwarm", s=80, edgecolor='k')

        for i, txt in enumerate(starts):
            plt.annotate(str(txt), (emb_2d[i, 0], emb_2d[i, 1]), fontsize=8, alpha=0.6)
        # plt.colorbar(scatter, label="Group Label")/
        # plt.grid(True)

        save_path = os.path.join(self.checkpoint_path,self.args.pretraining_run_name, "tsne_embedding_plot.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Saved] t-SNE plot saved to {save_path}")

