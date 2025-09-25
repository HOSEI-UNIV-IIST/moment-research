import logging
import warnings
from argparse import Namespace
from math import ceil

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


from moment.common import TASKS
from moment.data.base import TimeseriesOutputs
from moment.utils.masking import Masking
from moment.utils.utils import (
    NamespaceWithDefaults,
    get_anomaly_criterion,
    get_huggingface_model_dimensions,
)

from .layers.embed import PatchEmbedding, Patching
from .layers.revin import RevIN

SUPPORTED_HUGGINGFACE_MODELS = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x seq_len], where seq_len = n_patches * patch_len
        """
        # x = x.transpose(2, 3)                 # [batch_size x n_channels x n_patches x d_model]
        x = self.linear(
            self.dropout(x)
        )  # [batch_size x n_channels x n_patches x patch_len]
        x = x.flatten(start_dim=2, end_dim=3)  # [batch_size x n_patches x seq_len]
        return x


class MOMENT_re(nn.Module):
    def __init__(self, configs: Namespace | dict, **kwargs: dict):
        super().__init__()
        configs = self._update_inputs(configs, **kwargs)
        configs = self._validate_inputs(configs)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len

        # Normalization, patching and embedding
        self.normalizer = RevIN(
            num_features=1, affine=configs.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=configs.patch_len, stride=configs.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            seq_len=configs.seq_len,
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=configs.getattr("dropout", 0.1),
            add_positional_embedding=configs.getattr("add_positional_embedding", True),
            value_embedding_bias=configs.getattr("value_embedding_bias", False),
            orth_gain=configs.getattr("orth_gain", 1.41),
        ).to(configs.device)
        self.mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0))

        # Transformer backbone
        self.encoder = self._get_transformer_backbone(configs)

        # Prediction Head
        self.head = self._get_head(self.task_name)

    def _update_inputs(
        self, configs: Namespace | dict, **kwargs
    ) -> NamespaceWithDefaults:
        if isinstance(configs, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**configs, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(configs)

    def _validate_inputs(self, configs: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            configs.transformer_backbone == "PatchTST"
            and configs.transformer_type != "encoder_only"
        ):
            warnings.warn("PatchTST only supports encoder-only transformer backbones.")
            configs.transformer_type = "encoder_only"
        if (
            configs.transformer_backbone != "PatchTST"
            and configs.transformer_backbone not in SUPPORTED_HUGGINGFACE_MODELS
        ):
            raise NotImplementedError(
                f"Transformer backbone {configs.transformer_backbone} not supported."
                f"Please choose from {SUPPORTED_HUGGINGFACE_MODELS} or PatchTST."
            )
        if (
            configs.d_model is None
            and configs.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            configs.d_model = get_huggingface_model_dimensions(
                configs.transformer_backbone
            )
            logging.info("Setting d_model to {}".format(configs.d_model))
        elif configs.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone \
                             unless transformer backbone is a Huggingface model."
            )

        if configs.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of ['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if configs.patch_stride_len != configs.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return configs

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name in {
            TASKS.PRETRAINING,
            TASKS.ANOMALY_DETECTION,
            TASKS.IMPUTATION,
        } or (
            task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.finetuning_mode == "zero-shot"
        ):
            return PretrainHead(
                self.configs.d_model,
                self.configs.patch_len,
                self.configs.getattr("dropout", 0.1),
                self.configs.getattr("orth_gain", 1.41),
            )
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, configs):
        if configs.transformer_backbone == "PatchTST":
            return self._get_patchtst_encoder(configs)
        else:
            return self._get_huggingface_transformer(configs)

    def _get_huggingface_transformer(self, configs):
        from transformers import T5Config, T5EncoderModel, T5Model

        if configs.getattr("randomly_initialize_backbone", False):
            model_config = T5Config.from_pretrained(configs.transformer_backbone)
            transformer_backbone = T5Model(model_config)
            logging.info(f"Initializing randomly initialized\
                          transformer from {configs.transformer_backbone}.")
        else:
            transformer_backbone = T5EncoderModel.from_pretrained(
                configs.transformer_backbone
            )
            logging.info(f"Initializing pre-trained \
                          transformer from {configs.transformer_backbone}.")

        if configs.transformer_type == "encoder_only":
            transformer_backbone = transformer_backbone.get_encoder()
        elif configs.transformer_type == "decoder_only":
            transformer_backbone = transformer_backbone.get_decoder()

        if configs.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def _get_patchtst_encoder(self, configs):
        from .layers.self_attention_family import AttentionLayer, FullAttention
        from .layers.transformer_encoder_decoder import Encoder, EncoderLayer

        encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            attention_dropout=configs.attention_dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        return encoder

#### pretrain moment encoder for reconstruction
    def pretraining(
        self,
        x_enc_re: torch.Tensor,
        x_enc_t2: torch.Tensor,
        x_enc_f1: torch.Tensor,
        x_enc_f2: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        mask_t2: torch.Tensor = None,
        mask_f1: torch.Tensor = None,
        mask_f2: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        batch_size, n_channels, _ = x_enc_re.shape
        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc_re, input_mask=input_mask)
            mask = mask.to(x_enc_re.device)  # mask: [batch_size x seq_len]

        # Normalization
        x_enc_re = self.normalizer(x=x_enc_re, mask=mask * input_mask, mode="norm")
        # x_enc = self.normalizer(x=x_enc, mask=input_mask, mode='norm')
        x_enc_re = torch.nan_to_num(x_enc_re, nan=0, posinf=0, neginf=0)
        # Some time-series are too short, so masking them out results in NaNs.

        # [batch_size x n_channels x seq_len]
        x_enc_re = self.tokenizer(x=x_enc_re)
        x_enc_t2 = self.tokenizer(x=x_enc_t2)
        x_enc_f1 = self.tokenizer(x=x_enc_f1)
        x_enc_f2 = self.tokenizer(x=x_enc_f2)
        # [batch_size x n_channels x n_patches x patch_len]

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc_re, mask=mask)
        enc_t2 = self.patch_embedding(x_enc_t2, mask=mask_t2)
        enc_f1 = self.patch_embedding(x_enc_f1, mask=mask_f1)
        enc_f2 = self.patch_embedding(x_enc_f2, mask=mask_f2)


        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        enc_t2 = enc_t2.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        enc_f1 = enc_f1.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        enc_f2 = enc_f2.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        # Encoder
        attention_mask = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        ).repeat_interleave(n_channels, dim=0)
        output1= self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out1 = output1.last_hidden_state
        enc_out1 = enc_out1.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]
        output2 = self.encoder(inputs_embeds=enc_t2)
        enc_out2 = output2.last_hidden_state
        enc_out2 = enc_out2.reshape((-1, n_channels, n_patches, self.configs.d_model))
        output3 = self.encoder(inputs_embeds=enc_f1)
        enc_out3 = output3.last_hidden_state
        enc_out3 = enc_out3.reshape((-1, n_channels, n_patches, self.configs.d_model))
        output4 = self.encoder(inputs_embeds=enc_f2)
        enc_out4 = output4.last_hidden_state
        enc_out4 = enc_out4.reshape((-1, n_channels, n_patches, self.configs.d_model))
        return (enc_out1, enc_out2, enc_out3, enc_out4, input_mask, mask, self.normalizer.mean, self.normalizer.stdev)


    def initialize_soft_prompt(self, **kwargs):
        n_soft_prompt_tokens = self.configs.n_soft_prompt_tokens
        self.soft_prompt = nn.Embedding(n_soft_prompt_tokens, self.configs.d_model)
        return self.soft_prompt

    def _cat_learned_embedding_to_input(self, prompt_embeds, enc_in) -> torch.Tensor:
        prompt_embeds = prompt_embeds.repeat(enc_in.size(0), 1, 1)
        enc_in = torch.cat([prompt_embeds, enc_in], dim=1)
        return enc_in

    def _extend_attention_mask(self, attention_mask, n_tokens):
        n_batches = attention_mask.shape[0]
        extension = torch.full((n_batches, n_tokens), 1).to(self.configs.device)
        return torch.cat([extension, attention_mask], dim=1)

    def reconstruct(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        if mask is None:
            mask = torch.ones_like(input_mask)

        batch_size, n_channels, _ = x_enc.shape
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        attention_mask = (
            Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
            .repeat_interleave(n_channels, dim=0)
            .to(x_enc.device)
        )

        n_tokens = 0
        if "prompt_embeds" in kwargs:
            prompt_embeds = kwargs["prompt_embeds"].to(x_enc.device)

            if isinstance(prompt_embeds, nn.Embedding):
                prompt_embeds = prompt_embeds.weight.data.unsqueeze(0)

            n_tokens = prompt_embeds.shape[1]

            enc_in = self._cat_learned_embedding_to_input(prompt_embeds, enc_in)

            attention_mask = self._extend_attention_mask(attention_mask, n_tokens)

        # Encoder
        if self.configs.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out[:, n_tokens:, :]

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out)

    def forward(
        self,
        x_enc_re: torch.Tensor,
        x_enc_t2: torch.Tensor,
        x_enc_f1: torch.Tensor,
        x_enc_f2: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        mask_t2: torch.Tensor = None,
        mask_f1: torch.Tensor = None,
        mask_f2: torch.Tensor = None,
        **kwargs,
    ):
        if self.task_name == TASKS.PRETRAINING:
            return self.pretraining(
                x_enc_re=x_enc_re, x_enc_t2=x_enc_t2, x_enc_f1=x_enc_f1, x_enc_f2=x_enc_f2, mask=mask, mask_t2=mask_t2, mask_f1=mask_f1, mask_f2=mask_f2, input_mask=input_mask, **kwargs
            )
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
        return

    def _check_model_weights_for_illegal_values(self):
        illegal_encoder_weights = (
            torch.stack([torch.isnan(p).any() for p in self.encoder.parameters()])
            .any()
            .item()
        )
        illegal_head_weights = (
            torch.stack([torch.isnan(p).any() for p in self.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_weights = (
            torch.stack(
                [torch.isnan(p).any() for p in self.patch_embedding.parameters()]
            )
            .any()
            .item()
        )

        return (
            illegal_encoder_weights
            or illegal_head_weights
            or illegal_patch_embedding_weights
        )




class MOMENT(nn.Module):
    def __init__(self, configs: Namespace | dict, **kwargs: dict):
        super().__init__()
        configs = self._update_inputs(configs, **kwargs)
        configs = self._validate_inputs(configs)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len

        # Normalization, patching and embedding
        self.normalizer = RevIN(
            num_features=1, affine=configs.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=configs.patch_len, stride=configs.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            seq_len=configs.seq_len,
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=configs.getattr("dropout", 0.1),
            add_positional_embedding=configs.getattr("add_positional_embedding", True),
            value_embedding_bias=configs.getattr("value_embedding_bias", False),
            orth_gain=configs.getattr("orth_gain", 1.41),
        ).to(configs.device)
        self.mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0))

        # Transformer backbone
        self.encoder = self._get_transformer_backbone(configs)

        # Prediction Head
        self.head = self._get_head(self.task_name)

    def _update_inputs(
        self, configs: Namespace | dict, **kwargs
    ) -> NamespaceWithDefaults:
        if isinstance(configs, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**configs, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(configs)

    def _validate_inputs(self, configs: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            configs.transformer_backbone == "PatchTST"
            and configs.transformer_type != "encoder_only"
        ):
            warnings.warn("PatchTST only supports encoder-only transformer backbones.")
            configs.transformer_type = "encoder_only"
        if (
            configs.transformer_backbone != "PatchTST"
            and configs.transformer_backbone not in SUPPORTED_HUGGINGFACE_MODELS
        ):
            raise NotImplementedError(
                f"Transformer backbone {configs.transformer_backbone} not supported."
                f"Please choose from {SUPPORTED_HUGGINGFACE_MODELS} or PatchTST."
            )
        if (
            configs.d_model is None
            and configs.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            configs.d_model = get_huggingface_model_dimensions(
                configs.transformer_backbone
            )
            logging.info("Setting d_model to {}".format(configs.d_model))
        elif configs.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone \
                             unless transformer backbone is a Huggingface model."
            )

        if configs.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of ['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if configs.patch_stride_len != configs.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return configs

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name in {
            TASKS.PRETRAINING,
            TASKS.ANOMALY_DETECTION,
            TASKS.IMPUTATION,
        } or (
            task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.finetuning_mode == "zero-shot"
        ):
            return PretrainHead(
                self.configs.d_model,
                self.configs.patch_len,
                self.configs.getattr("dropout", 0.1),
                self.configs.getattr("orth_gain", 1.41),
            )
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, configs):
        if configs.transformer_backbone == "PatchTST":
            return self._get_patchtst_encoder(configs)
        else:
            return self._get_huggingface_transformer(configs)

    def _get_huggingface_transformer(self, configs):
        from transformers import T5Config, T5EncoderModel, T5Model

        if configs.getattr("randomly_initialize_backbone", False):
            model_config = T5Config.from_pretrained(configs.transformer_backbone)
            transformer_backbone = T5Model(model_config)
            logging.info(f"Initializing randomly initialized\
                          transformer from {configs.transformer_backbone}.")
        else:
            transformer_backbone = T5EncoderModel.from_pretrained(
                configs.transformer_backbone
            )
            logging.info(f"Initializing pre-trained \
                          transformer from {configs.transformer_backbone}.")

        if configs.transformer_type == "encoder_only":
            transformer_backbone = transformer_backbone.get_encoder()
        elif configs.transformer_type == "decoder_only":
            transformer_backbone = transformer_backbone.get_decoder()

        if configs.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def _get_patchtst_encoder(self, configs):
        from .layers.self_attention_family import AttentionLayer, FullAttention
        from .layers.transformer_encoder_decoder import Encoder, EncoderLayer

        encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            attention_dropout=configs.attention_dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        return encoder

#### pretrain moment encoder for others
    def pretraining(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        """
        batch_size, n_channels, _ = x_enc.shape


        # [batch_size x n_channels x seq_len]
        x_enc = self.tokenizer(x=x_enc)
        # [batch_size x n_channels x n_patches x patch_len]

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        # Encoder

        outputs = self.encoder(inputs_embeds=enc_in)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]
        return (enc_out)


    def initialize_soft_prompt(self, **kwargs):
        n_soft_prompt_tokens = self.configs.n_soft_prompt_tokens
        self.soft_prompt = nn.Embedding(n_soft_prompt_tokens, self.configs.d_model)
        return self.soft_prompt

    def _cat_learned_embedding_to_input(self, prompt_embeds, enc_in) -> torch.Tensor:
        prompt_embeds = prompt_embeds.repeat(enc_in.size(0), 1, 1)
        enc_in = torch.cat([prompt_embeds, enc_in], dim=1)
        return enc_in

    def _extend_attention_mask(self, attention_mask, n_tokens):
        n_batches = attention_mask.shape[0]
        extension = torch.full((n_batches, n_tokens), 1).to(self.configs.device)
        return torch.cat([extension, attention_mask], dim=1)

    def reconstruct(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        if mask is None:
            mask = torch.ones_like(input_mask)

        batch_size, n_channels, _ = x_enc.shape
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        attention_mask = (
            Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
            .repeat_interleave(n_channels, dim=0)
            .to(x_enc.device)
        )

        n_tokens = 0
        if "prompt_embeds" in kwargs:
            prompt_embeds = kwargs["prompt_embeds"].to(x_enc.device)

            if isinstance(prompt_embeds, nn.Embedding):
                prompt_embeds = prompt_embeds.weight.data.unsqueeze(0)

            n_tokens = prompt_embeds.shape[1]

            enc_in = self._cat_learned_embedding_to_input(prompt_embeds, enc_in)

            attention_mask = self._extend_attention_mask(attention_mask, n_tokens)

        # Encoder
        if self.configs.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out[:, n_tokens:, :]

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out)

    def forward(
        self,
        x_enc: torch.Tensor,
        mask: torch.Tensor = None,
        input_mask: torch.Tensor = None,
        **kwargs,
    ):
        if self.task_name == TASKS.PRETRAINING:
            return self.pretraining(
                x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs
            )
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
        return

    def _check_model_weights_for_illegal_values(self):
        illegal_encoder_weights = (
            torch.stack([torch.isnan(p).any() for p in self.encoder.parameters()])
            .any()
            .item()
        )
        illegal_head_weights = (
            torch.stack([torch.isnan(p).any() for p in self.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_weights = (
            torch.stack(
                [torch.isnan(p).any() for p in self.patch_embedding.parameters()]
            )
            .any()
            .item()
        )

        return (
            illegal_encoder_weights
            or illegal_head_weights
            or illegal_patch_embedding_weights
        )

class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        """
        x: [batch_size x n_channels x n_patches x d_model]
        output: [batch_size x n_channels x seq_len], where seq_len = n_patches * patch_len
        """
        # x = x.transpose(2, 3)                 # [batch_size x n_channels x n_patches x d_model]
        x = self.linear(
            self.dropout(x)
        )  # [batch_size x n_channels x n_patches x patch_len]
        x = x.flatten(start_dim=2, end_dim=3)  # [batch_size x n_patches x seq_len]
        return x

class MOMENTDecoder(nn.Module):
    def __init__(self, configs: Namespace | dict, **kwargs: dict):
        super().__init__()
        configs = self._update_inputs(configs, **kwargs)
        configs = self._validate_inputs(configs)
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len

        # Normalization, patching and embedding
        self.normalizer = RevIN(
            num_features=1, affine=configs.getattr("revin_affine", False)
        )
        self.tokenizer = Patching(
            patch_len=configs.patch_len, stride=configs.patch_stride_len
        )
        self.patch_embedding = PatchEmbedding(
            d_model=configs.d_model,
            seq_len=configs.seq_len,
            patch_len=configs.patch_len,
            stride=configs.patch_stride_len,
            dropout=configs.getattr("dropout", 0.1),
            add_positional_embedding=configs.getattr("add_positional_embedding", True),
            value_embedding_bias=configs.getattr("value_embedding_bias", False),
            orth_gain=configs.getattr("orth_gain", 1.41),
        ).to(configs.device)
        self.mask_generator = Masking(mask_ratio=configs.getattr("mask_ratio", 0.0))

        # Transformer backbone
        self.encoder = self._get_transformer_backbone(configs)

        # Prediction Head
        self.head = self._get_head(self.task_name)

    def _update_inputs(
        self, configs: Namespace | dict, **kwargs
    ) -> NamespaceWithDefaults:
        if isinstance(configs, dict) and "model_kwargs" in kwargs:
            return NamespaceWithDefaults(**{**configs, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(configs)

    def _validate_inputs(self, configs: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            configs.transformer_backbone == "PatchTST"
            and configs.transformer_type != "encoder_only"
        ):
            warnings.warn("PatchTST only supports encoder-only transformer backbones.")
            configs.transformer_type = "encoder_only"
        if (
            configs.transformer_backbone != "PatchTST"
            and configs.transformer_backbone not in SUPPORTED_HUGGINGFACE_MODELS
        ):
            raise NotImplementedError(
                f"Transformer backbone {configs.transformer_backbone} not supported."
                f"Please choose from {SUPPORTED_HUGGINGFACE_MODELS} or PatchTST."
            )
        if (
            configs.d_model is None
            and configs.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            configs.d_model = get_huggingface_model_dimensions(
                configs.transformer_backbone
            )
            logging.info("Setting d_model to {}".format(configs.d_model))
        elif configs.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone \
                             unless transformer backbone is a Huggingface model."
            )

        if configs.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of ['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if configs.patch_stride_len != configs.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return configs

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name in {
            TASKS.PRETRAINING,
            TASKS.ANOMALY_DETECTION,
            TASKS.IMPUTATION,
        } or (
            task_name == TASKS.SHORT_HORIZON_FORECASTING
            and self.configs.finetuning_mode == "zero-shot"
        ):
            return PretrainHead(
                self.configs.d_model,
                self.configs.patch_len,
                self.configs.getattr("dropout", 0.1),
                self.configs.getattr("orth_gain", 1.41),
            )
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, configs):
        if configs.transformer_backbone == "PatchTST":
            return self._get_patchtst_encoder(configs)
        else:
            return self._get_huggingface_transformer(configs)

    def _get_huggingface_transformer(self, configs):
        from transformers import T5Config, T5EncoderModel, T5Model

        if configs.getattr("randomly_initialize_backbone", False):
            model_config = T5Config.from_pretrained(configs.transformer_backbone)
            transformer_backbone = T5Model(model_config)
            logging.info(f"Initializing randomly initialized\
                          transformer from {configs.transformer_backbone}.")
        else:
            transformer_backbone = T5EncoderModel.from_pretrained(
                configs.transformer_backbone
            )
            logging.info(f"Initializing pre-trained \
                          transformer from {configs.transformer_backbone}.")

        if configs.transformer_type == "encoder_only":
            transformer_backbone = transformer_backbone.get_encoder()
        elif configs.transformer_type == "decoder_only":
            transformer_backbone = transformer_backbone.get_decoder()

        if configs.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def _get_patchtst_encoder(self, configs):
        from .layers.self_attention_family import AttentionLayer, FullAttention
        from .layers.transformer_encoder_decoder import Encoder, EncoderLayer

        encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            attention_dropout=configs.attention_dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        return encoder

#### pretrain moment decoder  
    def pretraining(
        self,
        enc_out: torch.Tensor,
        input_mask: torch.Tensor = None,
        pretrain_mask: torch.Tensor = None,
        mean: torch.Tensor = None,
        stdev: torch.Tensor = None,
        **kwargs,
    ):
        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]

        # De-Normalization
        # decoder 내부에서 mean, stdev 수동 세팅
        self.normalizer.mean = mean
        self.normalizer.stdev = stdev
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        illegal_output = (
            self._check_model_weights_for_illegal_values()
            if self.configs.debug
            else None
        )

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            pretrain_mask=pretrain_mask,
            illegal_output=illegal_output,
        )


    def initialize_soft_prompt(self, **kwargs):
        n_soft_prompt_tokens = self.configs.n_soft_prompt_tokens
        self.soft_prompt = nn.Embedding(n_soft_prompt_tokens, self.configs.d_model)
        return self.soft_prompt

    def _cat_learned_embedding_to_input(self, prompt_embeds, enc_in) -> torch.Tensor:
        prompt_embeds = prompt_embeds.repeat(enc_in.size(0), 1, 1)
        enc_in = torch.cat([prompt_embeds, enc_in], dim=1)
        return enc_in

    def _extend_attention_mask(self, attention_mask, n_tokens):
        n_batches = attention_mask.shape[0]
        extension = torch.full((n_batches, n_tokens), 1).to(self.configs.device)
        return torch.cat([extension, attention_mask], dim=1)

    def reconstruct(
        self,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        x_enc : [batch_size x n_channels x seq_len]
            Time-series data
        mask  : [batch_size x seq_len]
            Data that is masked but still attended to via
            mask-tokens
        input_mask : [batch_size x seq_len]
            Input mask for the time-series data that is
            unobserved. This is typically padded data,
            that is not attended to.
        """
        if mask is None:
            mask = torch.ones_like(input_mask)

        batch_size, n_channels, _ = x_enc.shape
        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)

        # Patching and embedding
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.configs.d_model)
        )
        # [batch_size * n_channels x n_patches x d_model]

        attention_mask = (
            Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
            .repeat_interleave(n_channels, dim=0)
            .to(x_enc.device)
        )

        n_tokens = 0
        if "prompt_embeds" in kwargs:
            prompt_embeds = kwargs["prompt_embeds"].to(x_enc.device)

            if isinstance(prompt_embeds, nn.Embedding):
                prompt_embeds = prompt_embeds.weight.data.unsqueeze(0)

            n_tokens = prompt_embeds.shape[1]

            enc_in = self._cat_learned_embedding_to_input(prompt_embeds, enc_in)

            attention_mask = self._extend_attention_mask(attention_mask, n_tokens)

        # Encoder
        if self.configs.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        enc_out = enc_out[:, n_tokens:, :]

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.configs.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        # Decoder
        dec_out = self.head(enc_out)  # z: [batch_size x n_channels x seq_len]

        # De-Normalization
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, reconstruction=dec_out)

    def forward(
        self,
        enc_out: torch.Tensor,
        input_mask: torch.Tensor = None,
        pretrain_mask: torch.Tensor = None,
        **kwargs,
    ):
        if self.task_name == TASKS.PRETRAINING:
            return self.pretraining(
                enc_out=enc_out, input_mask=input_mask, pretrain_mask=pretrain_mask, **kwargs
            )
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")
        return

    def _check_model_weights_for_illegal_values(self):
        illegal_encoder_weights = (
            torch.stack([torch.isnan(p).any() for p in self.encoder.parameters()])
            .any()
            .item()
        )
        illegal_head_weights = (
            torch.stack([torch.isnan(p).any() for p in self.head.parameters()])
            .any()
            .item()
        )
        illegal_patch_embedding_weights = (
            torch.stack(
                [torch.isnan(p).any() for p in self.patch_embedding.parameters()]
            )
            .any()
            .item()
        )

        return (
            illegal_encoder_weights
            or illegal_head_weights
            or illegal_patch_embedding_weights
        )


class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()
        self.projection = nn.Linear(49152, 512)
        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=2, )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,nhead=2,)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        B, C, P, D = x_in_t.shape
        print(x_in_t.shape)
        x_in_t = x_in_t.reshape(B, C, P * D) 
        x_in_t = x_in_t.view(B * C, P * D)    # [B * C, 49152]
        x_in_t = self.projection(x_in_t)     # [B * C, 512]
        x_in_t = x_in_t.view(B, C, 512)
        # [batch_size x n_channels * d_model x n_patches]
        print(x_in_t.shape)
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)
        
        B, C, P, D = x_in_f.shape
        x_in_f = x_in_f.reshape(B, C, P * D) 
        x_in_f = x_in_f.view(B * C, P * D)    # [B * C, 49152]
        x_in_f = self.projection(x_in_f)     # [B * C, 512]
        x_in_f = x_in_f.view(B, C, 512)
        # [batch_size x n_channels * d_model x n_patches]
        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq