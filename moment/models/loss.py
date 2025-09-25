import torch
import torch.nn.functional as F
from argparse import Namespace
import logging
import numpy as np

from torch import nn
from moment.utils.utils import (
    NamespaceWithDefaults,
    get_anomaly_criterion,
    get_huggingface_model_dimensions,
)

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

class NTXentLoss(torch.nn.Module):
    def __init__(self, configs: Namespace | dict, **kwargs: dict):
        super().__init__()
        configs = self._update_inputs(configs, **kwargs)
        configs = self._validate_inputs(configs)
        self.batch_size = configs.train_batch_size
        self.temperature = configs.temperature
        self.device = configs.device
        self.use_cosine_similarity = configs.use_cosine_similarity

        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(self.use_cosine_similarity)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
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

class NTXentLoss_poly(nn.Module):
    def __init__(self, configs: Namespace | dict):
        super().__init__()
        configs = self._update_inputs(configs)
        configs = self._validate_inputs(configs)
        self.batch_size = configs.train_batch_size
        self.temperature = configs.temperature
        self.device = configs.device
        self.use_cosine_similarity = configs.use_cosine_similarity

        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(self.use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),torch.zeros(2 * self.batch_size, negatives.shape[-1])),dim=-1).to(self.device).long()
        # Add poly loss
        pt = torch.mean(onehot_label* torch.nn.functional.softmax(logits,dim=-1))

        epsilon = self.batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1/self.batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss

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

