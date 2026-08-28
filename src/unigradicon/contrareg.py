"""ContraReg-style frozen autoencoder pair for registration finetuning."""

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from .contrastive import PREOP_MODALITIES, US_MODALITIES


def _encode_without_grad_block(encoder: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """Run encoder forward without no_grad so gradients reach inputs."""
    if hasattr(encoder, "encode") and callable(getattr(encoder, "encode")):
        features = encoder.encode(x)
    else:
        output = encoder(x)
        if (
            isinstance(output, (list, tuple))
            and len(output) == 2
            and isinstance(output[1], (list, tuple))
        ):
            features = output[1]
        elif isinstance(output, (list, tuple)):
            features = output
        else:
            raise TypeError(f"Unsupported encoder output type: {type(output)}")
    if not isinstance(features, (list, tuple)):
        raise TypeError("Encoder must return a sequence of feature maps.")
    return list(features)


class FrozenContraRegEncoderPair(nn.Module):
    """Two frozen per-modality autoencoders; encode runs without no_grad so
    grads reach inputs.

    Encoders are keyed by modality (preop vs US), not by registration role
    (fixed vs moving): the paired sampler can put either modality in the
    A/moving or B/fixed slot, so routing must follow each image's modality
    string rather than its role in the pair.
    """

    def __init__(self, preop_encoder: nn.Module, us_encoder: nn.Module):
        super().__init__()
        self.preop_encoder = preop_encoder
        self.us_encoder = us_encoder
        self._freeze_encoder(self.preop_encoder)
        self._freeze_encoder(self.us_encoder)

    @staticmethod
    def _freeze_encoder(encoder: nn.Module) -> None:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "FrozenContraRegEncoderPair":
        """Keep the frozen encoders in eval mode even when the parent network
        is switched to train mode (``net.train()`` recurses into children)."""
        super().train(mode)
        self.preop_encoder.eval()
        self.us_encoder.eval()
        return self

    @staticmethod
    def resolve_modality_key(modality: str) -> str:
        """Map a raw modality string to 'preop' or 'us'; raise on unknown."""
        normalized = (modality or "").lower()
        if normalized in PREOP_MODALITIES:
            return "preop"
        if normalized in US_MODALITIES:
            return "us"
        raise ValueError(
            f"Unknown modality '{modality}' for ContraReg encoder routing. "
            f"Expected one of {sorted(PREOP_MODALITIES | US_MODALITIES)}."
        )

    def encode(self, x: torch.Tensor, modality: str) -> List[torch.Tensor]:
        key = self.resolve_modality_key(modality)
        encoder = self.preop_encoder if key == "preop" else self.us_encoder
        return _encode_without_grad_block(encoder, x)


def contrareg_config_snapshot(settings: Any) -> Dict[str, Any]:
    """Extract ContraReg-related training settings for checkpoint metadata."""
    return {
        "contrareg_enabled": settings.contrareg_enabled,
        "contrareg_weight": settings.contrareg_weight,
        "contrareg_num_patches": settings.contrareg_num_patches,
        "contrareg_temperature": settings.contrareg_temperature,
        "contrareg_embed_dim": settings.contrareg_embed_dim,
        "contrareg_feature_channels": list(settings.contrareg_feature_channels),
        "contrareg_bidirectional": settings.contrareg_bidirectional,
        "contrareg_roi_patches": settings.contrareg_roi_patches,
        "contrareg_preop_ae_checkpoint": settings.contrareg_preop_ae_checkpoint,
        "contrareg_us_ae_checkpoint": settings.contrareg_us_ae_checkpoint,
    }
