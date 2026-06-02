"""ContraReg-style frozen autoencoder pair for registration finetuning."""

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn


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
    """Two frozen autoencoders; encode runs without no_grad so grads reach inputs."""

    def __init__(self, fixed_encoder: nn.Module, moving_encoder: nn.Module):
        super().__init__()
        self.fixed_encoder = fixed_encoder
        self.moving_encoder = moving_encoder
        self._freeze_encoder(self.fixed_encoder)
        self._freeze_encoder(self.moving_encoder)

    @staticmethod
    def _freeze_encoder(encoder: nn.Module) -> None:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    def encode_fixed(self, x: torch.Tensor) -> List[torch.Tensor]:
        return _encode_without_grad_block(self.fixed_encoder, x)

    def encode_moving(self, x: torch.Tensor) -> List[torch.Tensor]:
        return _encode_without_grad_block(self.moving_encoder, x)


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
        "contrareg_fixed_ae_checkpoint": settings.contrareg_fixed_ae_checkpoint,
        "contrareg_moving_ae_checkpoint": settings.contrareg_moving_ae_checkpoint,
    }
