from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_PREOP_MODALITIES = {"ct", "mri", "mr", "preop"}
_US_MODALITIES = {"us", "ultrasound"}


def _extract_state_dict(raw_state):
    if not isinstance(raw_state, dict):
        raise ValueError("Expected checkpoint dict with a 'state_dict' or 'model_state_dict' key.")
    if "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
        return raw_state["state_dict"]
    if "model_state_dict" in raw_state and isinstance(raw_state["model_state_dict"], dict):
        return raw_state["model_state_dict"]
    raise ValueError("Checkpoint dict must include 'state_dict' or 'model_state_dict'.")


class FrozenContrastiveFeatureExtractor(nn.Module):
    def __init__(
        self,
        preop_encoder: nn.Module,
        us_encoder: nn.Module,
        preop_checkpoint: Optional[str] = None,
        us_checkpoint: Optional[str] = None,
        feature_level: int = -2,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.preop_encoder = preop_encoder
        self.us_encoder = us_encoder
        self.feature_level = feature_level
        self.normalize_features = normalize_features

        if preop_checkpoint:
            self._load_checkpoint(self.preop_encoder, preop_checkpoint)
        if us_checkpoint:
            self._load_checkpoint(self.us_encoder, us_checkpoint)

        self._freeze_encoder(self.preop_encoder)
        self._freeze_encoder(self.us_encoder)

    @staticmethod
    def _load_checkpoint(encoder: nn.Module, checkpoint_path: str) -> None:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(state)
        encoder.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _freeze_encoder(encoder: nn.Module) -> None:
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    @staticmethod
    def _select_from_output(output, feature_level: int):
        if torch.is_tensor(output):
            return output
        if (
            isinstance(output, (list, tuple))
            and len(output) == 2
            and torch.is_tensor(output[0])
            and isinstance(output[1], (list, tuple))
        ):
            output = output[1]
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise ValueError("Encoder output is empty; cannot extract features.")
            return output[feature_level]
        raise TypeError(f"Unsupported encoder output type: {type(output)}")

    def _run_encoder(self, encoder: nn.Module, image: torch.Tensor) -> torch.Tensor:
        if hasattr(encoder, "encode") and callable(getattr(encoder, "encode")):
            features = encoder.encode(image)
        else:
            features = self._select_from_output(encoder(image), self.feature_level)
        if isinstance(features, (list, tuple)):
            features = features[self.feature_level]
        if not torch.is_tensor(features):
            raise TypeError("Extracted features must be a torch.Tensor.")
        if self.normalize_features:
            features = F.normalize(features, dim=1)
        return features

    def _select_encoder(self, modality: str) -> nn.Module:
        normalized = (modality or "").lower()
        if normalized in _PREOP_MODALITIES:
            return self.preop_encoder
        if normalized in _US_MODALITIES:
            return self.us_encoder
        raise ValueError(
            f"Unknown modality '{modality}'. Expected one of "
            f"{sorted(_PREOP_MODALITIES | _US_MODALITIES)}."
        )

    def forward(self, image: torch.Tensor, modality: str) -> torch.Tensor:
        encoder = self._select_encoder(modality)
        with torch.no_grad():
            return self._run_encoder(encoder, image)


def resize_features_to_phi(features: torch.Tensor, phi_vectorfield: torch.Tensor) -> torch.Tensor:
    target_size = phi_vectorfield.shape[2:]
    if features.shape[2:] == target_size:
        return features
    if features.dim() == 5:
        mode = "trilinear"
    elif features.dim() == 4:
        mode = "bilinear"
    elif features.dim() == 3:
        mode = "linear"
    else:
        raise ValueError(f"Unsupported feature tensor rank: {features.dim()}")
    return F.interpolate(features, size=target_size, mode=mode, align_corners=True)


class DenseInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1, num_samples: int = 2048):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        self.temperature = temperature
        self.num_samples = num_samples

    def forward(
        self,
        z_src: torch.Tensor,
        z_tgt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z_src.shape != z_tgt.shape:
            raise ValueError(f"z_src and z_tgt must match shapes, got {z_src.shape} and {z_tgt.shape}.")

        z_src = F.normalize(z_src, dim=1)
        z_tgt = F.normalize(z_tgt, dim=1)

        batch_size = z_src.shape[0]
        spatial_size = int(torch.tensor(z_src.shape[2:]).prod().item())
        z_src_flat = z_src.flatten(start_dim=2)
        z_tgt_flat = z_tgt.flatten(start_dim=2)

        if mask is None:
            valid_mask = torch.ones((batch_size, spatial_size), dtype=torch.bool, device=z_src.device)
        else:
            if mask.dim() == z_src.dim():
                mask = mask[:, 0]
            valid_mask = mask.flatten(start_dim=1) > 0

        losses = []
        for b in range(batch_size):
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(1)
            if valid_indices.numel() < 2:
                continue
            k = min(self.num_samples, int(valid_indices.numel()))
            perm = torch.randperm(valid_indices.numel(), device=z_src.device)[:k]
            sampled_idx = valid_indices[perm]

            src_samples = z_src_flat[b, :, sampled_idx].transpose(0, 1)  # [K, C]
            tgt_samples = z_tgt_flat[b, :, sampled_idx].transpose(0, 1)  # [K, C]

            logits = (src_samples @ tgt_samples.transpose(0, 1)) / self.temperature
            labels = torch.arange(k, device=z_src.device)

            ce_ab = F.cross_entropy(logits, labels)
            ce_ba = F.cross_entropy(logits.transpose(0, 1), labels)
            losses.append(0.5 * (ce_ab + ce_ba))

        if not losses:
            return torch.zeros((), dtype=z_src.dtype, device=z_src.device)
        return torch.stack(losses).mean()
