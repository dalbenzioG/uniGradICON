"""ContraReg-style patch projection and multi-scale PatchNCE (standalone)."""

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class PatchProjector3D(nn.Module):
    def __init__(
        self,
        feature_channels: Sequence[int] = (32, 64, 128),
        embed_dim: int = 256,
    ):
        super().__init__()
        self.feature_channels = tuple(feature_channels)
        self.embed_dim = embed_dim
        self.mlps = nn.ModuleList(
            [ProjectionMLP(in_channels, embed_dim) for in_channels in self.feature_channels]
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        num_patches: int,
        patch_ids: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if len(features) != len(self.feature_channels):
            raise ValueError(
                f"Expected {len(self.feature_channels)} feature maps, got {len(features)}."
            )
        if patch_ids is not None and len(patch_ids) != len(features):
            raise ValueError(
                f"Expected {len(features)} patch_id tensors, got {len(patch_ids)}."
            )

        projected: List[torch.Tensor] = []
        out_patch_ids: List[torch.Tensor] = []

        for level_idx, (feat, expected_channels, mlp) in enumerate(
            zip(features, self.feature_channels, self.mlps)
        ):
            if feat.dim() != 5:
                raise ValueError(f"Feature level {level_idx} must be 5D [B, C, D, H, W], got {feat.shape}.")
            if feat.shape[1] != expected_channels:
                raise ValueError(
                    f"Feature level {level_idx} expected {expected_channels} channels, got {feat.shape[1]}."
                )

            batch_size = feat.shape[0]
            spatial_size = int(feat.shape[2:].numel())
            num_sampled = min(num_patches, spatial_size)

            feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(batch_size, spatial_size, expected_channels)

            if patch_ids is None:
                level_patch_ids = torch.randperm(spatial_size, device=feat.device)[:num_sampled]
            else:
                level_patch_ids = patch_ids[level_idx]
                if not torch.is_tensor(level_patch_ids):
                    level_patch_ids = torch.tensor(level_patch_ids, dtype=torch.long, device=feat.device)
                else:
                    level_patch_ids = level_patch_ids.to(device=feat.device, dtype=torch.long)

            sampled = feat_flat[:, level_patch_ids, :]
            projected.append(mlp(sampled.reshape(batch_size * num_sampled, expected_channels)))
            out_patch_ids.append(level_patch_ids)

        return projected, out_patch_ids


def patch_nce_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    batch_size: int,
    num_patches: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    q = q.view(batch_size, num_patches, -1)
    k = k.view(batch_size, num_patches, -1)
    logits = torch.bmm(q, k.transpose(1, 2)) / temperature
    labels = torch.arange(num_patches, device=q.device)

    losses = [F.cross_entropy(logits[b], labels) for b in range(batch_size)]
    return torch.stack(losses).mean()


def multi_scale_patch_nce_loss(
    q_levels: Sequence[torch.Tensor],
    k_levels: Sequence[torch.Tensor],
    batch_size: int,
    num_patches: int,
    temperature: float = 0.07,
) -> torch.Tensor:
    if len(q_levels) != len(k_levels):
        raise ValueError(
            f"q_levels and k_levels must have the same length, got {len(q_levels)} and {len(k_levels)}."
        )
    if not q_levels:
        raise ValueError("q_levels must be non-empty.")

    level_losses = []
    for q, k in zip(q_levels, k_levels):
        level_num_patches = q.shape[0] // batch_size
        if level_num_patches <= 0:
            raise ValueError(
                f"Invalid patch count for batch_size={batch_size}: q.shape={q.shape}."
            )
        level_losses.append(
            patch_nce_loss(q, k, batch_size, level_num_patches, temperature=temperature)
        )
    return torch.stack(level_losses).mean()
