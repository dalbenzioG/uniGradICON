"""ContraReg-style 3D convolutional autoencoders for contrastive registration.

Each encoder exposes ``encode(x) -> [conv1, conv2, conv3]`` and
``forward(x) -> (reconstruction, features)``. Contrastive finetuning reads
features at ``feature_level`` (default ``-2``, i.e. 64 channels when
``base_channels=32``).
"""

from typing import Callable, Dict, Optional

import torch.nn as nn


def ae_conv_block(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    instance_norm=True,
    leaky_relu=True,
):
    layers = [
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    ]
    if instance_norm:
        layers.append(nn.InstanceNorm3d(out_channels))
    if leaky_relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, feature_dim=128):
        super().__init__()

        self.conv1 = ae_conv_block(in_channels, base_channels, 3, 2, 1)
        self.conv2 = ae_conv_block(base_channels, base_channels * 2, 3, 2, 1)
        self.conv3 = ae_conv_block(base_channels * 2, feature_dim, 3, 2, 1)

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ae_conv_block(feature_dim, base_channels * 2, 3, 1, 1),
        )
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ae_conv_block(base_channels * 2, base_channels, 3, 1, 1),
        )
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ae_conv_block(base_channels, base_channels, 3, 1, 1),
        )

        self.conv4 = ae_conv_block(base_channels, base_channels * 2, 3, 1, 1)
        self.conv5 = nn.Conv3d(base_channels * 2, in_channels, 3, 1, 1)

    def encode(self, x):
        features = []

        x = self.conv1(x)
        features.append(x)

        x = self.conv2(x)
        features.append(x)

        x = self.conv3(x)
        features.append(x)

        return features

    def decode(self, z):
        x = self.upconv1(z)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        features = self.encode(x)
        reconstruction = self.decode(features[-1])
        return reconstruction, features


ENCODER_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "AutoEncoder": AutoEncoder,
    "AutoEncoder3D": AutoEncoder,
}


def build_encoder_from_arch(arch: str, arch_kwargs: Optional[dict] = None) -> nn.Module:
    if arch not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder arch '{arch}'. Available: {sorted(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[arch](**(arch_kwargs or {}))
