"""Illumination-Adaptive Feature Modulation (IAFM).

A FiLM-style feature re-calibration module: a small MLP maps per-image
luminance statistics (mean, std) to a per-channel scale (gamma) and shift
(beta), which are applied to a backbone feature map. This lets the network
adapt its features to the lighting condition of each input image.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

# ImageNet normalization constants the upstream data pipeline
# (data/fer_dataset.py) uses to normalize images before they reach the
# backbone. compute_luminance_stats() assumes its input was normalized
# with exactly these constants.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# ITU-R BT.601 luma weights, used to convert de-normalized RGB back to a
# single-channel grayscale/luminance signal.
LUMINANCE_WEIGHTS: Tuple[float, float, float] = (0.299, 0.587, 0.114)


def compute_luminance_stats(images: torch.Tensor) -> torch.Tensor:
    """Compute per-image (mean, std) grayscale luminance for a batch.

    Assumption (documented, must stay consistent with the data pipeline):
    ``images`` is a batch of ImageNet-normalized RGB images, i.e.
    ``images = (raw_rgb - IMAGENET_MEAN) / IMAGENET_STD`` with
    ``raw_rgb`` in ``[0, 1]``. This function first approximately inverts
    that normalization (``raw_rgb = images * IMAGENET_STD + IMAGENET_MEAN``)
    to recover a "raw-ish" ``[0, 1]``-scale image, then converts to
    grayscale via the standard ITU-R BT.601 luma weights, and finally
    computes the per-image spatial mean and standard deviation of that
    grayscale map. The whole computation is fully vectorized (no
    per-sample Python loop).

    Args:
        images: Batch of ImageNet-normalized images, shape ``[B, 3, H, W]``.

    Returns:
        A ``[B, 2]`` tensor where column 0 is the per-image luminance mean
        and column 1 is the per-image luminance std.

    Raises:
        ValueError: If ``images`` does not have 4 dimensions with 3
            channels.
    """
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected images of shape [B, 3, H, W], got {tuple(images.shape)}")

    device, dtype = images.device, images.dtype
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    luma_weights = torch.tensor(LUMINANCE_WEIGHTS, device=device, dtype=dtype).view(1, 3, 1, 1)

    # Approximate de-normalization back to a raw-ish [0, 1] RGB space.
    raw_rgb = images * std + mean

    # Weighted RGB -> single-channel luminance, still fully vectorized.
    grayscale = (raw_rgb * luma_weights).sum(dim=1)  # [B, H, W]

    batch_size = grayscale.shape[0]
    flat = grayscale.reshape(batch_size, -1)
    luminance_mean = flat.mean(dim=1)
    luminance_std = flat.std(dim=1, unbiased=False)

    return torch.stack([luminance_mean, luminance_std], dim=1)


class IAFM(nn.Module):
    """FiLM-style illumination-adaptive feature modulation.

    A 2-layer MLP (Linear(2, 512) -> ReLU -> Linear(512, 2*C)) maps
    per-image luminance statistics to per-channel ``gamma``/``beta``
    modulation parameters, which are applied to a feature map as
    ``F' = gamma * F + beta`` (broadcast over spatial dimensions).

    The final linear layer is initialized so that, at the start of
    training, ``gamma`` is near 1 and ``beta`` is near 0 -- i.e. IAFM
    starts as an (approximate) identity transform and does not
    destabilize a pretrained backbone.
    """

    def __init__(self, num_channels: int = 1280, hidden_dim: int = 512) -> None:
        """
        Args:
            num_channels: Number of feature-map channels ``C`` to
                modulate (1280 for an EfficientNet-B0 pooled/feature map).
            hidden_dim: Hidden layer size of the MLP.
        """
        super().__init__()
        self.num_channels = num_channels

        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        self._init_identity()

    def _init_identity(self) -> None:
        """Initialize the final layer for a near-identity transform."""
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)
        with torch.no_grad():
            bias = torch.zeros(2 * self.num_channels)
            bias[: self.num_channels] = 1.0  # gamma starts near 1
            bias[self.num_channels :] = 0.0  # beta starts near 0
            final_layer.bias.copy_(bias)

    def forward(
        self, feature_map: torch.Tensor, luminance_stats: torch.Tensor
    ) -> torch.Tensor:
        """Apply illumination-adaptive modulation to a feature map.

        Args:
            feature_map: Backbone feature map, shape ``[B, C, H, W]``.
            luminance_stats: Per-image ``(mean, std)`` luminance, shape
                ``[B, 2]`` (e.g. from :func:`compute_luminance_stats`).

        Returns:
            The modulated feature map ``F' = gamma * F + beta``, with the
            same shape as ``feature_map``.

        Raises:
            ValueError: If ``feature_map``'s channel dimension does not
                match ``self.num_channels``.
        """
        if feature_map.shape[1] != self.num_channels:
            raise ValueError(
                f"feature_map has {feature_map.shape[1]} channels, "
                f"expected {self.num_channels}"
            )

        params = self.mlp(luminance_stats)  # [B, 2*C]
        gamma, beta = torch.chunk(params, 2, dim=1)  # each [B, C]

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        return gamma * feature_map + beta


if __name__ == "__main__":
    # Shape/sanity test using random tensors.
    torch.manual_seed(0)

    batch_size, num_channels, height, width = 4, 1280, 7, 7
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_features = torch.randn(batch_size, num_channels, height, width)

    stats = compute_luminance_stats(dummy_images)
    print(f"luminance_stats shape: {tuple(stats.shape)}")
    assert stats.shape == (batch_size, 2)
    assert not torch.isnan(stats).any(), "luminance_stats contains NaN"

    iafm = IAFM(num_channels=num_channels)
    output = iafm(dummy_features, stats)
    print(f"IAFM output shape: {tuple(output.shape)}")
    assert output.shape == dummy_features.shape

    with torch.no_grad():
        params = iafm.mlp(stats)
        gamma, beta = torch.chunk(params, 2, dim=1)
    assert not torch.isnan(gamma).any(), "gamma contains NaN"
    assert not torch.isnan(beta).any(), "beta contains NaN"

    print(f"gamma mean (should start near 1): {gamma.mean().item():.4f}")
    print(f"beta mean (should start near 0): {beta.mean().item():.4f}")
    print("Sanity check passed.")
