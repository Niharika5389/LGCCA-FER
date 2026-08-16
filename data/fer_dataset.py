"""FER-2013 dataset pipeline with precomputed landmark-heatmap conditioning.

This module implements the Step 1 data pipeline described in the
LGCA-Net implementation guide:

1. ``load_fer_csv``          - parse the raw FER-2013 CSV.
2. ``build_landmark_cache``  - offline MediaPipe FaceMesh landmark-heatmap
                                cache builder (run once, saved to disk).
3. ``FERDataset``             - a ``torch.utils.data.Dataset`` returning
                                (image, heatmap, label) triples, with
                                albumentations-based augmentation that is
                                applied identically to the image and the
                                heatmap.
4. ``compute_class_weights`` - inverse-frequency class weights for the
                                known FER-2013 class imbalance.

Preprocessing contract (must stay stable for IAFM/LGCCA downstream):
    48x48 grayscale -> bicubic upsample to 224x224 -> replicate to 3
    channels -> ImageNet mean/std normalization. Landmark heatmaps are
    56x56x1, single-channel, min-max normalized to [0, 1].

Run as a standalone script (``python -m data.fer_dataset`` or
``python data/fer_dataset.py``) to execute a visual sanity check that
overlays cached heatmaps on their source images.
"""

from __future__ import annotations

import argparse
import logging
import json
import os
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from configs.config import DATASET_PATH, IMAGE_SIZE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FER-2013's native ``Usage`` column values mapped to the split names used
# throughout this project. Native split is used (rather than a fresh
# stratified re-split) to stay comparable to published FER-2013 results:
# PublicTest is treated as validation, PrivateTest as test.
SPLIT_MAP = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}

FER_IMAGE_SIZE = 48  # native FER-2013 image resolution
HEATMAP_SIZE = 56  # landmark heatmap resolution
GAUSSIAN_SIGMA = 2.5  # std-dev (in heatmap pixels) of each landmark blob

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Documented MediaPipe FaceMesh (468-point topology) landmark indices for
# the eyes, eyebrows, mouth and nose regions. These are the standard
# contour/region index sets published in MediaPipe's face mesh
# documentation (https://github.com/google/mediapipe) and are the ones
# commonly used for attention-region heatmaps in FER literature.
LEFT_EYE_INDICES: List[int] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_INDICES: List[int] = [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
]
LEFT_EYEBROW_INDICES: List[int] = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_INDICES: List[int] = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
MOUTH_INDICES: List[int] = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
]
NOSE_INDICES: List[int] = [1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 220, 275, 440]

LANDMARK_INDICES: List[int] = sorted(
    set(
        LEFT_EYE_INDICES
        + RIGHT_EYE_INDICES
        + LEFT_EYEBROW_INDICES
        + RIGHT_EYEBROW_INDICES
        + MOUTH_INDICES
        + NOSE_INDICES
    )
)


# ---------------------------------------------------------------------------
# Step 1.1 - CSV loading
# ---------------------------------------------------------------------------


def load_fer_csv(path: str) -> pd.DataFrame:
    """Parse the raw FER-2013 CSV into image arrays, labels and splits.

    Args:
        path: Path to the raw FER-2013 ``fer2013.csv`` file with columns
            ``emotion`` (int 0-6), ``pixels`` (space-separated 48x48
            grayscale ints) and ``Usage``
            (``Training``/``PublicTest``/``PrivateTest``).

    Returns:
        A DataFrame with columns:
            - ``emotion``: int label (0-6).
            - ``image``: object column of ``np.ndarray`` (48, 48) uint8.
            - ``Usage``: original FER-2013 split string.
            - ``split``: mapped split name (``train``/``val``/``test``).

    Raises:
        ValueError: If required columns are missing or an unrecognized
            ``Usage`` value is encountered.
    """
    df = pd.read_csv(path)

    required_cols = {"emotion", "pixels", "Usage"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"FER-2013 CSV at {path!r} is missing columns: {missing}")

    def _parse_pixels(pixel_str: str) -> np.ndarray:
        values = np.fromstring(pixel_str, dtype=np.uint8, sep=" ")
        if values.size != FER_IMAGE_SIZE * FER_IMAGE_SIZE:
            raise ValueError(
                f"Expected {FER_IMAGE_SIZE * FER_IMAGE_SIZE} pixel values, "
                f"got {values.size}"
            )
        return values.reshape(FER_IMAGE_SIZE, FER_IMAGE_SIZE)

    df = df.copy()
    df["image"] = df["pixels"].apply(_parse_pixels)
    df["emotion"] = df["emotion"].astype(int)
    df["split"] = df["Usage"].map(SPLIT_MAP)

    if df["split"].isna().any():
        unknown = sorted(df.loc[df["split"].isna(), "Usage"].unique().tolist())
        raise ValueError(f"Unrecognized Usage value(s) in CSV: {unknown}")

    df = df.drop(columns=["pixels"]).reset_index(drop=True)
    logger.info(
        "Loaded FER-2013 CSV: %d rows (train=%d, val=%d, test=%d)",
        len(df),
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 1.2 - Landmark heatmap cache
# ---------------------------------------------------------------------------


def _gaussian_blob(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Render a single 2D Gaussian blob on a ``size x size`` canvas.

    Args:
        size: Canvas side length.
        cx: Blob center x-coordinate (in canvas pixels).
        cy: Blob center y-coordinate (in canvas pixels).
        sigma: Gaussian standard deviation, in canvas pixels.

    Returns:
        A ``(size, size)`` float32 array with the rendered Gaussian.
    """
    ys, xs = np.mgrid[0:size, 0:size]
    blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2)))
    return blob.astype(np.float32)


def build_landmark_cache(
    images: np.ndarray,
    output_path: str,
    canonical_fallback: np.ndarray,
) -> np.ndarray:
    """Build (and cache to disk) a landmark-heatmap for every image.

    For each 48x48 grayscale image: bicubic-upsample to 224x224, convert
    to 3-channel RGB, run MediaPipe FaceMesh, render 2D Gaussian blobs at
    the eyes/eyebrows/mouth/nose landmark indices onto a 56x56 canvas, sum
    and min-max normalize to [0, 1]. On detection failure the
    ``canonical_fallback`` heatmap is substituted.

    Args:
        images: ``(N, 48, 48)`` uint8 array of grayscale FER-2013 images.
        output_path: Where to save the resulting ``(N, 56, 56)`` float32
            ``.npy`` array. A ``<output_path>.stats.json`` sidecar with the
            fallback count/fraction is written alongside it.
        canonical_fallback: ``(56, 56)`` float32 array used whenever
            MediaPipe fails to detect a face (e.g. the mean heatmap over
            all successfully-detected training samples).

    Returns:
        The ``(N, 56, 56)`` float32 array of landmark heatmaps.

    Raises:
        ImportError: If ``mediapipe`` is not installed.
    """
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "mediapipe is required for build_landmark_cache(); "
            "install it via `pip install mediapipe`."
        ) from exc

    if canonical_fallback.shape != (HEATMAP_SIZE, HEATMAP_SIZE):
        raise ValueError(
            f"canonical_fallback must have shape ({HEATMAP_SIZE}, {HEATMAP_SIZE}), "
            f"got {canonical_fallback.shape}"
        )

    num_images = images.shape[0]
    heatmaps = np.zeros((num_images, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
    fallback_count = 0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )

    try:
        for i in range(num_images):
            img_224 = cv2.resize(
                images[i], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
            )
            img_rgb = cv2.cvtColor(img_224, cv2.COLOR_GRAY2RGB)

            results = face_mesh.process(img_rgb)

            if not results.multi_face_landmarks:
                heatmaps[i] = canonical_fallback
                fallback_count += 1
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            canvas = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
            for idx in LANDMARK_INDICES:
                lm = landmarks[idx]
                cx, cy = lm.x * HEATMAP_SIZE, lm.y * HEATMAP_SIZE
                canvas += _gaussian_blob(HEATMAP_SIZE, cx, cy, GAUSSIAN_SIGMA)

            c_min, c_max = float(canvas.min()), float(canvas.max())
            if c_max - c_min > 1e-8:
                canvas = (canvas - c_min) / (c_max - c_min)
                heatmaps[i] = canvas
            else:
                # Degenerate (all-zero) render - treat as a failure.
                heatmaps[i] = canonical_fallback
                fallback_count += 1

            if (i + 1) % 1000 == 0:
                logger.info("Processed %d/%d images", i + 1, num_images)
    finally:
        face_mesh.close()

    fallback_fraction = fallback_count / num_images if num_images else 0.0
    logger.info(
        "MediaPipe FaceMesh fallback used for %d/%d images (%.2f%%)",
        fallback_count,
        num_images,
        100.0 * fallback_fraction,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, heatmaps)
    stats = {
        "num_images": int(num_images),
        "fallback_count": int(fallback_count),
        "fallback_fraction": fallback_fraction,
    }
    with open(f"{output_path}.stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return heatmaps


# ---------------------------------------------------------------------------
# Step 1.3/1.4 - Dataset + augmentation
# ---------------------------------------------------------------------------


def _build_transforms(augment: bool) -> A.Compose:
    """Build the albumentations pipeline shared by image and heatmap.

    ``additional_targets={'heatmap': 'mask'}`` ensures horizontal flip
    (and any future geometric transform) is applied identically to the
    image and the heatmap, while pixel-level transforms (gamma/color
    jitter, coarse dropout) are applied to the image only. Mixup/cutmix
    are intentionally not implemented (label-mixing is not semantically
    valid for FER).

    Args:
        augment: Whether to include training-time augmentation.

    Returns:
        An ``albumentations.Compose`` pipeline.
    """
    if augment:
        transform_list = [
            A.HorizontalFlip(p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                p=0.1,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    else:
        transform_list = [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    # The heatmap (56x56) is a different resolution from the image (224x224)
    # by design; only shape-agnostic geometric ops (e.g. horizontal flip)
    # are shared via additional_targets, so the built-in equal-shape check
    # must be disabled.
    return A.Compose(
        transform_list,
        additional_targets={"heatmap": "mask"},
        is_check_shapes=False,
    )


class FERDataset(Dataset):
    """FER-2013 dataset returning (image, landmark heatmap, label) triples.

    Images are bicubic-upsampled from 48x48 grayscale to 224x224 RGB and
    normalized with ImageNet mean/std. Heatmaps are read from the
    precomputed ``heatmap_array`` (aligned by row index to ``csv_df``) and
    are kept at 56x56x1. Horizontal flip (and any other geometric
    augmentation) is applied identically to both via albumentations.

    Args:
        csv_df: DataFrame as returned by :func:`load_fer_csv` (must
            contain ``image``, ``emotion`` and ``split`` columns).
        heatmap_array: ``(N, 56, 56)`` array aligned by row index to
            ``csv_df`` (i.e. the full, unfiltered array produced by
            :func:`build_landmark_cache`).
        split: Which split to expose (``train``/``val``/``test``).
        augment: Whether to apply training-time augmentation.
    """

    def __init__(
        self,
        csv_df: pd.DataFrame,
        heatmap_array: np.ndarray,
        split: str,
        augment: bool = False,
    ) -> None:
        if "split" not in csv_df.columns:
            raise ValueError("csv_df must contain a 'split' column (see load_fer_csv).")

        self.full_df = csv_df.reset_index(drop=True)
        if len(self.full_df) != len(heatmap_array):
            raise ValueError(
                f"csv_df has {len(self.full_df)} rows but heatmap_array has "
                f"{len(heatmap_array)} entries; they must be row-aligned."
            )

        self.heatmap_array = heatmap_array
        self.split = split
        self.augment = augment
        self.transform = _build_transforms(augment)

        split_mask = self.full_df["split"] == split
        self.indices = self.full_df.index[split_mask].to_numpy()
        if len(self.indices) == 0:
            logger.warning("FERDataset: split %r has 0 samples.", split)
        self.csv_df = self.full_df.loc[self.indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.csv_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        original_idx = int(self.indices[idx])
        row = self.csv_df.iloc[idx]

        img_48 = row["image"]
        img_224 = cv2.resize(
            img_48, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC
        )
        img_rgb = cv2.cvtColor(img_224, cv2.COLOR_GRAY2RGB)

        heatmap = np.asarray(self.heatmap_array[original_idx], dtype=np.float32)

        transformed = self.transform(image=img_rgb, heatmap=heatmap)
        image_tensor = transformed["image"].float()

        heatmap_tensor = transformed["heatmap"]
        if not torch.is_tensor(heatmap_tensor):
            heatmap_tensor = torch.from_numpy(np.asarray(heatmap_tensor))
        heatmap_tensor = heatmap_tensor.float()
        if heatmap_tensor.dim() == 2:
            heatmap_tensor = heatmap_tensor.unsqueeze(0)

        label = int(row["emotion"])
        return image_tensor, heatmap_tensor, label


# ---------------------------------------------------------------------------
# Step 1.3 (imbalance) - class weights
# ---------------------------------------------------------------------------


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for ``CrossEntropyLoss``.

    Args:
        labels: 1D array of integer class labels.

    Returns:
        A ``torch.FloatTensor`` of per-class weights, indexed 0..max(labels),
        normalized so weights sum to ``num_classes`` (keeps the effective
        loss scale roughly comparable to unweighted CE).
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("compute_class_weights received an empty labels array.")

    classes, counts = np.unique(labels, return_counts=True)
    num_classes = int(classes.max()) + 1

    freq = np.zeros(num_classes, dtype=np.float64)
    freq[classes] = counts
    # Absent classes get a neutral weight of 1 rather than exploding to inf.
    freq[freq == 0] = 1.0

    inv_freq = 1.0 / freq
    weights = inv_freq / inv_freq.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def _sanity_check(
    csv_path: str,
    heatmap_path: str,
    output_dir: str,
    num_samples: int = 12,
) -> None:
    """Visualize random samples with their heatmap overlaid and save a grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load_fer_csv(csv_path)
    heatmap_array = np.load(heatmap_path)
    dataset = FERDataset(df, heatmap_array, split="train", augment=False)

    if len(dataset) == 0:
        raise RuntimeError("No training samples available for the sanity check.")

    num_samples = min(num_samples, len(dataset))
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(len(dataset), size=num_samples, replace=False)

    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)

    cols = 4
    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, sample_indices):
        image_tensor, heatmap_tensor, label = dataset[int(idx)]
        image = image_tensor.numpy() * std + mean
        image = np.clip(image, 0.0, 1.0).transpose(1, 2, 0)
        heatmap = heatmap_tensor.numpy()[0]

        ax.imshow(image)
        ax.imshow(
            cv2.resize(heatmap, (image.shape[1], image.shape[0])),
            cmap="jet",
            alpha=0.45,
        )
        ax.set_title(f"label={label}")
        ax.axis("off")

    for ax in axes[num_samples:]:
        ax.axis("off")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "heatmap_sanity_check.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved heatmap sanity-check grid to %s", out_path)


def main() -> None:
    """CLI entry point for the FER-2013 dataset sanity check."""
    parser = argparse.ArgumentParser(description="FER-2013 dataset sanity check")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=os.path.join(DATASET_PATH, "fer2013.csv"),
        help="Path to the raw FER-2013 CSV.",
    )
    parser.add_argument(
        "--heatmap-path",
        type=str,
        default=os.path.join("data", "processed", "landmark_heatmaps.npy"),
        help="Path to the precomputed (N, 56, 56) landmark heatmap .npy array.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("reports", "figures"),
        help="Directory to save heatmap_sanity_check.png into.",
    )
    args = parser.parse_args()
    _sanity_check(args.csv_path, args.heatmap_path, args.output_dir)


if __name__ == "__main__":
    main()
