"""Training script for the EfficientNet-B0 FER classification baseline.

Implements Step 2 of the LGCA-Net implementation guide: trains
``models.backbone.EfficientNetB0Classifier`` on FER-2013 (via
``data.fer_dataset.FERDataset``) with class-weighted, label-smoothed
cross-entropy, a linear-warmup + cosine-annealing LR schedule, and
early stopping on validation macro-F1.

Usage:
    python train_baseline.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from configs.config import DEVICE, NUM_CLASSES
from data.fer_dataset import FERDataset, compute_class_weights, load_fer_csv
from models.backbone import EfficientNetB0Classifier
from utils.seed import set_seed


def load_config(config_path: str) -> Dict:
    """Load a YAML training config.

    Args:
        config_path: Path to a YAML file with ``data``, ``train`` and
            ``output`` sections (see ``configs/baseline.yaml``).

    Returns:
        The parsed config as a nested dict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, pd.DataFrame]:
    """Build train/val DataLoaders from the FER-2013 CSV + heatmap cache.

    Args:
        cfg: Parsed YAML config.

    Returns:
        A tuple ``(train_loader, val_loader, train_df)`` where ``train_df``
        is the raw (unfiltered) parsed CSV, useful for computing class
        weights.
    """
    df = load_fer_csv(cfg["data"]["csv_path"])
    heatmap_array = np.load(cfg["data"]["heatmap_path"])

    train_dataset = FERDataset(df, heatmap_array, split="train", augment=True)
    val_dataset = FERDataset(df, heatmap_array, split="val", augment=False)

    num_workers = cfg["data"].get("num_workers", 4)
    batch_size = cfg["train"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, df


def get_class_weights(cfg: Dict, train_df) -> torch.Tensor:
    """Load class weights from disk, computing and caching them if absent.

    Args:
        cfg: Parsed YAML config (uses ``output.class_weights_path``).
        train_df: Full parsed FER-2013 DataFrame (used to compute weights
            from the training split if no cached file exists yet).

    Returns:
        A ``[NUM_CLASSES]`` tensor of class weights.
    """
    weights_path = cfg["output"]["class_weights_path"]
    if os.path.exists(weights_path):
        return torch.load(weights_path, map_location="cpu")

    train_labels = train_df.loc[train_df["split"] == "train", "emotion"].to_numpy()
    weights = compute_class_weights(train_labels)
    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    torch.save(weights, weights_path)
    return weights


def build_scheduler(
    optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int
) -> torch.optim.lr_scheduler.SequentialLR:
    """Build a 5-epoch linear-warmup -> cosine-annealing LR schedule.

    Args:
        optimizer: The optimizer to schedule.
        warmup_epochs: Number of linear warmup epochs.
        total_epochs: Total planned number of epochs.

    Returns:
        A ``SequentialLR`` combining ``LinearLR`` (warmup) and
        ``CosineAnnealingLR`` (decay over the remaining epochs).
    """
    warmup_epochs = max(warmup_epochs, 1)
    remaining_epochs = max(total_epochs - warmup_epochs, 1)

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average training loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, _heatmaps, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, _embedding = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Run validation and return (val_loss, val_accuracy, val_macro_f1)."""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, _heatmaps, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits, _embedding = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        num_batches += 1

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = running_loss / max(num_batches, 1)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    val_accuracy = correct / max(len(all_labels), 1)
    val_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return val_loss, val_accuracy, val_macro_f1


def plot_curves(log_rows: List[Dict], output_path: str) -> None:
    """Plot loss and macro-F1 curves and save them to disk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in log_rows]
    train_loss = [row["train_loss"] for row in log_rows]
    val_loss = [row["val_loss"] for row in log_rows]
    val_f1 = [row["val_macro_f1"] for row in log_rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="train loss")
    axes[0].plot(epochs, val_loss, label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="val macro-F1", color="tab:green")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("macro-F1")
    axes[1].set_title("Validation macro-F1")
    axes[1].legend()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(cfg: Dict) -> None:
    """Run the full baseline training loop, per the loaded config.

    Args:
        cfg: Parsed YAML config (see ``configs/baseline.yaml``).
    """
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    device = DEVICE
    experiment_dir = cfg["output"]["experiment_dir"]
    figures_dir = cfg["output"]["figures_dir"]
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    train_loader, val_loader, train_df = build_dataloaders(cfg)
    class_weights = get_class_weights(cfg, train_df).to(device)

    model = EfficientNetB0Classifier(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg["train"]["label_smoothing"]
    )
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = build_scheduler(
        optimizer, cfg["train"]["warmup_epochs"], cfg["train"]["epochs"]
    )

    log_path = os.path.join(experiment_dir, "log.csv")
    best_path = os.path.join(experiment_dir, "best.pt")
    final_path = os.path.join(experiment_dir, "final.pt")

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    patience = cfg["train"]["patience"]
    log_rows: List[Dict] = []

    with open(log_path, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"],
        )
        writer.writeheader()

        for epoch in range(1, cfg["train"]["epochs"] + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy, val_macro_f1 = validate(model, val_loader, criterion, device)
            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
            }
            log_rows.append(row)
            writer.writerow(row)
            log_file.flush()

            print(
                f"Epoch {epoch}/{cfg['train']['epochs']} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_accuracy:.4f} | val_macro_f1={val_macro_f1:.4f}"
            )

            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_macro_f1": val_macro_f1,
                        "val_accuracy": val_accuracy,
                        "config": cfg,
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch}: no val macro-F1 improvement "
                    f"for {patience} epochs (best={best_macro_f1:.4f})."
                )
                break

        torch.save(
            {
                "epoch": row["epoch"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": row["val_macro_f1"],
                "val_accuracy": row["val_accuracy"],
                "config": cfg,
            },
            final_path,
        )

    plot_curves(log_rows, os.path.join(figures_dir, "baseline_curves.png"))
    print(f"Training complete. Best val macro-F1: {best_macro_f1:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Final checkpoint: {final_path}")
    print(f"Log CSV: {log_path}")


def main() -> None:
    """CLI entry point: parse --config and run training."""
    parser = argparse.ArgumentParser(description="Train the FER EfficientNet-B0 baseline")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("configs", "baseline.yaml"),
        help="Path to the YAML training config.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
