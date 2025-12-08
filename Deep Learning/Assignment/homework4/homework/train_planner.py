"""
Usage examples (run from the project root):

# MLP planner (state-only)
python3 -m homework.train_planner --model mlp_planner

# Transformer planner (state-only)
python3 -m homework.train_planner --model transformer_planner

# CNN planner (image-based)
python3 -m homework.train_planner --model cnn_planner
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .models import load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def masked_mse_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor,
) -> torch.Tensor:
    """
    MSE over only the valid waypoints.

    preds:  (B, n, 2)
    labels: (B, n, 2)
    labels_mask: (B, n) bool
    """
    diff = preds - labels
    diff2 = diff ** 2

    # (B, n, 2) * (B, n, 1)
    mask = labels_mask[..., None].float()
    diff2 = diff2 * mask

    # average over valid entries & both coords
    denom = mask.sum() * 2.0 + 1e-8  # *2 for x,y
    loss = diff2.sum() / denom
    return loss


def get_dataloaders(
    model_name: str,
    dataset_root: str = "drive_data",
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train/val dataloaders with the right transform pipeline.
    """
    dataset_root = Path(dataset_root)
    train_path = dataset_root / "train"
    val_path = dataset_root / "val"

    # MLP / Transformer use only track+waypoints; CNN uses images too
    if model_name in ("mlp_planner", "transformer_planner"):
        transform_pipeline = "state_only"
    else:
        transform_pipeline = "default"

    train_loader = load_data(
        str(train_path),
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = load_data(
        str(val_path),
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    count = 0

    for batch in dataloader:
        # Move tensors to device
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        optimizer.zero_grad()

        if "image" in batch:
            preds = model(image=batch["image"])
        else:
            preds = model(
                track_left=batch["track_left"],
                track_right=batch["track_right"],
            )

        labels = batch["waypoints"]
        labels_mask = batch["waypoints_mask"]

        loss = masked_mse_loss(preds, labels, labels_mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    return running_loss / max(count, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    metric = PlannerMetric()
    running_loss = 0.0
    count = 0

    for batch in dataloader:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)

        if "image" in batch:
            preds = model(image=batch["image"])
        else:
            preds = model(
                track_left=batch["track_left"],
                track_right=batch["track_right"],
            )

        labels = batch["waypoints"]
        labels_mask = batch["waypoints_mask"]

        loss = masked_mse_loss(preds, labels, labels_mask)
        running_loss += loss.item()
        count += 1

        metric.add(preds, labels, labels_mask)

    stats = metric.compute()
    stats["loss"] = running_loss / max(count, 1)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="mlp_planner",
        choices=["mlp_planner", "transformer_planner", "cnn_planner"],
        help="Which planner model to train",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="drive_data",
        help="Path to drive_data root (containing train/ and val/)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        model_name=args.model,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = load_model(args.model, with_weights=False)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_l1 = float("inf")
    best_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_stats = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_stats['loss']:.4f}, "
            f"longitudinal={val_stats['longitudinal_error']:.4f}, "
            f"lateral={val_stats['lateral_error']:.4f}"
        )

        # track best by L1 error
        if val_stats["l1_error"] < best_val_l1:
            best_val_l1 = val_stats["l1_error"]
            best_path = save_model(model)
            print(f"  -> New best model saved to {best_path} (L1={best_val_l1:.4f})")

    if best_path is None:
        # if we never improved, at least save final model
        best_path = save_model(model)
        print(f"No improvement tracked, saved final model to {best_path}")
    else:
        print(f"Best model already saved at {best_path} with L1={best_val_l1:.4f}")


if __name__ == "__main__":
    main()

