# train_detection.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from homework.models import Detector, save_model
from homework.metrics import DetectionMetric


# ----------------------------------------
# Utils
# ----------------------------------------
def get_drive_loader(path, batch_size, shuffle, num_workers):
    from homework.datasets.road_dataset import load_data
    return load_data(
        dataset_path=path,
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# ----------------------------------------
# Training
# ----------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, lambda_depth):
    model.train()

    seg_loss_fn = nn.CrossEntropyLoss()
    depth_loss_fn = nn.L1Loss()

    metric = DetectionMetric(num_classes=3)
    total_loss, n = 0.0, 0

    for batch in loader:
        x = batch["image"].to(device)               # (B,3,H,W)
        y_seg = batch["track"].long().to(device)    # (B,H,W)
        y_depth = batch["depth"].to(device)         # (B,H,W)

        optimizer.zero_grad(set_to_none=True)

        # AMP 
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits, depth_pred = model(x)

            loss_seg = seg_loss_fn(logits, y_seg)
            loss_depth = depth_loss_fn(depth_pred, y_depth)

            loss = loss_seg + lambda_depth * loss_depth

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

        # metrics
        preds = logits.argmax(1)
        metric.add(preds, y_seg, depth_pred, y_depth)

    return total_loss / max(1, n), metric.compute()


# ----------------------------------------
# Evaluation
# ----------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = DetectionMetric(num_classes=3)

    for batch in loader:
        x = batch["image"].to(device)
        y_seg = batch["track"].long().to(device)
        y_depth = batch["depth"].to(device)

        logits, depth_pred = model(x)
        preds = logits.argmax(1)

        metric.add(preds, y_seg, depth_pred, y_depth)

    return metric.compute()


# ----------------------------------------
# Main
# ----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="drive_data/train")
    ap.add_argument("--val_dir", default="drive_data/val")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adam","sgd","adamw"])
    ap.add_argument("--lambda_depth", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--mixed_precision", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = get_drive_loader(args.train_dir,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers)

    val_loader   = get_drive_loader(args.val_dir,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers)

    # Model
    model = Detector(in_channels=3, num_classes=3).to(device)

    # Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type == "cuda")

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.lambda_depth,
        )

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"[{epoch:03d}] "
            f"loss={tr_loss:.4f} "
            f"train_iou={tr_metrics['iou']:.4f} val_iou={val_metrics['iou']:.4f} "
            f"val_depth={val_metrics['abs_depth_error']:.4f} "
            f"val_tp_depth={val_metrics['tp_depth_error']:.4f}"
        )

        # Save best model by IoU
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            path = save_model(model)
            print(f"  ↳ saved best model to {path} (mIoU={best_iou:.4f})")


if __name__ == "__main__":
    main()
