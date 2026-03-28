"""Train the GateDetector CNN on the generated dataset.

Usage:
    # Quick test (few epochs)
    python scripts/train_detector.py --data data/gate_dataset --epochs 10

    # Full training
    python scripts/train_detector.py --data data/gate_dataset --epochs 50

    # Resume from checkpoint
    python scripts/train_detector.py --data data/gate_dataset --resume checkpoints/gate_detector.pt

Saves:
    checkpoints/gate_detector.pt          best validation checkpoint
    checkpoints/gate_detector_last.pt     final checkpoint

Prints a training summary table at the end.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    print("ERROR: torch is required for training.  pip install torch")
    sys.exit(1)

from aigrandprix.ml.model import GateDetector, gate_loss
from aigrandprix.ml.dataset import GateDataset, dataset_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(model: GateDetector, loader: DataLoader,
             device: str, conf_threshold: float = 0.5) -> dict:
    """Compute val metrics: loss, det accuracy, bbox MAE (positive only)."""
    model.eval()
    total_loss = det_correct = n_total = n_pos = bbox_mae = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss, _ = gate_loss(pred, labels)
            total_loss += loss.item() * images.shape[0]

            # Detection accuracy
            det_pred  = (torch.sigmoid(pred[:, 0]) >= conf_threshold).float()
            det_label = (labels[:, 0] >= 0.5).float()
            det_correct += (det_pred == det_label).sum().item()
            n_total += images.shape[0]

            # Bbox MAE on positive samples
            pos_mask = det_label > 0.5
            if pos_mask.any():
                bbox_pred  = pred[pos_mask, 1:]
                bbox_label = labels[pos_mask, 1:]
                bbox_mae += (bbox_pred - bbox_label).abs().mean().item() * pos_mask.sum().item()
                n_pos += pos_mask.sum().item()

    return {
        "loss": total_loss / max(n_total, 1),
        "det_acc": det_correct / max(n_total, 1),
        "bbox_mae": bbox_mae / max(n_pos, 1),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    input_h: int,
    input_w: int,
    checkpoint_dir: str,
    resume: str,
    num_workers: int,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Dataset — support comma-separated list of roots
    roots = [d.strip() for d in data_dir.split(",") if d.strip()]
    data_arg = roots if len(roots) > 1 else roots[0]
    stats = dataset_stats(roots[0])
    print(f"Dataset: {stats['total']} frames from {len(roots)} source(s)  "
          f"({100*stats['pos_frac']:.1f}% positive)")
    train_ds = GateDataset(data_arg, input_h=input_h, input_w=input_w,
                           augment=False, split="train")
    val_ds   = GateDataset(data_arg, input_h=input_h, input_w=input_w,
                           augment=False, split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device == "cuda"))

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = GateDetector(input_h=input_h, input_w=input_w).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt.get("model", ckpt))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "gate_detector.pt")
    last_path = os.path.join(checkpoint_dir, "gate_detector_last.pt")

    history = []
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  "
          f"{'Det Acc':>8}  {'BBox MAE':>9}  {'Time':>6}")
    print("-" * 60)

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        t0 = time.perf_counter()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(images)
            loss, _ = gate_loss(pred, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.shape[0]

        train_loss /= max(len(train_ds), 1)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        is_best = val_metrics["loss"] < best_val_loss

        if is_best:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
                "input_h": input_h,
                "input_w": input_w,
            }, best_path)

        print(f"{epoch+1:>6}  {train_loss:>10.4f}  "
              f"{val_metrics['loss']:>9.4f}  "
              f"{val_metrics['det_acc']:>8.4f}  "
              f"{val_metrics['bbox_mae']:>9.4f}  "
              f"{elapsed:>5.1f}s"
              + (" *" if is_best else ""), flush=True)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val_metrics,
        })

    # Save last checkpoint
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": start_epoch + epochs,
        "val_loss": val_metrics["loss"],
        "input_h": input_h,
        "input_w": input_w,
    }, last_path)

    print(f"\nBest checkpoint: {best_path}  (val loss {best_val_loss:.4f})")
    print(f"Last checkpoint: {last_path}")

    # Benchmark inference speed
    model.eval()
    dummy = torch.zeros(1, 3, input_h, input_w).to(device)
    # Warmup
    for _ in range(10):
        model(dummy)
    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            model(dummy)
    avg_ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"Inference speed: {avg_ms:.2f} ms/frame on {device}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train GateDetector CNN")
    parser.add_argument("--data", default="data/gate_dataset",
                        help="Dataset root (default: data/gate_dataset)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--input-h", type=int, default=128,
                        help="Model input height (default: 128)")
    parser.add_argument("--input-w", type=int, default=160,
                        help="Model input width (default: 160)")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Where to save checkpoints (default: checkpoints/)")
    parser.add_argument("--resume", default="",
                        help="Resume from a checkpoint file")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes (default: 0)")
    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        input_h=args.input_h,
        input_w=args.input_w,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
