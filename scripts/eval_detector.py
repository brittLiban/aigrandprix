"""Evaluate trained GateDetector and compare against HSV baseline.

Runs both detectors on the validation split and prints a side-by-side
comparison: detection accuracy, bbox MAE, and inference speed.

Usage:
    python scripts/eval_detector.py --model checkpoints/gate_detector.pt
    python scripts/eval_detector.py --model checkpoints/gate_detector.pt --data data/gate_dataset
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aigrandprix.config import default_config


# ---------------------------------------------------------------------------
# HSV baseline detector (mirrors VisionLobe._detect logic)
# ---------------------------------------------------------------------------

def hsv_detect(image: np.ndarray, cfg) -> dict:
    """Run HSV+contour detection on a single image.
    Returns dict with detected, cx, cy, bw, bh."""
    H_orig, W_orig = image.shape[:2]
    img = cv2.resize(image, (cfg.vision.resize_w, cfg.vision.resize_h))
    H, W = cfg.vision.resize_h, cfg.vision.resize_w

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array(cfg.vision.hsv_lower, dtype=np.uint8)
    upper = np.array(cfg.vision.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"detected": False, "cx": 0.5, "cy": 0.5, "bw": 0.0, "bh": 0.0}

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < cfg.vision.min_contour_area:
        return {"detected": False, "cx": 0.5, "cy": 0.5, "bw": 0.0, "bh": 0.0}

    x, y, w, h = cv2.boundingRect(best)
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    bw = w / W
    bh = h / H

    return {"detected": True, "cx": cx, "cy": cy, "bw": bw, "bh": bh}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_both(data_dir: str, model_path: str,
                  input_h: int, input_w: int,
                  conf_threshold: float) -> None:
    try:
        import torch
        _torch_ok = True
    except ImportError:
        _torch_ok = False
        print("WARNING: torch not available, skipping ML evaluation")

    from aigrandprix.ml.dataset import GateDataset, dataset_stats

    stats = dataset_stats(data_dir)
    print(f"Dataset: {stats['total']} frames  "
          f"({100*stats['pos_frac']:.1f}% positive)")

    val_ds = GateDataset(data_dir, input_h=input_h, input_w=input_w,
                         augment=False, split="val")
    print(f"Val set: {len(val_ds)} frames")

    cfg = default_config()

    # --- Load ML model ---
    ml_model = None
    if _torch_ok and model_path and Path(model_path).exists():
        from aigrandprix.ml.model import GateDetector
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)
        loaded_h = ckpt.get("input_h", input_h)
        loaded_w = ckpt.get("input_w", input_w)
        ml_model = GateDetector(input_h=loaded_h, input_w=loaded_w)
        ml_model.load_state_dict(state)
        ml_model.eval()
        print(f"Loaded model from {model_path}  (input {loaded_h}x{loaded_w})")
    elif model_path:
        print(f"WARNING: model not found at {model_path}")

    # --- Metrics accumulators ---
    hsv_tp = hsv_fp = hsv_tn = hsv_fn = 0
    hsv_bbox_err = hsv_pos_n = 0.0
    ml_tp = ml_fp = ml_tn = ml_fn = 0
    ml_bbox_err = ml_pos_n = 0.0

    hsv_times = []
    ml_times  = []

    import json
    labels_path = Path(data_dir) / "labels.jsonl"
    with open(labels_path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    val_rows = rows[-max(1, int(len(rows) * 0.15)):]

    for row in val_rows:
        img_path = Path(data_dir) / row["file"]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        gt_det = row["gate_detected"]
        gt_cx, gt_cy = row["cx"], row["cy"]
        gt_bw, gt_bh = row["bw"], row["bh"]

        # --- HSV ---
        t0 = time.perf_counter()
        hsv_res = hsv_detect(img_rgb, cfg)
        hsv_times.append((time.perf_counter() - t0) * 1000)

        if hsv_res["detected"] and gt_det:
            hsv_tp += 1
            err = (abs(hsv_res["cx"] - gt_cx) + abs(hsv_res["cy"] - gt_cy) +
                   abs(hsv_res["bw"] - gt_bw) + abs(hsv_res["bh"] - gt_bh)) / 4
            hsv_bbox_err += err
            hsv_pos_n += 1
        elif hsv_res["detected"] and not gt_det:
            hsv_fp += 1
        elif not hsv_res["detected"] and gt_det:
            hsv_fn += 1
        else:
            hsv_tn += 1

        # --- ML ---
        if ml_model is not None:
            img_small = cv2.resize(img_rgb, (loaded_w, loaded_h))
            t0 = time.perf_counter()
            tensor = (torch.from_numpy(img_small).permute(2, 0, 1)
                      .unsqueeze(0).float().div(255.0))
            with torch.no_grad():
                raw = ml_model(tensor)
            ml_times.append((time.perf_counter() - t0) * 1000)

            ml_conf = float(torch.sigmoid(raw[0, 0]).item())
            ml_det = ml_conf >= conf_threshold
            ml_cx, ml_cy = float(raw[0, 1]), float(raw[0, 2])
            ml_bw, ml_bh = float(raw[0, 3]), float(raw[0, 4])

            if ml_det and gt_det:
                ml_tp += 1
                err = (abs(ml_cx - gt_cx) + abs(ml_cy - gt_cy) +
                       abs(ml_bw - gt_bw) + abs(ml_bh - gt_bh)) / 4
                ml_bbox_err += err
                ml_pos_n += 1
            elif ml_det and not gt_det:
                ml_fp += 1
            elif not ml_det and gt_det:
                ml_fn += 1
            else:
                ml_tn += 1

    # --- Print results ---
    n = len(val_rows)
    print(f"\n{'Metric':<22}  {'HSV':>12}  {'ML':>12}")
    print("-" * 50)

    def pct(num, den):
        return f"{100*num/max(den,1):.1f}%"

    hsv_acc = (hsv_tp + hsv_tn) / max(n, 1)
    ml_acc  = (ml_tp  + ml_tn)  / max(n, 1)

    hsv_recall = hsv_tp / max(hsv_tp + hsv_fn, 1)
    ml_recall  = ml_tp  / max(ml_tp  + ml_fn,  1)

    hsv_prec = hsv_tp / max(hsv_tp + hsv_fp, 1)
    ml_prec  = ml_tp  / max(ml_tp  + ml_fp,  1)

    print(f"{'Accuracy':<22}  {pct(hsv_tp+hsv_tn, n):>12}  {pct(ml_tp+ml_tn, n):>12}")
    print(f"{'Recall (det rate)':<22}  {pct(hsv_tp, hsv_tp+hsv_fn):>12}  {pct(ml_tp, ml_tp+ml_fn):>12}")
    print(f"{'Precision':<22}  {pct(hsv_tp, hsv_tp+hsv_fp):>12}  {pct(ml_tp, ml_tp+ml_fp):>12}")
    print(f"{'False positives':<22}  {hsv_fp:>12}  {ml_fp:>12}")
    print(f"{'False negatives':<22}  {hsv_fn:>12}  {ml_fn:>12}")
    print(f"{'Bbox MAE (pos only)':<22}  {hsv_bbox_err/max(hsv_pos_n,1):>12.4f}  "
          f"{ml_bbox_err/max(ml_pos_n,1):>12.4f}")

    if hsv_times:
        ml_ms = f"{float(np.median(ml_times)):>12.2f}" if ml_times else f"{'N/A':>12}"
        print(f"{'Inference ms (median)':<22}  {float(np.median(hsv_times)):>12.2f}  {ml_ms}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gate detector")
    parser.add_argument("--model", default="checkpoints/gate_detector.pt",
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--data", default="data/gate_dataset",
                        help="Dataset root (default: data/gate_dataset)")
    parser.add_argument("--input-h", type=int, default=128)
    parser.add_argument("--input-w", type=int, default=160)
    parser.add_argument("--conf-threshold", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    args = parser.parse_args()

    evaluate_both(
        data_dir=args.data,
        model_path=args.model,
        input_h=args.input_h,
        input_w=args.input_w,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
