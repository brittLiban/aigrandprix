# Training the ML Detector

## Overview

The GateDetector CNN is trained entirely on **synthetic data** generated from the mock simulator. The training pipeline:

1. Generate labeled frames from sim episodes
2. Train CNN on combined dataset
3. Evaluate detection + centering accuracy
4. Validate end-to-end with the full pipeline

---

## Dataset Generation

```bash
# Basic dataset — white flat gates, easy sim (≈75K frames)
.venv-ml\Scripts\python scripts/gen_dataset.py \
  --seeds 200 \
  --aug-copies 2 \
  --output data/gate_dataset

# Hard dataset — orange perspective gates (≈75K frames)
.venv-ml\Scripts\python scripts/gen_dataset.py \
  --seeds 200 \
  --aug-copies 2 \
  --output data/gate_dataset_hard \
  --config configs/hard.yaml
```

**What the generator does:**
- Runs N episodes with varied random seeds
- Each episode: random pitch/throttle/roll actions to create diverse approach angles
- Labels each frame with `gate_detected`, `cx`, `cy`, `bw`, `bh`, `area`, `scale`
- Injects 20% blank negative frames (no gate) per episode
- Applies 2 photometric augmentation copies: brightness/contrast, Gaussian noise, gamma

**Label format** (`labels.jsonl`, one JSON per line):
```json
{"file": "images/frame_000000.jpg", "gate_detected": true,
 "cx": 0.52, "cy": 0.48, "bw": 0.14, "bh": 0.18,
 "area": 43008.0, "scale": 0.21}
```

---

## Training

```bash
# Train on single dataset
.venv-ml\Scripts\python scripts/train_detector.py \
  --data data/gate_dataset \
  --epochs 30

# Train on multiple datasets (no disk copy needed)
.venv-ml\Scripts\python scripts/train_detector.py \
  --data "data/gate_dataset,data/gate_dataset_hard" \
  --epochs 30 \
  --batch-size 512 \
  --lr 1e-3
```

**Key training details:**
- Optimizer: Adam, lr=1e-3, CosineAnnealingLR scheduler
- Loss: `BCEWithLogitsLoss(det) + 5.0 × SmoothL1(bbox)` — bbox loss only on positive samples
- Val split: last 15% of data, deterministic
- Best checkpoint saved by val loss (not last epoch)
- RAM cache: all images loaded into RAM at init — eliminates disk I/O bottleneck

**Training results (150K frames, 30 epochs, RTX 3060 Ti):**
```
Epoch   Train Loss   Val Loss   Det Acc   BBox MAE   Time
    1       0.2734     0.1554    0.9812     0.1105   37.7s
   10       0.0267     0.1246    0.9819     0.0924   34.7s
   27       0.0236     0.0367    0.9985     0.0731   34.8s  *best
   30       0.0235     0.0367    0.9985     0.0731   37.7s

Inference speed: 0.56 ms/frame on CUDA
```

### Resume Training

```bash
.venv-ml\Scripts\python scripts/train_detector.py \
  --data "data/gate_dataset,data/gate_dataset_hard" \
  --epochs 20 \
  --resume checkpoints/gate_detector.pt
```

---

## Evaluation

### Detection Accuracy

```bash
.venv-ml\Scripts\python scripts/eval_detector.py --data data/gate_dataset
```

Reports precision, recall, F1 at various confidence thresholds.

### Gate Centering Accuracy

Measures how accurately the drone centers on gates at pass time across unseen seeds:

```bash
.venv-ml\Scripts\python scripts/eval_centering.py \
  --seeds 50 \
  --seed-start 5000 \
  --config configs/base.yaml configs/aggressive.yaml \
           configs/ml_vision.yaml configs/hard.yaml
```

**Current results (seeds 5000–5049, never seen during training):**
```
Detection rate at pass : 100.0%
cx error  mean=0.139  median=0.132  p90=0.288
cy error  mean=0.121  median=0.109  p90=0.240

Perfect  (<0.05) :  6.8%
Good     (<0.10) : 23.0%
OK       (<0.20) : 69.6%
Off-center(>0.20): 30.4%
```

### End-to-End Benchmark

```bash
.venv-ml\Scripts\python scripts/_validate.py
```

700 episodes across 7 configs × 100 seeds. Key metric: ML+Hard = 26.1s median, 0 recoveries.

---

## Fine-Tuning for Competition

When the official simulator arrives (May 2026):

```bash
# 1. Collect ~1000 frames from official sim
python scripts/gen_dataset.py \
  --adapter official \
  --seeds 10 \
  --output data/official_sim

# 2. Fine-tune — 5 epochs on official data mixed with synthetic
.venv-ml\Scripts\python scripts/train_detector.py \
  --data "data/gate_dataset,data/gate_dataset_hard,data/official_sim" \
  --epochs 5 \
  --lr 1e-4 \
  --resume checkpoints/gate_detector.pt
```

Even 1000 real frames mixed into 150K synthetic is typically enough to close the sim-to-real gap for detection.

---

## Preventing Overfitting

| Technique | Where |
|---|---|
| Train/val split (85/15) | `GateDataset(split="train"/"val")` |
| Best val loss checkpoint | `train_detector.py` saves on val improvement only |
| Photometric augmentation | Baked into dataset at gen time (not runtime) |
| 100 diverse seeds | Varied gate positions, approach angles |
| Small model (26K params) | Hard to badly overfit on 150K samples |
| Multi-domain training | White flat + orange perspective = prevents single-domain overfit |

**The real overfitting risk** is sim-to-real transfer, not statistical overfitting. Synthetic gates look different from real ones. The hard sim was specifically designed to close this gap (perspective rendering, varied colors, textured backgrounds).
