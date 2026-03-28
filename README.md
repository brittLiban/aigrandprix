# AI Grand Prix

Autonomous FPV drone racing pipeline built for the Anduril / DCL / Neros AI Grand Prix competition. Single camera, IMU, Python — no LiDAR, no GPS, no absolute position.

## Results

| Config | Completion | Median Lap | Recoveries |
|---|---|---|---|
| BASE (HSV, easy sim) | 100/100 | 45.0s | 0 |
| HSV + Aggressive | 100/100 | 26.7s | 0 |
| HSV + Stress (noise, dropped frames) | 100/100 | 30.9s | 0 |
| **ML + Aggressive** | **100/100** | **26.2s** | **0** |
| **ML + Stress** | **100/100** | **26.4s** | **0** |
| HSV + Hard (orange gates, perspective) | 100/100 | 61.4s | 100 |
| **ML + Hard** | **100/100** | **26.1s** | **0** |

ML model handles all conditions — including colored gates and perspective rendering — at the same lap time as the easy sim.

---

## Architecture

```
Camera frame + IMU
       │
       ▼
  VisionLobe          ← HSV contour detector OR ML CNN (26K params, 0.56ms/frame)
  StabilityLobe       ← IMU norms, tumble detection, instability spike tracking
  ProgressLobe        ← approach rate, alignment score, gate index
  RecoveryLobe        ← time since gate, directed recovery toward last known position
  RiskLobe            ← risk score, push level (0–3)
       │
       ▼
  FusionBrain         ← state machine: SEARCH → TRACK → APPROACH → COMMIT → RECOVER
       │
       ▼
  Controller          ← per-state PID profiles, anti-windup, throttle boost by push level
       │
       ▼
    Action            ← roll, pitch, yaw, throttle [-1, 1]
```

---

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run a single episode (HSV detector, easy sim)
python scripts/run_episode.py --config configs/base.yaml --seed 42

# Run with ML detector on hard sim
python scripts/run_episode.py \
  --config configs/base.yaml configs/aggressive.yaml \
           configs/ml_vision.yaml configs/hard.yaml

# Live demo window
python scripts/demo.py --config configs/base.yaml

# Analyze a run log
python scripts/analyze_log.py logs/<run_id>.jsonl
```

---

## ML Model

A lightweight CNN (26K parameters) that detects gates from a single RGB frame.

**Input:** 128×160 RGB image
**Output:** `[detection_confidence, cx, cy, bbox_width, bbox_height]`
**Speed:** 0.56ms/frame on RTX 3060 Ti

### Training

```bash
# Generate dataset (flat white gates, easy sim)
python scripts/gen_dataset.py --seeds 200 --aug-copies 2 --output data/gate_dataset

# Generate hard dataset (orange perspective gates)
python scripts/gen_dataset.py \
  --seeds 200 --aug-copies 2 \
  --output data/gate_dataset_hard \
  --config configs/hard.yaml

# Train on combined dataset (multi-root, no disk copy needed)
python scripts/train_detector.py \
  --data "data/gate_dataset,data/gate_dataset_hard" \
  --epochs 30

# Evaluate detector accuracy
python scripts/eval_detector.py --data data/gate_dataset
```

### Validation (700-episode benchmark)

```bash
python scripts/_validate.py
```

Runs 7 configs × 100 seeds in parallel (4 workers). Prints median lap time, min/max, p90, and recovery event count per config.

---

## Configs

| File | Purpose |
|---|---|
| `configs/base.yaml` | Default config — all settings, easy sim, HSV vision |
| `configs/aggressive.yaml` | Aggressive PID profiles, higher throttle |
| `configs/stress.yaml` | Sensor noise, dropped frames, latency spikes |
| `configs/ml_vision.yaml` | Switch vision backend to ML CNN |
| `configs/hard.yaml` | Hard sim: orange gates, perspective rendering, textured background |

Configs are deep-merged left to right:

```bash
python scripts/run_episode.py \
  --config configs/base.yaml configs/aggressive.yaml configs/ml_vision.yaml
```

---

## Mock Simulator

Drone-centric frame of reference — gate moves in the image as the drone moves.

**Flat mode:** axis-aligned white rectangle on dark background
**Perspective mode:** 3D pinhole projection with yaw/pitch rotation — produces realistic trapezoid shape when approaching off-center

Configurable: gate color, color jitter, background texture, Gaussian noise, exposure variation, dropped frames, latency spikes.

---

## Project Structure

```
aigrandprix/
├── adapters/       mock sim + official sim stub (wired in May)
├── brain/          state machine (FusionBrain) + gate trajectory planner
├── controller/     PID with per-state gain profiles
├── lobes/          vision, stability, progress, recovery, risk
├── logging/        JSONL run logger
├── ml/             GateDetector CNN + GateDataset
configs/            YAML config overlays
scripts/            run_episode, demo, gen_dataset, train_detector, validate
checkpoints/        trained model weights
data/               generated datasets (gitignored)
tests/              unit + integration tests
```

---

## Roadmap

- [ ] Webcam adapter — run full pipeline on real-world camera feed
- [ ] Fine-tune script for official sim data (arriving May 2026)
- [ ] Wire `OfficialSimAdapter` when spec released
- [ ] Swarm coordination layer (multiple drones, zone assignment)
- [ ] Velocity estimation from IMU + area growth rate
