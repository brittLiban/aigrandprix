# AI Grand Prix

Autonomous FPV drone racing AI built for the **Anduril / DCL / Neros AI Grand Prix** competition.

Single FPV camera + IMU, Python, ~100 TOPS onboard compute. No GPS, no LiDAR, no absolute position. Fastest valid time wins.

---

## Results (700-episode benchmark, 100 seeds each)

| Config | Completion | Median Lap | p90 | Recoveries |
|---|---|---|---|---|
| BASE — HSV, easy sim | 100/100 | 45.0s | 46.1s | 0 |
| HSV + Aggressive tuning | 100/100 | 26.7s | 28.0s | 0 |
| HSV + Stress (noise, drops) | 100/100 | 30.9s | 33.0s | 0 |
| ML + Aggressive | 100/100 | **26.2s** | 26.2s | 0 |
| ML + Stress | 100/100 | **26.4s** | 27.1s | 0 |
| HSV + Hard *(orange gates, perspective)* | 100/100 | 61.4s | 61.4s | **100** |
| **ML + Hard** *(orange gates, perspective)* | **100/100** | **26.1s** | 26.2s | **0** |

**Key takeaway:** HSV completely breaks on colored perspective gates (100 recoveries, 2.4× slower). The ML CNN handles all conditions at the same lap time — 26.1s median, zero recoveries, across every seed it has never seen.

---

## Pipeline Architecture

```
 ┌─────────────────────────────────────────────────────┐
 │                  Each control step (~3ms)            │
 │                                                      │
 │  Camera frame ──► VisionLobe ──► gate detected?     │
 │                   (HSV or ML)    cx, cy, area        │
 │                                                      │
 │  IMU accel/gyro ► StabilityLobe ► stability score   │
 │                                   tumble detection   │
 │                                                      │
 │  Vision + State ► ProgressLobe ► approach rate      │
 │                                   alignment score    │
 │                                                      │
 │  Vision + State ► RecoveryLobe ► time since gate    │
 │                                   directed yaw hint  │
 │                                                      │
 │  All above ─────► RiskLobe ────► risk score [0,1]  │
 │                                   push level [0–3]   │
 │                                                      │
 │  All above ─────► FusionBrain ─► DroneState         │
 │                   state machine   control target     │
 │                                                      │
 │  Target + State ► Controller ──► Action             │
 │                   per-state PID   roll/pitch/yaw/    │
 │                   + push boost    throttle           │
 └─────────────────────────────────────────────────────┘
```

### State Machine

```
SEARCH ──► TRACK ──► APPROACH ──► COMMIT ──► SEARCH (next gate)
   ▲                                │
   │         RECOVER ◄──────────────┘  (any state → RECOVER on tumble or gate lost)
   └─────────────┘
```

| State | Behaviour |
|---|---|
| SEARCH | Sweep yaw using gate trajectory planner hint; low throttle |
| TRACK | Gate acquired; stabilise, align, build confidence |
| APPROACH | Centred and stable; accelerate toward gate |
| COMMIT | Gate fills frame; max throttle, push through |
| RECOVER | Tumbling or gate lost; directed yaw toward last known position |

---

## ML Vision Model

A lightweight **26K-parameter CNN** trained on synthetic data from the mock simulator.

```
Input: 128×160 RGB frame
  └─► Conv-BN-ReLU  3→ 8, stride 2   (64×80)
  └─► Conv-BN-ReLU  8→16, stride 2   (32×40)
  └─► Conv-BN-ReLU 16→32, stride 2   (16×20)
  └─► Conv-BN-ReLU 32→64, stride 2   ( 8×10)
  └─► Global Average Pool             (64,)
  └─► Linear 64→32 → ReLU → Linear 32→5
Output: [det_logit, cx, cy, bw, bh]
```

**Speed:** 0.56ms/frame on RTX 3060 Ti — leaves >99% of the pipeline budget for everything else.

**Training data:** 150,000 frames — white flat gates + orange perspective gates, 2 augmentation copies each. Photometric augmentation: brightness, contrast, Gaussian noise, gamma.

**Accuracy:** 99.85% detection accuracy, 0.0367 val loss after 30 epochs.

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Live demo — easy sim, HSV detector
python scripts/demo.py --config configs/base.yaml --seed 42

# Live demo — hard sim, ML detector (requires .venv-ml with CUDA torch)
.venv-ml\Scripts\python scripts/demo.py \
  --config configs/base.yaml configs/aggressive.yaml \
           configs/ml_vision.yaml configs/hard.yaml \
  --seed 42

# Single episode (headless, writes JSONL log)
python scripts/run_episode.py --config configs/base.yaml --seed 42

# Analyze a run log
python scripts/analyze_log.py logs/<run_id>.jsonl

# 700-episode benchmark (all 7 configs × 100 seeds)
.venv-ml\Scripts\python scripts/_validate.py
```

---

## Config System

Configs are deep-merged left to right — later files override earlier ones.

| File | Purpose |
|---|---|
| `configs/base.yaml` | All defaults — HSV vision, easy sim, conservative PID |
| `configs/aggressive.yaml` | Higher throttle, tighter PID, faster state transitions |
| `configs/stress.yaml` | Sensor noise, dropped frames, latency spikes |
| `configs/ml_vision.yaml` | Switch vision backend to ML CNN |
| `configs/hard.yaml` | Hard sim: orange gates, perspective rendering, textured background, heavy noise |

```bash
# Stack configs to build any combination
python scripts/run_episode.py \
  --config configs/base.yaml \
           configs/aggressive.yaml \
           configs/ml_vision.yaml \
           configs/hard.yaml
```

---

## Training the ML Model

```bash
# 1. Generate training data
.venv-ml\Scripts\python scripts/gen_dataset.py \
  --seeds 200 --aug-copies 2 --output data/gate_dataset

# 2. Generate hard-sim data (orange perspective gates)
.venv-ml\Scripts\python scripts/gen_dataset.py \
  --seeds 200 --aug-copies 2 \
  --output data/gate_dataset_hard \
  --config configs/hard.yaml

# 3. Train on combined dataset (multi-root — no disk copy)
.venv-ml\Scripts\python scripts/train_detector.py \
  --data "data/gate_dataset,data/gate_dataset_hard" \
  --epochs 30

# 4. Evaluate detection accuracy
.venv-ml\Scripts\python scripts/eval_detector.py \
  --data data/gate_dataset

# 5. Evaluate centering accuracy at gate pass
.venv-ml\Scripts\python scripts/eval_centering.py \
  --seeds 50 --seed-start 5000 \
  --config configs/base.yaml configs/aggressive.yaml \
           configs/ml_vision.yaml configs/hard.yaml
```

Best checkpoint saved to `checkpoints/gate_detector.pt` automatically.

---

## Project Structure

```
aigrandprix/
├── adapters/
│   ├── base.py          AbstractAdapter — reset(), step(action), close()
│   ├── mock.py          MockSimAdapter — synthetic gates, perspective rendering, IMU
│   └── official.py      OfficialSimAdapter stub (wired when spec arrives May 2026)
├── brain/
│   ├── states.py        DroneState enum
│   ├── fusion.py        FusionBrain — state machine, stuck detection, hysteresis
│   └── planner.py       GatePlanner — recency-weighted gate trajectory prediction
├── controller/
│   └── pid.py           PID + per-state gain profiles + anti-windup + push boost
├── lobes/
│   ├── vision.py        HSV contour detector + confidence EMA
│   ├── vision_ml.py     ML CNN detector — same interface as VisionLobe
│   ├── stability.py     IMU norms, tumble detection, spike tracking
│   ├── progress.py      Approach rate, alignment score, gate index
│   ├── recovery.py      Time-since-gate, directed recovery yaw
│   └── risk.py          Risk score, push level, recovery penalty
├── logging/
│   └── run_logger.py    JSONL run logger — header + steps + footer metrics
├── ml/
│   ├── model.py         GateDetector CNN + gate_loss
│   └── dataset.py       GateDataset — multi-root, RAM cache, photometric augment
├── config.py            Pydantic config + YAML deep-merge loader
├── runner.py            PipelineRunner — wires all components, main loop
└── types.py             Observation, Action, VisionResult, etc.
configs/                 YAML config overlays
scripts/
├── demo.py              Live OpenCV demo window
├── run_episode.py       Headless single episode
├── gen_dataset.py       Parallel dataset generator
├── train_detector.py    CNN training loop
├── eval_detector.py     Detection accuracy evaluation
├── eval_centering.py    Gate centering accuracy evaluation
├── _validate.py         700-episode parallel benchmark
├── analyze_log.py       JSONL log parser + stats
└── sweep.py             Config sweep + manifest
checkpoints/             Model weights (gitignored)
data/                    Generated datasets (gitignored)
logs/                    Run logs (gitignored)
tests/                   Unit + integration tests
```

---

## Roadmap

- [ ] **Webcam adapter** — run full pipeline on real camera feed for demos
- [ ] **Centering improvement** — tune commit threshold so drone centers before committing
- [ ] **Bbox calibration** — ML model over-predicts bbox size; add calibration factor
- [ ] **Official sim adapter** — wire `OfficialSimAdapter` when spec arrives (May 2026)
- [ ] **Quick fine-tune script** — 1-epoch fine-tune on real sim data at competition
- [ ] **Velocity estimation** — use IMU + area growth for true closing speed
- [ ] **Swarm coordination layer** — zone assignment + mesh comms for multi-drone inspection use case
