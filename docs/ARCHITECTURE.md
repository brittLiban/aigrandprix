# Architecture Deep Dive

## Design Philosophy

The system is built around three constraints:
1. **No ground truth position** — the drone never knows where it is in the world, only what it sees
2. **15ms pipeline budget** — every lobe must finish before the next frame arrives
3. **Adapter pattern** — mock sim, official sim, and real drone are all swappable behind `AbstractAdapter`

---

## Data Flow

Every control step follows this exact sequence:

```python
vision_r    = vision(obs)           # what does the camera see?
stability_r = stability(obs)        # is the drone stable?
progress_r  = progress(obs, vision_r, state, time_in_state)  # making progress?
recovery_r  = recovery(vision_r, state)   # lost the gate?
risk_r      = risk(stability_r, progress_r, recovery_r, t)   # safe to push?
state, target = brain(vision_r, stability_r, progress_r, recovery_r, risk_r)
action      = controller(target, obs, dt, state, push_level=risk_r.push_level)
obs, info   = adapter.step(action)
```

Each lobe is **stateful** (maintains EMA, counters, history) but receives all inputs explicitly — no shared mutable state between lobes.

---

## VisionLobe

**HSV backend:** converts frame to HSV, thresholds for white/bright pixels, finds contours, selects the largest contour above `min_contour_area`. Returns normalized `(cx, cy, area, bbox)`.

**ML backend:** resizes frame to 128×160, runs GateDetector CNN, returns `(cx, cy, bw, bh)` from the sigmoid-activated bbox head. Confidence from `sigmoid(det_logit)`.

**Both backends share:**
- EMA smoothing on `confidence_ema` (`alpha=0.6`)
- Exponential decay when gate not detected (`decay_rate=2.0/s`)
- Position hold for N frames after detection loss (last known `cx, cy` held for up to 4 frames)

**Why EMA confidence?** Raw per-frame detection is noisy. A single missed frame shouldn't drop confidence to zero and trigger RECOVER — EMA smooths this out.

---

## FusionBrain State Machine

### Entry Conditions

| Transition | Condition |
|---|---|
| SEARCH → TRACK | `confidence_ema > 0.4` |
| TRACK → APPROACH | `aligned_score > 0.7 AND stability_score > 0.5` |
| APPROACH → COMMIT | `vision.area > 40000 px²` |
| COMMIT → SEARCH | gate passes (area spike then disappears) |
| Any → RECOVER | `is_tumbling OR frames_since_gate > 0.5s` |
| RECOVER → SEARCH | `stability_score > 0.5 AND frames_since_gate == 0` |

### Hysteresis

TRACK→APPROACH uses `approach_aligned_min * 0.8` as the exit threshold — stricter than entry. This prevents the drone from chattering between states when alignment is borderline.

### Stuck Detection

- SEARCH timeout: if in SEARCH for >3s without acquiring → force RECOVER
- APPROACH timeout: if in APPROACH for >2s without progressing → force RECOVER

### Control Targets per State

| State | Roll | Pitch | Yaw | Throttle |
|---|---|---|---|---|
| SEARCH | 0 | 0 | sweep + planner hint | 0.4 |
| TRACK | center | center | align | 0.5 |
| APPROACH | center | forward | align | 0.65 |
| COMMIT | 0 | max forward | 0 | 0.85 |
| RECOVER | 0 | 0 | last known direction | 0.35 |

Throttle is further boosted by `push_level × 0.03` from RiskLobe.

---

## Controller

Per-state PID gain profiles selected by `DroneState`. Separate PID instances per axis.

```
SEARCH:   yaw_kp=0.7  (fast scan)      throttle=0.4
TRACK:    yaw_kp=0.5  (smooth track)   throttle=0.5
APPROACH: yaw_kp=0.4  (precise)        throttle=0.65
COMMIT:   yaw_kp=0.2  (hold course)    throttle=0.85
RECOVER:  yaw_kp=0.8  (fast reacquire) throttle=0.35
```

**Anti-windup:** integral clamped at ±0.3 to prevent windup during long SEARCH phases.

**Integral reset on RECOVER:** previous gate's direction bias is cleared so recovery doesn't fight the new gate direction.

---

## RiskLobe

Risk score = `base_risk + recovery_penalty + spike_penalty`

- **Base risk:** function of `1 - stability_score * aligned_score`
- **Recovery penalty:** if last recovery was <2s ago, adds up to 0.3 extra risk
- **Spike penalty:** `instability_spike_count × 0.05`

Push levels:

| Push Level | Risk Score | Throttle Boost |
|---|---|---|
| 3 (max) | < 0.35 | +0.09 |
| 2 | < 0.60 | +0.06 |
| 1 | < 0.80 | +0.03 |
| 0 (conservative) | ≥ 0.80 | 0 |

**The key insight:** speed is only applied when the system is confident, stable, and hasn't recently crashed. This is what allows 26s lap times without any recoveries.

---

## GatePlanner

Builds an exponential recency-weighted history of gate `cx` positions. When entering SEARCH, predicts which direction the next gate is likely in.

```python
weights = [RECENCY_BASE ** i for i in range(len(history))]
pred_cx = weighted_average(history, weights)
# Regress toward center (gates are random in mock sim)
pred_cx = pred_cx * (1 - CENTER_PULL) + 0.5 * CENTER_PULL
search_yaw_hint = -(pred_cx - 0.5) * 2.0
```

**In the mock sim:** gates are randomly placed so the planner has limited impact. On a real course with a fixed layout, the planner will predict the next gate direction accurately by lap 2.

---

## Mock Simulator

### Flat Mode

Axis-aligned white rectangle on dark background. Gate position `(cx, cy, scale)` updated each step:

```python
cx    -= action.roll     * ROLL_GAIN     * dt   # 0.25
cy    += action.pitch    * PITCH_GAIN    * dt   # 0.20
scale += action.throttle * APPROACH_GAIN * dt   # 0.35
scale -= DECEL * dt                             # 0.05
```

Pass when `scale >= 0.85`.

### Perspective Mode (`configs/hard.yaml`)

Gate modelled as a physical square in 3D space. Camera at origin, gate at `(gx, gy, gz)` where `gz = 1/scale`. Gate rotated to face approach angle:

```python
yaw_off   = -(cx - 0.5) * 0.9    # gate tilts as drone goes off-axis
pitch_off =  (cy - 0.5) * 0.5
```

Four corners projected through pinhole camera to screen pixels. Rendered with `cv2.polylines`. Produces realistic trapezoid shape on off-axis approach.

---

## Timing

Every lobe is wrapped with `@timed(budget_ms)`. Overruns log a warning and increment `timing_violations` — they never abort a run.

| Lobe | Budget |
|---|---|
| VisionLobe (HSV) | 8ms |
| VisionLobe (ML) | 8ms (actual: ~3ms) |
| StabilityLobe | 2ms |
| ProgressLobe | 2ms |
| RecoveryLobe | 1ms |
| RiskLobe | 1ms |
| **Total pipeline** | **15ms** |

At 60fps (16.7ms/frame), this leaves ~1.7ms margin. In practice the ML pipeline runs in ~3ms total.
