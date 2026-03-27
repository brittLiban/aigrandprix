"""Core data contracts shared across the entire pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Observation:
    """Raw sensor data from the adapter (mock or official)."""
    t: float               # wall-clock time, seconds
    dt: float              # seconds since last observation (real wall-clock delta)
    image: np.ndarray      # (H, W, 3) uint8 RGB
    imu_accel: np.ndarray  # (3,) float64, m/s^2 in body frame
    imu_gyro: np.ndarray   # (3,) float64, rad/s in body frame
    motor_rpm: Optional[np.ndarray] = None  # (4,) float64, virtual only
    battery: Optional[float] = None         # volts or percent; physical stage
    meta: dict = field(default_factory=dict)


@dataclass
class Action:
    """Drone control command output by the Controller."""
    roll: float      # [-1, 1]
    pitch: float     # [-1, 1]
    yaw: float       # [-1, 1]
    throttle: float  # [0, 1]

    def clamp(self) -> "Action":
        """Return a new Action with all fields clamped to valid ranges."""
        return Action(
            roll=float(np.clip(self.roll, -1.0, 1.0)),
            pitch=float(np.clip(self.pitch, -1.0, 1.0)),
            yaw=float(np.clip(self.yaw, -1.0, 1.0)),
            throttle=float(np.clip(self.throttle, 0.0, 1.0)),
        )

    @staticmethod
    def zero() -> "Action":
        return Action(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.0)

    @staticmethod
    def hover() -> "Action":
        return Action(roll=0.0, pitch=0.0, yaw=0.0, throttle=0.5)


@dataclass
class VisionResult:
    """Output of VisionLobe: gate detection and tracking state."""
    gate_detected: bool
    cx: float               # gate center x, normalized [0, 1]; 0.5 = image center
    cy: float               # gate center y, normalized [0, 1]; 0.5 = image center
    area: float             # bounding-box area in pixels^2
    confidence: float       # raw detection confidence, [0, 1]
    confidence_ema: float   # EMA-smoothed confidence (temporal stability)
    last_seen_t: float      # wall time of last positive detection
    bbox: Optional[tuple]   # (x, y, w, h) pixel coords or None
    latency_ms: float       # VisionLobe processing time

    @staticmethod
    def null(t: float = 0.0) -> "VisionResult":
        """Empty result when no gate is detected (and no history)."""
        return VisionResult(
            gate_detected=False, cx=0.5, cy=0.5, area=0.0,
            confidence=0.0, confidence_ema=0.0, last_seen_t=t,
            bbox=None, latency_ms=0.0,
        )


@dataclass
class StabilityResult:
    """Output of StabilityLobe: IMU-based stability assessment."""
    accel_norm: float            # |accel| m/s^2
    gyro_norm: float             # |gyro| rad/s
    stability_score: float       # [0, 1]; 1 = perfectly stable
    is_tumbling: bool            # gyro_norm > tumble threshold
    instability_spike_count: int # spikes above threshold in recent window

    @staticmethod
    def stable() -> "StabilityResult":
        return StabilityResult(
            accel_norm=9.81, gyro_norm=0.0,
            stability_score=1.0, is_tumbling=False,
            instability_spike_count=0,
        )


@dataclass
class ProgressResult:
    """Output of ProgressLobe: gate approach and alignment metrics."""
    dx: float              # gate center offset x, normalized [-1, 1]; 0 = centered
    dy: float              # gate center offset y, normalized [-1, 1]; 0 = centered
    approach_rate: float   # d(area)/dt, EMA-smoothed; positive = closing
    aligned_score: float   # [0, 1]; 1 = perfectly centered
    progress_score: float  # approach_rate * aligned_score; overall closing quality
    gate_index: int        # monotonic gate counter inferred from area transitions
    time_in_state: float   # seconds spent in the current FusionBrain state


@dataclass
class RecoveryResult:
    """Output of RecoveryLobe: lost-gate detection and directed recovery."""
    frames_since_gate: float  # seconds since last positive gate detection
    in_recovery: bool
    suggested_yaw: float      # directed yaw hint: -1=left, 0=no pref, +1=right
    last_known_cx: float      # last cx when gate was positively detected
    last_known_cy: float      # last cy when gate was positively detected


@dataclass
class PlannerResult:
    """Output of GatePlanner: prediction of next gate position."""
    predicted_cx: float    # predicted gate center x [0, 1]
    predicted_cy: float    # predicted gate center y [0, 1]
    search_yaw_hint: float # suggested initial yaw toward predicted gate [-1, 1]
    confidence: float      # prediction confidence [0, 1]; 0 = no history
    gates_seen: int        # total gates acquired so far


@dataclass
class RiskResult:
    """Output of RiskLobe: risk score and aggression scheduling."""
    risk_score: float              # [0, 1]; 0 = safe, 1 = critical
    push_level: int                # 0=conservative, 1=moderate, 2=aggressive, 3=max
    safe_to_push: bool
    recent_recovery_penalty: float # extra risk contribution from recent recovery events
