"""MockSimAdapter — synthetic FPV simulator for pre-official-spec development.

The simulation uses a drone-centric frame of reference: the gate moves in
the image as the drone moves.  This is geometrically simplified but fully
sufficient for testing the vision, control, and state-machine pipeline.

Gate model
----------
A single gate is represented as (cx, cy, scale):
  cx, cy   — normalized [0,1] gate center in the image
  scale    — controls rendered size; grows as drone approaches

When scale >= GATE_PASS_THRESHOLD the gate is "passed" and the next gate
is drawn.  When all gates are passed, done=True.

Frame generation
----------------
A dark background with a white hollow rectangle (gate).  Optional noise,
exposure variation, dropped frames, and latency spikes are configurable.

IMU generation
--------------
Simplified but action-correlated.  dt is always real wall-clock.
"""
from __future__ import annotations

import random
import time
from typing import Optional

import cv2
import numpy as np

from aigrandprix.adapters.base import AbstractAdapter
from aigrandprix.config import SimConfig
from aigrandprix.types import Action, Observation

# ---------------------------------------------------------------------------
# Physics constants (tuned so a moderate action closes in ~2–4 seconds)
# ---------------------------------------------------------------------------
_ROLL_GAIN = 0.25        # cx shift per unit roll per second
_PITCH_GAIN = 0.20       # cy shift per unit pitch per second
_APPROACH_GAIN = 0.35    # scale growth per unit throttle per second
_DECEL = 0.05            # passive scale deceleration per second
_INITIAL_SCALE = 0.08    # starting gate scale (far away)
_GATE_PASS_THRESHOLD = 0.85  # scale at which gate is considered passed
_MAX_DEVIATION = 0.95    # soft-crash if gate center drifts this far from image


class MockSimAdapter(AbstractAdapter):
    """Synthetic FPV environment for pipeline development."""

    def __init__(self, config: SimConfig):
        self._cfg = config
        self._rng: Optional[np.random.Generator] = None
        self._py_rng: Optional[random.Random] = None
        # State (set by reset)
        self._gate_cx = 0.5
        self._gate_cy = 0.5
        self._gate_scale = _INITIAL_SCALE
        self._gate_index = 0
        self._gate_passed_this_step = False
        self._prev_frame: Optional[np.ndarray] = None
        self._t_start: float = 0.0
        self._t_last: float = 0.0
        self._done = False
        # Generate gate lateral offsets at reset time (seeded)
        self._gate_offsets: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # AbstractAdapter interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Observation:
        effective_seed = seed if seed is not None else self._cfg.seed
        self._rng = np.random.default_rng(effective_seed)
        self._py_rng = random.Random(effective_seed)

        # Pre-generate gate positions for the full sequence
        self._gate_offsets = [
            (
                float(np.clip(
                    self._rng.normal(0.5, self._cfg.lateral_std), 0.1, 0.9)),
                float(np.clip(
                    self._rng.normal(0.5, self._cfg.lateral_std * 0.5), 0.2, 0.8)),
            )
            for _ in range(self._cfg.total_gates)
        ]

        self._gate_cx, self._gate_cy = self._gate_offsets[0]
        self._gate_scale = _INITIAL_SCALE
        self._gate_index = 0
        self._gate_passed_this_step = False
        self._done = False
        self._sim_t = 0.0              # nominal simulated time (seconds)
        self._nominal_dt = 1.0 / max(self._cfg.fps, 1)
        self._prev_frame = None

        return self._make_observation(dt=0.0)

    def step(self, action: Action) -> tuple[Observation, dict]:
        # Use nominal dt for physics (so the sim runs at fps-equivalent speed
        # regardless of real wall-clock — critical for fast batch experiments).
        dt = self._nominal_dt
        self._sim_t += dt

        # Optional latency spike (sleep to simulate real latency)
        if (self._cfg.latency_spike_prob > 0 and
                self._py_rng.random() < self._cfg.latency_spike_prob):
            spike_s = self._py_rng.uniform(0, self._cfg.max_latency_ms) / 1000.0
            time.sleep(spike_s)

        # --- Update gate position ---
        clamped = action.clamp()
        self._gate_cx -= clamped.roll * _ROLL_GAIN * dt
        self._gate_cy += clamped.pitch * _PITCH_GAIN * dt
        self._gate_scale += clamped.throttle * _APPROACH_GAIN * dt - _DECEL * dt
        self._gate_scale = max(self._gate_scale, 0.01)

        # Clamp gate center to image
        self._gate_cx = float(np.clip(self._gate_cx, 0.0, 1.0))
        self._gate_cy = float(np.clip(self._gate_cy, 0.0, 1.0))

        # --- Check gate pass ---
        self._gate_passed_this_step = False
        if self._gate_scale >= _GATE_PASS_THRESHOLD:
            self._gate_passed_this_step = True
            self._gate_index += 1
            if self._gate_index >= self._cfg.total_gates:
                self._done = True
            else:
                cx, cy = self._gate_offsets[self._gate_index]
                self._gate_cx = cx
                self._gate_cy = cy
                self._gate_scale = _INITIAL_SCALE

        obs = self._make_observation(dt=dt)
        lap_time = self._sim_t if self._done else 0.0

        info = {
            "done": self._done,
            "gate_passed": self._gate_passed_this_step,
            "gate_index": self._gate_index,
            "lap_time": lap_time,
        }
        return obs, info

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self, dt: float) -> Observation:
        t = self._sim_t
        H, W = self._cfg.resolution[0], self._cfg.resolution[1]

        # Dropped frame: return previous frame
        if (self._prev_frame is not None and
                self._cfg.drop_frame_prob > 0 and
                self._py_rng.random() < self._cfg.drop_frame_prob):
            image = self._prev_frame.copy()
        else:
            image = self._render_frame(H, W)
            self._prev_frame = image.copy()

        return Observation(
            t=t,
            dt=dt,
            image=image,
            imu_accel=self._make_accel(),
            imu_gyro=self._make_gyro(),
        )

    def _render_frame(self, H: int, W: int) -> np.ndarray:
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)   # dark gray background

        # Gate as white hollow rectangle
        gate_w = int(self._gate_scale * W * 0.7)
        gate_h = int(self._gate_scale * H * 0.7)
        cx_px = int(self._gate_cx * W)
        cy_px = int(self._gate_cy * H)
        x1 = max(0, cx_px - gate_w // 2)
        y1 = max(0, cy_px - gate_h // 2)
        x2 = min(W - 1, cx_px + gate_w // 2)
        y2 = min(H - 1, cy_px + gate_h // 2)
        thickness = max(2, int(self._gate_scale * 12))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness)

        # Gaussian noise
        if self._cfg.noise_std > 0:
            noise = self._rng.normal(0, self._cfg.noise_std, frame.shape)
            frame = np.clip(frame.astype(np.int16) + noise.astype(np.int16),
                            0, 255).astype(np.uint8)

        # Exposure variation (multiplicative)
        if self._cfg.exposure_variation > 0:
            factor = self._rng.uniform(
                1.0 - self._cfg.exposure_variation,
                1.0 + self._cfg.exposure_variation)
            frame = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return frame

    def _make_accel(self) -> np.ndarray:
        """IMU accelerometer: gravity + action-correlated component + noise."""
        # We don't have the current action here; return gravity + noise
        # The controller state is reflected in gate movement, not IMU directly
        accel = np.array([0.0, 0.0, 9.81])
        accel += self._rng.normal(0, self._cfg.accel_noise_std, 3)
        return accel

    def _make_gyro(self) -> np.ndarray:
        """IMU gyroscope: near-zero + noise (no action passthrough here)."""
        gyro = self._rng.normal(0, self._cfg.gyro_noise_std, 3)
        return gyro

    # ------------------------------------------------------------------
    # Test / debug helpers
    # ------------------------------------------------------------------

    @property
    def gate_state(self) -> dict:
        """Expose internal gate state for unit tests."""
        return {
            "cx": self._gate_cx,
            "cy": self._gate_cy,
            "scale": self._gate_scale,
            "index": self._gate_index,
            "done": self._done,
        }
