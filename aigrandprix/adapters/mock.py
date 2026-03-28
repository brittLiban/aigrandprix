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

import math
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
        if self._cfg.render_mode == "perspective":
            frame = self._render_perspective(H, W)
        else:
            frame = self._render_flat(H, W)

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

    def _gate_color(self) -> tuple:
        """Return gate color with optional per-frame jitter."""
        r, g, b = self._cfg.gate_color
        if self._cfg.gate_color_jitter > 0:
            j = self._cfg.gate_color_jitter * 255
            r = int(np.clip(r + self._rng.uniform(-j, j), 0, 255))
            g = int(np.clip(g + self._rng.uniform(-j, j), 0, 255))
            b = int(np.clip(b + self._rng.uniform(-j, j), 0, 255))
        return (b, g, r)   # OpenCV uses BGR

    def _render_background(self, H: int, W: int) -> np.ndarray:
        """Flat or gradient+texture background."""
        if not self._cfg.bg_texture:
            frame = np.full((H, W, 3), 20, dtype=np.uint8)
            return frame

        # Random gradient: pick two corner colors from a dark palette
        c1 = int(self._rng.integers(5, 40))
        c2 = int(self._rng.integers(5, 40))
        c3 = int(self._rng.integers(5, 40))
        c4 = int(self._rng.integers(5, 40))
        # Bilinear gradient across the frame
        xs = np.linspace(0, 1, W, dtype=np.float32)
        ys = np.linspace(0, 1, H, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        gray = ((1-xg)*(1-yg)*c1 + xg*(1-yg)*c2 +
                (1-xg)*yg*c3 + xg*yg*c4).astype(np.uint8)
        frame = np.stack([gray, gray, gray], axis=2)

        # Add subtle texture noise
        tex = self._rng.normal(0, 4, (H, W, 3))
        frame = np.clip(frame.astype(np.int16) + tex.astype(np.int16),
                        0, 255).astype(np.uint8)
        return frame

    def _render_flat(self, H: int, W: int) -> np.ndarray:
        """Original flat rendering: axis-aligned rectangle."""
        frame = self._render_background(H, W)
        gate_w = int(self._gate_scale * W * 0.7)
        gate_h = int(self._gate_scale * H * 0.7)
        cx_px = int(self._gate_cx * W)
        cy_px = int(self._gate_cy * H)
        x1 = max(0, cx_px - gate_w // 2)
        y1 = max(0, cy_px - gate_h // 2)
        x2 = min(W - 1, cx_px + gate_w // 2)
        y2 = min(H - 1, cy_px + gate_h // 2)
        thickness = max(2, int(self._gate_scale * 12))
        cv2.rectangle(frame, (x1, y1), (x2, y2), self._gate_color(), thickness)
        return frame

    def _render_perspective(self, H: int, W: int) -> np.ndarray:
        """Perspective-correct rendering.

        Models the gate as a physical square at 3D position derived from
        (cx, cy, scale).  The gate is rotated to face the approach angle,
        producing a realistic trapezoid when the drone is off-center.

        Coordinate system: camera at origin, +Z forward, +X right, +Y down.
        """
        frame = self._render_background(H, W)
        scale = self._gate_scale
        cx, cy = self._gate_cx, self._gate_cy

        # 3D gate center — distance ∝ 1/scale
        gz = 1.0 / max(scale, 1e-3)
        gx = (cx - 0.5) * 2.0 * gz   # FOV factor = 1 → tan(45°)
        gy = (cy - 0.5) * 2.0 * gz * (H / W)

        # Gate faces the camera with a yaw/pitch offset proportional to lateral
        # displacement — simulates off-axis approach angle.
        yaw_off   = -(cx - 0.5) * 0.9    # radians, max ±0.45
        pitch_off =  (cy - 0.5) * 0.5

        cy_r, sy_r = math.cos(yaw_off),   math.sin(yaw_off)
        cp_r, sp_r = math.cos(pitch_off), math.sin(pitch_off)

        # Gate half-size in world units (produces correct screen size at center)
        hx = 0.35
        hy = 0.35 * (H / W)

        # 4 corners in gate-local space (gate normal = +Z_local)
        local = [(-hx, -hy, 0.0),
                 (+hx, -hy, 0.0),
                 (+hx, +hy, 0.0),
                 (-hx, +hy, 0.0)]

        screen_pts = []
        for (lx, ly, lz) in local:
            # Yaw rotation (around Y axis)
            rx  =  lx * cy_r + lz * sy_r
            rz1 = -lx * sy_r + lz * cy_r
            # Pitch rotation (around X axis)
            ry  =  ly * cp_r - rz1 * sp_r
            rz  =  ly * sp_r + rz1 * cp_r

            # Translate to world position
            wx, wy, wz = rx + gx, ry + gy, rz + gz

            if wz <= 0.001:
                screen_pts.append((W // 2, H // 2))
                continue

            # Pinhole projection → screen pixels
            sx = int((wx / wz + 1.0) / 2.0 * W)
            sy = int((wy / wz + 1.0) / 2.0 * H)
            screen_pts.append((sx, sy))

        pts = np.array(screen_pts, dtype=np.int32).reshape((-1, 1, 2))
        thickness = max(2, int(scale * 14))
        cv2.polylines(frame, [pts], isClosed=True,
                      color=self._gate_color(), thickness=thickness)
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
