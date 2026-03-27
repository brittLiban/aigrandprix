"""StabilityLobe — IMU-based stability assessment with spike tracking."""
from __future__ import annotations

from collections import deque

import numpy as np

from aigrandprix.config import StabilityConfig
from aigrandprix.timing import timed
from aigrandprix.types import Observation, StabilityResult


class StabilityLobe:
    """Compute stability score and detect IMU instability spikes."""

    def __init__(self, config: StabilityConfig, budget_ms: float = 2.0):
        self._cfg = config
        self._budget_ms = budget_ms
        self._violation_count = 0
        # Sliding window of (t, is_spike) for spike counting
        self._spike_window: deque[tuple[float, bool]] = deque()

    @timed("StabilityLobe")
    def __call__(self, obs: Observation) -> StabilityResult:
        accel_norm = float(np.linalg.norm(obs.imu_accel))
        gyro_norm = float(np.linalg.norm(obs.imu_gyro))

        is_tumbling = gyro_norm > self._cfg.gyro_tumble_threshold

        # Spike: gyro exceeds half the tumble threshold OR accel very high
        is_spike = (gyro_norm > self._cfg.gyro_tumble_threshold * 0.5 or
                    accel_norm > self._cfg.accel_high_threshold)

        # Update sliding window
        self._spike_window.append((obs.t, is_spike))
        cutoff = obs.t - self._cfg.spike_window_s
        while self._spike_window and self._spike_window[0][0] < cutoff:
            self._spike_window.popleft()

        spike_count = sum(1 for _, s in self._spike_window if s)

        # Stability score: 1.0 = hovering still; degrades with gyro
        gyro_clipped = min(gyro_norm, self._cfg.gyro_tumble_threshold)
        gyro_score = 1.0 - gyro_clipped / self._cfg.gyro_tumble_threshold
        # Also penalise extreme accel deviation from gravity
        accel_dev = abs(accel_norm - 9.81)
        accel_score = max(0.0, 1.0 - accel_dev / 15.0)
        stability_score = float(0.7 * gyro_score + 0.3 * accel_score)

        return StabilityResult(
            accel_norm=accel_norm,
            gyro_norm=gyro_norm,
            stability_score=stability_score,
            is_tumbling=is_tumbling,
            instability_spike_count=spike_count,
        )

    def reset(self) -> None:
        self._spike_window.clear()
