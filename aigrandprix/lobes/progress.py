"""ProgressLobe — gate alignment, approach rate, and gate index inference."""
from __future__ import annotations

import math

from aigrandprix.brain.states import DroneState
from aigrandprix.config import ProgressConfig
from aigrandprix.timing import timed
from aigrandprix.types import Observation, ProgressResult, VisionResult

# Area spike: gate just passed if area drops by this factor in one step
_AREA_DROP_FACTOR = 0.4


class ProgressLobe:
    """Track alignment and approach to the current gate."""

    def __init__(self, config: ProgressConfig, budget_ms: float = 2.0):
        self._cfg = config
        self._budget_ms = budget_ms
        self._violation_count = 0
        # Stateful
        self._prev_area: float = 0.0
        self._approach_rate_ema: float = 0.0
        self._gate_index: int = 0
        self._in_commit: bool = False

    @timed("ProgressLobe")
    def __call__(self, obs: Observation, vision: VisionResult,
                 current_state: DroneState, time_in_state: float) -> ProgressResult:
        # Alignment: gate center offset from image center
        dx = (vision.cx - 0.5) * 2.0   # [-1, 1], 0 = centered
        dy = (vision.cy - 0.5) * 2.0

        # Aligned score: 1 = perfectly centered, 0 = at edge
        raw_dist = math.sqrt(dx ** 2 + dy ** 2)
        max_dist = math.sqrt(2.0)   # diagonal = worst case
        aligned_score = float(max(0.0, 1.0 - raw_dist / max_dist))

        # Approach rate: d(area)/dt with EMA smoothing
        alpha = self._cfg.area_ema_alpha
        prev_area = self._prev_area  # snapshot before update

        if obs.dt > 0 and vision.gate_detected:
            raw_rate = (vision.area - prev_area) / obs.dt
            self._approach_rate_ema = (alpha * raw_rate
                                       + (1 - alpha) * self._approach_rate_ema)
        elif not vision.gate_detected:
            self._approach_rate_ema *= 0.9  # decay when lost

        if vision.gate_detected:
            self._prev_area = vision.area

        # Gate index inference: if we were in COMMIT and the gate disappears
        # OR area drops sharply, gate has been passed.
        # Reset _in_commit immediately after incrementing to prevent double-count.
        current_area = vision.area if vision.gate_detected else 0.0
        if (self._in_commit and prev_area > 0 and
                current_area < prev_area * _AREA_DROP_FACTOR):
            self._gate_index += 1
            self._in_commit = False   # prevent double-count on next step
            self._prev_area = 0.0    # clear so we don't re-trigger
        else:
            self._in_commit = (current_state == DroneState.COMMIT)

        progress_score = float(
            max(0.0, self._approach_rate_ema) * aligned_score)

        return ProgressResult(
            dx=float(dx),
            dy=float(dy),
            approach_rate=float(self._approach_rate_ema),
            aligned_score=aligned_score,
            progress_score=progress_score,
            gate_index=self._gate_index,
            time_in_state=float(time_in_state),
        )

    def reset(self) -> None:
        self._prev_area = 0.0
        self._approach_rate_ema = 0.0
        self._gate_index = 0
        self._in_commit = False
