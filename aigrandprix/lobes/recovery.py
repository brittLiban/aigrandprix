"""RecoveryLobe — lost-gate detection with directed recovery suggestions."""
from __future__ import annotations

from aigrandprix.brain.states import DroneState
from aigrandprix.config import RecoveryConfig
from aigrandprix.timing import timed
from aigrandprix.types import RecoveryResult, VisionResult


class RecoveryLobe:
    """Track lost-gate time and suggest directed yaw for recovery."""

    def __init__(self, config: RecoveryConfig, budget_ms: float = 1.0):
        self._cfg = config
        self._budget_ms = budget_ms
        self._violation_count = 0
        # Stateful
        self._frames_since_gate: float = 0.0
        self._last_known_cx: float = 0.5
        self._last_known_cy: float = 0.5
        self._last_dt: float = 1 / 60.0

    @timed("RecoveryLobe")
    def __call__(self, vision: VisionResult,
                 state: DroneState) -> RecoveryResult:
        if vision.gate_detected:
            self._last_known_cx = vision.cx
            self._last_known_cy = vision.cy
            self._frames_since_gate = 0.0
        else:
            self._frames_since_gate += self._last_dt

        in_recovery = (self._frames_since_gate >= self._cfg.lost_gate_s or
                       state == DroneState.RECOVER)

        # Directed yaw: turn toward where the gate was last seen
        cx = self._last_known_cx
        if cx < 0.4:
            suggested_yaw = -1.0   # gate was left → yaw left
        elif cx > 0.6:
            suggested_yaw = 1.0    # gate was right → yaw right
        else:
            suggested_yaw = 0.0    # gate was centered → slow scan

        return RecoveryResult(
            frames_since_gate=self._frames_since_gate,
            in_recovery=in_recovery,
            suggested_yaw=suggested_yaw,
            last_known_cx=self._last_known_cx,
            last_known_cy=self._last_known_cy,
        )

    def update_dt(self, dt: float) -> None:
        """Called by runner each step so recovery tracks real time."""
        self._last_dt = dt

    def reset(self) -> None:
        self._frames_since_gate = 0.0
        self._last_known_cx = 0.5
        self._last_known_cy = 0.5
