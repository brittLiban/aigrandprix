"""GatePlanner — predicts next gate position from course history.

Learns where gates tend to appear by recording the cx/cy at first
acquisition for each gate.  After a gate is passed, provides a
search_yaw_hint so the drone points toward the predicted next gate
instead of starting a blind sinusoidal sweep.

Design
------
- Exponentially-weighted average of recent gate positions (last 4)
- Regresses toward center (0.5) to avoid overconfident predictions
- Confidence grows with gates_seen, saturates at 1.0 after 5 gates
- Zero external dependencies — pure Python

Integration
-----------
FusionBrain calls:
  planner.record_gate(cx, cy)   — on every SEARCH -> TRACK transition
  result = planner()             — every frame, passes PlannerResult to brain
"""
from __future__ import annotations

from aigrandprix.types import PlannerResult


class GatePlanner:
    """Predicts next gate position from observed gate history."""

    # How strongly to pull prediction back toward screen center.
    # 0.0 = pure history, 1.0 = always predict center.
    _CENTER_PULL = 0.35

    # Exponential base for recency weighting (higher = more weight on recent)
    _RECENCY_BASE = 2.0

    # How many recent gates to use for prediction
    _WINDOW = 4

    def __init__(self):
        self._cx_history: list[float] = []
        self._cy_history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_gate(self, cx: float, cy: float) -> None:
        """Record a newly acquired gate.  Call on SEARCH -> TRACK transition."""
        self._cx_history.append(cx)
        self._cy_history.append(cy)

    def __call__(self) -> PlannerResult:
        """Return prediction for the next gate."""
        n = min(len(self._cx_history), self._WINDOW)

        if n == 0:
            return PlannerResult(
                predicted_cx=0.5, predicted_cy=0.5,
                search_yaw_hint=0.0, confidence=0.0,
                gates_seen=0,
            )

        recent_cx = self._cx_history[-n:]
        recent_cy = self._cy_history[-n:]

        # Exponential recency weights: oldest=base^0, newest=base^(n-1)
        weights = [self._RECENCY_BASE ** i for i in range(n)]
        total_w = sum(weights)

        raw_cx = sum(w * cx for w, cx in zip(weights, recent_cx)) / total_w
        raw_cy = sum(w * cy for w, cy in zip(weights, recent_cy)) / total_w

        # Regress toward center to avoid overcorrection
        pred_cx = (1 - self._CENTER_PULL) * raw_cx + self._CENTER_PULL * 0.5
        pred_cy = (1 - self._CENTER_PULL) * raw_cy + self._CENTER_PULL * 0.5

        # Yaw hint: negative = turn left if gate expected on left
        search_yaw_hint = -(pred_cx - 0.5) * 2.0

        # Confidence saturates after 5 gates
        confidence = min(len(self._cx_history) / 5.0, 1.0)

        return PlannerResult(
            predicted_cx=pred_cx,
            predicted_cy=pred_cy,
            search_yaw_hint=search_yaw_hint,
            confidence=confidence,
            gates_seen=len(self._cx_history),
        )

    def reset(self) -> None:
        self._cx_history.clear()
        self._cy_history.clear()
