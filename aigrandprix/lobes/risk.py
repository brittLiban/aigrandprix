"""RiskLobe — compute risk score and schedule aggression (push level)."""
from __future__ import annotations

import bisect

from aigrandprix.config import RiskConfig
from aigrandprix.timing import timed
from aigrandprix.types import ProgressResult, RecoveryResult, RiskResult, StabilityResult


class RiskLobe:
    """Synthesise stability + recovery history into a risk score and push level."""

    def __init__(self, config: RiskConfig, budget_ms: float = 1.0):
        self._cfg = config
        self._budget_ms = budget_ms
        self._violation_count = 0
        # Stateful: time of last recovery event
        self._last_recovery_t: float = -999.0
        self._prev_in_recovery: bool = False

    @timed("RiskLobe")
    def __call__(self, stability: StabilityResult, progress: ProgressResult,
                 recovery: RecoveryResult, obs_t: float) -> RiskResult:
        # Track recovery events (transitions into recovery)
        if recovery.in_recovery and not self._prev_in_recovery:
            self._last_recovery_t = obs_t
        self._prev_in_recovery = recovery.in_recovery

        # --- Base risk from stability ---
        gyro_risk = 1.0 - stability.stability_score
        tumble_risk = 1.0 if stability.is_tumbling else 0.0
        base_risk = 0.6 * gyro_risk + 0.4 * tumble_risk

        # --- Recovery penalty: recent recovery → be more conservative ---
        time_since_recovery = obs_t - self._last_recovery_t
        window = self._cfg.recovery_penalty_window
        if time_since_recovery < window:
            recovery_penalty = self._cfg.max_recovery_penalty * (
                1.0 - time_since_recovery / window)
        else:
            recovery_penalty = 0.0

        # --- Spike penalty ---
        spike_penalty = (stability.instability_spike_count
                         * self._cfg.spike_penalty_weight)

        risk_score = float(min(1.0, base_risk + recovery_penalty + spike_penalty))

        # Push level: thresholds are for (1 - risk_score), so higher safety → higher push
        safety = 1.0 - risk_score
        thresholds = sorted(self._cfg.push_level_thresholds, reverse=True)
        # push_level 0 if safety < lowest threshold; increases from there
        push_level = 0
        for thresh in reversed(thresholds):
            if safety >= thresh:
                push_level += 1

        push_level = min(push_level, 3)
        safe_to_push = push_level >= 2

        return RiskResult(
            risk_score=risk_score,
            push_level=push_level,
            safe_to_push=safe_to_push,
            recent_recovery_penalty=recovery_penalty,
        )

    def reset(self) -> None:
        self._last_recovery_t = -999.0
        self._prev_in_recovery = False
