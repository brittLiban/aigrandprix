"""FusionBrain — state machine that decides drone mode and target action.

States:  SEARCH → TRACK → APPROACH → COMMIT → RECOVER

Key design choices:
  - Hysteresis on TRACK↔APPROACH prevents chattering
  - Stuck detection: APPROACH times out if progress_score stays low
  - RECOVER is triggered from any state on instability or long gate loss
  - PID integral reset happens here (state transition to RECOVER)
  - time_in_state is tracked internally and passed to ProgressLobe
"""
from __future__ import annotations

import time

from aigrandprix.brain.states import DroneState
from aigrandprix.config import StateMachineConfig
from aigrandprix.types import (
    Action, ProgressResult, RecoveryResult, RiskResult,
    StabilityResult, VisionResult,
)


class FusionBrain:
    """State machine: reads lobe outputs, returns (new_state, target_action)."""

    def __init__(self, config: StateMachineConfig):
        self._cfg = config
        self._state = DroneState.SEARCH
        self._time_in_state: float = 0.0
        self._t_last: float = 0.0   # sim time of last call
        self._commit_area_peak: float = 0.0
        self._integral_reset_requested: bool = False

    @property
    def state(self) -> DroneState:
        return self._state

    @property
    def time_in_state(self) -> float:
        return self._time_in_state

    @property
    def integral_reset_requested(self) -> bool:
        """True on the step when RECOVER is first entered; consumed by Controller."""
        val = self._integral_reset_requested
        self._integral_reset_requested = False
        return val

    def __call__(
        self,
        vision: VisionResult,
        stability: StabilityResult,
        progress: ProgressResult,
        recovery: RecoveryResult,
        risk: RiskResult,
        sim_t: float = 0.0,
    ) -> tuple[DroneState, Action]:
        dt = sim_t - self._t_last if self._t_last > 0 else 0.016
        self._t_last = sim_t
        self._time_in_state += dt

        cfg = self._cfg
        prev_state = self._state

        # ------------------------------------------------------------------
        # Global RECOVER override (highest priority — any state)
        # ------------------------------------------------------------------
        # Require BOTH time-since-gate AND cold EMA before triggering RECOVER.
        # If EMA is still warm, the gate signal is degraded (noise/dropped frame)
        # not truly lost — don't pay the full RECOVER cost for a sensor glitch.
        gate_truly_lost = (recovery.frames_since_gate > cfg.search_timeout_s and
                           vision.confidence_ema < cfg.track_confidence_min * 0.3)
        if stability.is_tumbling or gate_truly_lost:
            new_state = DroneState.RECOVER
        else:
            new_state = self._transition(vision, stability, progress, recovery, risk)

        if new_state != prev_state:
            self._time_in_state = 0.0
            if new_state == DroneState.RECOVER:
                self._integral_reset_requested = True
            if new_state == DroneState.COMMIT:
                self._commit_area_peak = vision.area

        self._state = new_state
        target = self._compute_target(new_state, vision, recovery, risk)
        return new_state, target

    # ------------------------------------------------------------------
    # Transition logic
    # ------------------------------------------------------------------

    def _transition(
        self,
        vision: VisionResult,
        stability: StabilityResult,
        progress: ProgressResult,
        recovery: RecoveryResult,
        risk: RiskResult,
    ) -> DroneState:
        cfg = self._cfg
        state = self._state

        if state == DroneState.RECOVER:
            # Exit RECOVER when stable and gate reacquired
            if (stability.stability_score >= cfg.min_stable and
                    recovery.frames_since_gate == 0.0):
                return DroneState.SEARCH
            return DroneState.RECOVER

        if state == DroneState.SEARCH:
            if vision.confidence_ema >= cfg.track_confidence_min:
                return DroneState.TRACK
            # Stuck in SEARCH: only recover if EMA is also cold (truly lost)
            if (self._time_in_state > cfg.search_timeout_s and
                    vision.confidence_ema < cfg.track_confidence_min * 0.3):
                return DroneState.RECOVER
            return DroneState.SEARCH

        if state == DroneState.TRACK:
            # Lose gate → back to SEARCH
            if vision.confidence_ema < cfg.track_confidence_min * 0.5:
                return DroneState.SEARCH
            # Advance to APPROACH when aligned + stable
            if (progress.aligned_score >= cfg.approach_aligned_min and
                    stability.stability_score >= cfg.min_stable):
                return DroneState.APPROACH
            return DroneState.TRACK

        if state == DroneState.APPROACH:
            # Exit via hysteresis (stricter than entry)
            exit_threshold = cfg.approach_aligned_min * cfg.hysteresis_factor
            if (progress.aligned_score < exit_threshold or
                    vision.confidence_ema < cfg.track_confidence_min * 0.5):
                return DroneState.TRACK
            # Stuck: approach not making progress
            if (self._time_in_state > cfg.approach_timeout_s and
                    progress.progress_score < cfg.approach_aligned_min * 0.1):
                return DroneState.RECOVER
            # Advance to COMMIT when gate fills frame AND roughly centred.
            # Prevents committing while the drone is still flying at an angle.
            if (vision.area >= cfg.commit_area_threshold and
                    progress.aligned_score >= cfg.approach_aligned_min * 0.9):
                return DroneState.COMMIT
            return DroneState.APPROACH

        if state == DroneState.COMMIT:
            # Update area peak to detect gate pass
            if vision.gate_detected:
                self._commit_area_peak = max(self._commit_area_peak, vision.area)
            # Gate passed: area drops sharply after peak
            passed = (self._commit_area_peak > cfg.commit_area_threshold and
                      vision.area < self._commit_area_peak * 0.4)
            if passed or not vision.gate_detected:
                self._commit_area_peak = 0.0
                # Skip SEARCH if the next gate is already visible — go straight to TRACK
                if vision.gate_detected and vision.confidence_ema >= cfg.track_confidence_min:
                    return DroneState.TRACK
                return DroneState.SEARCH
            return DroneState.COMMIT

        return state   # fallback (should not reach)

    # ------------------------------------------------------------------
    # Target action per state
    # ------------------------------------------------------------------

    def _compute_target(
        self,
        state: DroneState,
        vision: VisionResult,
        recovery: RecoveryResult,
        risk: RiskResult,
    ) -> Action:
        """Compute desired motion as unnormalized targets for the Controller."""
        if state == DroneState.SEARCH:
            # If we have a direction hint, use it; otherwise sweep yaw sinusoidally
            # so the drone actively scans instead of hovering.
            if abs(recovery.suggested_yaw) > 0.05:
                yaw = recovery.suggested_yaw * 0.7
            else:
                import math
                yaw = math.sin(self._time_in_state * 1.2) * 0.6
            return Action(roll=0.0, pitch=0.0, yaw=yaw, throttle=0.45)

        if state == DroneState.TRACK:
            dx = vision.cx - 0.5   # error: + = gate right, - = gate left
            # High roll gain saturates the output for far-off gates — centers
            # them in ~70 frames even at extreme lateral offsets.
            return Action(
                roll=dx * 2.5,
                pitch=0.18,        # keep charging forward while correcting
                yaw=-dx * 0.7,
                throttle=0.6,
            )

        if state == DroneState.APPROACH:
            dx = vision.cx - 0.5
            # Keep the same strong roll as TRACK — don't let correction slow
            # just because we entered a new state.
            return Action(
                roll=dx * 2.0,
                pitch=0.25,
                yaw=-dx * 0.6,
                throttle=0.7,
            )

        if state == DroneState.COMMIT:
            dx = vision.cx - 0.5 if vision.gate_detected else 0.0
            # Small lateral correction even in COMMIT — don't fly blind
            return Action(
                roll=dx * 0.4,
                pitch=0.25,
                yaw=-dx * 0.3,
                throttle=0.85,
            )

        if state == DroneState.RECOVER:
            return Action(
                roll=0.0,
                pitch=0.0,
                yaw=recovery.suggested_yaw * 0.4,
                throttle=0.4,
            )

        return Action.hover()

    def reset(self) -> None:
        self._state = DroneState.SEARCH
        self._time_in_state = 0.0
        self._t_last = 0.0
        self._commit_area_peak = 0.0
        self._integral_reset_requested = False
