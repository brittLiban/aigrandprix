"""Unit tests for FusionBrain state machine transitions."""
import pytest

from aigrandprix.brain.fusion import FusionBrain
from aigrandprix.brain.states import DroneState
from aigrandprix.config import default_config
from aigrandprix.types import (
    ProgressResult, RecoveryResult, RiskResult,
    StabilityResult, VisionResult,
)


def vision(detected=True, cx=0.5, cy=0.5, area=5000.0,
           confidence=0.9, ema=0.8):
    return VisionResult(
        gate_detected=detected, cx=cx, cy=cy, area=area,
        confidence=confidence, confidence_ema=ema, last_seen_t=1.0,
        bbox=(100, 100, 200, 200), latency_ms=1.0,
    )


def stability(score=1.0, tumbling=False):
    return StabilityResult(
        accel_norm=9.81, gyro_norm=0.0 if not tumbling else 12.0,
        stability_score=score, is_tumbling=tumbling,
        instability_spike_count=0,
    )


def progress(aligned=0.9, rate=500.0, progress_score=400.0):
    return ProgressResult(
        dx=0.0, dy=0.0, approach_rate=rate,
        aligned_score=aligned, progress_score=progress_score,
        gate_index=0, time_in_state=0.0,
    )


def recovery(in_recovery=False, frames_since=0.0, yaw=0.0):
    return RecoveryResult(
        frames_since_gate=frames_since, in_recovery=in_recovery,
        suggested_yaw=yaw, last_known_cx=0.5, last_known_cy=0.5,
    )


def risk():
    return RiskResult(
        risk_score=0.1, push_level=2, safe_to_push=True,
        recent_recovery_penalty=0.0,
    )


def step(brain, **kwargs):
    v = kwargs.get("v", vision())
    s = kwargs.get("s", stability())
    p = kwargs.get("p", progress())
    r = kwargs.get("r", recovery())
    ri = kwargs.get("ri", risk())
    t = kwargs.get("t", 0.016)
    return brain(v, s, p, r, ri, sim_t=t)


class TestFusionBrainTransitions:
    def test_initial_state_is_search(self):
        brain = FusionBrain(default_config().state_machine)
        assert brain.state == DroneState.SEARCH

    def test_search_to_track_on_detection(self):
        brain = FusionBrain(default_config().state_machine)
        cfg = default_config().state_machine
        # confidence_ema above threshold
        state, _ = step(brain, v=vision(ema=cfg.track_confidence_min + 0.1), t=0.1)
        assert state == DroneState.TRACK

    def test_search_stays_search_when_no_gate(self):
        brain = FusionBrain(default_config().state_machine)
        state, _ = step(brain, v=vision(ema=0.0), t=0.1)
        assert state == DroneState.SEARCH

    def test_track_to_approach_when_aligned_and_stable(self):
        brain = FusionBrain(default_config().state_machine)
        cfg = default_config().state_machine
        # First get to TRACK
        step(brain, v=vision(ema=cfg.track_confidence_min + 0.1), t=0.1)
        assert brain.state == DroneState.TRACK
        # Now aligned + stable
        state, _ = step(brain,
                        v=vision(ema=cfg.track_confidence_min + 0.1),
                        s=stability(score=cfg.min_stable + 0.1),
                        p=progress(aligned=cfg.approach_aligned_min + 0.05),
                        t=0.2)
        assert state == DroneState.APPROACH

    def test_approach_to_commit_on_large_area(self):
        brain = FusionBrain(default_config().state_machine)
        cfg = default_config().state_machine
        # Drive to APPROACH
        step(brain, v=vision(ema=cfg.track_confidence_min + 0.1), t=0.1)
        step(brain,
             v=vision(ema=cfg.track_confidence_min + 0.1),
             s=stability(score=cfg.min_stable + 0.1),
             p=progress(aligned=cfg.approach_aligned_min + 0.05),
             t=0.2)
        assert brain.state == DroneState.APPROACH
        # Now large gate area
        state, _ = step(brain,
                        v=vision(area=cfg.commit_area_threshold + 1000),
                        s=stability(score=cfg.min_stable + 0.1),
                        p=progress(aligned=cfg.approach_aligned_min + 0.05),
                        t=0.3)
        assert state == DroneState.COMMIT

    def test_any_state_to_recover_on_tumbling(self):
        brain = FusionBrain(default_config().state_machine)
        state, _ = step(brain, s=stability(tumbling=True), t=0.1)
        assert state == DroneState.RECOVER

    def test_any_state_to_recover_on_long_gate_loss(self):
        brain = FusionBrain(default_config().state_machine)
        cfg = default_config().state_machine
        state, _ = step(brain,
                        r=recovery(frames_since=cfg.search_timeout_s + 1.0),
                        t=0.1)
        assert state == DroneState.RECOVER

    def test_recover_to_search_when_stable_and_gate_back(self):
        brain = FusionBrain(default_config().state_machine)
        cfg = default_config().state_machine
        # Enter RECOVER
        step(brain, s=stability(tumbling=True), t=0.1)
        assert brain.state == DroneState.RECOVER
        # Now stable + gate detected
        state, _ = step(brain,
                        s=stability(score=cfg.min_stable + 0.1),
                        v=vision(ema=cfg.track_confidence_min + 0.1),
                        r=recovery(frames_since=0.0),
                        t=0.2)
        assert state == DroneState.SEARCH

    def test_integral_reset_requested_on_recover_entry(self):
        brain = FusionBrain(default_config().state_machine)
        # Consume any initial flag
        _ = brain.integral_reset_requested
        step(brain, s=stability(tumbling=True), t=0.1)
        assert brain.integral_reset_requested

    def test_time_in_state_resets_on_transition(self):
        brain = FusionBrain(default_config().state_machine)
        cfg = default_config().state_machine
        step(brain, v=vision(ema=0.0), t=0.1)
        step(brain, v=vision(ema=0.0), t=0.2)
        step(brain, v=vision(ema=cfg.track_confidence_min + 0.1), t=0.3)
        # Just transitioned to TRACK; time_in_state should be near 0
        assert brain.time_in_state < 0.1

    def test_approach_hysteresis_prevents_chattering(self):
        brain = FusionBrain(default_config().state_machine)
        cfg = default_config().state_machine
        # Drive to APPROACH
        step(brain, v=vision(ema=cfg.track_confidence_min + 0.1), t=0.1)
        step(brain,
             v=vision(ema=cfg.track_confidence_min + 0.1),
             s=stability(score=cfg.min_stable + 0.1),
             p=progress(aligned=cfg.approach_aligned_min + 0.05),
             t=0.2)
        assert brain.state == DroneState.APPROACH
        # Alignment slightly below entry threshold but above hysteresis exit threshold
        exit_thresh = cfg.approach_aligned_min * cfg.hysteresis_factor
        mid_aligned = (exit_thresh + cfg.approach_aligned_min) / 2
        state, _ = step(brain,
                        v=vision(ema=cfg.track_confidence_min + 0.1),
                        s=stability(score=cfg.min_stable + 0.1),
                        p=progress(aligned=mid_aligned),
                        t=0.3)
        # Should stay in APPROACH due to hysteresis
        assert state == DroneState.APPROACH

    def test_reset(self):
        brain = FusionBrain(default_config().state_machine)
        cfg = default_config().state_machine
        step(brain, s=stability(tumbling=True), t=0.1)
        brain.reset()
        assert brain.state == DroneState.SEARCH
        assert brain.time_in_state == pytest.approx(0.0)
