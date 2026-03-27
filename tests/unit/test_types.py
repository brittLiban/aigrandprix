"""Unit tests for aigrandprix.types — core data contracts."""
import numpy as np
import pytest

from aigrandprix.types import (
    Action, Observation, VisionResult, StabilityResult,
    ProgressResult, RecoveryResult, RiskResult,
)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class TestAction:
    def test_clamp_in_range(self):
        a = Action(roll=0.5, pitch=-0.3, yaw=0.1, throttle=0.7)
        c = a.clamp()
        assert c.roll == pytest.approx(0.5)
        assert c.throttle == pytest.approx(0.7)

    def test_clamp_out_of_range(self):
        a = Action(roll=2.0, pitch=-3.0, yaw=1.5, throttle=-0.5)
        c = a.clamp()
        assert c.roll == pytest.approx(1.0)
        assert c.pitch == pytest.approx(-1.0)
        assert c.yaw == pytest.approx(1.0)
        assert c.throttle == pytest.approx(0.0)

    def test_zero(self):
        a = Action.zero()
        assert a.roll == 0.0 and a.pitch == 0.0 and a.yaw == 0.0 and a.throttle == 0.0

    def test_hover(self):
        a = Action.hover()
        assert a.throttle == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TestObservation:
    def test_construction(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.zeros(3)
        obs = Observation(t=1.0, dt=0.016, image=img, imu_accel=accel, imu_gyro=gyro)
        assert obs.t == pytest.approx(1.0)
        assert obs.motor_rpm is None
        assert obs.battery is None
        assert obs.meta == {}

    def test_image_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        obs = Observation(t=0.0, dt=0.0, image=img,
                          imu_accel=np.zeros(3), imu_gyro=np.zeros(3))
        assert obs.image.shape == (480, 640, 3)
        assert obs.image.dtype == np.uint8


# ---------------------------------------------------------------------------
# VisionResult
# ---------------------------------------------------------------------------

class TestVisionResult:
    def test_null(self):
        r = VisionResult.null(t=5.0)
        assert r.gate_detected is False
        assert r.confidence == 0.0
        assert r.confidence_ema == 0.0
        assert r.last_seen_t == pytest.approx(5.0)
        assert r.bbox is None

    def test_detected(self):
        r = VisionResult(
            gate_detected=True, cx=0.5, cy=0.5, area=10000.0,
            confidence=0.9, confidence_ema=0.75, last_seen_t=1.0,
            bbox=(100, 100, 200, 200), latency_ms=3.2,
        )
        assert r.gate_detected is True
        assert r.area == pytest.approx(10000.0)


# ---------------------------------------------------------------------------
# StabilityResult
# ---------------------------------------------------------------------------

class TestStabilityResult:
    def test_stable_factory(self):
        r = StabilityResult.stable()
        assert r.stability_score == pytest.approx(1.0)
        assert r.is_tumbling is False
        assert r.instability_spike_count == 0

    def test_fields(self):
        r = StabilityResult(
            accel_norm=9.81, gyro_norm=12.0,
            stability_score=0.1, is_tumbling=True,
            instability_spike_count=3,
        )
        assert r.is_tumbling is True
        assert r.instability_spike_count == 3


# ---------------------------------------------------------------------------
# ProgressResult
# ---------------------------------------------------------------------------

class TestProgressResult:
    def test_centered(self):
        r = ProgressResult(
            dx=0.0, dy=0.0, approach_rate=500.0,
            aligned_score=1.0, progress_score=500.0,
            gate_index=0, time_in_state=0.5,
        )
        assert r.aligned_score == pytest.approx(1.0)
        assert r.gate_index == 0


# ---------------------------------------------------------------------------
# RecoveryResult
# ---------------------------------------------------------------------------

class TestRecoveryResult:
    def test_not_in_recovery(self):
        r = RecoveryResult(
            frames_since_gate=0.0, in_recovery=False,
            suggested_yaw=0.0, last_known_cx=0.5, last_known_cy=0.5,
        )
        assert r.in_recovery is False

    def test_directed_yaw(self):
        # Gate was on left → yaw left
        r = RecoveryResult(
            frames_since_gate=0.6, in_recovery=True,
            suggested_yaw=-1.0, last_known_cx=0.2, last_known_cy=0.5,
        )
        assert r.suggested_yaw == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# RiskResult
# ---------------------------------------------------------------------------

class TestRiskResult:
    def test_safe(self):
        r = RiskResult(
            risk_score=0.1, push_level=3,
            safe_to_push=True, recent_recovery_penalty=0.0,
        )
        assert r.push_level == 3
        assert r.safe_to_push is True

    def test_risky(self):
        r = RiskResult(
            risk_score=0.9, push_level=0,
            safe_to_push=False, recent_recovery_penalty=0.25,
        )
        assert r.push_level == 0
        assert r.safe_to_push is False
