"""Unit tests for RecoveryLobe."""
import pytest

from aigrandprix.brain.states import DroneState
from aigrandprix.config import default_config
from aigrandprix.lobes.recovery import RecoveryLobe
from aigrandprix.types import VisionResult


def detected(cx=0.5, cy=0.5):
    return VisionResult(
        gate_detected=True, cx=cx, cy=cy, area=10000.0,
        confidence=0.9, confidence_ema=0.8, last_seen_t=1.0,
        bbox=(100, 100, 200, 200), latency_ms=1.0,
    )


def no_gate():
    return VisionResult.null()


class TestRecoveryLobe:
    def test_not_in_recovery_when_gate_detected(self):
        lobe = RecoveryLobe(default_config().recovery)
        result = lobe(detected(), DroneState.TRACK)
        assert not result.in_recovery
        assert result.frames_since_gate == pytest.approx(0.0)

    def test_in_recovery_after_lost_gate_time(self):
        lobe = RecoveryLobe(default_config().recovery)
        lost_gate_s = default_config().recovery.lost_gate_s
        # Accumulate enough lost time
        frames = int(lost_gate_s / 0.016) + 5
        for _ in range(frames):
            lobe.update_dt(0.016)
            result = lobe(no_gate(), DroneState.TRACK)
        assert result.in_recovery

    def test_recovery_clears_when_gate_redetected(self):
        lobe = RecoveryLobe(default_config().recovery)
        for _ in range(50):
            lobe.update_dt(0.016)
            lobe(no_gate(), DroneState.TRACK)
        result = lobe(detected(0.5, 0.5), DroneState.RECOVER)
        assert result.frames_since_gate == pytest.approx(0.0)

    def test_directed_yaw_left_when_gate_was_left(self):
        lobe = RecoveryLobe(default_config().recovery)
        # Gate was last seen on left
        lobe(detected(cx=0.2), DroneState.TRACK)
        # Gate lost
        result = lobe(no_gate(), DroneState.RECOVER)
        assert result.suggested_yaw == pytest.approx(-1.0)

    def test_directed_yaw_right_when_gate_was_right(self):
        lobe = RecoveryLobe(default_config().recovery)
        lobe(detected(cx=0.8), DroneState.TRACK)
        result = lobe(no_gate(), DroneState.RECOVER)
        assert result.suggested_yaw == pytest.approx(1.0)

    def test_directed_yaw_neutral_when_gate_was_centered(self):
        lobe = RecoveryLobe(default_config().recovery)
        lobe(detected(cx=0.5), DroneState.TRACK)
        result = lobe(no_gate(), DroneState.RECOVER)
        assert result.suggested_yaw == pytest.approx(0.0)

    def test_last_known_cx_updated(self):
        lobe = RecoveryLobe(default_config().recovery)
        lobe(detected(cx=0.3, cy=0.7), DroneState.TRACK)
        result = lobe(no_gate(), DroneState.RECOVER)
        assert result.last_known_cx == pytest.approx(0.3)
        assert result.last_known_cy == pytest.approx(0.7)

    def test_reset(self):
        lobe = RecoveryLobe(default_config().recovery)
        for _ in range(50):
            lobe.update_dt(0.016)
            lobe(no_gate(), DroneState.TRACK)
        lobe.reset()
        assert lobe._frames_since_gate == pytest.approx(0.0)
