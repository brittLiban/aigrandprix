"""Unit tests for RiskLobe."""
import pytest

from aigrandprix.config import default_config
from aigrandprix.lobes.risk import RiskLobe
from aigrandprix.types import ProgressResult, RecoveryResult, StabilityResult


def progress_r(approach_rate=500.0):
    return ProgressResult(
        dx=0.0, dy=0.0, approach_rate=approach_rate,
        aligned_score=0.9, progress_score=approach_rate * 0.9,
        gate_index=0, time_in_state=0.5,
    )


def recovery_r(in_recovery=False, frames_since_gate=0.0):
    return RecoveryResult(
        frames_since_gate=frames_since_gate, in_recovery=in_recovery,
        suggested_yaw=0.0, last_known_cx=0.5, last_known_cy=0.5,
    )


class TestRiskLobe:
    def test_stable_no_recovery_low_risk(self):
        lobe = RiskLobe(default_config().risk)
        result = lobe(
            StabilityResult.stable(), progress_r(), recovery_r(), obs_t=5.0)
        assert result.risk_score < 0.3
        assert result.push_level >= 2

    def test_tumbling_high_risk(self):
        lobe = RiskLobe(default_config().risk)
        stability = StabilityResult(
            accel_norm=9.81, gyro_norm=12.0,
            stability_score=0.0, is_tumbling=True,
            instability_spike_count=0,
        )
        result = lobe(stability, progress_r(), recovery_r(), obs_t=5.0)
        assert result.risk_score > 0.7
        assert result.push_level == 0
        assert not result.safe_to_push

    def test_recent_recovery_penalty(self):
        lobe = RiskLobe(default_config().risk)
        # Trigger a recovery event
        lobe(StabilityResult.stable(), progress_r(),
             recovery_r(in_recovery=True), obs_t=0.0)
        # Immediately after, check penalty is applied
        result = lobe(StabilityResult.stable(), progress_r(),
                      recovery_r(in_recovery=True), obs_t=0.1)
        assert result.recent_recovery_penalty > 0.0

    def test_recovery_penalty_decays(self):
        lobe = RiskLobe(default_config().risk)
        lobe(StabilityResult.stable(), progress_r(),
             recovery_r(in_recovery=True), obs_t=0.0)
        window = default_config().risk.recovery_penalty_window
        # After penalty window, penalty should be 0
        result = lobe(StabilityResult.stable(), progress_r(),
                      recovery_r(in_recovery=False), obs_t=window + 0.1)
        assert result.recent_recovery_penalty == pytest.approx(0.0)

    def test_spike_penalty_adds_risk(self):
        lobe = RiskLobe(default_config().risk)
        no_spike = StabilityResult(
            accel_norm=9.81, gyro_norm=0.0, stability_score=1.0,
            is_tumbling=False, instability_spike_count=0,
        )
        with_spike = StabilityResult(
            accel_norm=9.81, gyro_norm=0.0, stability_score=1.0,
            is_tumbling=False, instability_spike_count=5,
        )
        r_no = lobe(no_spike, progress_r(), recovery_r(), obs_t=10.0)
        r_sp = lobe(with_spike, progress_r(), recovery_r(), obs_t=10.0)
        assert r_sp.risk_score > r_no.risk_score

    def test_push_level_range(self):
        lobe = RiskLobe(default_config().risk)
        result = lobe(StabilityResult.stable(), progress_r(),
                      recovery_r(), obs_t=5.0)
        assert 0 <= result.push_level <= 3

    def test_reset(self):
        lobe = RiskLobe(default_config().risk)
        lobe(StabilityResult.stable(), progress_r(),
             recovery_r(in_recovery=True), obs_t=0.0)
        lobe.reset()
        assert lobe._last_recovery_t == pytest.approx(-999.0)
