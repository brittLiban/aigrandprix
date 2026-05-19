"""Unit tests for ProgressLobe."""
import numpy as np
import pytest

from aigrandprix.brain.states import DroneState
from aigrandprix.config import default_config
from aigrandprix.lobes.progress import ProgressLobe
from aigrandprix.types import Observation, VisionResult


def make_obs(t=1.0, dt=0.016):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return Observation(t=t, dt=dt, image=img,
                       imu_accel=np.array([0, 0, 9.81]),
                       imu_gyro=np.zeros(3))


def gate_at(cx, cy, area=10000.0, detected=True):
    return VisionResult(
        gate_detected=detected, cx=cx, cy=cy, area=area,
        confidence=0.9, confidence_ema=0.8, last_seen_t=1.0,
        bbox=(0, 0, 100, 100), latency_ms=3.0,
    )


class TestProgressLobe:
    def test_centered_gate_dx_dy_near_zero(self):
        lobe = ProgressLobe(default_config().progress)
        result = lobe(make_obs(), gate_at(0.5, 0.5), DroneState.TRACK, 0.1)
        assert abs(result.dx) < 0.05
        assert abs(result.dy) < 0.05

    def test_gate_top_left(self):
        lobe = ProgressLobe(default_config().progress)
        result = lobe(make_obs(), gate_at(0.2, 0.2), DroneState.TRACK, 0.1)
        assert result.dx < 0       # gate is left of center
        assert result.dy < 0       # gate is above center

    def test_gate_bottom_right(self):
        lobe = ProgressLobe(default_config().progress)
        result = lobe(make_obs(), gate_at(0.8, 0.8), DroneState.TRACK, 0.1)
        assert result.dx > 0
        assert result.dy > 0

    def test_aligned_score_centered(self):
        lobe = ProgressLobe(default_config().progress)
        result = lobe(make_obs(), gate_at(0.5, 0.5), DroneState.TRACK, 0.1)
        assert result.aligned_score > 0.9

    def test_aligned_score_edge(self):
        lobe = ProgressLobe(default_config().progress)
        result = lobe(make_obs(), gate_at(0.0, 0.0), DroneState.TRACK, 0.1)
        assert result.aligned_score < 0.3

    def test_approach_rate_positive_when_closing(self):
        lobe = ProgressLobe(default_config().progress)
        # Step 1: small area
        lobe(make_obs(t=0.0), gate_at(0.5, 0.5, area=5000.0), DroneState.APPROACH, 0.1)
        # Step 2: larger area (got closer)
        result = lobe(make_obs(t=0.016), gate_at(0.5, 0.5, area=8000.0),
                      DroneState.APPROACH, 0.1)
        assert result.approach_rate > 0.0

    def test_approach_rate_decays_when_lost(self):
        lobe = ProgressLobe(default_config().progress)
        # Build up rate
        lobe(make_obs(t=0.0), gate_at(0.5, 0.5, area=5000.0), DroneState.APPROACH, 0.1)
        lobe(make_obs(t=0.016), gate_at(0.5, 0.5, area=8000.0), DroneState.APPROACH, 0.1)
        rate_before = lobe._approach_rate_ema
        # Gate lost
        lobe(make_obs(t=0.032), gate_at(0.5, 0.5, detected=False), DroneState.RECOVER, 0.1)
        assert lobe._approach_rate_ema < rate_before

    def test_gate_index_starts_zero(self):
        lobe = ProgressLobe(default_config().progress)
        result = lobe(make_obs(), gate_at(0.5, 0.5), DroneState.TRACK, 0.1)
        assert result.gate_index == 0

    def test_progress_score_zero_when_no_approach(self):
        lobe = ProgressLobe(default_config().progress)
        result = lobe(make_obs(), gate_at(0.5, 0.5, area=0.0, detected=False),
                      DroneState.SEARCH, 0.1)
        assert result.progress_score == pytest.approx(0.0, abs=0.1)

    def test_reset(self):
        lobe = ProgressLobe(default_config().progress)
        lobe(make_obs(), gate_at(0.5, 0.5, area=5000.0), DroneState.APPROACH, 1.0)
        lobe.reset()
        assert lobe._gate_index == 0
        assert lobe._prev_area == 0.0

    def test_cy_tilt_offset_zero_default(self):
        """Default config: cy_tilt_offset=0, gate at 0.5 gives dy≈0."""
        cfg = default_config()
        assert cfg.progress.cy_tilt_offset == 0.0
        lobe = ProgressLobe(cfg.progress)
        result = lobe(make_obs(), gate_at(0.5, 0.5), DroneState.TRACK, 0.1)
        assert abs(result.dy) < 0.05

    def test_cy_tilt_offset_shifts_centered_dy(self):
        """With cy_tilt_offset=0.16, gate at cy=0.66 should give dy≈0."""
        from aigrandprix.config import ProgressConfig
        cfg = ProgressConfig(cy_tilt_offset=0.16)
        lobe = ProgressLobe(cfg)
        # Gate at cy=0.66 = 0.5 + 0.16 → should be "centered" → dy ≈ 0
        result = lobe(make_obs(), gate_at(0.5, 0.66), DroneState.TRACK, 0.1)
        assert abs(result.dy) < 0.05

    def test_cy_tilt_offset_improves_aligned_score(self):
        """With camera tilt, a gate at cy=0.66 should score higher with offset than without."""
        from aigrandprix.config import ProgressConfig
        lobe_no_offset  = ProgressLobe(ProgressConfig(cy_tilt_offset=0.0))
        lobe_with_offset = ProgressLobe(ProgressConfig(cy_tilt_offset=0.16))
        gate = gate_at(0.5, 0.66)  # gate below center due to camera tilt

        result_no  = lobe_no_offset(make_obs(), gate, DroneState.TRACK, 0.1)
        result_yes = lobe_with_offset(make_obs(), gate, DroneState.TRACK, 0.1)

        # With offset, dy≈0 → aligned_score close to 1.0
        # Without offset, dy>0 → aligned_score lower
        assert result_yes.aligned_score > result_no.aligned_score
        assert result_yes.aligned_score > 0.95
