"""Unit tests for StabilityLobe."""
import numpy as np
import pytest

from aigrandprix.config import default_config
from aigrandprix.lobes.stability import StabilityLobe
from aigrandprix.types import Observation


def obs_with_imu(accel, gyro, t=0.0):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return Observation(t=t, dt=0.016, image=img,
                       imu_accel=np.array(accel), imu_gyro=np.array(gyro))


class TestStabilityLobe:
    def test_stable_hover(self):
        lobe = StabilityLobe(default_config().stability)
        result = lobe(obs_with_imu([0, 0, 9.81], [0, 0, 0]))
        assert result.stability_score > 0.8
        assert not result.is_tumbling
        assert result.instability_spike_count == 0

    def test_tumbling_high_gyro(self):
        lobe = StabilityLobe(default_config().stability)
        # gyro_tumble_threshold default = 8.0 rad/s
        result = lobe(obs_with_imu([0, 0, 9.81], [0, 0, 10.0]))
        assert result.is_tumbling
        assert result.stability_score < 0.5

    def test_not_tumbling_moderate_gyro(self):
        lobe = StabilityLobe(default_config().stability)
        result = lobe(obs_with_imu([0, 0, 9.81], [0, 0, 3.0]))
        assert not result.is_tumbling

    def test_spike_count_increments(self):
        lobe = StabilityLobe(default_config().stability)
        # Spike threshold = gyro_tumble / 2 = 4.0 rad/s default
        for i in range(3):
            lobe(obs_with_imu([0, 0, 9.81], [0, 0, 5.0], t=float(i) * 0.016))
        result = lobe(obs_with_imu([0, 0, 9.81], [0, 0, 5.0], t=3 * 0.016))
        assert result.instability_spike_count >= 3

    def test_spike_window_slides(self):
        lobe = StabilityLobe(default_config().stability)
        cfg = default_config().stability
        # Add spikes, then advance time past window
        for i in range(5):
            lobe(obs_with_imu([0, 0, 9.81], [0, 0, 5.0], t=float(i) * 0.016))
        # Now at t = spike_window + some, spikes should drop out
        result = lobe(obs_with_imu([0, 0, 9.81], [0, 0, 0.0],
                                   t=cfg.spike_window_s + 1.0))
        assert result.instability_spike_count == 0

    def test_accel_norm(self):
        lobe = StabilityLobe(default_config().stability)
        result = lobe(obs_with_imu([0, 0, 9.81], [0, 0, 0]))
        assert result.accel_norm == pytest.approx(9.81, abs=0.01)

    def test_reset_clears_spikes(self):
        lobe = StabilityLobe(default_config().stability)
        for i in range(5):
            lobe(obs_with_imu([0, 0, 9.81], [0, 0, 5.0], t=float(i) * 0.016))
        lobe.reset()
        result = lobe(obs_with_imu([0, 0, 9.81], [0, 0, 5.0], t=10.0))
        assert result.instability_spike_count == 1  # only current frame
