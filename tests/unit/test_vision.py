"""Unit tests for VisionLobe."""
import cv2
import numpy as np
import pytest

from aigrandprix.config import default_config
from aigrandprix.lobes.vision import VisionLobe
from aigrandprix.types import Observation


def make_obs(image: np.ndarray, t: float = 1.0) -> Observation:
    return Observation(
        t=t, dt=0.016,
        image=image,
        imu_accel=np.array([0.0, 0.0, 9.81]),
        imu_gyro=np.zeros(3),
    )


def gate_image(cx: float, cy: float, scale: float = 0.3,
               H: int = 480, W: int = 640) -> np.ndarray:
    """Synthetic white gate rectangle on dark background."""
    img = np.full((H, W, 3), 20, dtype=np.uint8)
    gate_w = int(scale * W * 0.7)
    gate_h = int(scale * H * 0.7)
    x1 = int(cx * W) - gate_w // 2
    y1 = int(cy * H) - gate_h // 2
    x2 = x1 + gate_w
    y2 = y1 + gate_h
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 8)
    return img


class TestVisionLobeDetection:
    def test_detects_centered_gate(self):
        lobe = VisionLobe(default_config().vision)
        obs = make_obs(gate_image(0.5, 0.5))
        result = lobe(obs)
        assert result.gate_detected
        assert abs(result.cx - 0.5) < 0.1
        assert abs(result.cy - 0.5) < 0.1

    def test_no_detection_on_blank_image(self):
        lobe = VisionLobe(default_config().vision)
        blank = np.full((480, 640, 3), 20, dtype=np.uint8)
        result = lobe(make_obs(blank))
        assert not result.gate_detected
        assert result.confidence == pytest.approx(0.0)

    def test_gate_positions(self):
        lobe = VisionLobe(default_config().vision)
        for cx, cy in [(0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.5, 0.3)]:
            lobe.reset()
            obs = make_obs(gate_image(cx, cy, scale=0.25))
            result = lobe(obs)
            assert result.gate_detected, f"Not detected at cx={cx}, cy={cy}"
            assert abs(result.cx - cx) < 0.15, f"cx off: {result.cx} vs {cx}"
            assert abs(result.cy - cy) < 0.15, f"cy off: {result.cy} vs {cy}"

    def test_small_gate_below_min_area(self):
        lobe = VisionLobe(default_config().vision)
        # Very small gate should not be detected
        obs = make_obs(gate_image(0.5, 0.5, scale=0.01))
        result = lobe(obs)
        assert not result.gate_detected

    def test_confidence_positive_when_detected(self):
        lobe = VisionLobe(default_config().vision)
        obs = make_obs(gate_image(0.5, 0.5))
        result = lobe(obs)
        if result.gate_detected:
            assert result.confidence > 0.0


class TestVisionLobeEMA:
    def test_ema_builds_up_on_repeated_detections(self):
        lobe = VisionLobe(default_config().vision)
        obs = make_obs(gate_image(0.5, 0.5))
        for _ in range(5):
            result = lobe(obs)
        assert result.confidence_ema > 0.0

    def test_ema_decays_when_gate_lost(self):
        lobe = VisionLobe(default_config().vision)
        # Build up EMA
        for i in range(5):
            lobe(make_obs(gate_image(0.5, 0.5), t=float(i) * 0.016))
        ema_before = lobe._confidence_ema
        assert ema_before > 0.1

        # Now lose gate for 1 second
        blank = np.full((480, 640, 3), 20, dtype=np.uint8)
        lobe(make_obs(blank, t=5.016))  # 1 second later
        assert lobe._confidence_ema < ema_before

    def test_ema_resets(self):
        lobe = VisionLobe(default_config().vision)
        for i in range(5):
            lobe(make_obs(gate_image(0.5, 0.5), t=float(i) * 0.016))
        lobe.reset()
        assert lobe._confidence_ema == pytest.approx(0.0)
