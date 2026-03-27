"""Shared pytest fixtures."""
import numpy as np
import pytest

from aigrandprix.brain.states import DroneState
from aigrandprix.config import default_config
from aigrandprix.types import Action, Observation, VisionResult, StabilityResult


@pytest.fixture
def cfg():
    return default_config()


@pytest.fixture
def blank_obs():
    """Observation with a plain dark image (no gate)."""
    img = np.full((480, 640, 3), 20, dtype=np.uint8)
    return Observation(
        t=1.0, dt=0.016,
        image=img,
        imu_accel=np.array([0.0, 0.0, 9.81]),
        imu_gyro=np.zeros(3),
    )


@pytest.fixture
def gate_obs():
    """Observation with a white gate rectangle in the center."""
    img = np.full((480, 640, 3), 20, dtype=np.uint8)
    # Draw white rectangle in center
    import cv2
    cv2.rectangle(img, (220, 150), (420, 330), (255, 255, 255), 8)
    return Observation(
        t=1.0, dt=0.016,
        image=img,
        imu_accel=np.array([0.0, 0.0, 9.81]),
        imu_gyro=np.zeros(3),
    )


@pytest.fixture
def stable_stability():
    return StabilityResult.stable()


@pytest.fixture
def detected_vision():
    return VisionResult(
        gate_detected=True, cx=0.5, cy=0.5, area=10000.0,
        confidence=0.8, confidence_ema=0.7, last_seen_t=1.0,
        bbox=(200, 140, 200, 200), latency_ms=3.0,
    )


@pytest.fixture
def no_gate_vision():
    return VisionResult.null(t=1.0)
