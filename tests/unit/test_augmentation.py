"""Unit tests for augmentation transforms + detection rate under augmentation."""
import cv2
import numpy as np
import pytest

from aigrandprix.augmentation.transforms import (
    ALL_TRANSFORMS, augment, brightness_contrast, gamma,
    gaussian_noise, jpeg_artifacts, motion_blur, random_occlusion,
)
from aigrandprix.config import default_config
from aigrandprix.lobes.vision import VisionLobe
from aigrandprix.types import Observation


def gate_image(cx=0.5, cy=0.5, scale=0.35, H=480, W=640):
    img = np.full((H, W, 3), 20, dtype=np.uint8)
    gate_w = int(scale * W * 0.7)
    gate_h = int(scale * H * 0.7)
    x1 = int(cx * W) - gate_w // 2
    y1 = int(cy * H) - gate_h // 2
    cv2.rectangle(img, (x1, y1), (x1 + gate_w, y1 + gate_h),
                  (255, 255, 255), 8)
    return img


def obs_from(image):
    return Observation(
        t=1.0, dt=0.016, image=image,
        imu_accel=np.array([0., 0., 9.81]),
        imu_gyro=np.zeros(3),
    )


class TestTransformShapePreservation:
    """Each transform must preserve shape and dtype."""

    def _check(self, fn, image, rng):
        out = fn(image, rng)
        assert out.shape == image.shape, f"{fn.__name__} changed shape"
        assert out.dtype == np.uint8, f"{fn.__name__} changed dtype"

    def test_brightness_contrast(self):
        rng = np.random.default_rng(0)
        self._check(brightness_contrast, gate_image(), rng)

    def test_gamma(self):
        rng = np.random.default_rng(0)
        self._check(gamma, gate_image(), rng)

    def test_motion_blur(self):
        rng = np.random.default_rng(0)
        self._check(motion_blur, gate_image(), rng)

    def test_jpeg(self):
        rng = np.random.default_rng(0)
        self._check(jpeg_artifacts, gate_image(), rng)

    def test_occlusion(self):
        rng = np.random.default_rng(0)
        self._check(random_occlusion, gate_image(), rng)

    def test_gaussian_noise(self):
        rng = np.random.default_rng(0)
        self._check(gaussian_noise, gate_image(), rng)


class TestAugmentDispatch:
    def test_all_transforms_run(self):
        img = gate_image()
        out = augment(img, seed=42, transforms=ALL_TRANSFORMS)
        assert out.shape == img.shape

    def test_single_transform(self):
        img = gate_image()
        out = augment(img, seed=0, transforms=["brightness"])
        assert out.shape == img.shape

    def test_deterministic_with_seed(self):
        img = gate_image()
        out1 = augment(img, seed=7, transforms=["noise"])
        out2 = augment(img, seed=7, transforms=["noise"])
        assert np.array_equal(out1, out2)

    def test_different_seed_gives_different_output(self):
        img = gate_image()
        out1 = augment(img, seed=1, transforms=["noise"])
        out2 = augment(img, seed=2, transforms=["noise"])
        assert not np.array_equal(out1, out2)


class TestDetectionRateUnderAugmentation:
    """VisionLobe must detect gate in >= 70% of augmented frames per transform."""
    N_SAMPLES = 50
    THRESHOLD = 0.70

    @pytest.mark.slow
    @pytest.mark.parametrize("transform_name", ALL_TRANSFORMS)
    def test_detection_rate(self, transform_name):
        lobe = VisionLobe(default_config().vision)
        rng = np.random.default_rng(42)
        # Vary gate position and scale for a realistic spread
        detected = 0
        for i in range(self.N_SAMPLES):
            cx = rng.uniform(0.25, 0.75)
            cy = rng.uniform(0.25, 0.75)
            scale = rng.uniform(0.25, 0.45)
            img = gate_image(cx=cx, cy=cy, scale=scale)
            aug = augment(img, seed=i, transforms=[transform_name])
            lobe.reset()
            result = lobe(obs_from(aug))
            if result.gate_detected:
                detected += 1
        rate = detected / self.N_SAMPLES
        assert rate >= self.THRESHOLD, (
            f"Detection rate under '{transform_name}': {rate:.0%} < {self.THRESHOLD:.0%}"
        )
