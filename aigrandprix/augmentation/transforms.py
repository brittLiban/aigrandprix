"""Image augmentation transforms for robustness testing.

Each transform takes (image: np.ndarray, rng: np.random.Generator, **params)
and returns a transformed image with the same shape and dtype.

Usage:
    from aigrandprix.augmentation.transforms import augment
    augmented = augment(image, seed=42, transforms=["brightness", "blur"])
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def brightness_contrast(image: np.ndarray, rng: np.random.Generator,
                         brightness_delta: float = 40.0,
                         contrast_factor_range: tuple = (0.7, 1.3)) -> np.ndarray:
    """Random brightness shift and contrast scaling."""
    img = image.astype(np.float32)
    # Contrast
    factor = rng.uniform(*contrast_factor_range)
    img = img * factor
    # Brightness
    delta = rng.uniform(-brightness_delta, brightness_delta)
    img = img + delta
    return np.clip(img, 0, 255).astype(np.uint8)


def gamma(image: np.ndarray, rng: np.random.Generator,
          gamma_range: tuple = (0.5, 2.0)) -> np.ndarray:
    """Random gamma correction."""
    g = rng.uniform(*gamma_range)
    table = np.array([(i / 255.0) ** (1.0 / g) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def motion_blur(image: np.ndarray, rng: np.random.Generator,
                kernel_sizes: tuple = (3, 5, 7, 9)) -> np.ndarray:
    """Horizontal or vertical motion blur."""
    k = int(rng.choice(list(kernel_sizes)))
    direction = int(rng.integers(0, 2))  # 0=horizontal, 1=vertical
    kernel = np.zeros((k, k), dtype=np.float32)
    if direction == 0:
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k
    return cv2.filter2D(image, -1, kernel)


def jpeg_artifacts(image: np.ndarray, rng: np.random.Generator,
                   quality_range: tuple = (20, 60)) -> np.ndarray:
    """JPEG compression at random quality."""
    quality = int(rng.integers(*quality_range))
    _, encoded = cv2.imencode(".jpg", image,
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def random_occlusion(image: np.ndarray, rng: np.random.Generator,
                     max_rect_fraction: float = 0.2) -> np.ndarray:
    """Add a random dark rectangle (occlusion)."""
    H, W = image.shape[:2]
    rw = int(rng.uniform(0.05, max_rect_fraction) * W)
    rh = int(rng.uniform(0.05, max_rect_fraction) * H)
    x = int(rng.integers(0, max(1, W - rw)))
    y = int(rng.integers(0, max(1, H - rh)))
    out = image.copy()
    out[y:y + rh, x:x + rw] = 0
    return out


def gaussian_noise(image: np.ndarray, rng: np.random.Generator,
                   std_range: tuple = (5.0, 25.0)) -> np.ndarray:
    """Add Gaussian noise."""
    std = rng.uniform(*std_range)
    noise = rng.normal(0, std, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Registry and dispatch
# ---------------------------------------------------------------------------

_TRANSFORMS = {
    "brightness": brightness_contrast,
    "gamma": gamma,
    "motion_blur": motion_blur,
    "jpeg": jpeg_artifacts,
    "occlusion": random_occlusion,
    "noise": gaussian_noise,
}

ALL_TRANSFORMS = list(_TRANSFORMS.keys())


def augment(image: np.ndarray, seed: Optional[int] = None,
            transforms: Optional[list[str]] = None) -> np.ndarray:
    """Apply one or more named transforms to an image.

    Args:
        image:      Input image, (H, W, 3) uint8.
        seed:       RNG seed for reproducibility.
        transforms: List of transform names to apply in order.
                    Defaults to all transforms.

    Returns:
        Augmented image with same shape and dtype.
    """
    rng = np.random.default_rng(seed)
    if transforms is None:
        transforms = ALL_TRANSFORMS
    out = image.copy()
    for name in transforms:
        fn = _TRANSFORMS.get(name)
        if fn is not None:
            out = fn(out, rng)
    return out
