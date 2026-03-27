"""VisionLobe — gate detection and tracking from FPV camera frames.

v1: Classical HSV + contour detection.
    Designed to be swapped for ML (YOLO/segmentation) in v2 by keeping
    the same __call__ interface.

Key behaviors:
  - Resizes input image for fast processing
  - HSV thresholding to isolate gate color (configurable)
  - Contour filtering by area and aspect ratio
  - Confidence = combination of area and aspect score
  - EMA smoothing of confidence over time
  - Exponential decay of confidence_ema when gate not detected
"""
from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from aigrandprix.config import VisionConfig
from aigrandprix.timing import timed
from aigrandprix.types import Observation, VisionResult


class VisionLobe:
    """Classical gate detector with temporal confidence smoothing."""

    def __init__(self, config: VisionConfig, budget_ms: float = 8.0):
        self._cfg = config
        self._budget_ms = budget_ms
        self._violation_count = 0
        # Stateful: EMA smoothing
        self._confidence_ema: float = 0.0
        self._last_seen_t: float = -999.0
        # Position hold: persist last known gate estimate on brief dropouts
        self._hold_cx: float = 0.5
        self._hold_cy: float = 0.5
        self._hold_area: float = 0.0
        self._hold_bbox: object = None
        self._hold_count: int = 0   # frames since last real detection

    @timed("VisionLobe")
    def __call__(self, obs: Observation) -> VisionResult:
        t0 = time.perf_counter()
        result = self._detect(obs)
        result = self._update_ema(result, obs.t)
        result.latency_ms = (time.perf_counter() - t0) * 1000.0
        return result

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect(self, obs: Observation) -> VisionResult:
        cfg = self._cfg
        H_orig, W_orig = obs.image.shape[:2]

        # Resize for speed
        image = cv2.resize(obs.image, (cfg.resize_w, cfg.resize_h))
        H, W = cfg.resize_h, cfg.resize_w

        # Convert to HSV and threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lower = np.array(cfg.hsv_lower, dtype=np.uint8)
        upper = np.array(cfg.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        detected = False
        if contours:
            best = max(contours, key=cv2.contourArea)
            if cv2.contourArea(best) >= cfg.min_contour_area:
                detected = True

        if not detected:
            # Position hold: for the first N frames after dropout, return the
            # last known position with reduced confidence instead of cx=0.5/area=0.
            # This prevents frames_since_gate from accumulating on brief noise/
            # exposure dropouts where the gate is physically still there.
            max_hold = cfg.position_hold_frames
            if max_hold > 0 and self._hold_count < max_hold and self._hold_area > 0:
                self._hold_count += 1
                held_conf = self._confidence_ema * 0.5   # reduced confidence
                return VisionResult(
                    gate_detected=True,    # held — don't trip frames_since_gate
                    cx=self._hold_cx,
                    cy=self._hold_cy,
                    area=self._hold_area,
                    confidence=held_conf,
                    confidence_ema=self._confidence_ema,
                    last_seen_t=self._last_seen_t,
                    bbox=self._hold_bbox,
                    latency_ms=0.0,
                )
            self._hold_count = max_hold  # saturate so we stop holding
            return VisionResult(
                gate_detected=False, cx=0.5, cy=0.5, area=0.0,
                confidence=0.0, confidence_ema=self._confidence_ema,
                last_seen_t=self._last_seen_t, bbox=None, latency_ms=0.0,
            )

        x, y, w, h = cv2.boundingRect(best)
        cx_norm = (x + w / 2) / W
        cy_norm = (y + h / 2) / H
        area_px2 = float(w * h)

        # Scale bbox back to original resolution
        sx = W_orig / W
        sy = H_orig / H
        bbox = (int(x * sx), int(y * sy), int(w * sx), int(h * sy))

        confidence = self._compute_confidence(area_px2, w, h, W, H)

        # Update hold state with fresh detection
        self._hold_cx    = float(cx_norm)
        self._hold_cy    = float(cy_norm)
        self._hold_area  = float(area_px2 * (sx * sy))
        self._hold_bbox  = bbox
        self._hold_count = 0

        return VisionResult(
            gate_detected=True,
            cx=float(cx_norm),
            cy=float(cy_norm),
            area=area_px2 * (sx * sy),
            confidence=confidence,
            confidence_ema=self._confidence_ema,
            last_seen_t=self._last_seen_t,
            bbox=bbox,
            latency_ms=0.0,
        )

    def _compute_confidence(self, area: float, w: int, h: int,
                             W: int, H: int) -> float:
        """Confidence in [0,1] based on area coverage and aspect ratio."""
        # Area score: how much of the frame is the gate
        max_area = W * H
        area_score = min(area / (max_area * 0.5), 1.0)

        # Aspect score: gate should be roughly square (0.5–2.0 is fine)
        aspect = w / max(h, 1)
        aspect_score = 1.0 - min(abs(math.log(max(aspect, 0.1))), 1.0)

        return float(0.6 * area_score + 0.4 * aspect_score)

    # ------------------------------------------------------------------
    # EMA / temporal smoothing
    # ------------------------------------------------------------------

    def _update_ema(self, result: VisionResult, t: float) -> VisionResult:
        alpha = self._cfg.confidence_ema_alpha
        decay = self._cfg.confidence_decay_rate

        if result.gate_detected:
            self._confidence_ema = (alpha * result.confidence
                                    + (1 - alpha) * self._confidence_ema)
            self._last_seen_t = t
        else:
            elapsed = max(t - self._last_seen_t, 0.0)
            self._confidence_ema *= math.exp(-decay * elapsed)

        # Write updated values into result (dataclass is mutable)
        result.confidence_ema = self._confidence_ema
        result.last_seen_t = self._last_seen_t
        return result

    def reset(self) -> None:
        """Reset stateful EMA (call at episode start)."""
        self._confidence_ema = 0.0
        self._last_seen_t = -999.0
        self._hold_cx = 0.5
        self._hold_cy = 0.5
        self._hold_area = 0.0
        self._hold_bbox = None
        self._hold_count = 0
