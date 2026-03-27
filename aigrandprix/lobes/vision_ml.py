"""MLVisionLobe — drop-in ML replacement for VisionLobe.

Same __call__(obs) -> VisionResult interface as VisionLobe.
Uses a trained GateDetector CNN instead of HSV+contour.

Requires: torch (optional import — falls back to HSV if not available or
          if config.vision.model_path is empty).

Load via config:
    vision:
      backend: "ml"
      model_path: "checkpoints/gate_detector.pt"
      ml_input_h: 128
      ml_input_w: 160
      ml_conf_threshold: 0.5
      # EMA and hold settings still apply on top:
      confidence_ema_alpha: 0.6
      position_hold_frames: 4
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

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class MLVisionLobe:
    """Gate detector backed by a trained GateDetector CNN.

    Falls back to a warning + zero result if model_path is empty or torch
    is not installed.  Call .is_ready() to check before use.
    """

    def __init__(self, config: VisionConfig, budget_ms: float = 8.0):
        self._cfg = config
        self._budget_ms = budget_ms
        self._model = None
        self._device = "cpu"

        # Stateful (same as VisionLobe)
        self._confidence_ema: float = 0.0
        self._last_seen_t: float = -999.0
        self._hold_cx: float = 0.5
        self._hold_cy: float = 0.5
        self._hold_area: float = 0.0
        self._hold_bbox: object = None
        self._hold_count: int = 0

        self._load_model()

    def _load_model(self) -> None:
        if not _TORCH_AVAILABLE:
            return
        path = self._cfg.model_path
        if not path:
            return
        try:
            from aigrandprix.ml.model import GateDetector
            self._model = GateDetector(
                input_h=self._cfg.ml_input_h,
                input_w=self._cfg.ml_input_w,
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            # Support both raw state_dict and {"model": state_dict} checkpoints
            state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            self._model.load_state_dict(state)
            self._model.eval()
        except Exception as e:
            print(f"[MLVisionLobe] failed to load model from {path}: {e}")
            self._model = None

    def is_ready(self) -> bool:
        return self._model is not None

    @timed("MLVisionLobe")
    def __call__(self, obs: Observation) -> VisionResult:
        t0 = time.perf_counter()
        if self._model is None:
            result = VisionResult(
                gate_detected=False, cx=0.5, cy=0.5, area=0.0,
                confidence=0.0, confidence_ema=self._confidence_ema,
                last_seen_t=self._last_seen_t, bbox=None, latency_ms=0.0,
            )
        else:
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

        # Resize to model input size
        img_small = cv2.resize(obs.image, (cfg.ml_input_w, cfg.ml_input_h))

        # (H, W, 3) uint8 → (1, 3, H, W) float32 in [0, 1]
        tensor = (
            torch.from_numpy(img_small)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div(255.0)
        )

        with torch.no_grad():
            raw = self._model(tensor)   # (1, 5)

        conf = float(torch.sigmoid(raw[0, 0]).item())
        cx   = float(raw[0, 1].item())
        cy   = float(raw[0, 2].item())
        bw   = float(raw[0, 3].item())
        bh   = float(raw[0, 4].item())

        detected = conf >= cfg.ml_conf_threshold

        if not detected:
            # Position hold (same logic as VisionLobe)
            max_hold = cfg.position_hold_frames
            if max_hold > 0 and self._hold_count < max_hold and self._hold_area > 0:
                self._hold_count += 1
                held_conf = self._confidence_ema * 0.5
                return VisionResult(
                    gate_detected=True,
                    cx=self._hold_cx, cy=self._hold_cy, area=self._hold_area,
                    confidence=held_conf, confidence_ema=self._confidence_ema,
                    last_seen_t=self._last_seen_t, bbox=self._hold_bbox,
                    latency_ms=0.0,
                )
            self._hold_count = max_hold
            return VisionResult(
                gate_detected=False, cx=0.5, cy=0.5, area=0.0,
                confidence=0.0, confidence_ema=self._confidence_ema,
                last_seen_t=self._last_seen_t, bbox=None, latency_ms=0.0,
            )

        # Convert normalized bbox to pixel coords in original resolution
        cx_px = cx * W_orig
        cy_px = cy * H_orig
        w_px  = bw * W_orig
        h_px  = bh * H_orig
        x1 = int(cx_px - w_px / 2)
        y1 = int(cy_px - h_px / 2)
        bbox = (max(0, x1), max(0, y1), int(w_px), int(h_px))
        area = w_px * h_px

        # Update position hold state
        self._hold_cx    = cx
        self._hold_cy    = cy
        self._hold_area  = area
        self._hold_bbox  = bbox
        self._hold_count = 0

        return VisionResult(
            gate_detected=True,
            cx=cx, cy=cy, area=area,
            confidence=conf,
            confidence_ema=self._confidence_ema,
            last_seen_t=self._last_seen_t,
            bbox=bbox,
            latency_ms=0.0,
        )

    # ------------------------------------------------------------------
    # EMA — identical to VisionLobe
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

        result.confidence_ema = self._confidence_ema
        result.last_seen_t = self._last_seen_t
        return result

    def reset(self) -> None:
        self._confidence_ema = 0.0
        self._last_seen_t = -999.0
        self._hold_cx = 0.5
        self._hold_cy = 0.5
        self._hold_area = 0.0
        self._hold_bbox = None
        self._hold_count = 0
