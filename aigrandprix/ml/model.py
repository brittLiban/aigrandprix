"""GateDetector — lightweight CNN for gate detection and localization.

Architecture
------------
Tiny backbone (4 × conv stride-2 blocks) + global average pool + two FC layers.
~26 K parameters.  Runs in <2 ms on CPU at 128×160 input.

Output format (5 values per image)
-----------------------------------
  [0]  det_logit  — raw logit; apply sigmoid → P(gate present)
  [1]  cx         — gate center x, normalized [0, 1]
  [2]  cy         — gate center y, normalized [0, 1]
  [3]  bw         — gate bbox width, normalized [0, 1]
  [4]  bh         — gate bbox height, normalized [0, 1]

All bbox outputs are sigmoid-activated (guaranteed in [0, 1]).

Training
--------
See scripts/train_detector.py.  Loss = BCEWithLogitsLoss(det) +
bbox_weight * SmoothL1(bbox) masked to positive samples only.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class GateDetector(nn.Module):
    """Tiny gate detector.

    Args:
        input_h: Expected input height (default 128).
        input_w: Expected input width (default 160).
    """

    def __init__(self, input_h: int = 128, input_w: int = 160):
        super().__init__()
        self.input_h = input_h
        self.input_w = input_w

        self.backbone = nn.Sequential(
            _ConvBnRelu(3,  8,  stride=2),   # → 64 × 80
            _ConvBnRelu(8,  16, stride=2),   # → 32 × 40
            _ConvBnRelu(16, 32, stride=2),   # → 16 × 20
            _ConvBnRelu(32, 64, stride=2),   # →  8 × 10
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # → 64

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float32 in [0, 1].
        Returns (B, 5): [det_logit, cx, cy, bw, bh].
        cx/cy/bw/bh are passed through sigmoid.
        """
        feat = self.pool(self.backbone(x)).flatten(1)   # (B, 64)
        out = self.head(feat)                            # (B, 5)
        # det_logit stays raw; bbox outputs → sigmoid
        bbox = torch.sigmoid(out[:, 1:])                # (B, 4)
        return torch.cat([out[:, :1], bbox], dim=1)     # (B, 5)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor,
                conf_threshold: float = 0.5) -> list[dict]:
        """Run inference on a batch, return list of result dicts.

        Args:
            x: (B, 3, H, W) float32 in [0, 1].
            conf_threshold: Min sigmoid(det_logit) to declare detection.

        Returns:
            List of dicts with keys: detected, confidence, cx, cy, bw, bh.
        """
        self.eval()
        with torch.no_grad():
            raw = self(x)
        results = []
        for i in range(raw.shape[0]):
            det_logit = raw[i, 0].item()
            conf = float(torch.sigmoid(raw[i:i+1, 0]).item())
            results.append({
                "detected": conf >= conf_threshold,
                "confidence": conf,
                "cx": float(raw[i, 1].item()),
                "cy": float(raw[i, 2].item()),
                "bw": float(raw[i, 3].item()),
                "bh": float(raw[i, 4].item()),
            })
        return results


def gate_loss(pred: torch.Tensor, labels: torch.Tensor,
              bbox_weight: float = 5.0) -> tuple[torch.Tensor, dict]:
    """Combined detection + bbox regression loss.

    Args:
        pred:   (B, 5) model output — [det_logit, cx, cy, bw, bh].
        labels: (B, 5) ground truth — [det_float, cx, cy, bw, bh].
                det_float is 1.0 for present, 0.0 for absent.
        bbox_weight: Weight for bbox loss relative to detection loss.

    Returns:
        (total_loss, {"det": ..., "bbox": ...})
    """
    det_pred  = pred[:, 0]         # raw logit
    det_label = labels[:, 0]       # 0 or 1

    det_loss = nn.functional.binary_cross_entropy_with_logits(
        det_pred, det_label, reduction="mean"
    )

    # Bbox loss only on positive (gate present) samples
    pos_mask = det_label > 0.5
    if pos_mask.any():
        bbox_pred  = pred[pos_mask, 1:]     # (N+, 4)
        bbox_label = labels[pos_mask, 1:]   # (N+, 4)
        bbox_loss = nn.functional.smooth_l1_loss(bbox_pred, bbox_label,
                                                 reduction="mean")
    else:
        bbox_loss = torch.tensor(0.0, device=pred.device)

    total = det_loss + bbox_weight * bbox_loss
    return total, {"det": det_loss.item(), "bbox": bbox_loss.item()}
