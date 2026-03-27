"""Real-time drone racing demo — cv2 visualization for team presentations.

Usage:
    python scripts/demo.py [--config configs/base.yaml] [--seed 42] [--fps 20]

Keys:
    q  — quit
    r  — restart episode
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from aigrandprix.brain.fusion import FusionBrain
from aigrandprix.brain.states import DroneState
from aigrandprix.config import Config, default_config, load_config
from aigrandprix.controller.pid import Controller
from aigrandprix.lobes.progress import ProgressLobe
from aigrandprix.lobes.recovery import RecoveryLobe
from aigrandprix.lobes.risk import RiskLobe
from aigrandprix.lobes.stability import StabilityLobe
from aigrandprix.lobes.vision import VisionLobe
from aigrandprix.types import Action, Observation, VisionResult

# ── Layout constants ──────────────────────────────────────────────────────────
WIN_W, WIN_H = 960, 540
FPV_W, FPV_H = 640, 480
DASH_W = WIN_W - FPV_W  # 320
DASH_H = WIN_H           # 540

# ── State colours (BGR) ───────────────────────────────────────────────────────
STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "SEARCH":   (0,   150, 200),
    "TRACK":    (255, 200,   0),
    "APPROACH": (0,   220,   0),
    "COMMIT":   (150, 255,   0),
    "RECOVER":  (0,     0, 220),
}
STATE_ORDER = ["SEARCH", "TRACK", "APPROACH", "COMMIT", "RECOVER"]

RISK_COLORS = [
    (0, 200, 0),     # push 0 — green
    (0, 200, 200),   # push 1 — yellow
    (0, 140, 255),   # push 2 — orange
    (0, 0, 255),     # push 3 — red
]

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


# ── Demo state ────────────────────────────────────────────────────────────────
@dataclass
class DemoState:
    state_name:   str   = "SEARCH"
    gate_index:   int   = 0
    total_gates:  int   = 10
    lap_time:     float = 0.0
    push_level:   int   = 0
    risk_score:   float = 0.0
    aligned:      float = 0.0
    approach_rate: float = 0.0
    pipeline_ms:  float = 0.0
    in_recovery:  bool  = False
    recovery_events: int = 0
    gates_passed: int   = 0
    done:         bool  = False
    action:       Action = field(default_factory=Action.zero)
    fpv_frame:    Optional[np.ndarray] = None   # latest RGB frame from adapter
    bbox:         Optional[tuple]      = None   # (x,y,w,h) or None
    # rolling history for action bars (last 60 frames)
    roll_hist:     list[float] = field(default_factory=list)
    pitch_hist:    list[float] = field(default_factory=list)
    throttle_hist: list[float] = field(default_factory=list)


# ── Visualiser ────────────────────────────────────────────────────────────────
class DemoVisualizer:
    def __init__(self, total_gates: int = 10):
        self._total = total_gates
        cv2.namedWindow("AI Grand Prix — Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Grand Prix — Demo", WIN_W, WIN_H)

    def render(self, ds: DemoState) -> int:
        """Draw frame, show it, return cv2.waitKey result (1 ms)."""
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        self._draw_fpv(canvas, ds)
        self._draw_dashboard(canvas, ds)
        if ds.done:
            self._draw_completion_overlay(canvas, ds)
        cv2.imshow("AI Grand Prix — Demo", canvas)
        return cv2.waitKey(1) & 0xFF

    # ── FPV panel ─────────────────────────────────────────────────────────────
    def _draw_fpv(self, canvas: np.ndarray, ds: DemoState) -> None:
        if ds.fpv_frame is not None:
            # Scale to FPV_W × FPV_H
            bgr = cv2.cvtColor(ds.fpv_frame, cv2.COLOR_RGB2BGR)
            view = cv2.resize(bgr, (FPV_W, FPV_H))
        else:
            view = np.zeros((FPV_H, FPV_W, 3), dtype=np.uint8)

        # Gate bounding box
        if ds.bbox is not None:
            x, y, w, h = ds.bbox
            # scale from original image coords to FPV_W×FPV_H
            orig_h, orig_w = (ds.fpv_frame.shape[:2]
                              if ds.fpv_frame is not None else (FPV_H, FPV_W))
            sx = FPV_W / orig_w
            sy = FPV_H / orig_h
            px, py = int(x * sx), int(y * sy)
            pw, ph = int(w * sx), int(h * sy)
            col = STATE_COLORS.get(ds.state_name, (255, 255, 255))
            cv2.rectangle(view, (px, py), (px + pw, py + ph), col, 2)
            cv2.putText(view, "GATE", (px, max(py - 6, 12)),
                        FONT, 0.45, col, 1, cv2.LINE_AA)

        # Crosshair at image centre
        cx_px, cy_px = FPV_W // 2, FPV_H // 2
        cv2.line(view, (cx_px - 20, cy_px), (cx_px + 20, cy_px), (80, 80, 80), 1)
        cv2.line(view, (cx_px, cy_px - 20), (cx_px, cy_px + 20), (80, 80, 80), 1)

        # State banner at top of FPV
        col = STATE_COLORS.get(ds.state_name, (200, 200, 200))
        cv2.rectangle(view, (0, 0), (FPV_W, 32), (20, 20, 20), -1)
        cv2.putText(view, ds.state_name, (10, 23),
                    FONT_BOLD, 0.75, col, 2, cv2.LINE_AA)

        # HUD — bottom strip
        hud_y = FPV_H - 10
        cv2.putText(view,
                    f"Gate {ds.gate_index}/{ds.total_gates}  "
                    f"t={ds.lap_time:.1f}s  "
                    f"pipe={ds.pipeline_ms:.1f}ms",
                    (8, hud_y), FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # Paste FPV into canvas
        canvas[0:FPV_H, 0:FPV_W] = view

        # Recovery flash border
        if ds.in_recovery:
            cv2.rectangle(canvas, (0, 0), (FPV_W - 1, FPV_H - 1),
                          (0, 0, 200), 4)

    # ── Dashboard panel ───────────────────────────────────────────────────────
    def _draw_dashboard(self, canvas: np.ndarray, ds: DemoState) -> None:
        ox = FPV_W  # x offset for dashboard
        bg = np.full((DASH_H, DASH_W, 3), (18, 18, 18), dtype=np.uint8)

        y = 20
        # Title
        cv2.putText(bg, "AI GRAND PRIX", (10, y),
                    FONT_BOLD, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        y += 30

        # ── State machine boxes ──────────────────────────────────────────────
        cv2.putText(bg, "STATE MACHINE", (10, y), FONT, 0.38, (120, 120, 120), 1)
        y += 16
        box_w = 58
        box_h = 24
        gap   = 4
        total_w = len(STATE_ORDER) * box_w + (len(STATE_ORDER) - 1) * gap
        start_x = (DASH_W - total_w) // 2
        for i, sname in enumerate(STATE_ORDER):
            bx = start_x + i * (box_w + gap)
            active = (sname == ds.state_name)
            col = STATE_COLORS[sname]
            if active:
                cv2.rectangle(bg, (bx, y), (bx + box_w, y + box_h), col, -1)
                txt_col = (10, 10, 10)
            else:
                cv2.rectangle(bg, (bx, y), (bx + box_w, y + box_h), col, 1)
                txt_col = tuple(c // 2 for c in col)
            label = sname[:4]
            tw, _ = cv2.getTextSize(label, FONT, 0.32, 1)[0], None
            cv2.putText(bg, label, (bx + (box_w - tw[0]) // 2, y + 16),
                        FONT, 0.32, txt_col, 1, cv2.LINE_AA)
        y += box_h + 14

        # ── Gate progress bar ────────────────────────────────────────────────
        cv2.putText(bg, f"GATES  {ds.gate_index} / {ds.total_gates}",
                    (10, y), FONT, 0.38, (120, 120, 120), 1)
        y += 14
        bar_x, bar_w_total, bar_h = 10, DASH_W - 20, 16
        cv2.rectangle(bg, (bar_x, y), (bar_x + bar_w_total, y + bar_h),
                      (50, 50, 50), -1)
        filled = int(bar_w_total * ds.gate_index / max(ds.total_gates, 1))
        if filled > 0:
            cv2.rectangle(bg, (bar_x, y), (bar_x + filled, y + bar_h),
                          (0, 200, 100), -1)
        pct = 100 * ds.gate_index // max(ds.total_gates, 1)
        cv2.putText(bg, f"{pct}%",
                    (bar_x + bar_w_total // 2 - 12, y + 12),
                    FONT, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
        y += bar_h + 14

        # ── Risk / push level ────────────────────────────────────────────────
        cv2.putText(bg, f"RISK  {ds.risk_score:.2f}   PUSH LV {ds.push_level}",
                    (10, y), FONT, 0.38, (120, 120, 120), 1)
        y += 14
        push_col = RISK_COLORS[min(ds.push_level, 3)]
        for lv in range(4):
            bx2 = 10 + lv * 50
            filled2 = (lv <= ds.push_level)
            cv2.rectangle(bg, (bx2, y), (bx2 + 44, y + 18),
                          push_col if filled2 else (40, 40, 40), -1 if filled2 else 1)
            cv2.putText(bg, str(lv), (bx2 + 16, y + 13),
                        FONT, 0.38, (10, 10, 10) if filled2 else (80, 80, 80), 1)
        y += 30

        # ── Key metrics ──────────────────────────────────────────────────────
        cv2.putText(bg, "METRICS", (10, y), FONT, 0.38, (120, 120, 120), 1)
        y += 16
        metrics = [
            ("Lap time",     f"{ds.lap_time:.1f} s"),
            ("Aligned",      f"{ds.aligned:.2f}"),
            ("Approach rate",f"{ds.approach_rate:.0f} px/s"),
            ("Pipeline",     f"{ds.pipeline_ms:.1f} ms"),
            ("Recoveries",   str(ds.recovery_events)),
        ]
        for label, val in metrics:
            cv2.putText(bg, label, (12, y), FONT, 0.36, (160, 160, 160), 1)
            cv2.putText(bg, val,   (190, y), FONT, 0.36, (220, 220, 220), 1)
            y += 18
        y += 6

        # ── Action history bars ───────────────────────────────────────────────
        cv2.putText(bg, "CONTROLS (last 60 frames)", (10, y),
                    FONT, 0.36, (120, 120, 120), 1)
        y += 14

        bar_configs = [
            ("ROLL",     ds.roll_hist,     (-1, 1),  (0, 200, 255)),
            ("PITCH",    ds.pitch_hist,    (-1, 1),  (0, 255, 150)),
            ("THROTTLE", ds.throttle_hist, (0, 1),   (0, 140, 255)),
        ]
        hist_bar_h = 28
        hist_w     = DASH_W - 20
        n_hist     = 60
        for name, hist, (lo, hi), col in bar_configs:
            # label
            cv2.putText(bg, name, (10, y + 10), FONT, 0.32, (140, 140, 140), 1)
            # background
            cv2.rectangle(bg, (10, y + 14), (10 + hist_w, y + 14 + hist_bar_h),
                          (35, 35, 35), -1)
            # draw bars
            if hist:
                bar_unit = hist_w / n_hist
                for i, v in enumerate(hist[-n_hist:]):
                    norm = (v - lo) / (hi - lo)  # [0,1]
                    bh = int(norm * hist_bar_h)
                    bx3 = 10 + int(i * bar_unit)
                    bw3 = max(int(bar_unit), 1)
                    cv2.rectangle(bg,
                                  (bx3, y + 14 + hist_bar_h - bh),
                                  (bx3 + bw3, y + 14 + hist_bar_h),
                                  col, -1)
            # zero line (only for signed axes)
            if lo < 0 < hi:
                zero_y = y + 14 + hist_bar_h // 2
                cv2.line(bg, (10, zero_y), (10 + hist_w, zero_y),
                         (70, 70, 70), 1)
            # current value
            cur = hist[-1] if hist else 0.0
            cv2.putText(bg, f"{cur:+.2f}" if lo < 0 else f"{cur:.2f}",
                        (10 + hist_w + 4, y + 14 + hist_bar_h - 4),
                        FONT, 0.32, col, 1)
            y += hist_bar_h + 18

        # ── Hotkey reminder ───────────────────────────────────────────────────
        cv2.putText(bg, "Q=quit  R=restart", (10, DASH_H - 12),
                    FONT, 0.32, (70, 70, 70), 1)

        canvas[0:DASH_H, FPV_W:FPV_W + DASH_W] = bg

    # ── Completion overlay ────────────────────────────────────────────────────
    def _draw_completion_overlay(self, canvas: np.ndarray, ds: DemoState) -> None:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (FPV_W // 4, WIN_H // 3),
                      (FPV_W * 3 // 4, WIN_H * 2 // 3), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        lines = [
            ("COURSE COMPLETE", (0, 220, 100), 0.9, 2),
            (f"{ds.gate_index}/{ds.total_gates} gates  {ds.lap_time:.2f}s", (200, 200, 200), 0.55, 1),
            (f"recoveries: {ds.recovery_events}", (160, 160, 160), 0.42, 1),
            ("R = restart", (100, 100, 100), 0.38, 1),
        ]
        for i, (txt, col, scale, thick) in enumerate(lines):
            tw, th = cv2.getTextSize(txt, FONT_BOLD, scale, thick)[0]
            tx = (WIN_W - tw) // 2
            ty = WIN_H // 3 + 36 + i * int(th * 2.2)
            cv2.putText(canvas, txt, (tx, ty), FONT_BOLD, scale, col, thick, cv2.LINE_AA)

    def close(self) -> None:
        cv2.destroyAllWindows()


# ── Demo runner ───────────────────────────────────────────────────────────────
class DemoRunner:
    def __init__(self, config: Config, vis: DemoVisualizer,
                 seed: int = 42, display_fps: int = 20):
        self._cfg = config
        self._vis = vis
        self._seed = seed
        self._frame_interval = 1.0 / display_fps

        # Build adapter
        from aigrandprix.adapters.mock import MockSimAdapter
        self._adapter = MockSimAdapter(config.sim)

        # Build lobes
        self._vision    = VisionLobe(config.vision,
                                      budget_ms=config.lobes.vision.budget_ms)
        self._stability = StabilityLobe(config.stability,
                                         budget_ms=config.lobes.stability.budget_ms)
        self._progress  = ProgressLobe(config.progress,
                                        budget_ms=config.lobes.progress.budget_ms)
        self._recovery  = RecoveryLobe(config.recovery,
                                        budget_ms=config.lobes.recovery.budget_ms)
        self._risk      = RiskLobe(config.risk,
                                    budget_ms=config.lobes.risk.budget_ms)
        self._brain     = FusionBrain(config.state_machine)
        self._ctrl      = Controller(config.controller)

    def _reset(self) -> Observation:
        self._vision.reset()
        self._stability.reset()
        self._progress.reset()
        self._recovery.reset()
        self._risk.reset()
        self._brain.reset()
        self._ctrl.reset_all_integrals()
        return self._adapter.reset(seed=self._seed)

    def run_forever(self) -> None:
        ds = DemoState(total_gates=self._cfg.sim.total_gates)
        obs = self._reset()
        state = DroneState.SEARCH
        recovery_events = 0
        last_draw = time.perf_counter()

        while True:
            if not ds.done:
                t0 = time.perf_counter()

                # Pipeline
                vision_r    = self._vision(obs)
                stability_r = self._stability(obs)
                self._recovery.update_dt(obs.dt)
                progress_r  = self._progress(obs, vision_r, state,
                                              self._brain.time_in_state)
                recovery_r  = self._recovery(vision_r, state)
                risk_r      = self._risk(stability_r, progress_r,
                                          recovery_r, obs.t)
                prev_state  = state
                state, target = self._brain(
                    vision_r, stability_r, progress_r, recovery_r, risk_r,
                    sim_t=obs.t,
                )
                reset_int = self._brain.integral_reset_requested
                action = self._ctrl(
                    target, obs, obs.dt, state,
                    push_level=risk_r.push_level,
                    reset_integral=reset_int,
                )
                pipeline_ms = (time.perf_counter() - t0) * 1000.0

                if (prev_state != DroneState.RECOVER
                        and state == DroneState.RECOVER):
                    recovery_events += 1

                # Update demo state
                ds.state_name    = state.name
                ds.gate_index    = progress_r.gate_index
                ds.lap_time      = obs.t
                ds.push_level    = risk_r.push_level
                ds.risk_score    = risk_r.risk_score
                ds.aligned       = progress_r.aligned_score
                ds.approach_rate = progress_r.approach_rate
                ds.pipeline_ms   = pipeline_ms
                ds.in_recovery   = recovery_r.in_recovery
                ds.recovery_events = recovery_events
                ds.action        = action
                ds.fpv_frame     = obs.image
                ds.bbox          = vision_r.bbox
                ds.roll_hist.append(action.roll)
                ds.pitch_hist.append(action.pitch)
                ds.throttle_hist.append(action.throttle)

                # Step adapter
                obs, info = self._adapter.step(action)
                if info["done"]:
                    ds.gates_passed = info["gate_index"]
                    ds.lap_time     = info["lap_time"]
                    ds.done         = True

            # Render at display_fps
            now = time.perf_counter()
            if now - last_draw >= self._frame_interval:
                key = self._vis.render(ds)
                last_draw = now

                if key == ord('q'):
                    break
                if key == ord('r'):
                    ds = DemoState(total_gates=self._cfg.sim.total_gates)
                    obs = self._reset()
                    state = DroneState.SEARCH
                    recovery_events = 0
            else:
                # yield CPU briefly when display isn't needed
                time.sleep(0.001)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="AI Grand Prix live demo")
    parser.add_argument("--config", nargs="*", default=None,
                        help="YAML config files (stacked)")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--fps",   type=int, default=20,
                        help="Display FPS (pipeline runs at full speed)")
    args = parser.parse_args()

    cfg = load_config(*args.config) if args.config else default_config()

    vis    = DemoVisualizer(total_gates=cfg.sim.total_gates)
    runner = DemoRunner(cfg, vis, seed=args.seed, display_fps=args.fps)

    try:
        runner.run_forever()
    finally:
        vis.close()


if __name__ == "__main__":
    main()
