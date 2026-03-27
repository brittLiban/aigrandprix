"""Render mock simulator frames to a video file for visual sanity checking.

Usage:
    python scripts/visualize_mock.py [--seed 0] [--output /tmp/mock.mp4] [--seconds 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Add repo root to path so we can import aigrandprix without pip install
sys.path.insert(0, str(Path(__file__).parents[1]))

from aigrandprix.adapters.mock import MockSimAdapter
from aigrandprix.config import load_config, default_config
from aigrandprix.types import Action


def main():
    parser = argparse.ArgumentParser(description="Visualize mock sim to video")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="mock_preview.mp4")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps-out", type=int, default=30, help="Output video FPS")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else default_config()
    cfg.sim.seed = args.seed
    cfg.sim.drop_frame_prob = 0.0    # cleaner for visual inspection
    cfg.sim.noise_std = 5.0

    adapter = MockSimAdapter(cfg.sim)
    obs = adapter.reset(seed=args.seed)

    H, W = cfg.sim.resolution[0], cfg.sim.resolution[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, args.fps_out, (W, H))

    # Simple policy: always push forward with slight center-correction
    n_steps = int(args.seconds * cfg.sim.fps)
    gates_passed = 0

    for i in range(n_steps):
        # Naive centering + constant approach
        action = Action(
            roll=0.0,
            pitch=0.3,
            yaw=0.0,
            throttle=0.6,
        )
        obs, info = adapter.step(action)

        if info["gate_passed"]:
            gates_passed += 1
            print(f"  Gate {info['gate_index']} passed at t={obs.t:.2f}s")

        if info["done"]:
            print(f"All gates passed! lap_time={info['lap_time']:.2f}s")
            break

        # Overlay gate index and scale info
        frame_bgr = cv2.cvtColor(obs.image, cv2.COLOR_RGB2BGR)
        gs = adapter.gate_state
        cv2.putText(frame_bgr,
                    f"gate={gs['index']} scale={gs['scale']:.2f} t={obs.t:.2f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        out.write(frame_bgr)

    out.release()
    print(f"Written {args.output}  ({gates_passed} gates passed)")


if __name__ == "__main__":
    main()
