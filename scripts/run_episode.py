"""CLI entry point for a single episode.

Usage:
    python scripts/run_episode.py [--config configs/base.yaml] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from aigrandprix.config import default_config, load_config
from aigrandprix.runner import PipelineRunner


def main():
    parser = argparse.ArgumentParser(description="Run one AI Grand Prix episode")
    parser.add_argument("--config", nargs="*", default=None,
                        help="YAML config files (base first, then overrides)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--track-id", default="mock")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(*args.config)
    else:
        cfg = default_config()

    runner = PipelineRunner(cfg)
    print(f"Run ID: {runner.run_id}")
    summary = runner.run(seed=args.seed, track_id=args.track_id)

    print("\n--- Episode Summary ---")
    for k, v in summary.items():
        if k != "time_by_state":
            print(f"  {k}: {v}")
    if "time_by_state" in summary:
        print("  time_by_state:")
        for state, t in summary["time_by_state"].items():
            print(f"    {state}: {t:.3f}s")

    if runner.log_path:
        print(f"\nLog: {runner.log_path}")


if __name__ == "__main__":
    main()
