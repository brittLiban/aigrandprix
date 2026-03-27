"""Batch config sweep — resumable, manifest-first.

Usage:
    python scripts/sweep.py --config configs/base.yaml \\
        --sweep "state_machine.approach_aligned_min=[0.6,0.7,0.8]" \\
        --seeds 3 --episodes 5 --out sweep_results/my_sweep

The sweep script:
1. Writes a manifest JSON before any episodes run (resumable).
2. Skips completed runs by checking for existing log files.
3. Collects metrics from each run's footer line.
4. Writes a CSV sorted by median_pipeline_ms (best lap time proxy until
   we have real timing).
"""
from __future__ import annotations

import argparse
import ast
import csv
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))

from aigrandprix.config import default_config, load_config
from aigrandprix.runner import PipelineRunner


def _parse_sweep_arg(sweep_str: str) -> tuple[str, list]:
    """Parse 'key.path=[v1,v2,v3]' → ('key.path', [v1, v2, v3])."""
    key, _, values_str = sweep_str.partition("=")
    values = ast.literal_eval(values_str)
    if not isinstance(values, list):
        values = [values]
    return key.strip(), values


def _set_nested(d: Any, key_path: str, value: Any) -> None:
    """Set a nested attribute like 'state_machine.approach_aligned_min' on a dataclass."""
    parts = key_path.split(".")
    obj = d
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _read_footer(log_path: Path) -> dict | None:
    """Read the footer line from a JSONL log file."""
    try:
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        for line in reversed(lines):
            try:
                row = json.loads(line)
                if row.get("type") == "footer":
                    return row
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Batch config sweep")
    parser.add_argument("--config", nargs="*", default=None)
    parser.add_argument("--sweep", nargs="*", default=[],
                        help="Sweep args: 'key.path=[v1,v2]'")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds per config combo")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Episodes per (config, seed) — usually 1")
    parser.add_argument("--out", default="sweep_results/sweep",
                        help="Output directory prefix")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse sweep parameters
    sweep_params: dict[str, list] = {}
    for s in (args.sweep or []):
        key, values = _parse_sweep_arg(s)
        sweep_params[key] = values

    # Generate all config combinations
    if sweep_params:
        keys = list(sweep_params.keys())
        value_lists = [sweep_params[k] for k in keys]
        combos = list(itertools.product(*value_lists))
    else:
        keys = []
        combos = [()]

    seeds = list(range(args.seeds))
    all_runs = []
    for combo in combos:
        for seed in seeds:
            run_info = {"combo": dict(zip(keys, combo)), "seed": seed}
            all_runs.append(run_info)

    # Write manifest before any run
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"total_runs": len(all_runs), "runs": all_runs}, f, indent=2)
    print(f"Manifest written: {manifest_path}  ({len(all_runs)} runs)")

    # Load base config
    if args.config:
        base_cfg = load_config(*args.config)
    else:
        base_cfg = default_config()

    # Run sweep
    results = []
    for i, run_info in enumerate(all_runs):
        combo = run_info["combo"]
        seed = run_info["seed"]

        # Build run_id from combo + seed
        combo_str = "_".join(f"{k.split('.')[-1]}={v}" for k, v in combo.items())
        run_id = f"sweep_{combo_str}_seed{seed}" if combo_str else f"sweep_seed{seed}"
        run_id = run_id.replace(".", "").replace(" ", "")

        # Check if already done (resumable)
        cfg = load_config(*args.config) if args.config else default_config()
        for key, val in combo.items():
            _set_nested(cfg, key, val)
        cfg.logging.output_dir = str(out_dir / "logs")

        runner = PipelineRunner(cfg, run_id=run_id)
        expected_log = Path(cfg.logging.output_dir) / f"{run_id}.jsonl"
        if expected_log.exists():
            footer = _read_footer(expected_log)
            if footer:
                print(f"  [{i+1}/{len(all_runs)}] {run_id} — SKIP (already done)")
                row = {"run_id": run_id, "seed": seed, **combo, **footer}
                results.append(row)
                continue

        print(f"  [{i+1}/{len(all_runs)}] {run_id} ...", end=" ", flush=True)
        summary = runner.run(seed=seed)
        print(f"done  lap={summary.get('lap_time', '?')}s  "
              f"completion={summary.get('completion', '?')}")
        row = {"run_id": run_id, "seed": seed, **combo, **summary}
        results.append(row)

    # Write CSV sorted by lap_time (ascending; incomplete runs last)
    if results:
        results.sort(key=lambda r: (
            0 if r.get("completion") else 1,
            r.get("lap_time", 9999),
        ))
        csv_path = out_dir / "results.csv"
        # Flatten time_by_state
        flat_results = []
        for r in results:
            flat = {k: v for k, v in r.items() if k not in ("time_by_state", "type")}
            if "time_by_state" in r and isinstance(r["time_by_state"], dict):
                for state, t in r["time_by_state"].items():
                    flat[f"time_{state}"] = t
            flat_results.append(flat)

        all_keys = list(dict.fromkeys(k for r in flat_results for k in r))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_results)
        print(f"\nResults: {csv_path}  ({len(results)} rows)")


if __name__ == "__main__":
    main()
