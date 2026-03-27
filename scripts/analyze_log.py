"""Parse a JSONL run log and print a summary + per-gate breakdown.

Usage:
    python scripts/analyze_log.py logs/run_<id>.jsonl [--gates] [--states]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # truncated final line
    return rows


def main():
    parser = argparse.ArgumentParser(description="Analyze a run log")
    parser.add_argument("log", help="Path to JSONL log file")
    parser.add_argument("--gates", action="store_true",
                        help="Print per-gate breakdown")
    parser.add_argument("--states", action="store_true",
                        help="Print state transition log")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows = load_jsonl(path)
    header = next((r for r in rows if r["type"] == "header"), {})
    footer = next((r for r in rows if r["type"] == "footer"), {})
    steps = [r for r in rows if r["type"] == "step"]

    print(f"\n=== Run: {header.get('run_id', '?')} ===")
    print(f"  seed={header.get('seed', '?')}  "
          f"config_hash={header.get('config_hash', '?')}  "
          f"git={header.get('git_commit', '?')}")

    if footer:
        print("\n--- Summary ---")
        for k, v in footer.items():
            if k in ("type", "time_by_state"):
                continue
            print(f"  {k}: {v}")
        if "time_by_state" in footer:
            print("  time_by_state:")
            for state, t in sorted(footer["time_by_state"].items()):
                print(f"    {state}: {t:.3f}s")

    if not steps:
        print("\n(no step data)")
        return

    # Pipeline timing
    pipeline_ms = [s["pipeline_ms"] for s in steps]
    pipeline_ms.sort()
    n = len(pipeline_ms)
    print(f"\n--- Pipeline Latency ({n} steps) ---")
    print(f"  median: {pipeline_ms[n//2]:.2f}ms")
    print(f"  p90:    {pipeline_ms[int(n*0.9)]:.2f}ms")
    print(f"  max:    {pipeline_ms[-1]:.2f}ms")

    # Detection rate
    n_detected = sum(1 for s in steps if s["vision"]["detected"])
    print(f"\n--- Vision ---")
    print(f"  gate_detected: {n_detected}/{n} ({100*n_detected/n:.1f}%)")

    # Mode distribution
    mode_counts: dict[str, int] = defaultdict(int)
    for s in steps:
        mode_counts[s["mode"]] += 1
    print("\n--- Mode Distribution ---")
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"  {mode}: {count} ({100*count/n:.1f}%)")

    # Per-gate breakdown
    if args.gates:
        gate_steps: dict[int, list[dict]] = defaultdict(list)
        for s in steps:
            gate_steps[s["gate_index"]].append(s)
        print("\n--- Per-Gate Breakdown ---")
        for gate_idx in sorted(gate_steps.keys()):
            gs = gate_steps[gate_idx]
            if not gs:
                continue
            n_g = len(gs)
            t_start = gs[0]["t"]
            t_end = gs[-1]["t"]
            det = sum(1 for s in gs if s["vision"]["detected"])
            avg_pipe = sum(s["pipeline_ms"] for s in gs) / n_g
            print(f"  Gate {gate_idx}: {n_g} steps  "
                  f"t={t_start:.2f}->{t_end:.2f}s  "
                  f"det={100*det/n_g:.0f}%  "
                  f"pipe={avg_pipe:.1f}ms")

    # State transitions
    if args.states:
        print("\n--- State Transitions ---")
        for s in steps:
            if s.get("state_transition"):
                print(f"  t={s['t']:.3f}  {s['state_transition']}")


if __name__ == "__main__":
    main()
