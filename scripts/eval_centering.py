"""Evaluate how accurately the drone centers on gates at pass time.

Runs N random seeds, records vision cx/cy vs true gate cx/cy at each gate pass,
and reports alignment error statistics.

Usage:
    python scripts/eval_centering.py --seeds 50
    python scripts/eval_centering.py --seeds 50 --config configs/base.yaml configs/aggressive.yaml configs/hard.yaml
"""
from __future__ import annotations
import argparse, sys, os, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aigrandprix.config import default_config, load_config
from aigrandprix.runner import PipelineRunner
import logging
logging.disable(logging.CRITICAL)


def eval_seed(seed: int, config_paths: list[str]) -> list[dict]:
    """Run one episode, return per-gate pass records."""
    from aigrandprix.config import default_config, load_config
    from aigrandprix.adapters.mock import MockSimAdapter
    from aigrandprix.brain.states import DroneState
    from aigrandprix.lobes.vision import VisionLobe
    from aigrandprix.lobes.vision_ml import MLVisionLobe
    from aigrandprix.lobes.stability import StabilityLobe
    from aigrandprix.lobes.progress import ProgressLobe
    from aigrandprix.lobes.recovery import RecoveryLobe
    from aigrandprix.lobes.risk import RiskLobe
    from aigrandprix.brain.fusion import FusionBrain
    from aigrandprix.controller.pid import Controller

    cfg = default_config() if not config_paths else load_config(*config_paths)
    cfg.logging.output_dir = ""

    adapter = MockSimAdapter(cfg.sim)
    vision = (MLVisionLobe(cfg.vision, budget_ms=cfg.lobes.vision.budget_ms)
              if cfg.vision.backend == "ml"
              else VisionLobe(cfg.vision, budget_ms=cfg.lobes.vision.budget_ms))
    stability = StabilityLobe(cfg.stability, budget_ms=cfg.lobes.stability.budget_ms)
    progress  = ProgressLobe(cfg.progress,   budget_ms=cfg.lobes.progress.budget_ms)
    recovery  = RecoveryLobe(cfg.recovery,   budget_ms=cfg.lobes.recovery.budget_ms)
    risk      = RiskLobe(cfg.risk,           budget_ms=cfg.lobes.risk.budget_ms)
    brain     = FusionBrain(cfg.state_machine)
    ctrl      = Controller(cfg.controller)

    for lobe in [vision, stability, progress, recovery, risk]:
        lobe.reset()
    brain.reset(); ctrl.reset_all_integrals()

    obs = adapter.reset(seed=seed)
    state = DroneState.SEARCH
    records = []
    prev_gate_index = 0

    while True:
        vision_r    = vision(obs)
        stability_r = stability(obs)
        recovery.update_dt(obs.dt)
        progress_r  = progress(obs, vision_r, state, brain.time_in_state)
        recovery_r  = recovery(vision_r, state)
        risk_r      = risk(stability_r, progress_r, recovery_r, obs.t)
        state, target = brain(vision_r, stability_r, progress_r, recovery_r, risk_r, sim_t=obs.t)
        action = ctrl(target, obs, obs.dt, state,
                      push_level=risk_r.push_level,
                      reset_integral=brain.integral_reset_requested)

        # Snapshot true gate position BEFORE step (sim hasn't reset yet)
        true_cx_before = adapter.gate_state["cx"]
        true_cy_before = adapter.gate_state["cy"]

        obs, info = adapter.step(action)

        # Gate just passed — record centering using pre-step true position
        if info["gate_index"] > prev_gate_index or info["done"]:
            true_cx = true_cx_before
            true_cy = true_cy_before
            records.append({
                "seed": seed,
                "gate": prev_gate_index,
                "true_cx": true_cx,
                "true_cy": true_cy,
                "pred_cx": vision_r.cx,
                "pred_cy": vision_r.cy,
                "dx": abs(vision_r.cx - true_cx),
                "dy": abs(vision_r.cy - true_cy),
                "detected": vision_r.gate_detected,
                "conf_ema": vision_r.confidence_ema,
            })
            prev_gate_index = info["gate_index"]

        if info["done"]:
            break

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30,
                        help="Number of random seeds to evaluate (default: 30)")
    parser.add_argument("--seed-start", type=int, default=5000,
                        help="Starting seed (default: 5000, well outside training range)")
    parser.add_argument("--config", nargs="*", default=None)
    args = parser.parse_args()

    config_paths = args.config or []
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print(f"Evaluating {args.seeds} seeds ({seeds[0]}–{seeds[-1]}) "
          f"{'with ' + str(config_paths) if config_paths else 'default config'}...")
    print()

    all_records = []
    for i, seed in enumerate(seeds):
        recs = eval_seed(seed, config_paths)
        all_records.extend(recs)
        completed = i + 1
        if completed % 10 == 0 or completed == len(seeds):
            print(f"  {completed}/{args.seeds} seeds done  "
                  f"({len(all_records)} gate passes so far)...", flush=True)

    if not all_records:
        print("No records — did all episodes fail?")
        return

    dx_vals  = [r["dx"]  for r in all_records]
    dy_vals  = [r["dy"]  for r in all_records]
    det_rate = sum(1 for r in all_records if r["detected"]) / len(all_records)

    # Centering error buckets
    perfect   = sum(1 for r in all_records if r["dx"] < 0.05 and r["dy"] < 0.05)
    good      = sum(1 for r in all_records if r["dx"] < 0.10 and r["dy"] < 0.10)
    ok        = sum(1 for r in all_records if r["dx"] < 0.20 and r["dy"] < 0.20)
    total     = len(all_records)

    print(f"\n{'-'*55}")
    print(f"  Gate passes evaluated : {total}  ({args.seeds} seeds)")
    print(f"  Detection rate at pass: {det_rate*100:.1f}%")
    print()
    print(f"  Centering error (|pred - true|, normalized 0–1):")
    print(f"    cx  mean={statistics.mean(dx_vals):.3f}  "
          f"median={statistics.median(dx_vals):.3f}  "
          f"p90={sorted(dx_vals)[int(len(dx_vals)*0.9)]:.3f}")
    print(f"    cy  mean={statistics.mean(dy_vals):.3f}  "
          f"median={statistics.median(dy_vals):.3f}  "
          f"p90={sorted(dy_vals)[int(len(dy_vals)*0.9)]:.3f}")
    print()
    print(f"  Pass quality buckets (both cx AND cy within threshold):")
    print(f"    Perfect  (<0.05) : {perfect:>4}/{total}  ({100*perfect/total:.1f}%)")
    print(f"    Good     (<0.10) : {good:>4}/{total}  ({100*good/total:.1f}%)")
    print(f"    OK       (<0.20) : {ok:>4}/{total}  ({100*ok/total:.1f}%)")
    print(f"    Off-center (>=0.20): {total-ok:>4}/{total}  ({100*(total-ok)/total:.1f}%)")
    print(f"{'-'*55}")

    # Worst passes
    worst = sorted(all_records, key=lambda r: r["dx"] + r["dy"], reverse=True)[:5]
    print(f"\n  5 worst passes:")
    print(f"  {'seed':>6}  {'gate':>4}  {'dx':>6}  {'dy':>6}  {'det':>5}  {'conf':>6}")
    for r in worst:
        print(f"  {r['seed']:>6}  {r['gate']:>4}  {r['dx']:>6.3f}  "
              f"{r['dy']:>6.3f}  {'Y' if r['detected'] else 'N':>5}  "
              f"{r['conf_ema']:>6.3f}")


if __name__ == "__main__":
    main()
