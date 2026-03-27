"""Profile exactly where time goes per gate — fixed gate assignment."""
import logging, sys, io
logging.disable(logging.CRITICAL)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from collections import defaultdict
from aigrandprix.config import load_config
from aigrandprix.adapters.mock import MockSimAdapter
from aigrandprix.lobes.vision import VisionLobe
from aigrandprix.lobes.stability import StabilityLobe
from aigrandprix.lobes.progress import ProgressLobe
from aigrandprix.lobes.recovery import RecoveryLobe
from aigrandprix.lobes.risk import RiskLobe
from aigrandprix.brain.fusion import FusionBrain
from aigrandprix.controller.pid import Controller
from aigrandprix.brain.states import DroneState

cfg = load_config("configs/base.yaml", "configs/aggressive.yaml")

adapter  = MockSimAdapter(cfg.sim)
vision   = VisionLobe(cfg.vision, budget_ms=cfg.lobes.vision.budget_ms)
stability= StabilityLobe(cfg.stability, budget_ms=cfg.lobes.stability.budget_ms)
progress = ProgressLobe(cfg.progress, budget_ms=cfg.lobes.progress.budget_ms)
recovery = RecoveryLobe(cfg.recovery, budget_ms=cfg.lobes.recovery.budget_ms)
risk     = RiskLobe(cfg.risk, budget_ms=cfg.lobes.risk.budget_ms)
brain    = FusionBrain(cfg.state_machine)
ctrl     = Controller(cfg.controller)

obs = adapter.reset(seed=42)
state = DroneState.SEARCH

# Track gate correctly: a gate's frames are AFTER the previous pass until THIS pass
gate_state_frames = defaultdict(lambda: defaultdict(int))
current_gate = 0

while True:
    vision_r   = vision(obs)
    stability_r= stability(obs)
    recovery.update_dt(obs.dt)
    progress_r = progress(obs, vision_r, state, brain.time_in_state)
    recovery_r = recovery(vision_r, state)
    risk_r     = risk(stability_r, progress_r, recovery_r, obs.t)
    state, target = brain(vision_r, stability_r, progress_r, recovery_r, risk_r, sim_t=obs.t)
    action = ctrl(target, obs, obs.dt, state,
                  push_level=risk_r.push_level,
                  reset_integral=brain.integral_reset_requested)

    gate_state_frames[current_gate][state.name] += 1

    obs, info = adapter.step(action)
    if info["gate_passed"]:
        current_gate = info["gate_index"]   # move to NEXT gate index
    if info["done"]:
        break

fps = cfg.sim.fps
print(f"{'Gate':>5} {'SEARCH':>8} {'TRACK':>8} {'APPROACH':>10} {'COMMIT':>8} {'frames':>8} {'time':>8}")
print("-" * 65)
totals = defaultdict(int)
for g in range(cfg.sim.total_gates):
    d = gate_state_frames[g]
    total = sum(d.values())
    t = total / fps
    for s in ['SEARCH','TRACK','APPROACH','COMMIT','RECOVER']:
        totals[s] += d.get(s, 0)
    totals['TOTAL'] += total
    print(f"  {g:>3}  {d.get('SEARCH',0):>8} {d.get('TRACK',0):>8} "
          f"{d.get('APPROACH',0):>10} {d.get('COMMIT',0):>8} {total:>8} {t:>7.2f}s")
print("-" * 65)
total_all = totals['TOTAL']
print(f"  TOT  {totals['SEARCH']:>8} {totals['TRACK']:>8} "
      f"{totals['APPROACH']:>10} {totals['COMMIT']:>8} {total_all:>8} {total_all/fps:>7.2f}s")
print(f"\nPct   {100*totals['SEARCH']/total_all:>7.1f}%"
      f"  {100*totals['TRACK']/total_all:>7.1f}%"
      f"  {100*totals['APPROACH']/total_all:>9.1f}%"
      f"  {100*totals['COMMIT']/total_all:>7.1f}%")

max_throttle = 1.0
APPROACH_GAIN, DECEL, fps_ = 0.35, 0.05, fps
max_growth = (max_throttle * APPROACH_GAIN - DECEL) / fps_
min_steps = (0.85 - 0.08) / max_growth
print(f"\nPhysics ceiling: {min_steps:.0f} steps/gate = {min_steps/fps_:.2f}s/gate = {cfg.sim.total_gates*min_steps/fps_:.1f}s total")
print(f"Current:         {total_all/cfg.sim.total_gates:.0f} steps/gate = "
      f"{total_all/fps/cfg.sim.total_gates:.2f}s/gate = {total_all/fps:.1f}s total")
margin = (total_all/cfg.sim.total_gates - min_steps) / min_steps * 100
print(f"Margin above floor: {margin:.1f}%")
