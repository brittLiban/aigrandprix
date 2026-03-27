import statistics, logging
logging.disable(logging.CRITICAL)
from aigrandprix.config import load_config, default_config
from aigrandprix.runner import PipelineRunner

base_cfg = default_config()
agg_cfg  = load_config("configs/base.yaml", "configs/aggressive.yaml")

SEEDS = list(range(10))

for label, cfg in [("BASE      ", base_cfg), ("IMPROVED  ", agg_cfg)]:
    laps, rec, done = [], [], []
    for seed in SEEDS:
        r = PipelineRunner(cfg)
        s = r.run(seed=seed)
        done.append(s["completion"])
        laps.append(s["lap_time"])
        rec.append(s["recovery_events"])
    lc = [l for l, c in zip(laps, done) if c]
    print(f"{label}  complete={sum(done)}/{len(SEEDS)}"
          f"  median={statistics.median(lc):.1f}s"
          f"  min={min(lc):.1f}s  max={max(lc):.1f}s"
          f"  recoveries={sum(rec)}")
