"""Parallel 100-seed validation: HSV vs ML under normal and stress conditions."""
import statistics, logging, sys, io, multiprocessing as mp
logging.disable(logging.CRITICAL)

def run_episode(args):
    """Worker: run one episode, return (label, lap_time, completion, recovery_events)."""
    import sys, os, logging
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    logging.disable(logging.CRITICAL)
    label, config_paths, seed = args
    from aigrandprix.config import default_config, load_config
    from aigrandprix.runner import PipelineRunner
    cfg = default_config() if not config_paths else load_config(*config_paths)
    cfg.logging.flush_every_n_steps = 999999  # suppress frequent flushes
    cfg.logging.output_dir = ""               # empty = no log file written
    s = PipelineRunner(cfg).run(seed=seed)
    return label, s["lap_time"], s["completion"], s["recovery_events"]


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    SEEDS = list(range(100))

    configs = [
        ("BASE",         []),
        ("HSV+AGG",      ["configs/base.yaml", "configs/aggressive.yaml"]),
        ("HSV+STRESS",   ["configs/base.yaml", "configs/aggressive.yaml", "configs/stress.yaml"]),
        ("ML+AGG",       ["configs/base.yaml", "configs/aggressive.yaml", "configs/ml_vision.yaml"]),
        ("ML+STRESS",    ["configs/base.yaml", "configs/aggressive.yaml", "configs/stress.yaml", "configs/ml_vision.yaml"]),
        ("HSV+HARD",     ["configs/base.yaml", "configs/aggressive.yaml", "configs/hard.yaml"]),
        ("ML+HARD",      ["configs/base.yaml", "configs/aggressive.yaml", "configs/ml_vision.yaml", "configs/hard.yaml"]),
    ]

    tasks = [
        (label, paths, seed)
        for label, paths in configs
        for seed in SEEDS
    ]

    # ML workers load CUDA torch — cap to avoid paging file exhaustion
    has_ml = any("ml_vision" in str(paths) for _, paths in configs)
    n_workers = 4 if has_ml else max(1, mp.cpu_count() - 1)
    print(f"Running {len(tasks)} episodes across {n_workers} workers...", flush=True)

    results: dict[str, list] = {label: [] for label, _ in configs}
    with mp.Pool(processes=n_workers) as pool:
        for label, lap, done, rec in pool.imap_unordered(run_episode, tasks, chunksize=4):
            results[label].append((lap, done, rec))
            completed = sum(len(v) for v in results.values())
            if completed % 50 == 0:
                print(f"  {completed}/{len(tasks)} done...", flush=True)

    print(f"\n{'Config':<12} {'Done':>8} {'Median':>8} {'Min':>7} {'Max':>7} {'p90':>8} {'Rec':>5}")
    print("-" * 65)
    for label, _ in configs:
        rows = results[label]
        lc = sorted(l for l, c, _ in rows if c)
        rec_total = sum(r for _, _, r in rows)
        n = len(lc)
        if n == 0:
            print(f"{label:<12}   0/{len(SEEDS)}  --")
            continue
        median = statistics.median(lc)
        p90    = lc[int(n * 0.9)]
        print(f"{label:<12} {n:>4}/{len(SEEDS)}"
              f"  {median:>7.1f}s"
              f"  {min(lc):>6.1f}s"
              f"  {max(lc):>6.1f}s"
              f"  {p90:>7.1f}s"
              f"  {rec_total:>4}")


if __name__ == "__main__":
    main()
