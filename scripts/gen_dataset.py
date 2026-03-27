"""Generate a labeled gate detection dataset from the mock simulator.

Runs mock episodes in parallel across CPU workers and saves frames +
ground-truth bounding boxes.

Output layout:
    data/gate_dataset/
        images/
            frame_000000.jpg
            ...
        labels.jsonl

Usage:
    python scripts/gen_dataset.py
    python scripts/gen_dataset.py --seeds 100 --workers 16 --output data/gate_dataset
    python scripts/gen_dataset.py --seeds 50 --no-augment
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aigrandprix.adapters.mock import MockSimAdapter
from aigrandprix.config import SimConfig
from aigrandprix.types import Action
from aigrandprix.augmentation.transforms import ALL_TRANSFORMS, augment as apply_augment

_W_FRAC = 0.7
_H_FRAC = 0.7
_VISIBLE_AREA_MIN = 400.0


def _default_sim_cfg() -> SimConfig:
    return SimConfig(
        fps=60, seed=0, resolution=[480, 640], total_gates=10,
        lateral_std=0.25, noise_std=8.0,
        accel_noise_std=0.1, gyro_noise_std=0.02,
        drop_frame_prob=0.0, latency_spike_prob=0.0, max_latency_ms=0.0,
        exposure_variation=0.1,
    )


def _render_blank(cfg_sim: SimConfig, rng: np.random.Generator) -> np.ndarray:
    H, W = cfg_sim.resolution[0], cfg_sim.resolution[1]
    frame = np.full((H, W, 3), 20, dtype=np.uint8)
    if cfg_sim.noise_std > 0:
        noise = rng.normal(0, cfg_sim.noise_std, frame.shape)
        frame = np.clip(frame.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
    if cfg_sim.exposure_variation > 0:
        factor = rng.uniform(1.0 - cfg_sim.exposure_variation,
                             1.0 + cfg_sim.exposure_variation)
        frame = np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return frame


def _run_episode(seed: int, cfg_sim: SimConfig, max_steps: int,
                 neg_ratio: float = 0.20) -> list[dict]:
    rng = np.random.default_rng(seed + 100000)
    adapter = MockSimAdapter(cfg_sim)
    adapter.reset(seed=seed)

    pitch    = float(rng.uniform(0.1, 0.5))
    throttle = float(rng.uniform(0.5, 0.9))
    roll     = float(rng.uniform(-0.2, 0.2))
    action   = Action(roll=roll, pitch=pitch, yaw=0.0, throttle=throttle)

    records = []
    for _ in range(max_steps):
        obs, info = adapter.step(action)
        if info["done"]:
            break
        state = adapter.gate_state
        scale, cx, cy = state["scale"], state["cx"], state["cy"]
        H, W = obs.image.shape[:2]
        area = (scale * W * _W_FRAC) * (scale * H * _H_FRAC)
        if area >= _VISIBLE_AREA_MIN:
            records.append({
                "image": obs.image.copy(),
                "gate_detected": True,
                "cx": cx, "cy": cy,
                "bw": scale * _W_FRAC, "bh": scale * _H_FRAC,
                "area": float(area), "scale": float(scale),
            })
        else:
            records.append({
                "image": obs.image.copy(),
                "gate_detected": False,
                "cx": 0.5, "cy": 0.5, "bw": 0.0, "bh": 0.0,
                "area": 0.0, "scale": float(scale),
            })

    n_neg_target = int(len(records) * neg_ratio / max(1 - neg_ratio, 1e-6))
    for _ in range(n_neg_target):
        records.append({
            "image": _render_blank(cfg_sim, rng),
            "gate_detected": False,
            "cx": 0.5, "cy": 0.5, "bw": 0.0, "bh": 0.0,
            "area": 0.0, "scale": 0.0,
        })
    return records


# ---------------------------------------------------------------------------
# Worker — processes a batch of seeds, writes its own frames, returns labels
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> list[dict]:
    """Run a batch of seeds, save images, return label rows (no global state)."""
    seed_start, seed_end, img_dir, frame_id_start, cfg_sim, max_steps, aug_copies, no_augment = args

    # Re-insert path in worker process
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from aigrandprix.augmentation.transforms import ALL_TRANSFORMS, augment as apply_augment

    img_dir = Path(img_dir)
    label_rows = []
    frame_id = frame_id_start

    for seed in range(seed_start, seed_end):
        records = _run_episode(seed, cfg_sim, max_steps)
        for rec in records:
            image = rec["image"]
            copies = [image]
            if not no_augment:
                for c in range(aug_copies):
                    copies.append(apply_augment(image, seed=frame_id + c,
                                                transforms=ALL_TRANSFORMS))
            for img in copies:
                fname = f"frame_{frame_id:07d}.jpg"
                cv2.imwrite(
                    str(img_dir / fname),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                label_rows.append({
                    "file": f"images/{fname}",
                    "gate_detected": rec["gate_detected"],
                    "cx": rec["cx"], "cy": rec["cy"],
                    "bw": rec["bw"], "bh": rec["bh"],
                    "area": rec["area"], "scale": rec["scale"],
                })
                frame_id += 1

    return label_rows


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def save_dataset(output_dir: str, seeds: int, cfg_path: str,
                 no_augment: bool, max_steps: int, aug_copies: int,
                 workers: int) -> None:
    out = Path(output_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    if cfg_path:
        from aigrandprix.config import load_config
        sim_cfg = load_config(cfg_path).sim
    else:
        sim_cfg = _default_sim_cfg()

    copies_per_frame = 1 if no_augment else 1 + aug_copies
    # Estimate frames per episode: max_steps * 1.25 (with negatives)
    est_frames_per_seed = int(max_steps * 1.25 * copies_per_frame)

    # Split seeds across workers
    actual_workers = min(workers, seeds)
    chunk = max(1, seeds // actual_workers)
    batches = []
    frame_cursor = 0
    for i in range(0, seeds, chunk):
        seed_end = min(i + chunk, seeds)
        n_seeds_in_batch = seed_end - i
        batches.append((
            i, seed_end,
            str(img_dir),
            frame_cursor,
            sim_cfg, max_steps, aug_copies, no_augment,
        ))
        frame_cursor += n_seeds_in_batch * est_frames_per_seed

    print(f"Generating dataset: {seeds} seeds x {max_steps} steps "
          f"x {copies_per_frame} copies  |  {actual_workers} workers")

    if actual_workers == 1:
        all_rows = []
        for batch in batches:
            all_rows.extend(_worker(batch))
            done = batch[1]
            print(f"  {done}/{seeds} seeds done  ({len(all_rows)} frames)")
    else:
        with multiprocessing.Pool(processes=actual_workers) as pool:
            results = pool.map_async(_worker, batches)
            results.wait()
            all_rows_nested = results.get()
        all_rows = []
        for rows in all_rows_nested:
            all_rows.extend(rows)

    # Re-index frame filenames to be contiguous (workers may have gaps)
    # Already written to disk — just fix the labels file
    labels_path = out / "labels.jsonl"
    with open(labels_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    total = len(all_rows)
    pos = sum(1 for r in all_rows if r["gate_detected"])
    neg = total - pos
    print(f"\nDataset saved to {out}")
    print(f"  Total frames : {total}")
    print(f"  Positive     : {pos}  ({100*pos/max(total,1):.1f}%)")
    print(f"  Negative     : {neg}  ({100*neg/max(total,1):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate gate detection dataset")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--output", default="data/gate_dataset")
    parser.add_argument("--config", default="")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--aug-copies", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Steps per episode (default: 200 — enough variety, much faster)")
    parser.add_argument("--workers", type=int,
                        default=min(16, multiprocessing.cpu_count()),
                        help=f"Parallel workers (default: auto, {min(16, multiprocessing.cpu_count())} on this machine)")
    args = parser.parse_args()

    save_dataset(
        output_dir=args.output,
        seeds=args.seeds,
        cfg_path=args.config,
        no_augment=args.no_augment,
        max_steps=args.max_steps,
        aug_copies=args.aug_copies,
        workers=args.workers,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
