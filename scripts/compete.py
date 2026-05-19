"""Competition entry point for the AI Grand Prix Round 1 Qualifier.

Usage:
    python scripts/compete.py

Connects to the DCL simulator via MAVLink (port 14551) and vision UDP (port 5600).
Runs the full autonomous pipeline and logs results to logs/.

Pre-flight checks run before connecting — fix any reported issues first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aigrandprix.config import load_config


def _preflight(cfg) -> bool:
    """Run sanity checks before touching the simulator. Returns False if fatal."""
    ok = True

    # 1. pymavlink must be installed
    try:
        import pymavlink  # noqa: F401
    except ImportError:
        print("[FATAL] pymavlink not installed — run: pip install pymavlink")
        ok = False

    # 2. ML model must exist when backend is 'ml'
    if cfg.vision.backend == "ml":
        model_path = Path(cfg.vision.model_path)
        if not model_path.exists():
            print(f"[FATAL] ML model not found at: {model_path.resolve()}")
            print("  Options:")
            print("    1. Copy your trained .pt checkpoint to that path, or")
            print("    2. Edit official_r1.yaml and set vision.backend: hsv")
            ok = False
        else:
            try:
                import torch
                ckpt = torch.load(str(model_path), map_location="cpu",
                                  weights_only=True)
                print(f"[OK]    Model loaded: {model_path} "
                      f"({model_path.stat().st_size // 1024} KB)")
            except Exception as e:
                print(f"[FATAL] Model file exists but failed to load: {e}")
                ok = False

    # 3. Report GPU status
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            print(f"[OK]    GPU: {name} ({vram} MB VRAM) — ML inference on CUDA")
        else:
            print("[WARN]  No CUDA GPU detected — ML inference on CPU (slower)")
    except ImportError:
        pass

    # 4. Warn if cy_tilt_offset is 0 with official adapter
    if cfg.adapter.type == "official" and cfg.progress.cy_tilt_offset == 0.0:
        print("[WARN]  progress.cy_tilt_offset is 0.0 — camera is tilted 20° up.")
        print("        Gates may appear below image center; APPROACH transitions")
        print("        may be slow. Set cy_tilt_offset: 0.16 in official_r1.yaml")

    return ok


def main():
    cfg = load_config("configs/base.yaml", "configs/official_r1.yaml")

    print("=" * 60)
    print("AI Grand Prix — Round 1 Qualifier")
    print(f"Adapter : {cfg.adapter.type}")
    print(f"Vision  : {cfg.vision.backend}")
    print(f"MAVLink : {cfg.official.mavlink_connection}")
    print(f"Vision  : udp:{cfg.official.vision_host}:{cfg.official.vision_port}")
    print(f"Max run : {cfg.official.max_run_s:.0f}s")
    print(f"Tilt off: {cfg.progress.cy_tilt_offset:.3f}")
    print("-" * 60)

    if not _preflight(cfg):
        print("=" * 60)
        print("Pre-flight FAILED — fix issues above before running.")
        sys.exit(1)

    print("-" * 60)
    print("Connecting to simulator…")

    from aigrandprix.runner import PipelineRunner
    runner = PipelineRunner(cfg)
    result = runner.run(track_id="r1_qualifier")

    print()
    print("=" * 60)
    print("Run complete")
    print(f"  Lap time   : {result.get('lap_time', 0):.2f}s")
    print(f"  Gates      : {result.get('gate_count', 0)}")
    print(f"  Completed  : {result.get('completion', False)}")
    print(f"  Recoveries : {result.get('recovery_count', 0)}")
    print(f"  Avg ms/step: {result.get('mean_pipeline_ms', 0):.1f}ms")
    if runner.log_path:
        print(f"  Log        : {runner.log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
