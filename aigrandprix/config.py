"""Config system: dataclasses + YAML deep-merge loader.

Usage:
    cfg = load_config("configs/base.yaml", "configs/r1_qualifier.yaml")
    print(cfg.vision.hsv_lower)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


# ---------------------------------------------------------------------------
# Leaf config groups
# ---------------------------------------------------------------------------

@dataclass
class AdapterConfig:
    type: str = "mock"  # "mock" | "official"


@dataclass
class SimConfig:
    fps: int = 60
    seed: int = 42
    resolution: list = field(default_factory=lambda: [480, 640])
    total_gates: int = 10
    lateral_std: float = 0.15
    noise_std: float = 5.0
    accel_noise_std: float = 0.1
    gyro_noise_std: float = 0.02
    drop_frame_prob: float = 0.0
    latency_spike_prob: float = 0.0
    max_latency_ms: float = 20.0
    exposure_variation: float = 0.0
    # v2 rendering
    render_mode: str = "flat"           # "flat" | "perspective"
    gate_color: list = field(default_factory=lambda: [255, 255, 255])  # RGB
    bg_texture: bool = False            # gradient + noise background instead of flat gray
    gate_color_jitter: float = 0.0     # max per-channel jitter [0,1] for gate color variation


@dataclass
class LobeBudget:
    budget_ms: float = 8.0


@dataclass
class LobesConfig:
    vision: LobeBudget = field(default_factory=lambda: LobeBudget(8.0))
    stability: LobeBudget = field(default_factory=lambda: LobeBudget(2.0))
    progress: LobeBudget = field(default_factory=lambda: LobeBudget(2.0))
    recovery: LobeBudget = field(default_factory=lambda: LobeBudget(1.0))
    risk: LobeBudget = field(default_factory=lambda: LobeBudget(1.0))


@dataclass
class VisionConfig:
    resize_h: int = 240
    resize_w: int = 320
    hsv_lower: list = field(default_factory=lambda: [0, 0, 200])
    hsv_upper: list = field(default_factory=lambda: [180, 30, 255])
    min_contour_area: float = 500.0
    confidence_ema_alpha: float = 0.3
    confidence_decay_rate: float = 2.0   # per second
    position_hold_frames: int = 0        # hold last known position for N frames on dropout
    # ML backend
    backend: str = "hsv"                 # "hsv" | "ml"
    model_path: str = ""                 # path to trained .pt checkpoint
    ml_input_h: int = 90                 # 90×160 = 16:9, matches 640×360 camera
    ml_input_w: int = 160                # resize width fed to ML model
    ml_conf_threshold: float = 0.5       # detection sigmoid threshold


@dataclass
class StabilityConfig:
    gyro_tumble_threshold: float = 8.0   # rad/s
    accel_high_threshold: float = 20.0   # m/s^2
    spike_window_s: float = 0.5


@dataclass
class ProgressConfig:
    area_ema_alpha: float = 0.3
    min_progress_rate: float = 0.02
    # Camera upward tilt shifts the "centered" gate cy above 0.5.
    # Set to tan(tilt_deg) * fy / image_H / 2 for the competition camera.
    # Official cam: tan(20°)*320/360 ≈ 0.323 (level), ~0.15 at cruising pitch.
    cy_tilt_offset: float = 0.0


@dataclass
class RecoveryConfig:
    lost_gate_s: float = 0.5
    recovery_penalty_window: float = 2.0


@dataclass
class RiskConfig:
    push_level_thresholds: list = field(default_factory=lambda: [0.8, 0.6, 0.35])
    max_recovery_penalty: float = 0.3
    spike_penalty_weight: float = 0.05
    recovery_penalty_window: float = 2.0   # seconds after recovery to apply penalty


@dataclass
class StateMachineConfig:
    track_confidence_min: float = 0.4
    approach_aligned_min: float = 0.7
    min_stable: float = 0.5
    hysteresis_factor: float = 0.8
    commit_area_threshold: float = 40000.0
    search_timeout_s: float = 3.0
    approach_timeout_s: float = 2.0


@dataclass
class PIDProfile:
    yaw_kp: float = 0.5
    yaw_ki: float = 0.01
    yaw_kd: float = 0.02
    pitch_kp: float = 0.4
    pitch_ki: float = 0.01
    pitch_kd: float = 0.02
    roll_kp: float = 0.3
    roll_ki: float = 0.0
    roll_kd: float = 0.01
    throttle: float = 0.5   # base throttle for this state


@dataclass
class ControllerConfig:
    integral_clamp: float = 0.3
    push_throttle_step: float = 0.03
    profiles: dict = field(default_factory=lambda: {
        "SEARCH":   {"yaw_kp": 0.7, "yaw_ki": 0.0,  "yaw_kd": 0.02,
                     "pitch_kp": 0.2, "pitch_ki": 0.0, "pitch_kd": 0.01,
                     "roll_kp": 0.2, "roll_ki": 0.0, "roll_kd": 0.01,
                     "throttle": 0.4},
        "TRACK":    {"yaw_kp": 0.5, "yaw_ki": 0.01, "yaw_kd": 0.02,
                     "pitch_kp": 0.4, "pitch_ki": 0.01, "pitch_kd": 0.02,
                     "roll_kp": 0.3, "roll_ki": 0.0, "roll_kd": 0.01,
                     "throttle": 0.5},
        "APPROACH": {"yaw_kp": 0.4, "yaw_ki": 0.01, "yaw_kd": 0.02,
                     "pitch_kp": 0.6, "pitch_ki": 0.01, "pitch_kd": 0.02,
                     "roll_kp": 0.3, "roll_ki": 0.0, "roll_kd": 0.01,
                     "throttle": 0.65},
        "COMMIT":   {"yaw_kp": 0.2, "yaw_ki": 0.0,  "yaw_kd": 0.01,
                     "pitch_kp": 0.8, "pitch_ki": 0.0, "pitch_kd": 0.01,
                     "roll_kp": 0.2, "roll_ki": 0.0, "roll_kd": 0.01,
                     "throttle": 0.85},
        "RECOVER":  {"yaw_kp": 0.8, "yaw_ki": 0.0,  "yaw_kd": 0.03,
                     "pitch_kp": 0.1, "pitch_ki": 0.0, "pitch_kd": 0.01,
                     "roll_kp": 0.1, "roll_ki": 0.0, "roll_kd": 0.01,
                     "throttle": 0.35},
    })


@dataclass
class PipelineConfig:
    total_budget_ms: float = 15.0
    warn_budget_ms: float = 10.0


@dataclass
class LoggingConfig:
    output_dir: str = "logs/"
    flush_every_n_steps: int = 1
    save_video_snippets: bool = False


@dataclass
class OfficialAdapterConfig:
    # MAVLink connection string — see pymavlink docs.
    # "udpin:0.0.0.0:PORT"  listens on PORT (sim sends to us)
    # "udpout:HOST:PORT"     connects to sim at HOST:PORT
    mavlink_connection: str = "udpin:0.0.0.0:14551"
    vision_host: str = "0.0.0.0"
    vision_port: int = 5600
    target_system: int = 1      # MAVLink target system ID
    target_component: int = 1   # MAVLink target component ID
    connect_timeout_s: float = 20.0
    heartbeat_rate_hz: float = 4.0   # spec requires ≥ 2 Hz
    vision_recv_timeout_s: float = 2.0
    max_run_s: float = 480.0         # 8-minute race limit
    # Body-rate scaling: Action [-1, 1] → rad/s sent to SET_ATTITUDE_TARGET
    max_roll_rate_rad_s: float = 6.0    # ~344 deg/s
    max_pitch_rate_rad_s: float = 6.0
    max_yaw_rate_rad_s: float = 4.0     # ~229 deg/s
    # Camera is tilted 20° upward from body forward (spec §3.8).
    # Exposed in obs.meta so the pipeline can correct cy offsets if needed.
    camera_tilt_deg: float = 20.0


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    official: OfficialAdapterConfig = field(default_factory=OfficialAdapterConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    lobes: LobesConfig = field(default_factory=LobesConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    progress: ProgressConfig = field(default_factory=ProgressConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def config_hash(self) -> str:
        """MD5 of the serialized config for log reproducibility."""
        serialized = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (in-place, returns base)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _dict_to_config(d: dict) -> Config:
    """Convert a raw dict (from YAML) into a Config dataclass tree."""
    def _apply(dc_class, data: dict):
        if not isinstance(data, dict):
            return data
        kwargs = {}
        for f_name, f_obj in dc_class.__dataclass_fields__.items():
            if f_name not in data:
                continue
            f_type = f_obj.type
            raw = data[f_name]
            # Resolve forward references / string type hints
            if isinstance(f_type, str):
                f_type = globals().get(f_type, None)
            if f_type is not None and isinstance(raw, dict) and \
               hasattr(f_type, "__dataclass_fields__"):
                kwargs[f_name] = _apply(f_type, raw)
            else:
                kwargs[f_name] = raw
        return dc_class(**kwargs)

    return _apply(Config, d)


def load_config(*yaml_paths: str | Path) -> Config:
    """Load and deep-merge one or more YAML config files.

    The first file is the base; subsequent files override values.
    """
    merged: dict = {}
    for path in yaml_paths:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        _deep_merge(merged, data)
    return _dict_to_config(merged)


def default_config() -> Config:
    """Return a Config with all defaults (no YAML required)."""
    return Config()
