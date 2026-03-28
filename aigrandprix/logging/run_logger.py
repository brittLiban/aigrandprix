"""RunLogger — JSONL-format logging for every pipeline step + run metrics.

Log structure:
  line 1:  {"type": "header", "run_id": ..., "seed": ..., ...}
  line 2+: {"type": "step", "t": ..., "mode": ..., ...}
  last:    {"type": "footer", "completion": ..., "lap_time": ..., ...}

Parsers must handle truncated files (last line may be incomplete JSON).
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from aigrandprix.brain.states import DroneState
from aigrandprix.config import LoggingConfig
from aigrandprix.types import (
    Action, Observation, ProgressResult, RecoveryResult,
    RiskResult, StabilityResult, VisionResult,
)


class RunMetrics:
    """Accumulate per-step metrics; compute summary at end."""

    def __init__(self):
        self.pipeline_ms_list: list[float] = []
        self.timing_violations: int = 0
        self.recovery_events: int = 0
        self.recovery_durations: list[float] = []
        self._in_recovery: bool = False
        self._recovery_start_t: float = 0.0
        self.gate_count: int = 0
        self.completion: bool = False
        self.lap_time: float = 0.0
        self.state_time: dict[str, float] = defaultdict(float)
        self._prev_state: Optional[DroneState] = None
        self._last_t: float = 0.0

    def update(self, state: DroneState, pipeline_ms: float,
               recovery: RecoveryResult, obs_t: float, dt: float,
               timing_violation: bool = False) -> None:
        self.pipeline_ms_list.append(pipeline_ms)
        if timing_violation:
            self.timing_violations += 1

        # Recovery events
        if recovery.in_recovery and not self._in_recovery:
            self.recovery_events += 1
            self._recovery_start_t = obs_t
        if not recovery.in_recovery and self._in_recovery:
            self.recovery_durations.append(obs_t - self._recovery_start_t)
        self._in_recovery = recovery.in_recovery

        # Time per state
        self.state_time[state.name] += dt

    def summary(self) -> dict:
        ms = self.pipeline_ms_list
        avg_recovery = (statistics.mean(self.recovery_durations)
                        if self.recovery_durations else 0.0)
        gates_per_s = (self.gate_count / self.lap_time
                       if self.lap_time > 0 else 0.0)
        return {
            "completion": self.completion,
            "gates_passed": self.gate_count,
            "lap_time": round(self.lap_time, 3),
            "median_pipeline_ms": round(statistics.median(ms), 2) if ms else 0.0,
            "p90_pipeline_ms": round(
                sorted(ms)[int(len(ms) * 0.9)] if ms else 0.0, 2),
            "timing_violations": self.timing_violations,
            "recovery_events": self.recovery_events,
            "avg_recovery_s": round(avg_recovery, 3),
            "gates_per_second": round(gates_per_s, 3),
            "time_by_state": {k: round(v, 3) for k, v in self.state_time.items()},
        }


class RunLogger:
    """Write JSONL log: one header, N step lines, one footer."""

    def __init__(self, config: LoggingConfig, run_id: str):
        self._cfg = config
        self._run_id = run_id
        self._path: Optional[Path] = None
        self._fh = None
        self._step_count = 0

    def open(self) -> None:
        if not self._cfg.output_dir:
            return  # logging disabled (empty output_dir)
        out_dir = Path(self._cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / f"{self._run_id}.jsonl"
        self._fh = open(self._path, "w", encoding="utf-8")

    def write_header(self, run_id: str, track_id: str, seed: int,
                     git_commit: str, config_hash: str) -> None:
        self._write({"type": "header", "run_id": run_id,
                     "track_id": track_id, "seed": seed,
                     "git_commit": git_commit, "config_hash": config_hash})

    def write_step(
        self,
        obs: Observation,
        state: DroneState,
        vision: VisionResult,
        stability: StabilityResult,
        progress: ProgressResult,
        recovery: RecoveryResult,
        risk: RiskResult,
        target: Action,
        action: Action,
        pipeline_ms: float,
        time_in_state: float = 0.0,
        state_transition: Optional[str] = None,
        lobe_times_ms: Optional[dict] = None,
    ) -> None:
        row = {
            "type": "step",
            "t": round(obs.t, 4),
            "dt": round(obs.dt, 5),
            "gate_index": progress.gate_index,
            "mode": state.name,
            "time_in_state": round(time_in_state, 3),
            "state_transition": state_transition,
            "vision": {
                "detected": vision.gate_detected,
                "cx": round(vision.cx, 3),
                "cy": round(vision.cy, 3),
                "conf_ema": round(vision.confidence_ema, 3),
                "area": round(vision.area, 0),
            },
            "stability": {
                "gyro_norm": round(stability.gyro_norm, 3),
                "stability_score": round(stability.stability_score, 3),
                "spike_count": stability.instability_spike_count,
            },
            "progress": {
                "dx": round(progress.dx, 3),
                "dy": round(progress.dy, 3),
                "approach_rate": round(progress.approach_rate, 1),
                "aligned_score": round(progress.aligned_score, 3),
                "progress_score": round(progress.progress_score, 1),
            },
            "recovery": {
                "frames_since_gate": round(recovery.frames_since_gate, 3),
                "in_recovery": recovery.in_recovery,
            },
            "risk": {
                "risk_score": round(risk.risk_score, 3),
                "push_level": risk.push_level,
                "recovery_penalty": round(risk.recent_recovery_penalty, 3),
            },
            "target": {
                "roll": round(target.roll, 3),
                "pitch": round(target.pitch, 3),
                "yaw": round(target.yaw, 3),
                "throttle": round(target.throttle, 3),
            },
            "action": {
                "roll": round(action.roll, 3),
                "pitch": round(action.pitch, 3),
                "yaw": round(action.yaw, 3),
                "throttle": round(action.throttle, 3),
            },
            "lobe_times_ms": lobe_times_ms or {},
            "pipeline_ms": round(pipeline_ms, 2),
        }
        self._write(row)
        self._step_count += 1
        if self._cfg.flush_every_n_steps > 0 and \
                self._step_count % self._cfg.flush_every_n_steps == 0:
            self._fh.flush()

    def write_footer(self, metrics_summary: dict) -> None:
        self._write({"type": "footer", **metrics_summary})
        if self._fh:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def _write(self, obj: dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(obj) + "\n")
