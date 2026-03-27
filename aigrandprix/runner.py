"""PipelineRunner — wires all lobes into the main control loop."""
from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional

from aigrandprix.adapters.mock import MockSimAdapter
from aigrandprix.brain.fusion import FusionBrain
from aigrandprix.brain.states import DroneState
from aigrandprix.config import Config
from aigrandprix.controller.pid import Controller
from aigrandprix.lobes.progress import ProgressLobe
from aigrandprix.lobes.recovery import RecoveryLobe
from aigrandprix.lobes.risk import RiskLobe
from aigrandprix.lobes.stability import StabilityLobe
from aigrandprix.lobes.vision import VisionLobe
from aigrandprix.lobes.vision_ml import MLVisionLobe
from aigrandprix.logging.run_logger import RunLogger, RunMetrics
from aigrandprix.types import Action


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _build_adapter(config: Config):
    if config.adapter.type == "mock":
        return MockSimAdapter(config.sim)
    elif config.adapter.type == "official":
        from aigrandprix.adapters.official import OfficialSimAdapter
        return OfficialSimAdapter(config)
    else:
        raise ValueError(f"Unknown adapter type: {config.adapter.type!r}")


class PipelineRunner:
    """Full autonomous pipeline: adapter → lobes → brain → controller → log."""

    def __init__(self, config: Config, run_id: Optional[str] = None):
        self._cfg = config
        self._run_id = run_id or f"run_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"

        self.adapter = _build_adapter(config)
        if config.vision.backend == "ml":
            self.vision = MLVisionLobe(config.vision,
                                       budget_ms=config.lobes.vision.budget_ms)
        else:
            self.vision = VisionLobe(config.vision,
                                     budget_ms=config.lobes.vision.budget_ms)
        self.stability = StabilityLobe(config.stability,
                                        budget_ms=config.lobes.stability.budget_ms)
        self.progress = ProgressLobe(config.progress,
                                      budget_ms=config.lobes.progress.budget_ms)
        self.recovery = RecoveryLobe(config.recovery,
                                      budget_ms=config.lobes.recovery.budget_ms)
        self.risk = RiskLobe(config.risk,
                              budget_ms=config.lobes.risk.budget_ms)
        self.brain = FusionBrain(config.state_machine)
        self.controller = Controller(config.controller)
        self.logger = RunLogger(config.logging, self._run_id)

    def run(self, seed: Optional[int] = None, track_id: str = "mock") -> dict:
        """Execute one full episode and return run metrics."""
        effective_seed = seed if seed is not None else self._cfg.sim.seed

        # Reset all stateful components
        self.vision.reset()
        self.stability.reset()
        self.progress.reset()
        self.recovery.reset()
        self.risk.reset()
        self.brain.reset()
        self.controller.reset_all_integrals()

        obs = self.adapter.reset(seed=effective_seed)

        self.logger.open()
        self.logger.write_header(
            run_id=self._run_id,
            track_id=track_id,
            seed=effective_seed,
            git_commit=_git_commit(),
            config_hash=self._cfg.config_hash(),
        )

        metrics = RunMetrics()
        state = DroneState.SEARCH
        prev_state = DroneState.SEARCH

        while True:
            t0_wall = time.perf_counter()

            # --- Lobe pipeline ---
            vision_r = self.vision(obs)
            stability_r = self.stability(obs)
            self.recovery.update_dt(obs.dt)
            progress_r = self.progress(obs, vision_r, state, self.brain.time_in_state)
            recovery_r = self.recovery(vision_r, state)
            risk_r = self.risk(stability_r, progress_r, recovery_r, obs.t)

            # --- Brain ---
            prev_state = state
            state, target = self.brain(
                vision_r, stability_r, progress_r, recovery_r, risk_r,
                sim_t=obs.t,
            )
            reset_int = self.brain.integral_reset_requested

            # --- Controller ---
            action = self.controller(
                target, obs, obs.dt, state,
                push_level=risk_r.push_level,
                reset_integral=reset_int,
            )

            pipeline_ms = (time.perf_counter() - t0_wall) * 1000.0

            # --- Log ---
            transition = (f"{prev_state.name}->{state.name}"
                          if prev_state != state else None)
            self.logger.write_step(
                obs=obs,
                state=state,
                vision=vision_r,
                stability=stability_r,
                progress=progress_r,
                recovery=recovery_r,
                risk=risk_r,
                target=target,
                action=action,
                pipeline_ms=pipeline_ms,
                time_in_state=self.brain.time_in_state,
                state_transition=transition,
                lobe_times_ms={"vision": vision_r.latency_ms},
            )

            # --- Metrics ---
            metrics.update(
                state=state,
                pipeline_ms=pipeline_ms,
                recovery=recovery_r,
                obs_t=obs.t,
                dt=obs.dt,
            )

            # --- Step ---
            obs, info = self.adapter.step(action)

            if info["done"]:
                metrics.completion = True
                metrics.gate_count = info["gate_index"]
                metrics.lap_time = info["lap_time"]
                break

        summary = metrics.summary()
        self.logger.write_footer(summary)
        self.logger.close()
        return summary

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def log_path(self) -> Optional[Path]:
        return self.logger.path
