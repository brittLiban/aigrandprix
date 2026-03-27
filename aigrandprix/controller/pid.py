"""PID controller with per-state gain profiles and anti-windup."""
from __future__ import annotations

import numpy as np

from aigrandprix.brain.states import DroneState
from aigrandprix.config import ControllerConfig
from aigrandprix.types import Action, Observation


class PID:
    """Single-axis PID with integral clamping."""

    def __init__(self, kp: float, ki: float, kd: float,
                 integral_clamp: float = 0.3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral_clamp = integral_clamp
        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def step(self, error: float, dt: float) -> float:
        self._integral += error * dt
        self._integral = float(
            np.clip(self._integral, -self._integral_clamp, self._integral_clamp))
        derivative = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset_integral(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


class Controller:
    """Maps target Action → output Action using per-state PID gain profiles.

    The controller:
    1. Selects PID gains for the current DroneState.
    2. Computes error = target - current (for roll/pitch/yaw; throttle direct).
    3. Runs PID on each axis.
    4. Applies throttle boost from push_level.
    5. Clamps output to valid ranges.
    6. Resets integrals when brain signals RECOVER.
    """

    def __init__(self, config: ControllerConfig):
        self._cfg = config
        self._pids: dict[str, dict[str, PID]] = {}
        for state_name, profile in config.profiles.items():
            ic = config.integral_clamp
            self._pids[state_name] = {
                "roll":  PID(profile["roll_kp"], profile["roll_ki"],
                             profile["roll_kd"], ic),
                "pitch": PID(profile["pitch_kp"], profile["pitch_ki"],
                             profile["pitch_kd"], ic),
                "yaw":   PID(profile["yaw_kp"], profile["yaw_ki"],
                             profile["yaw_kd"], ic),
            }

    def __call__(self, target: Action, obs: Observation,
                 dt: float, state: DroneState,
                 push_level: int = 0,
                 reset_integral: bool = False) -> Action:
        state_name = state.name
        if state_name not in self._pids:
            state_name = "TRACK"  # fallback

        pids = self._pids[state_name]
        profile = self._cfg.profiles.get(state_name,
                                          self._cfg.profiles["TRACK"])

        if reset_integral:
            for pid in pids.values():
                pid.reset_integral()

        dt = max(dt, 1e-6)

        # For each axis, error = target - 0 (target IS the desired command)
        roll_out = pids["roll"].step(target.roll, dt)
        pitch_out = pids["pitch"].step(target.pitch, dt)
        yaw_out = pids["yaw"].step(target.yaw, dt)

        # Throttle: base from profile + push boost
        base_throttle = profile["throttle"]
        throttle_boost = push_level * self._cfg.push_throttle_step
        throttle_out = base_throttle + throttle_boost

        # Clamp
        action = Action(
            roll=float(np.clip(roll_out, -1.0, 1.0)),
            pitch=float(np.clip(pitch_out, -1.0, 1.0)),
            yaw=float(np.clip(yaw_out, -1.0, 1.0)),
            throttle=float(np.clip(throttle_out, 0.0, 1.0)),
        )
        return action

    def reset_all_integrals(self) -> None:
        for state_pids in self._pids.values():
            for pid in state_pids.values():
                pid.reset_integral()
