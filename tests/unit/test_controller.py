"""Unit tests for PID and Controller."""
import numpy as np
import pytest

from aigrandprix.brain.states import DroneState
from aigrandprix.config import default_config
from aigrandprix.controller.pid import Controller, PID
from aigrandprix.types import Action, Observation


def obs():
    return Observation(
        t=1.0, dt=0.016,
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        imu_accel=np.array([0.0, 0.0, 9.81]),
        imu_gyro=np.zeros(3),
    )


class TestPID:
    def test_proportional_only(self):
        pid = PID(kp=1.0, ki=0.0, kd=0.0)
        out = pid.step(error=0.5, dt=0.016)
        assert out == pytest.approx(0.5)

    def test_integral_accumulates(self):
        pid = PID(kp=0.0, ki=1.0, kd=0.0, integral_clamp=10.0)
        pid.step(error=1.0, dt=0.016)
        pid.step(error=1.0, dt=0.016)
        out = pid.step(error=1.0, dt=0.016)
        assert out == pytest.approx(3 * 0.016, abs=0.001)

    def test_integral_clamped(self):
        pid = PID(kp=0.0, ki=1.0, kd=0.0, integral_clamp=0.01)
        for _ in range(100):
            pid.step(error=1.0, dt=0.016)
        # Integral should be clamped at 0.01 → output = 0.01
        out = pid.step(error=0.0, dt=0.016)  # kp*0 + ki*clamp + kd*deriv
        assert abs(out) <= 0.011

    def test_integral_reset(self):
        pid = PID(kp=0.0, ki=1.0, kd=0.0)
        for _ in range(10):
            pid.step(1.0, 0.016)
        pid.reset_integral()
        out = pid.step(error=0.0, dt=0.016)
        assert out == pytest.approx(0.0, abs=1e-9)

    def test_derivative(self):
        pid = PID(kp=0.0, ki=0.0, kd=1.0)
        pid.step(error=0.0, dt=0.016)
        out = pid.step(error=0.016, dt=0.016)
        assert out == pytest.approx(1.0, abs=0.1)


class TestController:
    def test_output_within_valid_range(self):
        ctrl = Controller(default_config().controller)
        target = Action(roll=2.0, pitch=-3.0, yaw=1.5, throttle=0.7)
        action = ctrl(target, obs(), dt=0.016, state=DroneState.TRACK)
        assert -1.0 <= action.roll <= 1.0
        assert -1.0 <= action.pitch <= 1.0
        assert -1.0 <= action.yaw <= 1.0
        assert 0.0 <= action.throttle <= 1.0

    def test_throttle_base_comes_from_profile(self):
        ctrl = Controller(default_config().controller)
        cfg = default_config().controller
        target = Action.zero()
        for state in DroneState:
            action = ctrl(target, obs(), dt=0.016, state=state, push_level=0)
            expected = cfg.profiles[state.name]["throttle"]
            assert abs(action.throttle - expected) < 0.05, \
                f"throttle mismatch for {state.name}"

    def test_push_level_increases_throttle(self):
        ctrl = Controller(default_config().controller)
        target = Action.zero()
        action_0 = ctrl(target, obs(), dt=0.016, state=DroneState.APPROACH,
                        push_level=0)
        action_3 = ctrl(target, obs(), dt=0.016, state=DroneState.APPROACH,
                        push_level=3)
        assert action_3.throttle > action_0.throttle

    def test_integral_reset_zeroes_i_term(self):
        ctrl = Controller(default_config().controller)
        target = Action(roll=0.5, pitch=0.5, yaw=0.5, throttle=0.5)
        # Build up integral
        for _ in range(20):
            ctrl(target, obs(), dt=0.016, state=DroneState.APPROACH)
        # Reset
        action_after = ctrl(Action.zero(), obs(), dt=0.016,
                            state=DroneState.RECOVER, reset_integral=True)
        # With zero target and reset integral, output should be near 0 for roll/pitch/yaw
        assert abs(action_after.roll) < 0.1
        assert abs(action_after.pitch) < 0.1
        assert abs(action_after.yaw) < 0.1

    def test_different_states_give_different_outputs(self):
        ctrl = Controller(default_config().controller)
        target = Action(roll=0.3, pitch=0.3, yaw=0.3, throttle=0.5)
        ctrl.reset_all_integrals()
        a_search = ctrl(target, obs(), dt=0.016, state=DroneState.SEARCH)
        ctrl.reset_all_integrals()
        a_commit = ctrl(target, obs(), dt=0.016, state=DroneState.COMMIT)
        # COMMIT has higher yaw_kp than SEARCH in the COMMIT direction
        assert a_search.throttle != a_commit.throttle
