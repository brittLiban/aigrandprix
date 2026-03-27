"""Unit tests for aigrandprix.timing — @timed decorator and TimingViolation."""
import time

import pytest

from aigrandprix.timing import timed, TimingViolation, measure_ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FastLobe:
    _budget_ms = 50.0
    _violation_count = 0

    @timed("FastLobe")
    def __call__(self, x):
        return x * 2


class SlowLobe:
    _budget_ms = 5.0
    _violation_count = 0

    @timed("SlowLobe")
    def __call__(self, x):
        time.sleep(0.020)  # 20ms > 5ms budget
        return x


class SlowLobeRaises:
    _budget_ms = 5.0
    _violation_count = 0

    @timed("SlowLobeRaises", raise_in_test=True)
    def __call__(self, x):
        time.sleep(0.020)
        return x


class NoBudgetLobe:
    """Lobe with no budget attribute — timing still runs but no check."""
    @timed("NoBudgetLobe")
    def __call__(self, x):
        return x + 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTimedDecorator:
    def test_fast_lobe_returns_correctly(self):
        lobe = FastLobe()
        assert lobe(3) == 6

    def test_fast_lobe_no_violation(self):
        lobe = FastLobe()
        lobe(1)
        assert lobe._violation_count == 0

    def test_slow_lobe_logs_warning_not_crash(self):
        """Slow lobe must NOT raise — it only warns."""
        lobe = SlowLobe()
        result = lobe(42)   # should not raise
        assert result == 42
        assert lobe._violation_count == 1

    def test_slow_lobe_increments_violation_count(self):
        lobe = SlowLobe()
        lobe(1)
        lobe(2)
        assert lobe._violation_count == 2

    def test_raise_in_test_mode(self):
        lobe = SlowLobeRaises()
        with pytest.raises(TimingViolation):
            lobe(1)

    def test_no_budget_attr_does_not_crash(self):
        lobe = NoBudgetLobe()
        assert lobe(5) == 6

    def test_measure_ms_utility(self):
        result, ms = measure_ms(lambda: time.sleep(0.010))
        assert ms >= 10.0
        assert result is None
