"""Per-lobe timing enforcement.

Usage:
    from aigrandprix.timing import timed, TimingViolation

    class MyLobe:
        def __init__(self, budget_ms: float = 8.0):
            self._budget_ms = budget_ms
            self._violation_count = 0

        @timed("MyLobe")
        def __call__(self, obs):
            ...

The decorator measures wall-clock time, logs a WARNING if the budget is
exceeded, and increments a violation counter on the lobe instance if it has
a `_violation_count` attribute. It never raises or aborts the run.
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class TimingViolation(Exception):
    """Raised only in tests to assert timing budget enforcement."""
    pass


def timed(lobe_name: str, budget_ms_attr: str = "_budget_ms",
          raise_in_test: bool = False) -> Callable:
    """Decorator factory that enforces a per-lobe timing budget.

    Args:
        lobe_name: Human-readable name for log messages.
        budget_ms_attr: Name of the instance attribute holding budget_ms.
            If the attribute does not exist, timing measurement still runs
            but no budget check is performed.
        raise_in_test: If True, raise TimingViolation instead of warning.
            Used only in unit tests.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            t0 = time.perf_counter()
            result = fn(self, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            budget_ms: Optional[float] = getattr(self, budget_ms_attr, None)
            if budget_ms is not None and elapsed_ms > budget_ms:
                if hasattr(self, "_violation_count"):
                    self._violation_count += 1
                msg = (
                    f"[TIMING] {lobe_name} took {elapsed_ms:.1f}ms "
                    f"(budget {budget_ms:.1f}ms)"
                )
                if raise_in_test:
                    raise TimingViolation(msg)
                logger.warning(msg)

            # Attach latency to result if it has a latency_ms field
            if hasattr(result, "latency_ms"):
                object.__setattr__(result, "latency_ms", elapsed_ms) \
                    if hasattr(result, "__dataclass_fields__") \
                    else setattr(result, "latency_ms", elapsed_ms)

            return result
        return wrapper
    return decorator


def measure_ms(fn: Callable, *args, **kwargs) -> tuple:
    """Utility: call fn(*args, **kwargs), return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000.0
