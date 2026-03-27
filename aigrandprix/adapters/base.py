"""Abstract adapter interface — shared contract for mock and official adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from aigrandprix.types import Action, Observation


class AbstractAdapter(ABC):
    """Interface that all simulator adapters must implement.

    Adapters translate between the official (or mock) environment API and the
    internal Observation / Action contracts.  The pipeline never imports from
    adapters.official directly — it always goes through this interface.
    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset the environment and return the first observation."""

    @abstractmethod
    def step(self, action: Action) -> tuple[Observation, dict]:
        """Apply action, advance simulation, return (obs, info).

        info dict keys (all adapters must populate):
            done          bool  — episode finished (all gates passed or crashed)
            gate_passed   bool  — a gate was passed on THIS step
            gate_index    int   — current gate counter (for test assertions only;
                                  the pipeline infers this in ProgressLobe)
            lap_time      float — elapsed time since reset (or 0.0 if not done)
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the adapter."""
