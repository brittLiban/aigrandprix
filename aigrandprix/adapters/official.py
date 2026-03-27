"""OfficialSimAdapter stub — to be implemented when the interface spec drops (May 2026).

DO NOT use this adapter until the official SDK is available.
Set adapter.type = "mock" in your config for all development.
"""
from __future__ import annotations

from typing import Optional

from aigrandprix.adapters.base import AbstractAdapter
from aigrandprix.types import Action, Observation


class OfficialSimAdapter(AbstractAdapter):
    """Stub that fails loudly — prevents accidentally running mock in competition."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "OfficialSimAdapter is not yet implemented.\n"
            "The official AI Grand Prix interface spec is expected in May 2026.\n"
            "Set adapter.type = 'mock' in your config for development."
        )

    def reset(self, seed: Optional[int] = None) -> Observation:
        raise NotImplementedError

    def step(self, action: Action) -> tuple[Observation, dict]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
