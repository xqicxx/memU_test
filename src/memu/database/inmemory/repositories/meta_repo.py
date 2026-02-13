from __future__ import annotations

from typing import Any

from memu.database.inmemory.state import InMemoryState
from memu.database.repositories.meta import MetaRepo


class InMemoryMetaRepository(MetaRepo):
    def __init__(self, *, state: InMemoryState) -> None:
        self._state = state
        self.meta = self._state.meta

    def get_meta(self, key: str) -> dict[str, Any] | None:
        return self.meta.get(key)

    def set_meta(self, key: str, value: dict[str, Any]) -> dict[str, Any]:
        self.meta[key] = dict(value)
        return self.meta[key]


__all__ = ["InMemoryMetaRepository"]
