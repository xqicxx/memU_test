from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetaRepo(Protocol):
    """Repository contract for key/value metadata."""

    def get_meta(self, key: str) -> dict[str, Any] | None: ...

    def set_meta(self, key: str, value: dict[str, Any]) -> dict[str, Any]: ...

