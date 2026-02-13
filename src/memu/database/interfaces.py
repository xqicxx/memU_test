from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from memu.database.models import CategoryItem as CategoryItemRecord
from memu.database.models import MemoryCategory as MemoryCategoryRecord
from memu.database.models import MemoryItem as MemoryItemRecord
from memu.database.models import Resource as ResourceRecord
from memu.database.repositories import CategoryItemRepo, MemoryCategoryRepo, MemoryItemRepo, ResourceRepo
from memu.database.repositories import MetaRepo


@runtime_checkable
class Database(Protocol):
    """Backend-agnostic database contract."""

    resource_repo: ResourceRepo
    memory_category_repo: MemoryCategoryRepo
    memory_item_repo: MemoryItemRepo
    category_item_repo: CategoryItemRepo
    meta_repo: MetaRepo

    resources: dict[str, ResourceRecord]
    items: dict[str, MemoryItemRecord]
    categories: dict[str, MemoryCategoryRecord]
    relations: list[CategoryItemRecord]
    meta: dict[str, dict[str, Any]]

    def close(self) -> None: ...


__all__ = [
    "CategoryItemRecord",
    "Database",
    "MemoryCategoryRecord",
    "MemoryItemRecord",
    "ResourceRecord",
]
