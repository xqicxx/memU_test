"""SQLAlchemy schema definitions for SQLite backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from sqlalchemy import MetaData
from sqlmodel import SQLModel

from memu.database.sqlite.models import (
    SQLiteCategoryItemModel,
    SQLiteMetaModel,
    SQLiteMemoryCategoryModel,
    SQLiteMemoryItemModel,
    SQLiteResourceModel,
    build_sqlite_table_model,
)


@dataclass
class SQLiteSQLAModels:
    """Container for SQLite SQLAlchemy/SQLModel models."""

    Base: type[Any]
    Resource: type[Any]
    MemoryCategory: type[Any]
    MemoryItem: type[Any]
    CategoryItem: type[Any]
    Meta: type[Any]


_MODEL_CACHE: dict[tuple[type[Any], str], SQLiteSQLAModels] = {}


def get_sqlite_sqlalchemy_models(
    *,
    scope_model: type[BaseModel] | None = None,
    table_prefix: str = "memu_",
) -> SQLiteSQLAModels:
    """Build (and cache) SQLModel ORM models for SQLite storage.

    Args:
        scope_model: Optional Pydantic model defining user scope fields.
        table_prefix: Prefix for SQLite table names (avoid reserved "sqlite_" by default).

    Returns:
        SQLiteSQLAModels containing all table models.
    """
    scope = scope_model or BaseModel
    cache_key = (scope, table_prefix)
    cached = _MODEL_CACHE.get(cache_key)
    if cached:
        return cached

    metadata_obj = MetaData()

    def _tbl(name: str) -> str:
        return f"{table_prefix}{name}" if table_prefix else name

    resource_model = build_sqlite_table_model(
        scope,
        SQLiteResourceModel,
        tablename=_tbl("resources"),
        metadata=metadata_obj,
    )
    memory_category_model = build_sqlite_table_model(
        scope,
        SQLiteMemoryCategoryModel,
        tablename=_tbl("memory_categories"),
        metadata=metadata_obj,
    )
    memory_item_model = build_sqlite_table_model(
        scope,
        SQLiteMemoryItemModel,
        tablename=_tbl("memory_items"),
        metadata=metadata_obj,
    )
    category_item_model = build_sqlite_table_model(
        scope,
        SQLiteCategoryItemModel,
        tablename=_tbl("category_items"),
        metadata=metadata_obj,
    )
    meta_model = build_sqlite_table_model(
        BaseModel,
        SQLiteMetaModel,
        tablename=_tbl("meta"),
        metadata=metadata_obj,
    )

    class SQLiteBase(SQLModel):
        __abstract__ = True
        metadata = metadata_obj

    models = SQLiteSQLAModels(
        Base=SQLiteBase,
        Resource=resource_model,
        MemoryCategory=memory_category_model,
        MemoryItem=memory_item_model,
        CategoryItem=category_item_model,
        Meta=meta_model,
    )
    _MODEL_CACHE[cache_key] = models
    return models


def get_sqlite_metadata(scope_model: type[BaseModel] | None = None) -> MetaData:
    """Get SQLAlchemy metadata for SQLite tables.

    Args:
        scope_model: Optional Pydantic model defining user scope fields.

    Returns:
        SQLAlchemy MetaData object.
    """
    from typing import cast

    return cast(MetaData, get_sqlite_sqlalchemy_models(scope_model=scope_model).Base.metadata)


__all__ = ["SQLiteSQLAModels", "get_sqlite_metadata", "get_sqlite_sqlalchemy_models"]
