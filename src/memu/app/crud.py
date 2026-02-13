from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, cast, get_args

import pendulum

from pydantic import BaseModel

from memu.database.models import MemoryCategory, MemoryType
from memu.prompts.category_patch import CATEGORY_PATCH_PROMPT
from memu.workflow.step import WorkflowState, WorkflowStep

logger = logging.getLogger(__name__)

_QUERY_MAX_LIMIT = 200
_ITEM_FILTER_KEYS = {"memory_type", "category_id"}
_CATEGORY_FILTER_KEYS = {"category_id"}

if TYPE_CHECKING:
    from memu.app.service import Context
    from memu.app.settings import PatchConfig
    from memu.database.interfaces import Database


class CRUDMixin:
    if TYPE_CHECKING:
        _run_workflow: Callable[..., Awaitable[WorkflowState]]
        _get_context: Callable[[], Context]
        _get_database: Callable[[], Database]
        _get_step_llm_client: Callable[[Mapping[str, Any] | None], Any]
        _get_step_embedding_client: Callable[[Mapping[str, Any] | None], Any]
        _get_llm_client: Callable[..., Any]
        _model_dump_without_embeddings: Callable[[BaseModel], dict[str, Any]]
        _extract_json_blob: Callable[[str], str]
        _escape_prompt_value: Callable[[str], str]
        user_model: type[BaseModel]
        patch_config: PatchConfig
        _ensure_categories_ready: Callable[[Context, Database, Mapping[str, Any] | None], Awaitable[None]]

    async def list_memory_items(
        self,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self._get_context()
        store = self._get_database()
        where_filters = self._normalize_where(where)

        state: WorkflowState = {
            "ctx": ctx,
            "store": store,
            "where": where_filters,
        }

        result = await self._run_workflow("crud_list_memory_items", state)
        response = cast(dict[str, Any] | None, result.get("response"))
        if response is None:
            msg = "List memory items workflow failed to produce a response"
            raise RuntimeError(msg)
        return response

    async def query_memory_items(
        self,
        where: dict[str, Any],
        *,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
        time_range: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        scope_where, extra_filters = self._split_query_where(where)
        self._require_user_scope(scope_where)
        items_response = await self.list_memory_items(where=scope_where)
        items = list(items_response.get("items", []))
        items = self._apply_item_filters(items, extra_filters, scope_where, time_range)
        items = self._apply_ordering(items, order_by, default_field="created_at")
        items = self._apply_pagination(items, limit, offset)
        return items

    async def list_memory_categories(
        self,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self._get_context()
        store = self._get_database()
        where_filters = self._normalize_where(where)

        state: WorkflowState = {
            "ctx": ctx,
            "store": store,
            "where": where_filters,
        }
        result = await self._run_workflow("crud_list_memory_categories", state)
        response = cast(dict[str, Any] | None, result.get("response"))
        if response is None:
            msg = "List memory categories workflow failed to produce a response"
            raise RuntimeError(msg)
        return response

    async def query_memory_categories(
        self,
        where: dict[str, Any],
        *,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
        time_range: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        scope_where, extra_filters = self._split_query_where(where)
        self._require_user_scope(scope_where)
        categories_response = await self.list_memory_categories(where=scope_where)
        categories = list(categories_response.get("categories", []))
        categories = self._apply_category_filters(categories, extra_filters, time_range)
        categories = self._apply_ordering(categories, order_by, default_field="created_at")
        categories = self._apply_pagination(categories, limit, offset)
        return categories

    async def clear_memory(
        self,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self._get_context()
        store = self._get_database()
        where_filters = self._normalize_where(where)

        state: WorkflowState = {
            "ctx": ctx,
            "store": store,
            "where": where_filters,
        }

        result = await self._run_workflow("crud_clear_memory", state)
        response = cast(dict[str, Any] | None, result.get("response"))
        if response is None:
            msg = "Clear memory workflow failed to produce a response"
            raise RuntimeError(msg)
        return response

    def _build_list_memory_items_workflow(self) -> list[WorkflowStep]:
        steps = [
            WorkflowStep(
                step_id="list_memory_items",
                role="read_memories",
                handler=self._crud_list_memory_items,
                requires={"ctx", "store", "where"},
                produces={"items"},
                capabilities={"db"},
            ),
            WorkflowStep(
                step_id="build_response",
                role="emit",
                handler=self._crud_build_list_items_response,
                requires={"items", "ctx", "store"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    @staticmethod
    def _list_list_memories_initial_keys() -> set[str]:
        return {
            "ctx",
            "store",
            "where",
        }

    def _build_list_memory_categories_workflow(self) -> list[WorkflowStep]:
        steps = [
            WorkflowStep(
                step_id="list_memory_categories",
                role="read_categories",
                handler=self._crud_list_memory_categories,
                requires={"ctx", "store", "where"},
                produces={"categories"},
                capabilities={"db"},
            ),
            WorkflowStep(
                step_id="build_response",
                role="emit",
                handler=self._crud_build_list_categories_response,
                requires={"categories", "ctx", "store"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    def _build_clear_memory_workflow(self) -> list[WorkflowStep]:
        steps = [
            WorkflowStep(
                step_id="clear_memory_categories",
                role="delete_memories",
                handler=self._crud_clear_memory_categories,
                requires={"ctx", "store", "where"},
                produces={"deleted_categories"},
                capabilities={"db"},
            ),
            WorkflowStep(
                step_id="clear_memory_items",
                role="delete_memories",
                handler=self._crud_clear_memory_items,
                requires={"ctx", "store", "where"},
                produces={"deleted_items"},
                capabilities={"db"},
            ),
            WorkflowStep(
                step_id="clear_memory_resources",
                role="delete_memories",
                handler=self._crud_clear_memory_resources,
                requires={"ctx", "store", "where"},
                produces={"deleted_resources"},
                capabilities={"db"},
            ),
            WorkflowStep(
                step_id="build_response",
                role="emit",
                handler=self._crud_build_clear_memory_response,
                requires={"ctx", "store", "deleted_categories", "deleted_items", "deleted_resources"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    @staticmethod
    def _list_clear_memories_initial_keys() -> set[str]:
        return {
            "ctx",
            "store",
            "where",
        }

    def _normalize_where(self, where: Mapping[str, Any] | None) -> dict[str, Any]:
        """Validate and clean the `where` scope filters against the configured user model."""
        if not where:
            return {}

        valid_fields = set(getattr(self.user_model, "model_fields", {}).keys())
        cleaned: dict[str, Any] = {}

        for raw_key, value in where.items():
            if value is None:
                continue
            field = raw_key.split("__", 1)[0]
            if field not in valid_fields:
                msg = f"Unknown filter field '{field}' for current user scope"
                raise ValueError(msg)
            cleaned[raw_key] = value

        return cleaned

    def _split_query_where(
        self,
        where: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not where:
            return {}, {}
        scope_fields = set(getattr(self.user_model, "model_fields", {}).keys())
        scope_where: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, value in where.items():
            if value is None:
                continue
            field = key.split("__", 1)[0]
            if field in scope_fields:
                scope_where[key] = value
            else:
                extra[key] = value
        return scope_where, extra

    def _require_user_scope(self, scope_where: Mapping[str, Any]) -> None:
        scope_fields = set(getattr(self.user_model, "model_fields", {}).keys())
        if not scope_where:
            msg = "where must include at least one user scope field (e.g. user_id)"
            raise ValueError(msg)
        if "user_id" in scope_fields and "user_id" not in scope_where:
            msg = "where must include user_id"
            raise ValueError(msg)

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                # Support ISO with Z
                value = value.replace("Z", "+00:00")
                return datetime.fromisoformat(value)
            except Exception:
                try:
                    return pendulum.parse(value)
                except Exception:
                    return None
        return None

    def _apply_time_range(
        self, rows: list[dict[str, Any]], time_range: dict[str, str] | None
    ) -> list[dict[str, Any]]:
        if not time_range:
            return rows
        gte = self._parse_datetime(time_range.get("created_at__gte"))
        lte = self._parse_datetime(time_range.get("created_at__lte"))
        if gte is None and lte is None:
            return rows

        filtered: list[dict[str, Any]] = []
        for row in rows:
            created = self._parse_datetime(row.get("created_at"))
            if created is None:
                continue
            if gte is not None and created < gte:
                continue
            if lte is not None and created > lte:
                continue
            filtered.append(row)
        return filtered

    def _apply_item_filters(
        self,
        items: list[dict[str, Any]],
        extra_filters: dict[str, Any],
        scope_where: Mapping[str, Any],
        time_range: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        if not extra_filters and not time_range:
            return items
        unknown = set(extra_filters) - _ITEM_FILTER_KEYS
        if unknown:
            msg = f"Unsupported item filters: {sorted(unknown)}"
            raise ValueError(msg)

        memory_type = extra_filters.get("memory_type")
        if memory_type is not None:
            if isinstance(memory_type, (list, tuple, set)):
                allowed = set(memory_type)
                items = [item for item in items if item.get("memory_type") in allowed]
            else:
                items = [item for item in items if item.get("memory_type") == memory_type]

        category_id = extra_filters.get("category_id")
        if category_id is not None:
            store = self._get_database()
            relations = store.category_item_repo.list_relations(scope_where)
            allowed_item_ids = {rel.item_id for rel in relations if rel.category_id == category_id}
            items = [item for item in items if item.get("id") in allowed_item_ids]

        items = self._apply_time_range(items, time_range)
        return items

    def _apply_category_filters(
        self,
        categories: list[dict[str, Any]],
        extra_filters: dict[str, Any],
        time_range: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        if not extra_filters and not time_range:
            return categories
        unknown = set(extra_filters) - _CATEGORY_FILTER_KEYS
        if unknown:
            msg = f"Unsupported category filters: {sorted(unknown)}"
            raise ValueError(msg)
        category_id = extra_filters.get("category_id")
        if category_id is not None:
            categories = [cat for cat in categories if cat.get("id") == category_id]
        categories = self._apply_time_range(categories, time_range)
        return categories

    @staticmethod
    def _apply_pagination(
        rows: list[dict[str, Any]],
        limit: int | None,
        offset: int | None,
    ) -> list[dict[str, Any]]:
        if offset is None:
            offset = 0
        if offset < 0:
            msg = "offset must be >= 0"
            raise ValueError(msg)
        if limit is None:
            return rows[offset:]
        if limit < 0:
            msg = "limit must be >= 0"
            raise ValueError(msg)
        limit = min(limit, _QUERY_MAX_LIMIT)
        return rows[offset : offset + limit]

    @staticmethod
    def _normalize_order_by(order_by: str | None, default_field: str) -> tuple[str, bool]:
        if not order_by:
            return default_field, True
        field = order_by
        desc = False
        if order_by.startswith("-"):
            field = order_by[1:]
            desc = True
        return field, desc

    def _apply_ordering(
        self,
        rows: list[dict[str, Any]],
        order_by: str | None,
        *,
        default_field: str,
    ) -> list[dict[str, Any]]:
        field, desc = self._normalize_order_by(order_by, default_field)
        if field not in {"created_at", "updated_at", "salience"}:
            msg = f"Unsupported order_by: {field}"
            raise ValueError(msg)

        def get_val(row: dict[str, Any]) -> Any:
            if field == "salience":
                return row.get("salience") or (row.get("extra") or {}).get("salience")
            value = row.get(field)
            dt = self._parse_datetime(value)
            return dt if dt is not None else value

        values = [get_val(row) for row in rows]
        if all(v is None for v in values):
            return rows

        def sort_key(row: dict[str, Any]) -> tuple[int, Any]:
            val = get_val(row)
            if val is None:
                return (0, 0)
            return (1, val)

        return sorted(rows, key=sort_key, reverse=desc)

    def _crud_list_memory_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        where_filters = state.get("where") or {}
        store = state["store"]
        items = store.memory_item_repo.list_items(where_filters)
        state["items"] = items
        return state

    def _crud_list_memory_categories(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        where_filters = state.get("where") or {}
        store = state["store"]
        categories = store.memory_category_repo.list_categories(where_filters)
        state["categories"] = categories
        return state

    def _crud_build_list_items_response(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        items = state["items"]
        items_list = [self._model_dump_without_embeddings(item) for item in items.values()]
        response = {
            "items": items_list,
        }
        state["response"] = response
        return state

    def _crud_build_list_categories_response(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        categories = state["categories"]
        categories_list = [self._model_dump_without_embeddings(category) for category in categories.values()]
        response = {
            "categories": categories_list,
        }
        state["response"] = response
        return state

    def _crud_clear_memory_categories(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        where_filters = state.get("where") or {}
        store = state["store"]
        deleted = store.memory_category_repo.clear_categories(where_filters)
        state["deleted_categories"] = deleted
        return state

    def _crud_clear_memory_items(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        where_filters = state.get("where") or {}
        store = state["store"]
        deleted = store.memory_item_repo.clear_items(where_filters)
        state["deleted_items"] = deleted
        return state

    def _crud_clear_memory_resources(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        where_filters = state.get("where") or {}
        store = state["store"]
        deleted = store.resource_repo.clear_resources(where_filters)
        state["deleted_resources"] = deleted
        return state

    def _crud_build_clear_memory_response(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        deleted_categories = state.get("deleted_categories", {})
        deleted_items = state.get("deleted_items", {})
        deleted_resources = state.get("deleted_resources", {})
        response = {
            "deleted_categories": [self._model_dump_without_embeddings(cat) for cat in deleted_categories.values()],
            "deleted_items": [self._model_dump_without_embeddings(item) for item in deleted_items.values()],
            "deleted_resources": [self._model_dump_without_embeddings(res) for res in deleted_resources.values()],
        }
        state["response"] = response
        return state

    async def create_memory_item(
        self,
        *,
        memory_type: MemoryType,
        memory_content: str,
        memory_categories: list[str],
        user: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if memory_type not in get_args(MemoryType):
            msg = f"Invalid memory type: '{memory_type}', must be one of {get_args(MemoryType)}"
            raise ValueError(msg)

        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None
        await self._ensure_categories_ready(ctx, store, user_scope)

        state: WorkflowState = {
            "memory_payload": {
                "type": memory_type,
                "content": memory_content,
                "categories": memory_categories,
            },
            "ctx": ctx,
            "store": store,
            "category_ids": list(ctx.category_ids),
            "user": user_scope,
        }

        result = await self._run_workflow("patch_create", state)
        response = cast(dict[str, Any] | None, result.get("response"))
        if response is None:
            msg = "Create memory item workflow failed to produce a response"
            raise RuntimeError(msg)
        return response

    async def update_memory_item(
        self,
        *,
        memory_id: str,
        memory_type: MemoryType | None = None,
        memory_content: str | None = None,
        memory_categories: list[str] | None = None,
        user: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if all((memory_type is None, memory_content is None, memory_categories is None)):
            msg = "At least one of memory type, memory content, or memory categories is required for UPDATE operation"
            raise ValueError(msg)
        if memory_type and memory_type not in get_args(MemoryType):
            msg = f"Invalid memory type: '{memory_type}', must be one of {get_args(MemoryType)}"
            raise ValueError(msg)

        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None
        await self._ensure_categories_ready(ctx, store, user_scope)

        state: WorkflowState = {
            "memory_id": memory_id,
            "memory_payload": {
                "type": memory_type,
                "content": memory_content,
                "categories": memory_categories,
            },
            "ctx": ctx,
            "store": store,
            "category_ids": list(ctx.category_ids),
            "user": user_scope,
        }

        result = await self._run_workflow("patch_update", state)
        response = cast(dict[str, Any] | None, result.get("response"))
        if response is None:
            msg = "Update memory item workflow failed to produce a response"
            raise RuntimeError(msg)
        return response

    async def delete_memory_item(
        self,
        *,
        memory_id: str,
        user: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self._get_context()
        store = self._get_database()
        user_scope = self.user_model(**user).model_dump() if user is not None else None
        await self._ensure_categories_ready(ctx, store, user_scope)

        state: WorkflowState = {
            "memory_id": memory_id,
            "ctx": ctx,
            "store": store,
            "category_ids": list(ctx.category_ids),
            "user": user_scope,
        }

        result = await self._run_workflow("patch_delete", state)
        response = cast(dict[str, Any] | None, result.get("response"))
        if response is None:
            msg = "Delete memory item workflow failed to produce a response"
            raise RuntimeError(msg)
        return response

    def _build_create_memory_item_workflow(self) -> list[WorkflowStep]:
        steps = [
            WorkflowStep(
                step_id="create_memory_item",
                role="patch",
                handler=self._patch_create_memory_item,
                requires={"memory_payload", "ctx", "store", "user"},
                produces={"memory_item", "category_updates"},
                capabilities={"db", "llm"},
                config={"embed_llm_profile": "embedding"},
            ),
            WorkflowStep(
                step_id="persist_index",
                role="persist",
                handler=self._patch_persist_and_index,
                requires={"category_updates", "ctx", "store"},
                produces={"categories"},
                capabilities={"db", "llm"},
                config={"chat_llm_profile": "default"},
            ),
            WorkflowStep(
                step_id="build_response",
                role="emit",
                handler=self._patch_build_response,
                requires={"memory_item", "category_updates", "ctx", "store"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    @staticmethod
    def _list_create_memory_item_initial_keys() -> set[str]:
        return {
            "memory_payload",
            "ctx",
            "store",
            "user",
        }

    def _build_update_memory_item_workflow(self) -> list[WorkflowStep]:
        steps = [
            WorkflowStep(
                step_id="update_memory_item",
                role="patch",
                handler=self._patch_update_memory_item,
                requires={"memory_id", "memory_payload", "ctx", "store", "user"},
                produces={"memory_item", "category_updates"},
                capabilities={"db", "llm"},
                config={"embed_llm_profile": "embedding"},
            ),
            WorkflowStep(
                step_id="persist_index",
                role="persist",
                handler=self._patch_persist_and_index,
                requires={"category_updates", "ctx", "store"},
                produces={"categories"},
                capabilities={"db", "llm"},
                config={"chat_llm_profile": "default"},
            ),
            WorkflowStep(
                step_id="build_response",
                role="emit",
                handler=self._patch_build_response,
                requires={"memory_item", "category_updates", "ctx", "store"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    @staticmethod
    def _list_update_memory_item_initial_keys() -> set[str]:
        return {
            "memory_id",
            "memory_payload",
            "ctx",
            "store",
            "user",
        }

    def _build_delete_memory_item_workflow(self) -> list[WorkflowStep]:
        steps = [
            WorkflowStep(
                step_id="delete_memory_item",
                role="patch",
                handler=self._patch_delete_memory_item,
                requires={"memory_id", "ctx", "store", "user"},
                produces={"memory_item", "category_updates"},
                capabilities={"db"},
            ),
            WorkflowStep(
                step_id="persist_index",
                role="persist",
                handler=self._patch_persist_and_index,
                requires={"category_updates", "ctx", "store"},
                produces={"categories"},
                capabilities={"db", "llm"},
                config={"chat_llm_profile": "default"},
            ),
            WorkflowStep(
                step_id="build_response",
                role="emit",
                handler=self._patch_build_response,
                requires={"memory_item", "category_updates", "ctx", "store"},
                produces={"response"},
                capabilities=set(),
            ),
        ]
        return steps

    @staticmethod
    def _list_delete_memory_item_initial_keys() -> set[str]:
        return {
            "memory_id",
            "ctx",
            "store",
            "user",
        }

    async def _patch_create_memory_item(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        memory_payload = state["memory_payload"]
        ctx = state["ctx"]
        store = state["store"]
        user = state["user"]
        category_memory_updates: dict[str, tuple[Any, Any]] = {}

        embed_payload = [memory_payload["content"]]
        content_embedding = (await self._get_step_embedding_client(step_context).embed(embed_payload))[0]

        item = store.memory_item_repo.create_item(
            memory_type=memory_payload["type"],
            summary=memory_payload["content"],
            embedding=content_embedding,
            user_data=dict(user or {}),
        )
        cat_names = memory_payload["categories"]
        mapped_cat_ids = self._map_category_names_to_ids(cat_names, ctx)
        for cid in mapped_cat_ids:
            store.category_item_repo.link_item_category(item.id, cid, user_data=dict(user or {}))
            category_memory_updates[cid] = (None, memory_payload["content"])

        state.update({
            "memory_item": item,
            "category_updates": category_memory_updates,
        })
        return state

    async def _patch_update_memory_item(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        memory_id = state["memory_id"]
        memory_payload = state["memory_payload"]
        ctx = state["ctx"]
        store = state["store"]
        user = state["user"]
        category_memory_updates: dict[str, tuple[Any, Any]] = {}

        item = store.memory_item_repo.get_item(memory_id)
        if not item:
            msg = f"Memory item with id {memory_id} not found"
            raise ValueError(msg)
        old_content = item.summary
        old_item_categories = store.category_item_repo.get_item_categories(memory_id)
        mapped_old_cat_ids = [cat.category_id for cat in old_item_categories]

        if memory_payload["content"]:
            embed_payload = [memory_payload["content"]]
            content_embedding = (await self._get_step_embedding_client(step_context).embed(embed_payload))[0]
        else:
            content_embedding = None

        if memory_payload["type"] or memory_payload["content"]:
            item = store.memory_item_repo.update_item(
                item_id=memory_id,
                memory_type=memory_payload["type"],
                summary=memory_payload["content"],
                embedding=content_embedding,
            )
        new_cat_names = memory_payload["categories"]
        mapped_new_cat_ids = self._map_category_names_to_ids(new_cat_names, ctx)

        cats_to_remove = set(mapped_old_cat_ids) - set(mapped_new_cat_ids)
        cats_to_add = set(mapped_new_cat_ids) - set(mapped_old_cat_ids)
        for cid in cats_to_remove:
            store.category_item_repo.unlink_item_category(memory_id, cid)
            category_memory_updates[cid] = (old_content, None)
        for cid in cats_to_add:
            store.category_item_repo.link_item_category(memory_id, cid, user_data=dict(user or {}))
            category_memory_updates[cid] = (None, item.summary)

        if memory_payload["content"]:
            for cid in set(mapped_old_cat_ids) & set(mapped_new_cat_ids):
                category_memory_updates[cid] = (old_content, item.summary)

        state.update({
            "memory_item": item,
            "category_updates": category_memory_updates,
        })
        return state

    async def _patch_delete_memory_item(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        memory_id = state["memory_id"]
        store = state["store"]
        category_memory_updates: dict[str, tuple[Any, Any]] = {}

        item = store.memory_item_repo.get_item(memory_id)
        if not item:
            msg = f"Memory item with id {memory_id} not found"
            raise ValueError(msg)
        item_categories = store.category_item_repo.get_item_categories(memory_id)
        for cat in item_categories:
            category_memory_updates[cat.category_id] = (item.summary, None)
        store.memory_item_repo.delete_item(memory_id)

        state.update({
            "memory_item": item,
            "category_updates": category_memory_updates,
        })
        return state

    async def _patch_persist_and_index(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        llm_client = self._get_step_llm_client(step_context)
        await self._patch_category_summaries(
            state.get("category_updates", {}),
            ctx=state["ctx"],
            store=state["store"],
            llm_client=llm_client,
        )
        return state

    def _patch_build_response(self, state: WorkflowState, step_context: Any) -> WorkflowState:
        store = state["store"]
        item = self._model_dump_without_embeddings(state["memory_item"])
        category_updates_ids = list(state.get("category_updates", {}).keys())
        category_updates = [
            self._model_dump_without_embeddings(store.memory_category_repo.categories[c]) for c in category_updates_ids
        ]
        response = {
            "memory_item": item,
            "category_updates": category_updates,
        }
        state["response"] = response
        return state

    def _map_category_names_to_ids(self, names: list[str], ctx: Context) -> list[str]:
        if not names:
            return []
        mapped: list[str] = []
        seen: set[str] = set()
        for name in names:
            key = name.strip().lower()
            cid = ctx.category_name_to_id.get(key)
            if cid and cid not in seen:
                mapped.append(cid)
                seen.add(cid)
        return mapped

    async def _patch_category_summaries(
        self,
        updates: dict[str, list[str]],
        ctx: Context,
        store: Database,
        llm_client: Any | None = None,
    ) -> None:
        if not updates:
            return
        tasks = []
        target_ids: list[str] = []
        client = llm_client or self._get_llm_client()
        for cid, (content_before, content_after) in updates.items():
            cat = store.memory_category_repo.categories.get(cid)
            if not cat or (not content_before and not content_after):
                continue
            prompt = self._build_category_patch_prompt(
                category=cat, content_before=content_before, content_after=content_after
            )
            tasks.append(client.chat(prompt))
            target_ids.append(cid)
        if not tasks:
            return
        patches = await asyncio.gather(*tasks)
        for cid, patch in zip(target_ids, patches, strict=True):
            need_update, summary = self._parse_category_patch_response(patch)
            if not need_update:
                continue
            cat = store.memory_category_repo.categories.get(cid)
            store.memory_category_repo.update_category(
                category_id=cid,
                summary=summary.strip(),
            )

    def _build_category_patch_prompt(
        self, *, category: MemoryCategory, content_before: str | None, content_after: str | None
    ) -> str:
        if content_before and content_after:
            update_content = "\n".join([
                "The memory content before:",
                content_before,
                "The memory content after:",
                content_after,
            ])
        elif content_before:
            update_content = "\n".join([
                "This memory content is discarded:",
                content_before,
            ])
        elif content_after:
            update_content = "\n".join([
                "This memory content is newly added:",
                content_after,
            ])
        original_content = category.summary or ""
        prompt = CATEGORY_PATCH_PROMPT
        return prompt.format(
            category=self._escape_prompt_value(category.name),
            original_content=self._escape_prompt_value(original_content or ""),
            update_content=self._escape_prompt_value(update_content or ""),
        )

    def _parse_category_patch_response(self, response: str) -> tuple[bool, str]:
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError):
            return False, ""
        if not isinstance(data, dict):
            return False, ""
        if not data.get("updated_content"):
            return False, ""
        need_update = data.get("need_update", False)
        updated_content = data["updated_content"].strip()
        if updated_content == "empty":
            updated_content = ""
        return need_update, updated_content
