from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar
from datetime import datetime, timezone

from pydantic import BaseModel

from memu.app.crud import CRUDMixin
from memu.app.memorize import MemorizeMixin
from memu.app.retrieve import RetrieveMixin
from memu.app.settings import (
    BlobConfig,
    CategoryConfig,
    DatabaseConfig,
    LLMConfig,
    LLMProfilesConfig,
    MemorizeConfig,
    RetrieveConfig,
    UserConfig,
    load_enable_video_from_config,
    load_memory_categories_from_config,
    resolve_restart_marker_path,
    resolve_memory_category_config_path,
)
from memu.blob.local_fs import LocalFS
from memu.database.factory import build_database
from memu.database.interfaces import Database
from memu.llm.http_client import HTTPLLMClient
from memu.llm.wrapper import (
    LLMCallMetadata,
    LLMClientWrapper,
    LLMInterceptorHandle,
    LLMInterceptorRegistry,
)
from memu.workflow.interceptor import WorkflowInterceptorHandle, WorkflowInterceptorRegistry
from memu.workflow.pipeline import PipelineManager
from memu.workflow.runner import WorkflowRunner, resolve_workflow_runner
from memu.workflow.step import WorkflowState, WorkflowStep

TConfigModel = TypeVar("TConfigModel", bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass
class Context:
    categories_ready: bool = False
    category_ids: list[str] = field(default_factory=list)
    category_name_to_id: dict[str, str] = field(default_factory=dict)
    category_init_task: asyncio.Task | None = None


class MemoryService(MemorizeMixin, RetrieveMixin, CRUDMixin):
    def __init__(
        self,
        *,
        llm_profiles: LLMProfilesConfig | dict[str, Any] | None = None,
        blob_config: BlobConfig | dict[str, Any] | None = None,
        database_config: DatabaseConfig | dict[str, Any] | None = None,
        memorize_config: MemorizeConfig | dict[str, Any] | None = None,
        retrieve_config: RetrieveConfig | dict[str, Any] | None = None,
        workflow_runner: WorkflowRunner | str | None = None,
        user_config: UserConfig | dict[str, Any] | None = None,
    ):
        self.llm_profiles = self._validate_config(llm_profiles, LLMProfilesConfig)
        self.user_config = self._validate_config(user_config, UserConfig)
        self.user_model = self.user_config.model
        self.llm_config = self._validate_config(self.llm_profiles.default, LLMConfig)
        self.blob_config = self._validate_config(blob_config, BlobConfig)
        self.database_config = self._validate_config(database_config, DatabaseConfig)
        memo_categories = load_memory_categories_from_config()
        enable_video = load_enable_video_from_config()
        memorize_config = self._merge_memory_categories_config(memorize_config, memo_categories)
        memorize_config = self._merge_multimodal_config(memorize_config, enable_video)
        self.memorize_config = self._validate_config(memorize_config, MemorizeConfig)
        self.retrieve_config = self._validate_config(retrieve_config, RetrieveConfig)
        self._warn_missing_llm_api_keys()
        if memo_categories is not None:
            logger.info(
                "Loaded %d memory categories from %s",
                len(memo_categories),
                resolve_memory_category_config_path(),
            )
        else:
            logger.info(
                "Using default memory categories: %d",
                len(self.memorize_config.memory_categories or []),
            )
        if enable_video is not None:
            logger.info("Video modality enabled: %s", enable_video)

        self.fs = LocalFS(self.blob_config.resources_dir)
        self.category_configs: list[CategoryConfig] = list(self.memorize_config.memory_categories or [])
        self.category_config_map: dict[str, CategoryConfig] = {cfg.name: cfg for cfg in self.category_configs}
        self._category_prompt_str = self._format_categories_for_prompt(self.category_configs)

        self._context = Context(categories_ready=not bool(self.category_configs))

        self.database: Database = build_database(
            config=self.database_config,
            user_model=self.user_model,
        )
        # We need the concrete user scope (user_id: xxx) to initialize the categories
        # self._start_category_initialization(self._context, self.database)

        # Initialize client caches (lazy creation on first use)
        self._llm_clients: dict[str, Any] = {}
        self._llm_interceptors = LLMInterceptorRegistry()
        self._workflow_interceptors = WorkflowInterceptorRegistry()

        self._workflow_runner = resolve_workflow_runner(workflow_runner)

        self._pipelines = PipelineManager(
            available_capabilities={"llm", "vector", "db", "io", "vision"},
            llm_profiles=set(self.llm_profiles.profiles.keys()),
        )
        self._register_pipelines()

    @staticmethod
    def _merge_memory_categories_config(
        memorize_config: MemorizeConfig | dict[str, Any] | None,
        categories: list[CategoryConfig] | None,
    ) -> MemorizeConfig | dict[str, Any] | None:
        if not categories:
            return memorize_config
        if memorize_config is None:
            return {"memory_categories": categories}
        if isinstance(memorize_config, MemorizeConfig):
            # Explicit config passed; do not override for backward compatibility.
            return memorize_config
        if isinstance(memorize_config, Mapping):
            if "memory_categories" in memorize_config:
                return memorize_config
            merged = dict(memorize_config)
            merged["memory_categories"] = categories
            return merged
        return memorize_config

    @staticmethod
    def _merge_multimodal_config(
        memorize_config: MemorizeConfig | dict[str, Any] | None,
        enable_video: bool | None,
    ) -> MemorizeConfig | dict[str, Any] | None:
        if enable_video is None:
            return memorize_config
        if memorize_config is None:
            return {"multimodal": {"enable_video": enable_video}}
        if isinstance(memorize_config, MemorizeConfig):
            # Explicit config passed; do not override for backward compatibility.
            return memorize_config
        if isinstance(memorize_config, Mapping):
            existing = memorize_config.get("multimodal")
            if isinstance(existing, Mapping) and "enable_video" in existing:
                return memorize_config
            merged = dict(memorize_config)
            multimodal = dict(existing) if isinstance(existing, Mapping) else {}
            multimodal.setdefault("enable_video", enable_video)
            merged["multimodal"] = multimodal
            return merged
        return memorize_config

    @staticmethod
    def _is_missing_api_key(api_key: str | None) -> bool:
        if api_key is None:
            return True
        key = api_key.strip()
        if not key:
            return True
        return key in {
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "SILICONFLOW_API_KEY",
            "XAI_API_KEY",
        }

    @staticmethod
    def _infer_api_key_env(cfg: LLMConfig) -> str | None:
        key = (cfg.api_key or "").strip()
        if key in {"DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"}:
            return key
        base_url = (cfg.base_url or "").lower()
        if "deepseek" in base_url:
            return "DEEPSEEK_API_KEY"
        if "siliconflow" in base_url:
            return "SILICONFLOW_API_KEY"
        if "x.ai" in base_url or "xai" in base_url:
            return "XAI_API_KEY"
        if "openai" in base_url:
            return "OPENAI_API_KEY"
        return None

    def _warn_missing_llm_api_keys(self) -> None:
        for name, cfg in self.llm_profiles.profiles.items():
            if not self._is_missing_api_key(cfg.api_key):
                continue
            env_hint = self._infer_api_key_env(cfg)
            if env_hint is None:
                continue
            logger.warning(
                "LLM profile '%s' missing API key (provider=%s, base_url=%s). Set %s or pass api_key in llm_profiles.",
                name,
                cfg.provider,
                cfg.base_url,
                env_hint,
            )

    def _init_llm_client(self, config: LLMConfig | None = None) -> Any:
        """Initialize LLM client based on configuration."""
        cfg = config or self.llm_config
        backend = cfg.client_backend
        if backend == "sdk":
            from memu.llm.openai_sdk import OpenAISDKClient

            return OpenAISDKClient(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                chat_model=cfg.chat_model,
                embed_model=cfg.embed_model,
                embed_batch_size=cfg.embed_batch_size,
            )
        elif backend == "httpx":
            return HTTPLLMClient(
                base_url=cfg.base_url,
                api_key=cfg.api_key,
                chat_model=cfg.chat_model,
                provider=cfg.provider,
                endpoint_overrides=cfg.endpoint_overrides,
                embed_model=cfg.embed_model,
            )
        elif backend == "lazyllm_backend":
            from memu.llm.lazyllm_client import LazyLLMClient

            return LazyLLMClient(
                llm_source=cfg.lazyllm_source.llm_source or cfg.lazyllm_source.source,
                vlm_source=cfg.lazyllm_source.vlm_source or cfg.lazyllm_source.source,
                embed_source=cfg.lazyllm_source.embed_source or cfg.lazyllm_source.source,
                stt_source=cfg.lazyllm_source.stt_source or cfg.lazyllm_source.source,
                chat_model=cfg.chat_model,
                embed_model=cfg.embed_model,
                vlm_model=cfg.lazyllm_source.vlm_model,
                stt_model=cfg.lazyllm_source.stt_model,
            )
        else:
            msg = f"Unknown llm_client_backend '{cfg.client_backend}'"
            raise ValueError(msg)

    def _get_llm_base_client(self, profile: str | None = None) -> Any:
        """
        Lazily initialize and cache LLM clients per profile to avoid eager network setup.
        """
        name = profile or "default"
        client = self._llm_clients.get(name)
        if client is not None:
            return client
        cfg: LLMConfig | None = self.llm_profiles.profiles.get(name)
        if cfg is None:
            msg = f"Unknown llm profile '{name}'"
            raise KeyError(msg)
        client = self._init_llm_client(cfg)
        self._llm_clients[name] = client
        return client

    @staticmethod
    def _llm_call_metadata(profile: str, step_context: Mapping[str, Any] | None) -> LLMCallMetadata:
        if not isinstance(step_context, Mapping):
            return LLMCallMetadata(profile)
        operation = None
        for key in ("operation", "workflow_name"):
            value = step_context.get(key)
            if isinstance(value, str) and value.strip():
                operation = value.strip()
                break
        step_id = step_context.get("step_id") if isinstance(step_context.get("step_id"), str) else None
        trace_id = step_context.get("trace_id") if isinstance(step_context.get("trace_id"), str) else None
        tags = step_context.get("tags") if isinstance(step_context.get("tags"), Mapping) else None
        return LLMCallMetadata(profile=profile, operation=operation, step_id=step_id, trace_id=trace_id, tags=tags)

    def _wrap_llm_client(
        self,
        client: Any,
        *,
        profile: str | None = None,
        step_context: Mapping[str, Any] | None = None,
    ) -> Any:
        cfg: LLMConfig | None = self.llm_profiles.profiles.get(profile or "default")
        provider = cfg.provider if cfg is not None else None
        metadata = self._llm_call_metadata(profile or "default", step_context)
        return LLMClientWrapper(
            client,
            registry=self._llm_interceptors,
            metadata=metadata,
            provider=provider,
            chat_model=getattr(client, "chat_model", None),
            embed_model=getattr(client, "embed_model", None),
        )

    def _get_llm_client(self, profile: str | None = None, step_context: Mapping[str, Any] | None = None) -> Any:
        base_client = self._get_llm_base_client(profile)
        return self._wrap_llm_client(base_client, profile=profile, step_context=step_context)

    @property
    def llm_client(self) -> Any:
        """Default LLM client (lazy)."""
        return self._get_llm_client()

    @property
    def workflow_runner(self) -> WorkflowRunner:
        """Current workflow runner backend."""
        return self._workflow_runner

    @staticmethod
    def _llm_profile_from_context(
        step_context: Mapping[str, Any] | None, task: Literal["chat", "embedding"] = "chat"
    ) -> str | None:
        if not isinstance(step_context, Mapping):
            return None
        step_cfg = step_context.get("step_config")
        if not isinstance(step_cfg, Mapping):
            return None
        if task == "chat":
            profile = step_cfg.get("chat_llm_profile", step_cfg.get("llm_profile"))
        elif task == "embedding":
            profile = step_cfg.get("embed_llm_profile", step_cfg.get("llm_profile"))
        else:
            raise ValueError(task)
        if isinstance(profile, str) and profile.strip():
            return profile.strip()
        return None

    def _get_step_llm_client(self, step_context: Mapping[str, Any] | None) -> Any:
        profile = self._llm_profile_from_context(step_context, task="chat") or "default"
        return self._get_llm_client(profile, step_context=step_context)

    def _get_step_embedding_client(self, step_context: Mapping[str, Any] | None) -> Any:
        profile = self._llm_profile_from_context(step_context, task="embedding") or "embedding"
        return self._get_llm_client(profile, step_context=step_context)

    def intercept_before_llm_call(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        priority: int = 0,
        where: Mapping[str, Any] | Callable[..., Any] | None = None,
    ) -> LLMInterceptorHandle:
        return self._llm_interceptors.register_before(fn, name=name, priority=priority, where=where)

    def intercept_after_llm_call(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        priority: int = 0,
        where: Mapping[str, Any] | Callable[..., Any] | None = None,
    ) -> LLMInterceptorHandle:
        return self._llm_interceptors.register_after(fn, name=name, priority=priority, where=where)

    def intercept_on_error_llm_call(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        priority: int = 0,
        where: Mapping[str, Any] | Callable[..., Any] | None = None,
    ) -> LLMInterceptorHandle:
        return self._llm_interceptors.register_on_error(fn, name=name, priority=priority, where=where)

    def intercept_before_workflow_step(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
    ) -> WorkflowInterceptorHandle:
        """
        Register an interceptor to be called before each workflow step.

        The interceptor receives (step_context: WorkflowStepContext, state: WorkflowState).
        """
        return self._workflow_interceptors.register_before(fn, name=name)

    def intercept_after_workflow_step(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
    ) -> WorkflowInterceptorHandle:
        """
        Register an interceptor to be called after each workflow step.

        The interceptor receives (step_context: WorkflowStepContext, state: WorkflowState).
        """
        return self._workflow_interceptors.register_after(fn, name=name)

    def intercept_on_error_workflow_step(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
    ) -> WorkflowInterceptorHandle:
        """
        Register an interceptor to be called when a workflow step raises an exception.

        The interceptor receives (step_context: WorkflowStepContext, state: WorkflowState, error: Exception).
        """
        return self._workflow_interceptors.register_on_error(fn, name=name)

    def _get_context(self) -> Context:
        return self._context

    def _get_database(self) -> Database:
        return self.database

    async def health(
        self,
        *,
        user: dict[str, Any] | None = None,
        include_counts: bool = False,
    ) -> dict[str, Any]:
        """
        Lightweight health/status check for service readiness.

        Does not trigger category initialization or any LLM calls.
        """
        restart_status = self._read_restart_marker()
        status: dict[str, Any] = {
            "ok": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "db": {
                "provider": self.database_config.metadata_store.provider,
                "ok": True,
            },
            "categories": {
                "configured": len(self.category_configs),
                "initialized": self._context.categories_ready,
                "init_in_progress": self._context.category_init_task is not None,
            },
            "restart_required": restart_status.get("restart_required", False),
        }
        if "reason" in restart_status:
            status["restart_reason"] = restart_status["reason"]
        if "requested_at" in restart_status:
            status["restart_requested_at"] = restart_status["requested_at"]

        try:
            store = self._get_database()
            where_filters: dict[str, Any] | None = None
            if user:
                where_filters = self._normalize_where(user)
            categories = store.memory_category_repo.list_categories(where_filters)
            if include_counts:
                items = store.memory_item_repo.list_items(where_filters)
                status["counts"] = {"categories": len(categories), "items": len(items)}
        except Exception as exc:
            status["ok"] = False
            status["db"]["ok"] = False
            status["error"] = str(exc)

        return status

    def mark_restart_required(self, *, reason: str | None = None) -> dict[str, Any]:
        payload = {
            "restart_required": True,
            "reason": reason or "config_updated",
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_restart_marker(payload)
        return payload

    def is_restart_required(self) -> bool:
        return bool(self._read_restart_marker().get("restart_required", False))

    def clear_restart_required(self) -> bool:
        payload = {
            "restart_required": False,
            "reason": "cleared",
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        return self._write_restart_marker(payload)

    def _read_restart_marker(self) -> dict[str, Any]:
        meta_repo = getattr(self._get_database(), "meta_repo", None)
        if meta_repo is not None:
            try:
                data = meta_repo.get_meta("restart_required")
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                logger.warning("Failed to read restart marker from DB: %s", exc)

        file_data = self._read_restart_marker_file()
        if file_data is not None:
            if meta_repo is not None:
                try:
                    meta_repo.set_meta("restart_required", file_data)
                except Exception as exc:
                    logger.warning("Failed to sync restart marker into DB: %s", exc)
            return file_data
        return {"restart_required": False}

    def _read_restart_marker_file(self) -> dict[str, Any] | None:
        path = resolve_restart_marker_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
            return {"restart_required": True}
        except Exception as exc:
            logger.warning("Failed to read restart marker at %s: %s", path, exc)
            return {"restart_required": True}

    def _write_restart_marker(self, payload: dict[str, Any]) -> bool:
        wrote_db = False
        meta_repo = getattr(self._get_database(), "meta_repo", None)
        if meta_repo is not None:
            try:
                meta_repo.set_meta("restart_required", payload)
                wrote_db = True
            except Exception as exc:
                logger.warning("Failed to write restart marker to DB: %s", exc)

        wrote_file = self._write_restart_marker_file(payload)
        return wrote_db or wrote_file

    def _write_restart_marker_file(self, payload: dict[str, Any]) -> bool:
        path = resolve_restart_marker_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return True
        except Exception as exc:
            logger.warning("Failed to write restart marker at %s: %s", path, exc)
            return False

    def _provider_summary(self) -> dict[str, Any]:
        vector_provider = None
        if self.database_config.vector_index:
            vector_provider = self.database_config.vector_index.provider
        return {
            "llm_profiles": list(self.llm_profiles.profiles.keys()),
            "storage": {
                "metadata_store": self.database_config.metadata_store.provider,
                "vector_index": vector_provider,
            },
        }

    def _register_pipelines(self) -> None:
        memo_workflow = self._build_memorize_workflow()
        memo_initial_keys = self._list_memorize_initial_keys()
        self._pipelines.register("memorize", memo_workflow, initial_state_keys=memo_initial_keys)
        rag_workflow = self._build_rag_retrieve_workflow()
        retrieve_initial_keys = self._list_retrieve_initial_keys()
        self._pipelines.register("retrieve_rag", rag_workflow, initial_state_keys=retrieve_initial_keys)
        llm_workflow = self._build_llm_retrieve_workflow()
        self._pipelines.register("retrieve_llm", llm_workflow, initial_state_keys=retrieve_initial_keys)
        patch_create_workflow = self._build_create_memory_item_workflow()
        patch_create_initial_keys = CRUDMixin._list_create_memory_item_initial_keys()
        self._pipelines.register("patch_create", patch_create_workflow, initial_state_keys=patch_create_initial_keys)
        patch_update_workflow = self._build_update_memory_item_workflow()
        patch_update_initial_keys = CRUDMixin._list_update_memory_item_initial_keys()
        self._pipelines.register("patch_update", patch_update_workflow, initial_state_keys=patch_update_initial_keys)
        patch_delete_workflow = self._build_delete_memory_item_workflow()
        patch_delete_initial_keys = CRUDMixin._list_delete_memory_item_initial_keys()
        self._pipelines.register("patch_delete", patch_delete_workflow, initial_state_keys=patch_delete_initial_keys)
        crud_list_items_workflow = self._build_list_memory_items_workflow()
        crud_list_memories_initial_keys = CRUDMixin._list_list_memories_initial_keys()
        self._pipelines.register(
            "crud_list_memory_items", crud_list_items_workflow, initial_state_keys=crud_list_memories_initial_keys
        )
        crud_list_categories_workflow = self._build_list_memory_categories_workflow()
        self._pipelines.register(
            "crud_list_memory_categories",
            crud_list_categories_workflow,
            initial_state_keys=crud_list_memories_initial_keys,
        )
        crud_clear_memory_workflow = self._build_clear_memory_workflow()
        crud_clear_memory_initial_keys = CRUDMixin._list_clear_memories_initial_keys()
        self._pipelines.register(
            "crud_clear_memory", crud_clear_memory_workflow, initial_state_keys=crud_clear_memory_initial_keys
        )

    async def _run_workflow(self, workflow_name: str, initial_state: WorkflowState) -> WorkflowState:
        """Execute a workflow through the configured runner backend."""
        steps = self._pipelines.build(workflow_name)
        runner_context = {"workflow_name": workflow_name}
        return await self._workflow_runner.run(
            workflow_name,
            steps,
            initial_state,
            runner_context,
            interceptor_registry=self._workflow_interceptors,
        )

    @staticmethod
    def _extract_json_blob(raw: str) -> str:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            msg = "No JSON object found"
            raise ValueError(msg)
        return raw[start : end + 1]

    @staticmethod
    def _escape_prompt_value(value: str) -> str:
        return value.replace("{", "{{").replace("}", "}}")

    def _model_dump_without_embeddings(self, obj: BaseModel) -> dict[str, Any]:
        data = obj.model_dump(exclude={"embedding"})
        return data

    @staticmethod
    def _validate_config(
        config: Mapping[str, Any] | BaseModel | None,
        model_type: type[TConfigModel],
    ) -> TConfigModel:
        if isinstance(config, model_type):
            return config
        if config is None:
            return model_type()
        return model_type.model_validate(config)

    def configure_pipeline(self, *, step_id: str, configs: Mapping[str, Any], pipeline: str = "memorize") -> int:
        revision = self._pipelines.config_step(pipeline, step_id, dict(configs))
        return revision

    def insert_step_after(
        self,
        *,
        target_step_id: str,
        new_step: WorkflowStep,
        pipeline: str = "memorize",
    ) -> int:
        revision = self._pipelines.insert_after(pipeline, target_step_id, new_step)
        return revision

    def insert_step_before(
        self,
        *,
        target_step_id: str,
        new_step: WorkflowStep,
        pipeline: str = "memorize",
    ) -> int:
        revision = self._pipelines.insert_before(pipeline, target_step_id, new_step)
        return revision

    def replace_step(
        self,
        *,
        target_step_id: str,
        new_step: WorkflowStep,
        pipeline: str = "memorize",
    ) -> int:
        revision = self._pipelines.replace_step(pipeline, target_step_id, new_step)
        return revision

    def remove_step(self, *, target_step_id: str, pipeline: str = "memorize") -> int:
        revision = self._pipelines.remove_step(pipeline, target_step_id)
        return revision
