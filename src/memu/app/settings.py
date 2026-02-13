from collections.abc import Mapping
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, RootModel, StringConstraints, model_validator

logger = logging.getLogger(__name__)

from memu.prompts.category_summary import (
    DEFAULT_CATEGORY_SUMMARY_PROMPT_ORDINAL,
)
from memu.prompts.category_summary import (
    PROMPT as CATEGORY_SUMMARY_PROMPT,
)
from memu.prompts.memory_type import (
    DEFAULT_MEMORY_CUSTOM_PROMPT_ORDINAL,
    DEFAULT_MEMORY_TYPES,
)
from memu.prompts.memory_type import (
    PROMPTS as DEFAULT_MEMORY_TYPE_PROMPTS,
)


def normalize_value(v: str) -> str:
    if isinstance(v, str):
        return v.strip().lower()
    return v


Normalize = BeforeValidator(normalize_value)


def _default_memory_types() -> list[str]:
    return list(DEFAULT_MEMORY_TYPES)


def _default_memory_type_prompts() -> "dict[str, str | CustomPrompt]":
    return dict(DEFAULT_MEMORY_TYPE_PROMPTS)


class PromptBlock(BaseModel):
    label: str | None = None
    ordinal: int = Field(default=0)
    prompt: str | None = None


class CustomPrompt(RootModel[dict[str, PromptBlock]]):
    root: dict[str, PromptBlock] = Field(default_factory=dict)

    def get(self, key: str, default: PromptBlock | None = None) -> PromptBlock | None:
        return self.root.get(key, default)

    def items(self) -> list[tuple[str, PromptBlock]]:
        return list(self.root.items())


def complete_prompt_blocks(prompt: CustomPrompt, default_blocks: Mapping[str, int]) -> CustomPrompt:
    for key, ordinal in default_blocks.items():
        if key not in prompt.root:
            prompt.root[key] = PromptBlock(ordinal=ordinal)
    return prompt


CompleteMemoryTypePrompt = AfterValidator(lambda v: complete_prompt_blocks(v, DEFAULT_MEMORY_CUSTOM_PROMPT_ORDINAL))


CompleteCategoryPrompt = AfterValidator(lambda v: complete_prompt_blocks(v, DEFAULT_CATEGORY_SUMMARY_PROMPT_ORDINAL))


class CategoryConfig(BaseModel):
    name: str
    description: str = ""
    target_length: int | None = None
    summary_prompt: str | Annotated[CustomPrompt, CompleteCategoryPrompt] | None = None


def _default_memory_categories() -> list[CategoryConfig]:
    return [
        CategoryConfig.model_validate(cat)
        for cat in (
            {"name": "personal_info", "description": "Personal information about the user"},
            {"name": "preferences", "description": "User preferences, likes and dislikes"},
            {"name": "relationships", "description": "Information about relationships with others"},
            {"name": "activities", "description": "Activities, hobbies, and interests"},
            {"name": "goals", "description": "Goals, aspirations, and objectives"},
            {"name": "experiences", "description": "Past experiences and events"},
            {"name": "knowledge", "description": "Knowledge, facts, and learned information"},
            {"name": "opinions", "description": "Opinions, viewpoints, and perspectives"},
            {"name": "habits", "description": "Habits, routines, and patterns"},
            {"name": "work_life", "description": "Work-related information and professional life"},
        )
    ]


MEMU_CATEGORY_CONFIG_ENV = "MEMU_CATEGORY_CONFIG"
MEMU_CATEGORY_CONFIG_DEFAULT = Path("config") / "memory_categories.json"
MEMU_CONFIG_ENV = "MEMU_CONFIG_PATH"  # Backward compatibility
MEMU_CONFIG_DEFAULT = "config.json"  # Backward compatibility
MEMU_ENABLE_VIDEO_ENV = "MEMU_ENABLE_VIDEO"
MEMU_RESTART_MARKER_ENV = "MEMU_RESTART_MARKER"
MEMU_RESTART_MARKER_DEFAULT = Path("config") / "restart_required.json"


def resolve_memory_category_config_path() -> Path:
    """
    Resolve memory category config path.

    NanoBot can update this file and restart MemU to apply changes.
    """
    override = os.getenv(MEMU_CATEGORY_CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    return Path(MEMU_CATEGORY_CONFIG_DEFAULT).expanduser()


def resolve_memu_config_path() -> Path:
    """Backward-compatible alias for older config path usage."""
    override = os.getenv(MEMU_CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    return Path(MEMU_CONFIG_DEFAULT).expanduser()


def resolve_restart_marker_path() -> Path:
    override = os.getenv(MEMU_RESTART_MARKER_ENV)
    if override:
        return Path(override).expanduser()
    return Path(MEMU_RESTART_MARKER_DEFAULT).expanduser()


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Failed to load JSON config from %s: %s", path, exc)
        return None


def _parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    norm = value.strip().lower()
    if norm in {"1", "true", "yes", "y", "on"}:
        return True
    if norm in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _validate_category_entries(raw: list[Any]) -> list["CategoryConfig"]:
    categories: list[CategoryConfig] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw):
        try:
            cat = CategoryConfig.model_validate(item)
        except Exception as exc:
            logger.warning("Invalid memory_categories[%s]: %s", idx, exc)
            continue
        name = (cat.name or "").strip()
        if not name:
            logger.warning("Invalid memory_categories[%s]: name is required", idx)
            continue
        key = name.lower()
        if key in seen:
            logger.warning("Duplicate memory category name ignored: %s", name)
            continue
        seen.add(key)
        desc = (cat.description or "").strip()
        cat = cat.model_copy(update={"name": name, "description": desc})
        categories.append(cat)
    return categories


def load_memory_categories_from_config() -> list["CategoryConfig"] | None:
    """
    Load memory_categories from JSON config.

    Supported:
    - config/memory_categories.json (default)
    - MEMU_CATEGORY_CONFIG override
    - Fallback to legacy config.json (MEMU_CONFIG_PATH)
    """
    data = _load_json_file(resolve_memory_category_config_path())
    if data is None:
        # Backward compatibility: support legacy config.json shape
        legacy = _load_json_file(resolve_memu_config_path())
        data = legacy if legacy is not None else None

    if data is None:
        return None

    raw: Any = None
    if isinstance(data, dict):
        raw = data.get("memory_categories")
    elif isinstance(data, list):
        raw = data
    else:
        logger.warning("memu category config must be a list or object")
        return None

    if raw is None:
        return None
    if not isinstance(raw, list):
        logger.warning("memu category config memory_categories must be a list")
        return None
    categories = _validate_category_entries(raw)
    if not categories:
        logger.warning("No valid memory categories found in config")
        return None
    return categories


def load_enable_video_from_config() -> bool | None:
    """Load multimodal.enable_video from config or env override."""
    env_value = _parse_env_bool(os.getenv(MEMU_ENABLE_VIDEO_ENV))
    if env_value is not None:
        return env_value
    data = _load_json_file(resolve_memu_config_path())
    if not isinstance(data, dict):
        return None
    multimodal = data.get("multimodal")
    if not isinstance(multimodal, dict):
        return None
    raw = multimodal.get("enable_video")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return _parse_env_bool(raw)
    return None


class MultimodalConfig(BaseModel):
    enable_video: bool = Field(default=False, description="Enable video modality processing.")


class LazyLLMSource(BaseModel):
    source: str | None = Field(default=None, description="default source for lazyllm client backend")
    llm_source: str | None = Field(default=None, description="LLM source for lazyllm client backend")
    embed_source: str | None = Field(default=None, description="Embedding source for lazyllm client backend")
    vlm_source: str | None = Field(default=None, description="VLM source for lazyllm client backend")
    stt_source: str | None = Field(default=None, description="STT source for lazyllm client backend")
    vlm_model: str = Field(default="qwen-vl-plus", description="Vision language model for lazyllm client backend")
    stt_model: str = Field(default="qwen-audio-turbo", description="Speech-to-text model for lazyllm client backend")


class LLMConfig(BaseModel):
    provider: str = Field(
        default="openai",
        description="Identifier for the LLM provider implementation (used by HTTP client backend).",
    )
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="OPENAI_API_KEY")
    chat_model: str = Field(default="gpt-4o-mini")
    client_backend: str = Field(
        default="sdk",
        description="Which LLM client backend to use: 'httpx' (httpx), 'sdk' (official OpenAI), or 'lazyllm_backend' (for more LLM source like Qwen, Doubao, SIliconflow, etc.)",
    )
    lazyllm_source: LazyLLMSource = Field(default=LazyLLMSource())
    endpoint_overrides: dict[str, str] = Field(
        default_factory=dict,
        description="Optional overrides for HTTP endpoints (keys: 'chat'/'summary').",
    )
    embed_model: str = Field(
        default="text-embedding-3-small",
        description="Default embedding model used for vectorization.",
    )
    embed_batch_size: int = Field(
        default=1,
        description="Maximum batch size for embedding API calls (used by SDK client backends).",
    )

    @model_validator(mode="after")
    def set_provider_defaults(self) -> "LLMConfig":
        if self.provider == "grok":
            # If values match the OpenAI defaults, switch them to Grok defaults
            if self.base_url == "https://api.openai.com/v1":
                self.base_url = "https://api.x.ai/v1"
            if self.api_key == "OPENAI_API_KEY":
                self.api_key = "XAI_API_KEY"
            if self.chat_model == "gpt-4o-mini":
                self.chat_model = "grok-2-latest"
        return self


def _default_deepseek_config() -> LLMConfig:
    return LLMConfig(
        provider=os.getenv("DEEPSEEK_PROVIDER", "openai"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
        chat_model=os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
        client_backend=os.getenv("DEEPSEEK_CLIENT_BACKEND", "sdk"),
    )


def _default_siliconflow_embedding_config() -> LLMConfig:
    return LLMConfig(
        provider=os.getenv("SILICONFLOW_PROVIDER", "openai"),
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        api_key=os.getenv("SILICONFLOW_API_KEY", "SILICONFLOW_API_KEY"),
        chat_model=os.getenv("SILICONFLOW_CHAT_MODEL", os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")),
        embed_model=os.getenv("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3"),
        client_backend=os.getenv("SILICONFLOW_CLIENT_BACKEND", "sdk"),
    )


def _default_llm_profiles() -> dict[str, "LLMConfig"]:
    return {
        "default": _default_deepseek_config(),
        "embedding": _default_siliconflow_embedding_config(),
    }


class BlobConfig(BaseModel):
    provider: str = Field(default="local")
    resources_dir: str = Field(default="./data/resources")


class RetrieveCategoryConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to enable category retrieval.")
    top_k: int = Field(default=5, description="Total number of categories to retrieve.")


class RetrieveItemConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to enable item retrieval.")
    top_k: int = Field(default=5, description="Total number of items to retrieve.")
    # Reference-aware retrieval
    use_category_references: bool = Field(
        default=False,
        description="When category retrieval is insufficient, follow [ref:ITEM_ID] citations to fetch referenced items.",
    )
    # Salience-aware retrieval settings
    ranking: Literal["similarity", "salience"] = Field(
        default="salience",
        description="Ranking strategy: 'similarity' (cosine only) or 'salience' (weighted by reinforcement + recency).",
    )
    recency_decay_days: float = Field(
        default=30.0,
        description="Half-life in days for recency decay in salience scoring. After this many days, recency factor is ~0.5.",
    )


class RetrieveResourceConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to enable resource retrieval.")
    top_k: int = Field(default=5, description="Total number of resources to retrieve.")


class RetrieveConfig(BaseModel):
    """Configure retrieval behavior for `MemoryUser.retrieve`.

    Attributes:
        method: Retrieval strategy. Use "rag" for embedding-based vector search or
            "llm" to delegate ranking to the LLM.
        top_k: Maximum number of results to return per category (and per stage),
            controlling breadth of the retrieved context.
    """

    method: Annotated[Literal["rag", "llm"], Normalize] = "rag"
    # top_k: int = Field(
    #     default=5,
    #     description="Maximum number of results to return per category.",
    # )
    route_intention: bool = Field(
        default=True, description="Whether to route intention (judge needs retrieval & rewrite query)."
    )
    # route_intention_prompt: str = Field(default="", description="User prompt for route intention.")
    # route_intention_llm_profile: str = Field(default="default", description="LLM profile for route intention.")
    category: RetrieveCategoryConfig = Field(default=RetrieveCategoryConfig())
    item: RetrieveItemConfig = Field(default=RetrieveItemConfig())
    resource: RetrieveResourceConfig = Field(default=RetrieveResourceConfig())
    sufficiency_check: bool = Field(default=True, description="Whether to check sufficiency after each tier.")
    sufficiency_check_prompt: str = Field(default="", description="User prompt for sufficiency check.")
    sufficiency_check_llm_profile: str = Field(default="default", description="LLM profile for sufficiency check.")
    llm_ranking_llm_profile: str = Field(default="default", description="LLM profile for LLM ranking.")


class MemorizeConfig(BaseModel):
    category_assign_threshold: float = Field(default=0.25)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    multimodal_preprocess_prompts: dict[str, str | CustomPrompt] = Field(
        default_factory=dict,
        description="Optional mapping of modality -> preprocess system prompt.",
    )
    preprocess_llm_profile: str = Field(default="default", description="LLM profile for preprocess.")
    memory_types: list[str] = Field(
        default_factory=_default_memory_types,
        description="Ordered list of memory types (profile/event/knowledge/behavior by default).",
    )
    memory_type_prompts: dict[str, str | Annotated[CustomPrompt, CompleteMemoryTypePrompt]] = Field(
        default_factory=_default_memory_type_prompts,
        description="User prompt overrides for each memory type extraction.",
    )
    memory_extract_llm_profile: str = Field(default="default", description="LLM profile for memory extract.")
    memory_categories: list[CategoryConfig] = Field(
        default_factory=_default_memory_categories,
        description="Global memory category definitions embedded at service startup.",
    )
    # default_category_summary_prompt: str | CustomPrompt = Field(
    default_category_summary_prompt: str | Annotated[CustomPrompt, CompleteCategoryPrompt] = Field(
        default=CATEGORY_SUMMARY_PROMPT,
        description="Default system prompt for auto-generated category summaries.",
    )
    default_category_summary_target_length: int = Field(
        default=400,
        description="Target max length for auto-generated category summaries.",
    )
    category_update_llm_profile: str = Field(default="default", description="LLM profile for category summary.")
    # Reference tracking for category summaries
    enable_item_references: bool = Field(
        default=False,
        description="Enable inline [ref:ITEM_ID] citations in category summaries linking to source memory items.",
    )
    enable_item_reinforcement: bool = Field(
        default=True,
        description="Enable reinforcement tracking for memory items.",
    )


class PatchConfig(BaseModel):
    pass


class DefaultUserModel(BaseModel):
    user_id: str | None = None
    # Agent/session scoping for multi-agent and multi-session memory filtering
    # agent_id: str | None = None
    # session_id: str | None = None


class UserConfig(BaseModel):
    model: type[BaseModel] = Field(default=DefaultUserModel)


Key = Annotated[str, StringConstraints(min_length=1)]


class LLMProfilesConfig(RootModel[dict[Key, LLMConfig]]):
    root: dict[str, LLMConfig] = Field(default_factory=_default_llm_profiles)

    def get(self, key: str, default: LLMConfig | None = None) -> LLMConfig | None:
        return self.root.get(key, default)

    @model_validator(mode="before")
    @classmethod
    def ensure_default(cls, data: Any) -> Any:
        # if data is None:
        #     return {"default": LLMConfig()}
        # if isinstance(data, dict) and "default" not in data:
        #     data = dict(data)
        #     data["default"] = LLMConfig()
        # return data
        if data is None:
            data = {}
        elif isinstance(data, dict):
            data = dict(data)
        else:
            return data
        if "default" not in data:
            data["default"] = _default_deepseek_config()
        if "embedding" not in data:
            data["embedding"] = _default_siliconflow_embedding_config()
        return data

    @property
    def profiles(self) -> dict[str, LLMConfig]:
        return self.root

    @property
    def default(self) -> LLMConfig:
        return self.root.get("default", LLMConfig())


class MetadataStoreConfig(BaseModel):
    provider: Annotated[Literal["inmemory", "postgres", "sqlite"], Normalize] = "inmemory"
    ddl_mode: Annotated[Literal["create", "validate"], Normalize] = "create"
    dsn: str | None = Field(default=None, description="Database connection string (required for postgres/sqlite).")


class VectorIndexConfig(BaseModel):
    provider: Annotated[Literal["bruteforce", "pgvector", "none"], Normalize] = "bruteforce"
    dsn: str | None = Field(default=None, description="Postgres connection string when provider=pgvector.")


class DatabaseConfig(BaseModel):
    metadata_store: MetadataStoreConfig = Field(default_factory=MetadataStoreConfig)
    vector_index: VectorIndexConfig | None = Field(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.vector_index is None:
            if self.metadata_store.provider == "postgres":
                self.vector_index = VectorIndexConfig(provider="pgvector", dsn=self.metadata_store.dsn)
            else:
                self.vector_index = VectorIndexConfig(provider="bruteforce")
        elif self.vector_index.provider == "pgvector" and self.vector_index.dsn is None:
            self.vector_index = self.vector_index.model_copy(update={"dsn": self.metadata_store.dsn})
