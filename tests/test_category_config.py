import json
from pathlib import Path

from memu.app import MemoryService


def test_load_categories_from_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "memory_categories.json"
    config_path.write_text(
        json.dumps(
            {
                "memory_categories": [
                    {"name": "personal_info", "description": "用户个人信息"},
                    {"name": "preferences", "description": "用户偏好"},
                    {"name": "work", "description": "职业/工作信息"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMU_CATEGORY_CONFIG", str(config_path))

    service = MemoryService(
        database_config={"metadata_store": {"provider": "inmemory"}},
    )

    names = [c.name for c in service.category_configs]
    assert names == ["personal_info", "preferences", "work"]


def test_fallback_to_default_categories(monkeypatch) -> None:
    monkeypatch.delenv("MEMU_CATEGORY_CONFIG", raising=False)
    service = MemoryService(
        database_config={"metadata_store": {"provider": "inmemory"}},
    )
    names = [c.name for c in service.category_configs]
    assert "personal_info" in names
