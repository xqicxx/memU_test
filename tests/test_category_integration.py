import json
import os
from pathlib import Path

import pytest

from memu.app import MemoryService


def _has_valid_key(value: str | None) -> bool:
    if not value:
        return False
    placeholder = {"DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "OPENAI_API_KEY"}
    return value not in placeholder


@pytest.mark.asyncio
async def test_memory_category_integration(tmp_path: Path, monkeypatch) -> None:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    silicon_key = os.getenv("SILICONFLOW_API_KEY")
    if not (_has_valid_key(deepseek_key) and _has_valid_key(silicon_key)):
        pytest.skip("Integration test requires DEEPSEEK_API_KEY and SILICONFLOW_API_KEY.")

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

    conversation = [
        {"role": "user", "content": "我喜欢燕麦拿铁。"},
        {"role": "assistant", "content": "好的，我记住了。"},
        {"role": "user", "content": "我在星云科技做产品经理。"},
        {"role": "assistant", "content": "了解。"},
    ]
    conv_path = tmp_path / "conv.json"
    conv_path.write_text(json.dumps(conversation, ensure_ascii=False), encoding="utf-8")

    service = MemoryService(
        blob_config={"resources_dir": str(tmp_path / "resources")},
        database_config={
            "metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{tmp_path / 'memu_test.db'}"},
        },
        retrieve_config={
            "method": "rag",
            "route_intention": False,
            "sufficiency_check": False,
            "category": {"enabled": True, "top_k": 5},
            "item": {"enabled": True, "top_k": 5},
            "resource": {"enabled": False},
        },
    )

    await service.memorize(
        resource_url=str(conv_path),
        modality="conversation",
        user={"user_id": "test_user"},
    )

    result = await service.retrieve(
        queries=[{"role": "user", "content": {"text": "他喜欢喝什么？"}}],
        where={"user_id": "test_user"},
    )

    category_names = [c["name"] for c in result.get("categories", [])]
    item_summaries = [i.get("summary", "") for i in result.get("items", [])]

    assert "preferences" in category_names
    if item_summaries:
        assert any(s.strip() for s in item_summaries)
