import pytest

from memu.app import MemoryService


@pytest.mark.asyncio
async def test_video_disabled_skips_pipeline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMU_ENABLE_VIDEO", "false")

    service = MemoryService(
        blob_config={"resources_dir": str(tmp_path / "resources")},
        database_config={
            "metadata_store": {"provider": "sqlite", "dsn": f"sqlite:///{tmp_path / 'memu_test.db'}"},
        },
    )

    result = await service.memorize(
        resource_url=str(tmp_path / "dummy.mp4"),
        modality="video",
        user={"user_id": "test_user"},
    )

    assert result.get("skipped") is True
    assert result.get("reason") == "video_disabled"
    assert result.get("resources") == []
    assert result.get("items") == []
