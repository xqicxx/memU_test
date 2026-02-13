import pytest

from memu.app import MemoryService


@pytest.mark.asyncio
async def test_restart_marker_roundtrip(tmp_path, monkeypatch) -> None:
    marker = tmp_path / "restart_required.json"
    monkeypatch.setenv("MEMU_RESTART_MARKER", str(marker))

    service = MemoryService(
        database_config={"metadata_store": {"provider": "inmemory"}},
    )

    result = service.mark_restart_required(reason="category_updated")
    assert result["restart_required"] is True
    assert marker.exists()

    status = await service.health()
    assert status["restart_required"] is True
    assert status["restart_reason"] == "category_updated"

    cleared = service.clear_restart_required()
    assert cleared is True
    assert marker.exists() is True
    data = marker.read_text(encoding="utf-8")
    assert "restart_required" in data
