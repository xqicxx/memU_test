import pytest

from memu.app import MemoryService


@pytest.mark.asyncio
async def test_health_status_basic() -> None:
    service = MemoryService(
        database_config={"metadata_store": {"provider": "inmemory"}},
    )

    status = await service.health()

    assert status["ok"] is True
    assert status["db"]["ok"] is True
    assert "categories" in status
    assert status["categories"]["configured"] >= 0


@pytest.mark.asyncio
async def test_health_status_with_counts() -> None:
    service = MemoryService(
        database_config={"metadata_store": {"provider": "inmemory"}},
    )

    status = await service.health(include_counts=True)

    assert status["ok"] is True
    assert "counts" in status
    assert "items" in status["counts"]
    assert "categories" in status["counts"]
