import pendulum
import pytest

from memu.app import MemoryService


def _build_service() -> MemoryService:
    return MemoryService(
        database_config={"metadata_store": {"provider": "inmemory"}},
    )


def _seed(service: MemoryService):
    store = service.database
    user = {"user_id": "u1"}
    other = {"user_id": "u2"}

    cat = store.memory_category_repo.get_or_create_category(
        name="preferences",
        description="User preferences",
        embedding=[0.0],
        user_data=user,
    )

    item1 = store.memory_item_repo.create_item(
        resource_id="r1",
        memory_type="profile",
        summary="喜欢燕麦拿铁",
        embedding=[0.1],
        user_data=user,
    )
    item2 = store.memory_item_repo.create_item(
        resource_id="r2",
        memory_type="event",
        summary="昨天去跑步",
        embedding=[0.2],
        user_data=user,
    )
    _ = store.memory_item_repo.create_item(
        resource_id="r3",
        memory_type="profile",
        summary="其他用户的偏好",
        embedding=[0.3],
        user_data=other,
    )

    store.category_item_repo.link_item_category(item1.id, cat.id, user_data=user)

    item1.created_at = pendulum.datetime(2024, 1, 1, tz="UTC")
    item2.created_at = pendulum.datetime(2024, 1, 2, tz="UTC")

    return cat, item1, item2


@pytest.mark.asyncio
async def test_query_requires_user_scope() -> None:
    service = _build_service()
    with pytest.raises(ValueError):
        await service.query_memory_items(where={})


@pytest.mark.asyncio
async def test_query_item_filters_and_pagination() -> None:
    service = _build_service()
    _seed(service)

    items = await service.query_memory_items(where={"user_id": "u1", "memory_type": "profile"})
    assert len(items) == 1
    assert items[0]["memory_type"] == "profile"

    ordered = await service.query_memory_items(where={"user_id": "u1"}, order_by="-created_at")
    assert ordered[0]["summary"] == "昨天去跑步"

    paged = await service.query_memory_items(where={"user_id": "u1"}, order_by="-created_at", limit=1, offset=0)
    assert len(paged) == 1
    assert paged[0]["summary"] == "昨天去跑步"


@pytest.mark.asyncio
async def test_query_item_by_category() -> None:
    service = _build_service()
    cat, _, _ = _seed(service)

    items = await service.query_memory_items(where={"user_id": "u1", "category_id": cat.id})
    assert len(items) == 1
    assert items[0]["summary"] == "喜欢燕麦拿铁"


@pytest.mark.asyncio
async def test_query_categories_basic() -> None:
    service = _build_service()
    cat, _, _ = _seed(service)

    categories = await service.query_memory_categories(where={"user_id": "u1"})
    assert len(categories) == 1
    assert categories[0]["id"] == cat.id
