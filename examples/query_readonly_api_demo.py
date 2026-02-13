import asyncio

from memu.app import MemoryService


async def main() -> None:
    service = MemoryService(
        database_config={"metadata_store": {"provider": "inmemory"}},
    )

    # 手工写入一些演示数据（走 repo，避免依赖 LLM）
    store = service.database
    user = {"user_id": "demo_user"}
    cat = store.memory_category_repo.get_or_create_category(
        name="preferences",
        description="User preferences",
        embedding=[0.0],
        user_data=user,
    )
    item = store.memory_item_repo.create_item(
        resource_id="r1",
        memory_type="profile",
        summary="我喜欢燕麦拿铁",
        embedding=[0.1],
        user_data=user,
    )
    store.category_item_repo.link_item_category(item.id, cat.id, user_data=user)

    # 只读查询 API
    items = await service.query_memory_items(where={"user_id": "demo_user", "category_id": cat.id})
    categories = await service.query_memory_categories(where={"user_id": "demo_user"})

    print("items:", items)
    print("categories:", categories)


if __name__ == "__main__":
    asyncio.run(main())
