from __future__ import annotations

import asyncio
import time
from pathlib import Path

from memu.app import MemoryService

CONVERSATION_PATH = Path(__file__).with_name("locomo_simple_conversation.json")

TEST_CASES = [
    {"id": "T01", "query": "他住在哪里？", "expected_keywords": ["深圳南山"]},
    {"id": "T02", "query": "他最喜欢的咖啡是什么？", "expected_keywords": ["燕麦拿铁"]},
    {"id": "T03", "query": "他对什么过敏？", "expected_keywords": ["花生"]},
    {"id": "T04", "query": "他家里的猫叫什么名字？", "expected_keywords": ["摩卡"]},
    {"id": "T05", "query": "他的生日是哪一天？", "expected_keywords": ["1994年7月12日", "7月12日"]},
    {"id": "T06", "query": "他在哪家公司工作，做什么？", "expected_keywords": ["星云科技", "产品经理"]},
    {"id": "T07", "query": "他平时怎么通勤？", "expected_keywords": ["骑电动车", "电动车"]},
    {"id": "T08", "query": "他哪天去成都出差？", "expected_keywords": ["2026年3月10日", "3月10日"]},
    {"id": "T09", "query": "他最讨厌吃什么？", "expected_keywords": ["香菜"]},
    {"id": "T10", "query": "他最常用的编程语言是什么？", "expected_keywords": ["Python"]},
]


def _collect_haystack(result: dict) -> str:
    items = result.get("items", []) or []
    categories = result.get("categories", []) or []
    lines = []
    for item in items:
        summary = item.get("summary", "")
        if summary:
            lines.append(str(summary))
    for cat in categories:
        summary = cat.get("summary") or cat.get("description") or ""
        if summary:
            lines.append(str(summary))
    return " ".join(lines)


async def run() -> int:
    if not CONVERSATION_PATH.exists():
        raise FileNotFoundError(f"Conversation file not found: {CONVERSATION_PATH}")

    service = MemoryService(
        database_config={"metadata_store": {"provider": "inmemory"}},
        retrieve_config={
            "method": "rag",
            "route_intention": False,
            "sufficiency_check": False,
            "category": {"enabled": False},
            "resource": {"enabled": False},
            "item": {"enabled": True, "top_k": 5},
        },
    )

    user_id = "locomo_simple_user"

    print("\n" + "=" * 60)
    print("[LOCOMO] Memorizing conversation...")
    print("=" * 60)
    mem_start = time.perf_counter()
    await service.memorize(
        resource_url=str(CONVERSATION_PATH),
        modality="conversation",
        user={"user_id": user_id},
    )
    mem_ms = (time.perf_counter() - mem_start) * 1000
    print(f"[LOCOMO] Memorize latency: {mem_ms:.0f} ms\n")

    passed = 0
    latencies = []
    for case in TEST_CASES:
        queries = [{"role": "user", "content": {"text": case["query"]}}]
        t0 = time.perf_counter()
        result = await service.retrieve(queries=queries, where={"user_id": user_id})
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        haystack = _collect_haystack(result)
        ok = any(keyword in haystack for keyword in case["expected_keywords"])
        if ok:
            passed += 1

        print(f"[{case['id']}] {case['query']}")
        print(f"  pass: {ok} | latency: {latency_ms:.0f} ms")
        if not ok:
            print(f"  expected: {case['expected_keywords']}")
            preview = haystack[:160] + ("..." if len(haystack) > 160 else "")
            print(f"  got: {preview or '<empty>'}")

    total = len(TEST_CASES)
    avg_latency = sum(latencies) / total if total else 0.0
    print("\n" + "-" * 60)
    print(f"[LOCOMO] Score: {passed}/{total}")
    print(f"[LOCOMO] Avg retrieval latency: {avg_latency:.0f} ms")
    print("-" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
