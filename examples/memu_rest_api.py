from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from memu.app import MemoryService

app = FastAPI(title="MemU REST")

service = MemoryService()


class RestartRequest(BaseModel):
    restart_required: bool
    reason: str | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return await service.health(include_counts=True)


@app.get("/meta/restart_required")
async def get_restart_required() -> dict[str, Any]:
    status = await service.health()
    return {
        "restart_required": status.get("restart_required", False),
        "restart_reason": status.get("restart_reason"),
        "restart_requested_at": status.get("restart_requested_at"),
    }


@app.post("/meta/restart_required")
async def set_restart_required(payload: RestartRequest) -> dict[str, Any]:
    if payload.restart_required:
        return service.mark_restart_required(reason=payload.reason)
    cleared = service.clear_restart_required()
    if not cleared:
        raise HTTPException(status_code=500, detail="failed to clear restart flag")
    return {"restart_required": False, "reason": "cleared"}


# Future: read-only list endpoints
# @app.get("/memory/items")
# async def list_items(...):
#     return await service.query_memory_items(...)
