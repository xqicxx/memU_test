from __future__ import annotations

import asyncio
import json

import typer

from memu.app import MemoryService

app = typer.Typer()
service = MemoryService()


def _parse_bool(value: str) -> bool:
    norm = value.strip().lower()
    if norm in {"1", "true", "yes", "y", "on"}:
        return True
    if norm in {"0", "false", "no", "n", "off"}:
        return False
    raise typer.BadParameter("restart_required must be true/false")

@app.command()
def health() -> None:
    data = asyncio.run(service.health(include_counts=True))
    print(json.dumps(data, ensure_ascii=False, indent=2))


@app.command()
def meta_get() -> None:
    data = asyncio.run(service.health())
    print(
        json.dumps(
            {
                "restart_required": data.get("restart_required", False),
                "restart_reason": data.get("restart_reason"),
                "restart_requested_at": data.get("restart_requested_at"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def meta_set(
    restart_required: str = typer.Option("true", "--restart-required", help="true/false"),
    reason: str = typer.Option("config_updated", "--reason"),
) -> None:
    flag = _parse_bool(restart_required)
    if flag:
        data = service.mark_restart_required(reason=reason)
    else:
        service.clear_restart_required()
        data = {"restart_required": False, "reason": "cleared"}
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
