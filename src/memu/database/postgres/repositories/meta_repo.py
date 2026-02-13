from __future__ import annotations

import json
from typing import Any

from memu.database.postgres.repositories.base import PostgresRepoBase
from memu.database.postgres.session import SessionManager
from memu.database.repositories.meta import MetaRepo
from memu.database.state import DatabaseState


class PostgresMetaRepo(PostgresRepoBase, MetaRepo):
    def __init__(
        self,
        *,
        state: DatabaseState,
        meta_model: type[Any],
        sqla_models: Any,
        sessions: SessionManager,
        scope_fields: list[str],
    ) -> None:
        super().__init__(state=state, sqla_models=sqla_models, sessions=sessions, scope_fields=scope_fields)
        self._meta_model = meta_model
        self.meta = self._state.meta

    def get_meta(self, key: str) -> dict[str, Any] | None:
        cached = self.meta.get(key)
        if cached is not None:
            return cached
        with self._sessions.session() as session:
            row = session.get(self._meta_model, key)
            if row is None:
                return None
            try:
                value = json.loads(row.value_json)
            except Exception:
                value = {"value": row.value_json}
            self.meta[key] = value
            return value

    def set_meta(self, key: str, value: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(value, ensure_ascii=False)
        now = self._now()
        with self._sessions.session() as session:
            row = session.get(self._meta_model, key)
            if row is None:
                row = self._meta_model(key=key, value_json=payload, updated_at=now)
                session.add(row)
            else:
                row.value_json = payload
                row.updated_at = now
                session.add(row)
            session.commit()
            session.refresh(row)
        self.meta[key] = dict(value)
        return self.meta[key]


__all__ = ["PostgresMetaRepo"]
