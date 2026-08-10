"""SQLite-backed WorkOrder store. Stdlib only -- no new dependencies."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional

from .work_order import OrgStatus, WorkOrder


_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_orders (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    assigned_role TEXT,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL
)
"""


class WorkOrderStore:
    """Thin wrapper around sqlite3 for WorkOrder persistence."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, wo: WorkOrder) -> None:
        payload = json.dumps(wo.to_dict())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO work_orders (id, status, assigned_role, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    assigned_role = excluded.assigned_role,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (wo.id, wo.status.value, wo.assigned_role, payload, wo.updated_at),
            )

    def load(self, wo_id: str) -> Optional[WorkOrder]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT payload FROM work_orders WHERE id = ?", (wo_id,)
            ).fetchone()
        if row is None:
            return None
        return WorkOrder.from_dict(json.loads(row["payload"]))

    def list_active(self) -> list[WorkOrder]:
        """Return WorkOrders that are not in a terminal state."""
        terminal = (OrgStatus.COMPLETED.value, OrgStatus.FAILED.value)
        placeholders = ",".join("?" * len(terminal))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT payload FROM work_orders WHERE status NOT IN ({placeholders})",
                terminal,
            ).fetchall()
        return [WorkOrder.from_dict(json.loads(r["payload"])) for r in rows]

    def all(self) -> list[WorkOrder]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM work_orders ORDER BY updated_at DESC"
            ).fetchall()
        return [WorkOrder.from_dict(json.loads(r["payload"])) for r in rows]
