from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Generator


class JobState(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Job:
    id: str
    source_path: str
    state: JobState
    error: str | None
    meta_json: str | None
    created_at: float
    updated_at: float
    # Optional: token or API cost accounting (float units; meaning up to the app)
    cost_units: float | None = None

    @property
    def meta(self) -> dict[str, Any]:
        if not self.meta_json:
            return {}
        return json.loads(self.meta_json)


class JobStore:
    """SQLite-backed job queue. Safe for a single process writer; add WAL for concurrency later."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    state TEXT NOT NULL,
                    error TEXT,
                    meta_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    cost_units REAL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_state
                    ON jobs (state);
                """
            )

    def upsert_file(self, source_path: Path, state: JobState = JobState.PENDING) -> str:
        """Register a file if missing; return job id."""
        sp = str(Path(source_path).resolve())
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM jobs WHERE source_path = ?",
                (sp,),
            ).fetchone()
            if row:
                return str(row[0])
            job_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO jobs
                    (id, source_path, state, error, meta_json, created_at, updated_at, cost_units)
                VALUES (?, ?, ?, NULL, NULL, ?, ?, NULL)
                """,
                (job_id, sp, state.value, now, now),
            )
            return job_id

    def update_state(
        self,
        job_id: str,
        state: JobState,
        error: str | None = None,
        meta: dict[str, Any] | None = None,
        cost_units: float | None = None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            if meta is not None:
                conn.execute(
                    """
                    UPDATE jobs
                    SET state = ?, error = ?, meta_json = ?, updated_at = ?, cost_units = COALESCE(?, cost_units)
                    WHERE id = ?
                    """,
                    (state.value, error, json.dumps(meta), now, cost_units, job_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs
                    SET state = ?, error = ?, updated_at = ?, cost_units = COALESCE(?, cost_units)
                    WHERE id = ?
                    """,
                    (state.value, error, now, cost_units, job_id),
                )

    def register_scan(self, files: list[Path]) -> int:
        """Create pending jobs for each file; existing paths keep their current row (upsert is insert-only)."""
        for f in files:
            self.upsert_file(f, JobState.PENDING)
        return len(files)

    def list_all(self) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, source_path, state, error, meta_json, created_at, updated_at, cost_units
                FROM jobs ORDER BY source_path
                """
            ).fetchall()
        return [self._row_to_job(r) for r in rows]

    def list_by_state(self, state: JobState | None = None) -> list[Job]:
        with self._connect() as conn:
            if state is None:
                rows = conn.execute(
                    """
                    SELECT id, source_path, state, error, meta_json, created_at, updated_at, cost_units
                    FROM jobs ORDER BY source_path
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, source_path, state, error, meta_json, created_at, updated_at, cost_units
                    FROM jobs WHERE state = ? ORDER BY source_path
                    """,
                    (state.value,),
                ).fetchall()
        return [self._row_to_job(r) for r in rows]

    @staticmethod
    def _row_to_job(row: tuple[Any, ...]) -> Job:
        return Job(
            id=str(row[0]),
            source_path=str(row[1]),
            state=JobState(str(row[2])),
            error=row[3],
            meta_json=row[4],
            created_at=float(row[5]),
            updated_at=float(row[6]),
            cost_units=float(row[7]) if row[7] is not None else None,
        )

    def count_by_state(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT state, COUNT(*) FROM jobs GROUP BY state"
            ).fetchall()
        return {str(s): int(c) for s, c in rows}
