from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator


@dataclass(frozen=True)
class ReviewDecision:
    id: str
    project_slug: str
    volume_slug: str
    run_id: str
    page_stem: str
    artifact_type: str
    artifact_path: str
    decision: str
    notes: str
    created_at: float

    def to_meta(self) -> dict[str, object]:
        return {
            "id": self.id,
            "project_slug": self.project_slug,
            "volume_slug": self.volume_slug,
            "run_id": self.run_id,
            "page_stem": self.page_stem,
            "artifact_type": self.artifact_type,
            "artifact_path": self.artifact_path,
            "decision": self.decision,
            "notes": self.notes,
            "created_at": self.created_at,
        }


class ReviewStore:
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
                CREATE TABLE IF NOT EXISTS review_decisions (
                    id TEXT PRIMARY KEY,
                    project_slug TEXT NOT NULL,
                    volume_slug TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    page_stem TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_review_by_run
                    ON review_decisions (project_slug, volume_slug, run_id, page_stem);
                """
            )

    def add_decision(
        self,
        *,
        project_slug: str,
        volume_slug: str,
        run_id: str,
        page_stem: str,
        artifact_type: str,
        artifact_path: str,
        decision: str,
        notes: str = "",
    ) -> str:
        rid = str(uuid.uuid4())
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO review_decisions
                    (id, project_slug, volume_slug, run_id, page_stem, artifact_type, artifact_path, decision, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    project_slug,
                    volume_slug,
                    run_id,
                    page_stem,
                    artifact_type,
                    artifact_path,
                    decision,
                    notes,
                    now,
                ),
            )
        return rid

    def list_decisions(self, *, project_slug: str, volume_slug: str, run_id: str | None = None) -> list[ReviewDecision]:
        with self._connect() as conn:
            if run_id is None:
                rows = conn.execute(
                    """
                    SELECT id, project_slug, volume_slug, run_id, page_stem, artifact_type, artifact_path, decision, notes, created_at
                    FROM review_decisions
                    WHERE project_slug = ? AND volume_slug = ?
                    ORDER BY created_at
                    """,
                    (project_slug, volume_slug),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, project_slug, volume_slug, run_id, page_stem, artifact_type, artifact_path, decision, notes, created_at
                    FROM review_decisions
                    WHERE project_slug = ? AND volume_slug = ? AND run_id = ?
                    ORDER BY created_at
                    """,
                    (project_slug, volume_slug, run_id),
                ).fetchall()
        return [
            ReviewDecision(
                id=str(r[0]),
                project_slug=str(r[1]),
                volume_slug=str(r[2]),
                run_id=str(r[3]),
                page_stem=str(r[4]),
                artifact_type=str(r[5]),
                artifact_path=str(r[6]),
                decision=str(r[7]),
                notes=str(r[8]),
                created_at=float(r[9]),
            )
            for r in rows
        ]

    def export_jsonl(self, output_path: Path, *, project_slug: str, volume_slug: str, run_id: str | None = None) -> Path:
        rows = self.list_decisions(project_slug=project_slug, volume_slug=volume_slug, run_id=run_id)
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = "\n".join(json.dumps(row.to_meta(), ensure_ascii=False) for row in rows)
        out.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
        return out
