from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator


def _slugify(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    out = "-".join(part for part in out.split("-") if part)
    return out or "item"


@dataclass(frozen=True)
class ProjectRecord:
    id: str
    name: str
    slug: str
    corpus_root: str
    notes: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class VolumeRecord:
    id: str
    project_id: str
    name: str
    slug: str
    folder_path: str
    page_count: int
    processing_profile: str
    created_at: float
    updated_at: float


class ProjectStore:
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
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    slug TEXT NOT NULL UNIQUE,
                    corpus_root TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS volumes (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    slug TEXT NOT NULL,
                    folder_path TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    processing_profile TEXT NOT NULL DEFAULT 'balanced',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(project_id, name),
                    UNIQUE(project_id, slug)
                );
                CREATE INDEX IF NOT EXISTS idx_volumes_project
                    ON volumes (project_id);
                """
            )

    def upsert_project(self, name: str, corpus_root: Path, notes: str = "") -> str:
        now = time.time()
        nm = name.strip()
        slug = _slugify(nm)
        root = str(Path(corpus_root).expanduser().resolve())
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM projects WHERE name = ?", (nm,)).fetchone()
            if row:
                project_id = str(row[0])
                conn.execute(
                    "UPDATE projects SET corpus_root = ?, notes = ?, updated_at = ? WHERE id = ?",
                    (root, notes, now, project_id),
                )
                return project_id
            project_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO projects (id, name, slug, corpus_root, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (project_id, nm, slug, root, notes, now, now),
            )
            return project_id

    def add_volume(
        self,
        project_id: str,
        name: str,
        folder_path: Path,
        processing_profile: str = "balanced",
    ) -> str:
        now = time.time()
        nm = name.strip()
        slug = _slugify(nm)
        folder = str(Path(folder_path).expanduser().resolve())
        page_count = 0
        p = Path(folder)
        if p.is_dir():
            page_count = len(
                [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg"}]
            )
        with self._connect() as conn:
            volume_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO volumes (id, project_id, name, slug, folder_path, page_count, processing_profile, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (volume_id, project_id, nm, slug, folder, page_count, processing_profile, now, now),
            )
            return volume_id

    def list_projects(self) -> list[ProjectRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, slug, corpus_root, notes, created_at, updated_at FROM projects ORDER BY name"
            ).fetchall()
        return [
            ProjectRecord(
                id=str(r[0]),
                name=str(r[1]),
                slug=str(r[2]),
                corpus_root=str(r[3]),
                notes=str(r[4]),
                created_at=float(r[5]),
                updated_at=float(r[6]),
            )
            for r in rows
        ]

    def list_volumes(self, project_id: str) -> list[VolumeRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, project_id, name, slug, folder_path, page_count, processing_profile, created_at, updated_at
                FROM volumes WHERE project_id = ? ORDER BY name
                """,
                (project_id,),
            ).fetchall()
        return [
            VolumeRecord(
                id=str(r[0]),
                project_id=str(r[1]),
                name=str(r[2]),
                slug=str(r[3]),
                folder_path=str(r[4]),
                page_count=int(r[5]),
                processing_profile=str(r[6]),
                created_at=float(r[7]),
                updated_at=float(r[8]),
            )
            for r in rows
        ]

    def get_volume(self, project_slug: str, volume_slug: str) -> VolumeRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT v.id, v.project_id, v.name, v.slug, v.folder_path, v.page_count, v.processing_profile, v.created_at, v.updated_at
                FROM volumes v
                JOIN projects p ON p.id = v.project_id
                WHERE p.slug = ? AND v.slug = ?
                """,
                (project_slug, volume_slug),
            ).fetchone()
        if row is None:
            return None
        return VolumeRecord(
            id=str(row[0]),
            project_id=str(row[1]),
            name=str(row[2]),
            slug=str(row[3]),
            folder_path=str(row[4]),
            page_count=int(row[5]),
            processing_profile=str(row[6]),
            created_at=float(row[7]),
            updated_at=float(row[8]),
        )
