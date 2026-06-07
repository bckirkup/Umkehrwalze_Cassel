from __future__ import annotations

import tempfile
from pathlib import Path

from revprint.project_store import ProjectStore


def test_project_store_add_and_lookup_volume() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        corpus = root / "corpus"
        corpus.mkdir()
        (corpus / "p_0001.jpg").write_bytes(b"\xff\xd8\xff")
        db = root / "projects.sqlite"
        store = ProjectStore(db)
        store.init_schema()

        pid = store.upsert_project("Archive A", corpus, notes="demo")
        vid = store.add_volume(pid, "Vol 1", corpus, processing_profile="forensic")
        assert pid
        assert vid

        projects = store.list_projects()
        assert len(projects) == 1
        assert projects[0].slug == "archive-a"
        volumes = store.list_volumes(pid)
        assert len(volumes) == 1
        assert volumes[0].slug == "vol-1"
        assert volumes[0].page_count == 1
        rec = store.get_volume("archive-a", "vol-1")
        assert rec is not None
        assert rec.processing_profile == "forensic"
