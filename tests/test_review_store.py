from __future__ import annotations

import json
import tempfile
from pathlib import Path

from revprint.review_store import ReviewStore


def test_review_store_add_list_export() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        db = root / "reviews.sqlite"
        store = ReviewStore(db)
        store.init_schema()
        rid = store.add_decision(
            project_slug="archive-a",
            volume_slug="vol-1",
            run_id="run-1",
            page_stem="page_0001",
            artifact_type="edge_candidate",
            artifact_path=str(root / "a.png"),
            decision="accept",
            notes="looks good",
        )
        assert rid
        rows = store.list_decisions(project_slug="archive-a", volume_slug="vol-1", run_id="run-1")
        assert len(rows) == 1
        assert rows[0].decision == "accept"
        out = store.export_jsonl(root / "labels.jsonl", project_slug="archive-a", volume_slug="vol-1")
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["artifact_type"] == "edge_candidate"
