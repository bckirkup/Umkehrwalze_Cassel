from __future__ import annotations

import json
import tempfile
from pathlib import Path

from revprint.proof_review import build_proof_review_rubric, write_proof_review_rubric


def test_write_proof_review_rubric_from_manifest() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"run_id": "r1", "processed_pages": [{"source_path": "a.jpg"}, {"source_path": "b.jpg"}]}),
            encoding="utf-8",
        )
        out = write_proof_review_rubric(manifest)
        assert out.is_file()
        text = out.read_text(encoding="utf-8")
        assert "Run ID: `r1`" in text
        assert "Page count: `2`" in text


def test_build_proof_review_rubric_contains_checklist() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"run_id": "r2", "processed_pages": []}), encoding="utf-8")
        text = build_proof_review_rubric("Proof Review Rubric", manifest)
        assert "Reviewer Checklist" in text
        assert "Approve next iteration" in text
