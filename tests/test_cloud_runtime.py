from __future__ import annotations

import json
import tempfile
from pathlib import Path

from revprint.cloud_runtime import build_local_cloud_manifest


def test_build_local_cloud_manifest_writes_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        page = {
            "source_path": str(root / "src.jpg"),
            "ghost_suppression_enabled": True,
            "dewarp_enabled": False,
            "translation_en": "Hello",
            "translation_source_type": "manual",
            "ocr_draft": "Hallo",
            "edge_reconstruct_candidate_mask_path": str(root / "mask.png"),
            "ocr_reconstruct_hint_mask_path": str(root / "hint_mask.png"),
        }
        out = build_local_cloud_manifest(
            run_id="run-1",
            output_dir=root,
            processed_pages=[page],
            project_id="p",
            volume_id="v",
            profile="forensic",
        )
        assert out.path.is_file()
        payload = json.loads(out.path.read_text(encoding="utf-8"))
        assert payload["manifest_id"] == "run-1-local"
        assert payload["mode"] == "local-only"
        assert len(payload["jobs"]) == 1
        assert payload["jobs"][0]["inputs"]["ocr_hints_uri"].endswith("hint_mask.png")
