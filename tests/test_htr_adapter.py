from __future__ import annotations

import json
import tempfile
from pathlib import Path

from revprint.htr_adapter import load_htr_sidecar


def test_load_htr_sidecar_reads_segments() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "page_0001"
        payload = {
            "source_engine": "test-htr",
            "language": "de",
            "script": "kurrent",
            "segments": [
                {
                    "text": "Anno 1742",
                    "confidence": 0.81,
                    "bbox_xywh": [10, 20, 40, 12],
                    "language": "de",
                    "script": "kurrent",
                }
            ],
        }
        (root / f"{stem}.htr.json").write_text(json.dumps(payload), encoding="utf-8")
        ev, meta = load_htr_sidecar(root, stem, enabled=True)
        assert ev is not None
        assert "Anno 1742" in ev.full_text
        assert bool(meta.get("htr_used")) is True
