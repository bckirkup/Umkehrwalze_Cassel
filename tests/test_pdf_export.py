from __future__ import annotations

import tempfile
from pathlib import Path

from revprint.pdf_export import export_translation_pdf


def test_export_translation_pdf_writes_valid_pdf() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "tr.pdf"
        page = {
            "source_path": str(Path(d) / "p.jpg"),
            "translation_source_type": "gemini_seed",
            "translation_en": "English seed body.",
            "ocr_draft": "",
            "text_evidence": {
                "source_engine": "manual",
                "language": "de",
                "script": "unknown",
                "segments": [{"text": "German here", "confidence": 1.0, "bbox_xywh": [0, 0, 1, 1], "language": "de", "script": "unknown"}],
            },
        }
        export_translation_pdf([page], out)
        assert out.is_file()
        raw = out.read_bytes()
        assert len(raw) > 800
        assert raw.startswith(b"%PDF")
