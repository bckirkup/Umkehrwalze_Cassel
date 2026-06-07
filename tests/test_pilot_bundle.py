from __future__ import annotations

import tempfile
from pathlib import Path

from revprint.pilot_bundle import write_pilot_print_bundle


def test_write_pilot_print_bundle_creates_markdown() -> None:
    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "run1"
        run.mkdir()
        pdf = Path(d) / "rep.pdf"
        tr = Path(d) / "tr.pdf"
        pdf.write_bytes(b"x")
        tr.write_bytes(b"y")
        man = run / "manifest.json"
        man.write_text("{}", encoding="utf-8")
        out = write_pilot_print_bundle(
            run_dir=run,
            page_records=[
                {
                    "source_path": str(Path(d) / "a.jpg"),
                    "cleaned_grayscale_path": str(run / "a.cleaned_gray.png"),
                    "translation_source_type": "gemini_seed",
                }
            ],
            reproduction_pdf=pdf,
            translation_pdf=tr,
            manifest_path=man,
        )
        assert out.is_file()
        text = out.read_text(encoding="utf-8")
        assert "Pilot print bundle" in text
        assert "reproduction" in text.lower() or str(pdf) in text
