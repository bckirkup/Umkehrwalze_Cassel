from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.page_model import (
    _extract_scan_number,
    build_page_model,
)


def _make_test_image(path: Path, w: int = 100, h: int = 150) -> None:
    """Create a minimal gray test image."""
    arr = np.full((h, w), 220, dtype=np.uint8)
    # Make left side darker to simulate spine.
    arr[:, :15] = 160
    Image.fromarray(arr, mode="L").save(path)


def test_extract_scan_number() -> None:
    assert _extract_scan_number("hstam_4_h_nr_4156_0068.jpg") == 68
    assert _extract_scan_number("hstam_4_h_nr_4156_0190.jpg") == 190
    assert _extract_scan_number("random_file.txt") is None


def test_build_page_model_ordering() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        paths: list[Path] = []
        for n in [68, 69, 70, 71]:
            p = root / f"hstam_4_h_nr_4156_{n:04d}.jpg"
            _make_test_image(p)
            paths.append(p)

        model = build_page_model(paths, spine_detect_sample=0)
        assert model.page_count == 4
        assert model.pages[0].scan_number == 68
        assert model.pages[-1].scan_number == 71


def test_facing_page_pairing() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        paths: list[Path] = []
        for n in [68, 69, 70, 71]:
            p = root / f"hstam_4_h_nr_4156_{n:04d}.jpg"
            _make_test_image(p)
            paths.append(p)

        model = build_page_model(paths, spine_detect_sample=0)
        # Even pages are verso, odd are recto (default convention).
        p68 = model.by_scan_number[68]
        assert p68.side == "verso"
        assert p68.facing_path == paths[1]  # scan 69

        p69 = model.by_scan_number[69]
        assert p69.side == "recto"
        assert p69.facing_path == paths[0]  # scan 68


def test_json_enrichment() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        p = root / "hstam_4_h_nr_4156_0068.jpg"
        _make_test_image(p)

        json_data = [
            {
                "original_filename": "hstam_4_h_nr_4156_0068.jpg",
                "document_date": "1777-01-01",
                "notes": "test note",
            }
        ]
        json_path = root / "metadata.json"
        json_path.write_text(json.dumps(json_data), encoding="utf-8")

        model = build_page_model([p], json_path=json_path, spine_detect_sample=0)
        assert model.pages[0].json_entry is not None
        assert model.pages[0].json_entry["notes"] == "test note"
