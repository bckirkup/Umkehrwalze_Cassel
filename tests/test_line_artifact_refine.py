from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.line_artifact_refine import apply_line_artifact_refine


def test_line_refine_suppresses_long_border_line() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "scan_0001"
        cleaned = root / f"{stem}.cleaned_gray.png"
        arr = np.full((320, 220), 242, dtype=np.uint8)
        arr[:, 12:14] = 56  # strong vertical spine-like line
        Image.fromarray(arr, mode="L").save(cleaned)

        meta = apply_line_artifact_refine(
            stem=stem,
            pages_dir=root,
            cleaned_gray_path=cleaned,
            enable=True,
            min_length_ratio=0.5,
            border_band_ratio=0.18,
        )
        assert Path(str(meta["line_refine_before_path"])).is_file()
        assert Path(str(meta["line_refine_after_path"])).is_file()
        assert Path(str(meta["line_refine_removed_mask_path"])).is_file()
        assert bool(meta["line_refine_applied"]) is True
        assert int(meta["line_refine_removed_lines"]) >= 1
