from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.crease_refine import apply_crease_refine


def test_crease_refine_writes_artifacts() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "scan_0001"
        cleaned = root / f"{stem}.cleaned_gray.png"
        arr = np.full((300, 220), 230, dtype=np.uint8)
        arr[:, 20:24] = 120
        Image.fromarray(arr, mode="L").save(cleaned)
        meta = apply_crease_refine(
            stem=stem,
            pages_dir=root,
            cleaned_gray_path=cleaned,
            enable=True,
            darkness_threshold=8.0,
        )
        assert Path(str(meta["crease_before_path"])).is_file()
        assert Path(str(meta["crease_after_path"])).is_file()
        assert Path(str(meta["crease_mask_path"])).is_file()
