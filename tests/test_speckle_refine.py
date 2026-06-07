from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.speckle_refine import apply_speckle_refine


def test_speckle_refine_removes_small_dots_and_writes_assets() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "page_0001"
        arr = np.full((60, 80), 230, dtype=np.uint8)
        arr[10, 10] = 20
        arr[20, 30] = 35
        cleaned = root / f"{stem}.cleaned_gray.png"
        Image.fromarray(arr, mode="L").save(cleaned)
        meta = apply_speckle_refine(
            stem=stem,
            pages_dir=root,
            cleaned_gray_path=cleaned,
            enable=True,
            max_component_area=20,
            border_max_component_area=30,
            border_band_ratio=0.12,
        )
        assert Path(str(meta["speckle_before_path"])).is_file()
        assert Path(str(meta["speckle_after_path"])).is_file()
        assert Path(str(meta["speckle_removed_mask_path"])).is_file()
        assert int(meta["speckle_removed_components"]) >= 1
        assert int(meta["speckle_removed_components_border"]) >= 0
        assert int(meta["speckle_removed_components_inner"]) >= 0
