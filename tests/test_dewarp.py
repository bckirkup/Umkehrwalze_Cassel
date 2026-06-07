from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.dewarp import dewarp_grayscale_optional


def test_dewarp_disabled_no_file_side_effects() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "x"
        cleaned = root / f"{stem}.cleaned_gray.png"
        Image.fromarray(np.full((30, 40), 240, dtype=np.uint8), mode="L").save(cleaned)
        meta = dewarp_grayscale_optional(cleaned, root, stem, enable=False)
        assert meta["dewarp_enabled"] is False
        assert Path(root / f"{stem}.dewarped_gray.png").is_file() is False


def test_dewarp_writes_output_when_enabled() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "scan_0001"
        h, w = 200, 260
        arr = np.full((h, w), 245, dtype=np.uint8)
        arr[40:160, 60:200] = 30
        cleaned = root / f"{stem}.cleaned_gray.png"
        Image.fromarray(arr, mode="L").save(cleaned)
        meta = dewarp_grayscale_optional(cleaned, root, stem, enable=True)
        out = Path(str(meta["dewarped_grayscale_path"]))
        assert out.is_file()
        assert meta["dewarp_enabled"] is True
