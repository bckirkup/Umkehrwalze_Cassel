from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.stamp_suppress import detect_stamp_mask_chroma, suppress_stamps


def _make_stamped_image(path: Path) -> None:
    """Create a test RGB image with a coloured stamp region."""
    arr = np.full((200, 150, 3), 215, dtype=np.uint8)
    # Red stamp at top-left.
    arr[10:40, 10:40, 0] = 200
    arr[10:40, 10:40, 1] = 40
    arr[10:40, 10:40, 2] = 40
    # Dark ink.
    arr[80:85, 20:130, :] = 35
    Image.fromarray(arr, mode="RGB").save(path)


def test_detect_stamp_mask_chroma() -> None:
    arr = np.full((200, 150, 3), 215, dtype=np.uint8)
    arr[10:40, 10:40, 0] = 200
    arr[10:40, 10:40, 1] = 40
    arr[10:40, 10:40, 2] = 40
    mask = detect_stamp_mask_chroma(arr, chroma_threshold=15.0, min_component_area=10)
    # Stamp region should be detected.
    assert np.any(mask[15:35, 15:35] > 0)
    # Ink region should not be detected (it's achromatic/dark).
    assert np.all(mask[80:85, 20:130] == 0)


def test_suppress_stamps_writes_output() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        inp = root / "stamped.jpg"
        out = root / "cleaned.png"
        _make_stamped_image(inp)

        meta = suppress_stamps(inp, out)
        assert out.is_file()
        assert meta["stamp_suppress_applied"] is True
        assert meta["stamp_suppress_pixel_count"] > 0
        assert Path(str(meta["stamp_suppress_mask_path"])).is_file()
