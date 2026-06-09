from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.adaptive_clean import adaptive_clean_page, extract_ink_on_white, sauvola_ink_mask


def _make_gray_image(w: int = 200, h: int = 300) -> np.ndarray:
    """Create a test grayscale image with ink on paper."""
    arr = np.full((h, w), 215, dtype=np.uint8)
    arr[50:55, 30:170] = 40   # Dark line
    arr[80:85, 30:170] = 45   # Another line
    arr[:, :20] = 180          # Spine shadow
    return arr


def test_sauvola_ink_mask_basic() -> None:
    gray = _make_gray_image()
    mask = sauvola_ink_mask(gray, window_size=31, prefer_gpu=False)
    assert mask.shape == gray.shape
    assert mask.dtype == np.float32
    # Ink regions should be detected.
    assert float(np.mean(mask[52, 50:150])) > 0.5
    # Paper regions should not be detected.
    assert float(np.mean(mask[100, 50:150])) < 0.5


def test_extract_ink_on_white() -> None:
    gray = _make_gray_image()
    result = extract_ink_on_white(gray, window_size=31, prefer_gpu=False)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8
    # Paper should be near white.
    assert float(np.mean(result[100, 50:150])) > 240
    # Ink should be dark.
    assert float(np.mean(result[52, 50:150])) < 200


def test_adaptive_clean_page_writes_output() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        inp = root / "input.png"
        out = root / "output.png"
        gray = _make_gray_image()
        Image.fromarray(gray, mode="L").save(inp)

        meta = adaptive_clean_page(inp, out, window_size=31, prefer_gpu=False)
        assert out.is_file()
        assert meta["adaptive_clean_applied"] is True
        assert meta["adaptive_clean_gpu_used"] is False
