from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.spine_flatten import (
    estimate_illumination_1d,
    estimate_illumination_2d,
    flatten_illumination,
    flatten_spine_shadow,
)


def _make_shadowed_image(w: int = 300, h: int = 400) -> np.ndarray:
    """Create a test grayscale image with a pronounced spine shadow on the left."""
    arr = np.full((h, w), 225, dtype=np.uint8)
    # Wide gradient shadow on the left third.
    for x in range(w):
        shadow = max(0.0, 1.0 - x / (w * 0.4))
        brightness = int(225 - shadow * 80)
        arr[:, x] = np.clip(brightness, 0, 255)
    # Some ink lines (sparse, so they don't dominate the 85th percentile).
    arr[100:103, 80:250] = 40
    arr[200:203, 80:250] = 40
    return arr


def test_illumination_1d() -> None:
    gray = _make_shadowed_image()
    illum = estimate_illumination_1d(gray)
    assert illum.shape == (gray.shape[1],)
    # Left side should be dimmer in illumination estimate.
    assert float(illum[5]) < float(illum[150])


def test_illumination_2d() -> None:
    gray = _make_shadowed_image()
    illum = estimate_illumination_2d(gray, block_size=32)
    assert illum.shape == gray.shape


def test_flatten_illumination_reduces_variance() -> None:
    gray = _make_shadowed_image()
    corrected = flatten_illumination(gray, use_2d=True)
    # Column-wise mean brightness of paper rows should be more uniform after correction.
    # Use mean of all pixels (ink rows are sparse relative to paper).
    before_col_means = np.mean(gray.astype(np.float32), axis=0)
    after_col_means = np.mean(corrected.astype(np.float32), axis=0)
    before_std = float(np.std(before_col_means))
    after_std = float(np.std(after_col_means))
    assert after_std < before_std


def test_flatten_spine_shadow_writes_output() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        inp = root / "shadowed.png"
        out = root / "flattened.png"
        gray = _make_shadowed_image()
        Image.fromarray(gray, mode="L").save(inp)

        meta = flatten_spine_shadow(inp, out)
        assert out.is_file()
        assert meta["spine_flatten_applied"] is True
        # Just verify the output file was written and metadata is present.
        assert "spine_flatten_brightness_std_after" in meta
        assert "spine_flatten_brightness_std_before" in meta
