from __future__ import annotations

import numpy as np

from revprint.content_mask import apply_content_mask, compute_content_mask


def _make_page_with_dark_edges(w: int = 400, h: int = 600) -> np.ndarray:
    """Synthetic scan: bright paper interior, dark binding on left, dark edges."""
    arr = np.full((h, w), 210, dtype=np.uint8)
    # Dark binding shadow on the left
    arr[:, :40] = 60
    # Ragged dark top/bottom edges
    arr[:15, :] = 80
    arr[-15:, :] = 80
    # Dark right edge
    arr[:, -20:] = 90
    # Ink lines in the interior
    arr[100:105, 80:350] = 30
    arr[200:205, 80:350] = 35
    return arr


def test_compute_content_mask_shape_and_dtype() -> None:
    gray = _make_page_with_dark_edges()
    mask = compute_content_mask(gray)
    assert mask.shape == gray.shape
    assert mask.dtype == np.float32


def test_interior_is_high_exterior_is_low() -> None:
    gray = _make_page_with_dark_edges()
    mask = compute_content_mask(gray)
    # Centre of the page should be fully inside the mask.
    assert float(np.mean(mask[250:350, 150:250])) > 0.8
    # Left binding shadow should be mostly masked out.
    assert float(np.mean(mask[:, :20])) < 0.3


def test_uniform_page_returns_full_mask() -> None:
    """A uniformly bright image has no edges to remove."""
    gray = np.full((300, 200), 215, dtype=np.uint8)
    mask = compute_content_mask(gray)
    assert float(np.mean(mask)) > 0.7


def test_apply_content_mask_blends() -> None:
    gray = np.full((100, 100), 120, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:80, 20:80] = 1.0  # interior region
    result = apply_content_mask(gray, mask, fill_value=252.0)
    assert result.dtype == np.uint8
    # Exterior should be fill value.
    assert int(result[0, 0]) == 252
    # Interior should keep original value.
    assert int(result[50, 50]) == 120


def test_apply_content_mask_intermediate() -> None:
    """Fractional mask values produce a linear blend."""
    gray = np.full((50, 50), 100, dtype=np.uint8)
    mask = np.full((50, 50), 0.5, dtype=np.float32)
    result = apply_content_mask(gray, mask, fill_value=200.0)
    expected = int(100 * 0.5 + 200 * 0.5)
    assert abs(int(result[25, 25]) - expected) <= 1
