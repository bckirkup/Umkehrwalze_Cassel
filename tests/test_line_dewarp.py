from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.line_dewarp import _detect_baselines, compute_dewarp_map, dewarp_page


def _make_curved_text_image(w: int = 400, h: int = 600) -> np.ndarray:
    """Create a test grayscale image with curved text lines."""
    arr = np.full((h, w), 230, dtype=np.uint8)
    # Draw several curved text lines.
    for y_base in range(80, 500, 50):
        for x in range(40, w - 40):
            # Parabolic curvature: lines dip toward the left edge (simulating spine).
            curve = int(8 * ((x - w / 2) / (w / 2)) ** 2)
            y = y_base + curve
            if 0 <= y < h and 0 <= y + 2 < h:
                arr[y: y + 3, x] = 50
    return arr


def test_detect_baselines() -> None:
    gray = _make_curved_text_image()
    binary = gray < 150
    baselines = _detect_baselines(binary, min_line_height=10)
    # Should find several text lines.
    assert len(baselines) >= 3


def test_compute_dewarp_map_shape() -> None:
    gray = _make_curved_text_image()
    dy = compute_dewarp_map(gray, work_scale=0.5, min_line_height=15)
    assert dy.shape == gray.shape
    assert dy.dtype == np.float32


def test_dewarp_page_writes_output() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        inp = root / "curved.png"
        out = root / "dewarped.png"
        gray = _make_curved_text_image()
        Image.fromarray(gray, mode="L").save(inp)

        meta = dewarp_page(inp, out, work_scale=0.5, min_line_height=15)
        assert out.is_file()
        assert "line_dewarp_max_displacement_px" in meta
