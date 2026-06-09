from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.corpus_stats import CorpusStats, _compute_page_stats, compute_corpus_stats


def _make_test_image(path: Path, w: int = 200, h: int = 300) -> None:
    """Create a test RGB image with paper-like tones and some 'ink'."""
    arr = np.full((h, w, 3), 215, dtype=np.uint8)
    # Dark 'ink' strokes.
    arr[50:55, 30:170, :] = 40
    arr[80:85, 30:170, :] = 45
    # A red 'stamp' region.
    arr[10:25, 10:25, 0] = 200  # R
    arr[10:25, 10:25, 1] = 50   # G
    arr[10:25, 10:25, 2] = 50   # B
    # Left spine shadow.
    arr[:, :20, :] = 170
    Image.fromarray(arr, mode="RGB").save(path)


def test_page_stats_basic() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test_0001.jpg"
        _make_test_image(p)
        stats = _compute_page_stats(p, max_dim=200)
        assert stats.paper_median_l > 150
        assert stats.ink_median_l < stats.paper_median_l
        assert stats.spine_side in ("left", "right", "unknown")


def test_corpus_stats_aggregation() -> None:
    with tempfile.TemporaryDirectory() as d:
        paths: list[Path] = []
        for i in range(4):
            p = Path(d) / f"test_{i:04d}.jpg"
            _make_test_image(p)
            paths.append(p)

        stats = compute_corpus_stats(paths, max_dim=200, n_stamp_clusters=2)
        assert isinstance(stats, CorpusStats)
        assert len(stats.page_stats) == 4
        assert stats.corpus_paper_median_l > 0
        assert stats.corpus_ink_median_l > 0
