"""Corpus-level statistics: paper tone, ink range, stamp colour clusters.

A single O(N) pass over the corpus at reduced resolution to establish
data-driven thresholds for the per-page cleaning pipeline.  All heavy
per-pixel work uses NumPy/OpenCV — no Python pixel loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class PageStats:
    """Per-page statistics computed at reduced resolution."""

    path: Path
    paper_median_l: float
    paper_iqr_l: float
    ink_median_l: float
    ink_darkest_5pct_l: float
    spine_side: str  # "left", "right", or "unknown"
    spine_shadow_extent_px: int
    stamp_pixel_count: int


@dataclass
class CorpusStats:
    """Aggregated statistics across all pages."""

    page_stats: list[PageStats] = field(default_factory=list)
    corpus_paper_median_l: float = 0.0
    corpus_paper_iqr_l: float = 0.0
    corpus_ink_median_l: float = 0.0
    corpus_ink_darkest_5pct_l: float = 0.0
    stamp_centroids_lab: list[tuple[float, float, float]] = field(default_factory=list)
    stamp_centroid_radii: list[float] = field(default_factory=list)


def _load_reduced(path: Path, max_dim: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Load image at reduced resolution, return (gray_u8, lab_f32)."""
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        w, h = rgb.size
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            rgb = rgb.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
    arr = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2Lab).astype(np.float32)
    return gray, lab


def _detect_spine(gray: np.ndarray) -> tuple[str, int]:
    """Return (side, shadow_extent_px) from column brightness profile."""
    h, w = gray.shape
    col_means = np.mean(gray.astype(np.float32), axis=0)
    band = max(4, int(w * 0.15))
    left_mean = float(np.mean(col_means[:band]))
    right_mean = float(np.mean(col_means[-band:]))
    if abs(left_mean - right_mean) < 5.0:
        return "unknown", 0

    if left_mean < right_mean:
        side = "left"
        # Find where spine shadow ends (column mean rises to within 90% of center).
        center_mean = float(np.mean(col_means[w // 3: 2 * w // 3]))
        threshold = center_mean * 0.92
        extent = 0
        for x in range(min(w, band * 2)):
            if col_means[x] >= threshold:
                extent = x
                break
        else:
            extent = band
    else:
        side = "right"
        center_mean = float(np.mean(col_means[w // 3: 2 * w // 3]))
        threshold = center_mean * 0.92
        extent = 0
        for x in range(w - 1, max(0, w - band * 2), -1):
            if col_means[x] >= threshold:
                extent = w - x
                break
        else:
            extent = band
    return side, extent


def _compute_page_stats(path: Path, max_dim: int = 800) -> PageStats:
    gray, lab = _load_reduced(path, max_dim=max_dim)
    h, w = gray.shape
    flat = gray.ravel().astype(np.float32)

    # Paper: brightest 60% of pixels.
    paper_threshold = float(np.percentile(flat, 40))
    paper_pixels = flat[flat >= paper_threshold]
    paper_median = float(np.median(paper_pixels)) if paper_pixels.size > 0 else 200.0
    paper_iqr = float(np.percentile(paper_pixels, 75) - np.percentile(paper_pixels, 25)) if paper_pixels.size > 0 else 20.0

    # Ink: darkest 10% of pixels.
    ink_threshold = float(np.percentile(flat, 10))
    ink_pixels = flat[flat <= ink_threshold]
    ink_median = float(np.median(ink_pixels)) if ink_pixels.size > 0 else 80.0
    ink_dark5 = float(np.percentile(flat, 5))

    spine_side, spine_extent = _detect_spine(gray)

    # Stamp detection: pixels with high chroma (far from grey axis) in Lab space.
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]
    chroma = np.sqrt((a_chan - 128.0) ** 2 + (b_chan - 128.0) ** 2)
    # Stamps are colourful and not super dark (not ink).
    stamp_mask = (chroma > 18.0) & (l_chan > 30.0) & (l_chan < 200.0)
    stamp_count = int(np.count_nonzero(stamp_mask))

    return PageStats(
        path=path,
        paper_median_l=paper_median,
        paper_iqr_l=paper_iqr,
        ink_median_l=ink_median,
        ink_darkest_5pct_l=ink_dark5,
        spine_side=spine_side,
        spine_shadow_extent_px=spine_extent,
        stamp_pixel_count=stamp_count,
    )


def _cluster_stamp_colours(
    image_paths: list[Path],
    max_dim: int = 800,
    max_pages: int = 20,
    n_clusters: int = 4,
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """Cluster stamp-coloured pixels across a sample of pages.

    Returns (centroids_lab, radii) where each centroid is (L, a, b) and
    radius is the mean distance from centroid to its cluster members.
    """
    from sklearn.cluster import KMeans

    all_stamp_lab: list[np.ndarray] = []
    step = max(1, len(image_paths) // max_pages)
    sampled = image_paths[::step][:max_pages]
    for path in sampled:
        _, lab = _load_reduced(path, max_dim=max_dim)
        l_chan = lab[:, :, 0]
        a_chan = lab[:, :, 1]
        b_chan = lab[:, :, 2]
        chroma = np.sqrt((a_chan - 128.0) ** 2 + (b_chan - 128.0) ** 2)
        stamp_mask = (chroma > 18.0) & (l_chan > 30.0) & (l_chan < 200.0)
        if np.count_nonzero(stamp_mask) < 10:
            continue
        pixels = np.column_stack([
            l_chan[stamp_mask],
            a_chan[stamp_mask],
            b_chan[stamp_mask],
        ])
        # Subsample to cap memory.
        if pixels.shape[0] > 5000:
            idx = np.random.default_rng(42).choice(pixels.shape[0], 5000, replace=False)
            pixels = pixels[idx]
        all_stamp_lab.append(pixels)

    if not all_stamp_lab:
        return [], []

    combined = np.vstack(all_stamp_lab)
    actual_k = min(n_clusters, max(1, combined.shape[0] // 20))
    km = KMeans(n_clusters=actual_k, n_init=10, random_state=42)
    km.fit(combined)
    centroids = [tuple(float(v) for v in c) for c in km.cluster_centers_]
    # Compute mean distance per cluster.
    labels = km.labels_
    radii: list[float] = []
    for i in range(actual_k):
        members = combined[labels == i]
        if members.shape[0] == 0:
            radii.append(0.0)
            continue
        dists = np.linalg.norm(members - km.cluster_centers_[i], axis=1)
        radii.append(float(np.mean(dists)))
    return centroids, radii


def compute_corpus_stats(
    image_paths: list[Path],
    max_dim: int = 800,
    n_stamp_clusters: int = 4,
) -> CorpusStats:
    """Compute corpus-level statistics from all image paths.

    This is an O(N) pass at reduced resolution.  At 800 px max dimension,
    each page takes ~50 ms, so 600 pages ≈ 30 seconds.
    """
    page_stats_list: list[PageStats] = []
    for p in image_paths:
        try:
            ps = _compute_page_stats(p, max_dim=max_dim)
            page_stats_list.append(ps)
        except Exception:
            continue

    if not page_stats_list:
        return CorpusStats()

    paper_medians = np.array([ps.paper_median_l for ps in page_stats_list])
    ink_medians = np.array([ps.ink_median_l for ps in page_stats_list])
    ink_darks = np.array([ps.ink_darkest_5pct_l for ps in page_stats_list])
    paper_iqrs = np.array([ps.paper_iqr_l for ps in page_stats_list])

    centroids, radii = _cluster_stamp_colours(
        image_paths, max_dim=max_dim, n_clusters=n_stamp_clusters
    )

    return CorpusStats(
        page_stats=page_stats_list,
        corpus_paper_median_l=float(np.median(paper_medians)),
        corpus_paper_iqr_l=float(np.median(paper_iqrs)),
        corpus_ink_median_l=float(np.median(ink_medians)),
        corpus_ink_darkest_5pct_l=float(np.median(ink_darks)),
        stamp_centroids_lab=centroids,
        stamp_centroid_radii=radii,
    )
