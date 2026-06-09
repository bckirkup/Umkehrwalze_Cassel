"""Stamp suppression via Lab-space colour clustering.

Detects and removes librarian stamps (red, blue, purple, black archival marks)
using colour distance in CIE Lab space.  Stamp regions are inpainted with local
paper texture rather than simply set to white, preserving natural paper grain.

Uses corpus-level stamp colour centroids from :mod:`corpus_stats` so that the
same colour model applies consistently across all 600+ pages.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(rgb, dtype=np.uint8)


def detect_stamp_mask(
    rgb: np.ndarray,
    centroids_lab: list[tuple[float, float, float]],
    radii: list[float],
    radius_scale: float = 1.8,
    min_component_area: int = 50,
) -> np.ndarray:
    """Detect stamp pixels by colour distance to known stamp centroids.

    Parameters
    ----------
    rgb : uint8 RGB image
    centroids_lab : list of (L, a, b) cluster centres from corpus stats
    radii : mean distance from centroid to cluster members
    radius_scale : how many radii to include (1.8 = ~90% of cluster)
    min_component_area : ignore tiny components (probably noise)

    Returns
    -------
    mask : uint8 image, 255 = stamp pixel
    """
    if not centroids_lab:
        return np.zeros(rgb.shape[:2], dtype=np.uint8)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    combined_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

    for centroid, radius in zip(centroids_lab, radii):
        c_arr = np.array(centroid, dtype=np.float32).reshape(1, 1, 3)
        dist = np.linalg.norm(lab - c_arr, axis=-1)
        threshold = max(12.0, radius * radius_scale)
        combined_mask[dist < threshold] = 255

    # Remove tiny components (noise).
    if min_component_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            combined_mask, connectivity=8
        )
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_component_area:
                combined_mask[labels == i] = 0

    return combined_mask


def detect_stamp_mask_chroma(
    rgb: np.ndarray,
    chroma_threshold: float = 22.0,
    lightness_range: tuple[float, float] = (30.0, 210.0),
    min_component_area: int = 80,
) -> np.ndarray:
    """Fallback stamp detection using chroma threshold (no corpus centroids needed).

    Detects pixels that are more colourful than typical paper/ink.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab).astype(np.float32)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1] - 128.0
    b_chan = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a_chan ** 2 + b_chan ** 2)

    mask = (
        (chroma > chroma_threshold)
        & (l_chan > lightness_range[0])
        & (l_chan < lightness_range[1])
    ).astype(np.uint8) * 255

    if min_component_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_component_area:
                mask[labels == i] = 0

    return mask


def suppress_stamps(
    input_path: Path,
    output_path: Path,
    centroids_lab: list[tuple[float, float, float]] | None = None,
    radii: list[float] | None = None,
    radius_scale: float = 1.8,
    chroma_fallback_threshold: float = 22.0,
    inpaint_radius: int = 5,
) -> dict[str, object]:
    """Detect and inpaint stamp regions on a single page.

    If corpus centroids are provided, uses centroid-based detection.
    Otherwise falls back to chroma-threshold detection.

    Returns metadata dict.
    """
    rgb = _load_rgb(input_path)

    if centroids_lab and radii:
        stamp_mask = detect_stamp_mask(
            rgb,
            centroids_lab=centroids_lab,
            radii=radii,
            radius_scale=radius_scale,
        )
        method = "centroid"
    else:
        stamp_mask = detect_stamp_mask_chroma(
            rgb,
            chroma_threshold=chroma_fallback_threshold,
        )
        method = "chroma_fallback"

    stamp_pixel_count = int(np.count_nonzero(stamp_mask))

    if stamp_pixel_count > 0:
        # Dilate mask slightly for better inpainting coverage.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stamp_mask_dilated = cv2.dilate(stamp_mask, kernel, iterations=1)
        # Inpaint in grayscale (our pipeline operates on grayscale).
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        inpainted = cv2.inpaint(gray, stamp_mask_dilated, inpaint_radius, cv2.INPAINT_NS)
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        inpainted = gray

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(inpainted, mode="L").save(output_path)

    # Also save the mask for debugging.
    mask_path = output_path.parent / (output_path.stem + ".stamp_mask.png")
    Image.fromarray(stamp_mask, mode="L").save(mask_path)

    return {
        "stamp_suppress_applied": stamp_pixel_count > 0,
        "stamp_suppress_method": method,
        "stamp_suppress_pixel_count": stamp_pixel_count,
        "stamp_suppress_output_path": str(output_path),
        "stamp_suppress_mask_path": str(mask_path),
    }
