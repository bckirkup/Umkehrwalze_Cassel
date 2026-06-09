"""Spine shadow flattening via 1D illumination correction.

Replaces the heuristic crease brightening in ``crease_refine.py`` with a
principled approach: estimate the illumination gradient across the page
(primarily caused by the binding shadow) and divide it out.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        gray = ImageOps.exif_transpose(im).convert("L")
    return np.asarray(gray, dtype=np.uint8)


def estimate_illumination_1d(
    gray: np.ndarray,
    percentile: float = 85.0,
    smooth_sigma: float = 0.05,
) -> np.ndarray:
    """Estimate per-column illumination from bright pixels.

    For each column, take the ``percentile``-th brightness value (ignoring
    ink pixels which are dark).  Smooth with a Gaussian to get a stable
    1D illumination profile.

    Parameters
    ----------
    gray : uint8 grayscale image
    percentile : brightness percentile to sample (85 = paper, ignoring ink)
    smooth_sigma : Gaussian sigma as fraction of image width

    Returns
    -------
    illum_1d : float32 array of shape (w,), per-column illumination estimate
    """
    h, w = gray.shape
    col_profile = np.percentile(gray.astype(np.float32), percentile, axis=0)
    sigma_px = max(3.0, smooth_sigma * w)
    # Gaussian smoothing of the 1D profile.
    kernel_size = int(sigma_px * 6) | 1
    col_smooth = cv2.GaussianBlur(
        col_profile.reshape(1, -1), (kernel_size, 1), sigma_px
    ).ravel()
    return col_smooth


def estimate_illumination_2d(
    gray: np.ndarray,
    block_size: int = 64,
    percentile: float = 85.0,
) -> np.ndarray:
    """Estimate 2D illumination field from a grid of bright-pixel samples.

    More accurate than 1D for pages with both vertical (spine) and
    horizontal (crease) shadows.

    Parameters
    ----------
    gray : uint8 grayscale image
    block_size : grid cell size in pixels
    percentile : brightness percentile per cell

    Returns
    -------
    illum_2d : float32 array same shape as gray
    """
    h, w = gray.shape
    gray_f = gray.astype(np.float32)

    # Sample brightness on a coarse grid.
    rows = max(1, h // block_size)
    cols = max(1, w // block_size)
    grid = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            y0 = r * block_size
            y1 = min(h, (r + 1) * block_size)
            x0 = c * block_size
            x1 = min(w, (c + 1) * block_size)
            cell = gray_f[y0:y1, x0:x1]
            if cell.size == 0:
                grid[r, c] = 200.0
            else:
                grid[r, c] = float(np.percentile(cell, percentile))

    # Upscale grid to full resolution with bilinear interpolation.
    illum = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    return illum


def flatten_illumination(
    gray: np.ndarray,
    target_white: float = 245.0,
    use_2d: bool = True,
    block_size: int = 64,
    percentile: float = 85.0,
) -> np.ndarray:
    """Divide out illumination gradient from a grayscale page.

    Produces an image where paper brightness is uniform at ``target_white``
    while ink darkness is preserved relative to local paper tone.
    """
    gray_f = gray.astype(np.float32)

    if use_2d:
        illum = estimate_illumination_2d(gray, block_size=block_size, percentile=percentile)
    else:
        illum_1d = estimate_illumination_1d(gray, percentile=percentile)
        illum = np.broadcast_to(illum_1d.reshape(1, -1), gray.shape).astype(np.float32)

    # Avoid division by zero.
    illum = np.clip(illum, 10.0, 255.0)

    # Division normalization: pixel / local_bg * target.
    corrected = (gray_f / illum) * target_white
    return np.clip(corrected, 0.0, 255.0).astype(np.uint8)


def flatten_spine_shadow(
    input_path: Path,
    output_path: Path,
    target_white: float = 245.0,
    use_2d: bool = True,
) -> dict[str, object]:
    """Flatten spine/crease shadows on a single page.

    Returns metadata dict.
    """
    gray = _load_gray(input_path)
    corrected = flatten_illumination(gray, target_white=target_white, use_2d=use_2d)

    # Measure improvement: ratio of brightness variance before vs after.
    before_std = float(np.std(np.percentile(gray.astype(np.float32), 85, axis=0)))
    after_std = float(np.std(np.percentile(corrected.astype(np.float32), 85, axis=0)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrected, mode="L").save(output_path)

    return {
        "spine_flatten_applied": True,
        "spine_flatten_mode": "2d" if use_2d else "1d",
        "spine_flatten_target_white": target_white,
        "spine_flatten_brightness_std_before": before_std,
        "spine_flatten_brightness_std_after": after_std,
        "spine_flatten_output_path": str(output_path),
    }
