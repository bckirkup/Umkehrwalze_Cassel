"""True line-curvature dewarping via text-line baseline detection.

Corrects the curved text lines caused by book binding by:
1. Detecting text line baselines using horizontal projection profile
2. Fitting a polynomial curvature model to each baseline
3. Building a dense displacement map
4. Remapping the image to flatten all text lines

This replaces the rotation-only ``dewarp.py`` approach with actual
geometric unwrapping of binding curvature.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from scipy.signal import find_peaks
from skimage.filters import threshold_sauvola


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        gray = ImageOps.exif_transpose(im).convert("L")
    return np.asarray(gray, dtype=np.uint8)


def _detect_baselines(
    binary: np.ndarray,
    min_line_height: int = 8,
    peak_prominence: float = 0.1,
) -> list[int]:
    """Detect text line y-positions from horizontal projection profile.

    Parameters
    ----------
    binary : bool array where True = ink
    min_line_height : minimum distance between adjacent lines
    peak_prominence : prominence threshold for peak detection (fraction of max)
    """
    h, w = binary.shape
    proj = np.sum(binary.astype(np.float32), axis=1)
    if proj.max() == 0:
        return []

    proj_norm = proj / proj.max()
    prominence = max(peak_prominence, 0.05)
    peaks, properties = find_peaks(
        proj_norm,
        distance=min_line_height,
        prominence=prominence,
    )
    return sorted(int(p) for p in peaks)


def _trace_baseline(
    binary: np.ndarray,
    y_center: int,
    search_band: int = 20,
    n_samples: int = 40,
) -> list[tuple[int, int]]:
    """Trace the actual curved baseline of a text line.

    For each x-sample across the page, find the local centre of mass of
    ink pixels in a vertical band around the expected y position.
    """
    h, w = binary.shape
    band_half = search_band // 2
    step = max(1, w // n_samples)
    points: list[tuple[int, int]] = []

    for x in range(step // 2, w, step):
        y_lo = max(0, y_center - band_half)
        y_hi = min(h, y_center + band_half)
        col_slice = binary[y_lo:y_hi, max(0, x - 3): min(w, x + 4)]
        if col_slice.sum() == 0:
            continue
        # Weighted centroid in the y direction.
        ys = np.arange(y_lo, y_hi, dtype=np.float32)
        weights = col_slice.sum(axis=1).astype(np.float32)
        if weights.sum() > 0:
            y_com = float(np.average(ys, weights=weights))
            points.append((x, int(round(y_com))))

    return points


def _fit_baseline_poly(
    points: list[tuple[int, int]],
    degree: int = 3,
) -> np.ndarray | None:
    """Fit a polynomial to baseline points.

    Returns polynomial coefficients (highest degree first) or None if
    insufficient data.
    """
    if len(points) < degree + 2:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    try:
        coeffs = np.polyfit(xs, ys, degree)
        return coeffs
    except (np.linalg.LinAlgError, ValueError):
        return None


def compute_dewarp_map(
    gray: np.ndarray,
    window_size: int = 51,
    sauvola_k: float = 0.2,
    poly_degree: int = 3,
    min_line_height: int = 20,
    work_scale: float = 0.25,
) -> np.ndarray:
    """Compute a vertical displacement map to flatten text line curvature.

    Parameters
    ----------
    gray : uint8 grayscale image
    window_size : Sauvola window for binarization
    sauvola_k : Sauvola sensitivity
    poly_degree : polynomial degree for baseline fitting
    min_line_height : minimum px between text lines at work_scale
    work_scale : downscale factor for baseline detection (0.25 = quarter-res)

    Returns
    -------
    dy_map : float32 array same size as gray, per-pixel vertical displacement
    """
    h, w = gray.shape

    # Work at reduced resolution.
    sw = max(1, int(w * work_scale))
    sh = max(1, int(h * work_scale))
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)

    # Binarize at work scale.
    thresh = threshold_sauvola(small, window_size=max(7, window_size // 4 | 1), k=sauvola_k)
    binary = small < thresh

    # Detect text line y-positions.
    scaled_min_height = max(4, int(min_line_height * work_scale))
    line_ys = _detect_baselines(binary, min_line_height=scaled_min_height)

    if len(line_ys) < 3:
        # Not enough lines to build a curvature model.
        return np.zeros((h, w), dtype=np.float32)

    # Trace each baseline and fit polynomials.
    baseline_polys: list[tuple[int, np.ndarray]] = []
    for y_center in line_ys:
        points = _trace_baseline(binary, y_center, search_band=scaled_min_height)
        coeffs = _fit_baseline_poly(points, degree=poly_degree)
        if coeffs is not None:
            baseline_polys.append((y_center, coeffs))

    if len(baseline_polys) < 3:
        return np.zeros((h, w), dtype=np.float32)

    # Build displacement map at work scale.
    # For each baseline, compute its deviation from a straight line at y_center.
    dy_small = np.zeros((sh, sw), dtype=np.float32)
    xs = np.arange(sw, dtype=np.float64)

    for y_center, coeffs in baseline_polys:
        curve_y = np.polyval(coeffs, xs)
        deviation = curve_y - float(y_center)
        # Assign this deviation to a band around the baseline.
        band_half = scaled_min_height // 2
        y_lo = max(0, y_center - band_half)
        y_hi = min(sh, y_center + band_half + 1)
        for y in range(y_lo, y_hi):
            # Weight by proximity to the baseline.
            proximity = 1.0 - abs(y - y_center) / max(1.0, float(band_half))
            dy_small[y, :] = deviation * proximity

    # Smooth the displacement map to fill gaps between detected lines.
    dy_small = ndimage.gaussian_filter(dy_small, sigma=(scaled_min_height * 0.5, 3.0))

    # Upscale to full resolution.
    dy_map = cv2.resize(dy_small, (w, h), interpolation=cv2.INTER_LINEAR)
    dy_map *= (1.0 / work_scale)  # Scale displacement back to full-res pixels.

    return dy_map


def apply_dewarp(
    gray: np.ndarray,
    dy_map: np.ndarray,
) -> np.ndarray:
    """Apply a vertical displacement map to flatten text lines.

    Uses cv2.remap for sub-pixel accuracy.
    """
    h, w = gray.shape
    map_x = np.broadcast_to(
        np.arange(w, dtype=np.float32).reshape(1, w), (h, w)
    ).copy()
    map_y = np.broadcast_to(
        np.arange(h, dtype=np.float32).reshape(h, 1), (h, w)
    ).copy()
    map_y = map_y - dy_map
    return cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)


def dewarp_page(
    input_path: Path,
    output_path: Path,
    window_size: int = 51,
    poly_degree: int = 3,
    min_line_height: int = 20,
    work_scale: float = 0.25,
) -> dict[str, object]:
    """Dewarp a single page to correct text-line curvature.

    Returns metadata dict.
    """
    gray = _load_gray(input_path)
    dy_map = compute_dewarp_map(
        gray,
        window_size=window_size,
        poly_degree=poly_degree,
        min_line_height=min_line_height,
        work_scale=work_scale,
    )

    max_displacement = float(np.max(np.abs(dy_map)))
    if max_displacement < 1.5:
        # Negligible curvature; skip remapping.
        Image.fromarray(gray, mode="L").save(output_path)
        return {
            "line_dewarp_applied": False,
            "line_dewarp_reason": "negligible_curvature",
            "line_dewarp_max_displacement_px": max_displacement,
            "line_dewarp_output_path": str(output_path),
        }

    dewarped = apply_dewarp(gray, dy_map)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(dewarped, mode="L").save(output_path)

    return {
        "line_dewarp_applied": True,
        "line_dewarp_max_displacement_px": max_displacement,
        "line_dewarp_poly_degree": poly_degree,
        "line_dewarp_output_path": str(output_path),
    }
