"""Content-region masking: detect the paper interior and exclude ragged edges,
binding shadows, and scanner-bed artifacts.

Both the proof pipeline (``edge_refine``) and the batch pipeline
(``adaptive_clean``) call :func:`compute_content_mask` to obtain a soft
float32 mask in [0, 1] where 1 = safe interior paper and 0 = edge or
background that should become white.
"""

from __future__ import annotations

import cv2
import numpy as np


def _binding_margin_px(smooth: np.ndarray, paper_brightness: float) -> tuple[int, int, int, int]:
    """Detect dark edge margins via per-column/row brightness profiling.

    Returns (left, top, right, bottom) pixel widths of the detected dark
    margin on each side.  For a bound book the left (or right) margin is
    typically the largest.
    """
    h, w = smooth.shape[:2]
    target = paper_brightness * 0.90

    def _scan_from_edge(profile: np.ndarray) -> int:
        """Walk inward until brightness reaches *target* for a sustained run."""
        run = 0
        for i, v in enumerate(profile):
            if v >= target:
                run += 1
                if run >= 5:
                    return max(0, i - run)
            else:
                run = 0
        return 0

    col_meds = np.median(smooth[h // 4 : 3 * h // 4, :], axis=0).astype(np.float64)
    row_meds = np.median(smooth[:, w // 4 : 3 * w // 4], axis=1).astype(np.float64)

    left = _scan_from_edge(col_meds)
    right = _scan_from_edge(col_meds[::-1])
    top = _scan_from_edge(row_meds)
    bottom = _scan_from_edge(row_meds[::-1])
    return left, top, right, bottom


def compute_content_mask(
    gray: np.ndarray,
    erode_fraction: float = 0.025,
    feather_fraction: float = 0.03,
) -> np.ndarray:
    """Compute a feathered mask of the paper interior.

    The algorithm combines two complementary strategies:

    **Threshold-based** — heavy blur, threshold at 82 % of the central
    paper brightness, morphological close, largest connected component,
    erode, feather.

    **Profile-based** — per-column/row median brightness scanning from
    each edge inward to detect gradual binding shadows that the global
    threshold misses.

    The final mask is the intersection (min) of both, so neither a dark
    border nor a gradual shadow survives.

    Parameters
    ----------
    gray : uint8 grayscale image
    erode_fraction : fraction of min(h, w) to erode inward from the paper
        boundary (default 0.025 ≈ 2.5 %).
    feather_fraction : fraction of min(h, w) for the Gaussian feather
        radius (default 0.03 ≈ 3 %).

    Returns
    -------
    Float32 array in [0, 1] with the same shape as *gray*.
    """
    h, w = gray.shape[:2]
    dim = min(h, w)

    # --- 1. blur away ink ------------------------------------------------
    blur_r = max(15, (dim // 15) | 1)
    smooth = cv2.GaussianBlur(gray, (blur_r, blur_r), 0)

    # --- 2. paper brightness reference from centre -----------------------
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w // 4, 3 * w // 4
    paper_brightness = float(np.median(smooth[cy0:cy1, cx0:cx1]))

    # --- 3. threshold-based mask -----------------------------------------
    thresh = paper_brightness * 0.82
    paper_binary = (smooth > thresh).astype(np.uint8) * 255

    close_k = max(21, dim // 30) | 1
    paper_binary = cv2.morphologyEx(
        paper_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        paper_binary, connectivity=8,
    )
    if num <= 1:
        return np.ones((h, w), dtype=np.float32)

    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    paper = (labels == best).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(paper, contours, -1, 255, cv2.FILLED)

    erode_px = max(5, int(dim * erode_fraction))
    paper = cv2.erode(
        paper,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1),
        ),
    )

    feather_px = max(5, int(dim * feather_fraction))
    feather_k = feather_px * 2 + 1
    thresh_mask = cv2.GaussianBlur(
        paper.astype(np.float32), (feather_k, feather_k), 0,
    ) / 255.0

    # --- 4. profile-based binding/edge margin mask -----------------------
    left, top, right, bottom = _binding_margin_px(smooth, paper_brightness)
    # Add extra erosion on top of profile detection
    extra = max(5, int(dim * erode_fraction * 0.5))
    left += extra
    top += extra
    right += extra
    bottom += extra

    profile_mask = np.ones((h, w), dtype=np.float32)
    if left > 0:
        ramp = np.linspace(0.0, 1.0, min(left + feather_px, w))
        profile_mask[:, : len(ramp)] = np.minimum(
            profile_mask[:, : len(ramp)], ramp[np.newaxis, :],
        )
    if right > 0:
        ramp = np.linspace(0.0, 1.0, min(right + feather_px, w))
        profile_mask[:, -len(ramp) :] = np.minimum(
            profile_mask[:, -len(ramp) :], ramp[np.newaxis, ::-1],
        )
    if top > 0:
        ramp = np.linspace(0.0, 1.0, min(top + feather_px, h))
        profile_mask[: len(ramp), :] = np.minimum(
            profile_mask[: len(ramp), :], ramp[:, np.newaxis],
        )
    if bottom > 0:
        ramp = np.linspace(0.0, 1.0, min(bottom + feather_px, h))
        profile_mask[-len(ramp) :, :] = np.minimum(
            profile_mask[-len(ramp) :, :], ramp[::-1, np.newaxis],
        )

    # --- 5. combine: take the minimum (most aggressive) ------------------
    combined = np.minimum(thresh_mask, profile_mask)
    return np.clip(combined, 0.0, 1.0)


def apply_content_mask(
    gray: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 255.0,
) -> np.ndarray:
    """Blend *gray* toward *fill_value* using *mask*.

    Where mask == 1 the original pixel is kept; where mask == 0 the pixel
    becomes *fill_value* (pure white).  Intermediate values are linearly
    interpolated.
    """
    blended = gray.astype(np.float32) * mask + fill_value * (1.0 - mask)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
