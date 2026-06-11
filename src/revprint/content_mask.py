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


def compute_content_mask(
    gray: np.ndarray,
    erode_fraction: float = 0.012,
    feather_fraction: float = 0.015,
) -> np.ndarray:
    """Compute a feathered mask of the paper interior.

    The algorithm:
      1. Heavy Gaussian blur to erase ink marks.
      2. Reference paper brightness from the central 50 % of the image.
      3. Threshold at 72 % of paper brightness to separate interior paper
         from binding shadow, dark page edges, and scanner background.
      4. Morphological close to bridge small gaps.
      5. Largest connected component → paper region.
      6. Fill interior holes.
      7. Erode inward to cut past ragged edges.
      8. Gaussian feather for a smooth transition to white.

    Parameters
    ----------
    gray : uint8 grayscale image
    erode_fraction : fraction of min(h, w) to erode inward from the paper
        boundary (default 0.012 ≈ 1.2 % of the shorter dimension).
    feather_fraction : fraction of min(h, w) for the Gaussian feather
        radius (default 0.015).

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

    # --- 3. threshold ----------------------------------------------------
    thresh = paper_brightness * 0.72
    paper_binary = (smooth > thresh).astype(np.uint8) * 255

    # --- 4. morphological close ------------------------------------------
    close_k = max(21, dim // 30) | 1
    paper_binary = cv2.morphologyEx(
        paper_binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    )

    # --- 5. largest connected component ----------------------------------
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        paper_binary, connectivity=8,
    )
    if num <= 1:
        return np.ones((h, w), dtype=np.float32)

    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    paper = (labels == best).astype(np.uint8) * 255

    # --- 6. fill holes ---------------------------------------------------
    contours, _ = cv2.findContours(
        paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if contours:
        cv2.drawContours(paper, contours, -1, 255, cv2.FILLED)

    # --- 7. erode past ragged edges --------------------------------------
    erode_px = max(3, int(dim * erode_fraction))
    paper = cv2.erode(
        paper,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1),
        ),
    )

    # --- 8. feather ------------------------------------------------------
    feather_px = max(3, int(dim * feather_fraction))
    feather_k = feather_px * 2 + 1
    soft = cv2.GaussianBlur(
        paper.astype(np.float32), (feather_k, feather_k), 0,
    ) / 255.0

    return np.clip(soft, 0.0, 1.0)


def apply_content_mask(
    gray: np.ndarray,
    mask: np.ndarray,
    fill_value: float = 252.0,
) -> np.ndarray:
    """Blend *gray* toward *fill_value* using *mask*.

    Where mask == 1 the original pixel is kept; where mask == 0 the pixel
    becomes *fill_value* (paper-white).  Intermediate values are linearly
    interpolated.
    """
    blended = gray.astype(np.float32) * mask + fill_value * (1.0 - mask)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
