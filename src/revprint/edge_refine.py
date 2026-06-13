"""
OpenCV: inpaint border shadows (excluding ink via gradient) then flatten to 8-bit
grayscale for print.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from revprint.content_mask import apply_content_mask, compute_content_mask


def _pil_to_bgr(pil_rgb: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(pil_rgb.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)


def _paper_median(gray: np.ndarray) -> float:
    h, w = gray.shape
    y0, y1 = h // 5, 4 * h // 5
    x0, x1 = w // 5, 4 * w // 5
    return float(np.median(gray[y0:y1, x0:x1]))


def _border_band(h: int, w: int, width_px: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    b = min(width_px, w // 3, h // 3)
    m[:b, :] = True
    m[-b:, :] = True
    m[:, :b] = True
    m[:, -b:] = True
    return m


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return mag > 0.045


def _normalize_paper_to_white(gray: np.ndarray) -> np.ndarray:
    """Flatten paper tone while preserving stroke darkness as grayscale."""
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    # Division normalization: local paper tends toward white, ink remains dark.
    norm = cv2.divide(gray, bg, scale=245)
    return np.clip(norm, 0, 255).astype(np.uint8)


def _ink_on_white(gray: np.ndarray) -> np.ndarray:
    """Return grayscale ink over a pure white background."""
    norm = _normalize_paper_to_white(gray)
    g = norm.astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    # Dark strokes plus high-gradient hairlines. Keep this stricter than a
    # background-preserving facsimile: the product is ink on new white paper.
    dark = norm < 212
    hairline = (norm < 238) & (mag > 0.028)
    stroke = dark | hairline
    # Bridge tiny cracks in real ink without growing paper texture into gray haze.
    stroke = cv2.morphologyEx(stroke.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    stroke = stroke.astype(bool)

    out = np.full_like(norm, 255)
    # Stretch ink values slightly darker for laser-print legibility while
    # preserving mid-tone pen pressure.
    ink_vals = np.clip((norm.astype(np.int16) - 18), 0, 245).astype(np.uint8)
    out[stroke] = ink_vals[stroke]
    return out


def _largest_component_mask(binary_255: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_255 == 255).astype(np.uint8), connectivity=8
    )
    if num <= 1:
        return (binary_255 == 255)
    best, best_a = 1, stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num):
        if stats[i, cv2.CC_STAT_AREA] > best_a:
            best, best_a = i, stats[i, cv2.CC_STAT_AREA]
    return labels == best


def refine_to_grayscale(pil_rgb: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Return (8-bit L print image, L mask: 255 = combined inpaint regions for review).
    """
    bgr = _pil_to_bgr(pil_rgb)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    med = _paper_median(gray)
    ink0 = _ink_mask(gray)

    border_px = max(6, min(h, w) // 14)
    band = _border_band(h, w, border_px)
    dark0 = gray < int(np.clip(med - 8, 60, 210))
    m1 = (band & dark0 & ~ink0).astype(np.uint8) * 255
    m1 = cv2.dilate(m1, np.ones((5, 5), np.uint8), iterations=1)
    g1 = cv2.inpaint(gray, m1, 7, cv2.INPAINT_TELEA) if int(m1.max()) > 0 else gray

    paper_bin = (g1 > 80).astype(np.uint8) * 255
    k21 = np.ones((21, 21), np.uint8)
    paper_bin = cv2.morphologyEx(paper_bin, cv2.MORPH_CLOSE, k21)
    pmask = _largest_component_mask(paper_bin)
    dist = cv2.distanceTransform((pmask.astype(np.uint8) * 255), cv2.DIST_L2, 3)
    edge_ribbon = pmask & (dist < 0.11 * min(h, w)) & (dist > 0.5)
    ink1 = _ink_mask(g1)
    dark1 = g1 < int(np.clip(med - 2, 70, 230))
    m2 = (edge_ribbon & dark1 & ~ink1).astype(np.uint8) * 255
    m2 = cv2.dilate(m2, np.ones((3, 3), np.uint8), iterations=1)
    g2 = cv2.inpaint(g1, m2, 5, cv2.INPAINT_NS) if int(m2.max()) > 0 else g1

    combined = np.clip(m1.astype(np.int32) + m2.astype(np.int32), 0, 255).astype(np.uint8)
    print_gray = _ink_on_white(g2)

    # Mask non-paper regions (binding shadow, ragged edges, scanner bed)
    # to white *after* ink extraction.  Applying the mask before
    # _ink_on_white() causes division-normalization artifacts at the
    # mask boundary (the σ=35 Gaussian blur of _normalize_paper_to_white
    # sees the step from original pixels to fill-white and creates a
    # false edge).  Masking the final output avoids this.
    content = compute_content_mask(g2)
    print_gray = apply_content_mask(print_gray, content)

    mask_debug = Image.fromarray(combined, mode="L")
    return Image.fromarray(print_gray, mode="L"), mask_debug
