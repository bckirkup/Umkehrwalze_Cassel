from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(path)


def _apply_local_clear(
    gray: np.ndarray,
    out: np.ndarray,
    removed: np.ndarray,
    line_mask: np.ndarray,
    *,
    is_spine: bool,
    counts: dict[str, int],
) -> None:
    local = cv2.dilate(line_mask, np.ones((7, 7), np.uint8), iterations=1)
    line_dark = float(np.mean((255 - gray)[line_mask > 0])) if np.any(line_mask > 0) else 0.0
    local_dark_cov = (
        float(np.mean((gray[local > 0] < 150).astype(np.float32))) if np.any(local > 0) else 1.0
    )
    cov_limit = 0.52 if is_spine else 0.34
    if line_dark < 16.0 or local_dark_cov > cov_limit:
        return
    boost = 82 if is_spine else 60
    out[local > 0] = np.minimum(255, out[local > 0].astype(np.int16) + boost).astype(np.uint8)
    removed[local > 0] = 255
    counts["lines"] += 1
    counts["spine" if is_spine else "fold"] += 1


def _candidate_columns(gray: np.ndarray) -> list[tuple[int, int]]:
    h, w = gray.shape
    dark = (gray < 165).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((1, 23), np.uint8))
    candidates: list[tuple[int, int]] = []
    active_start = -1
    for x in range(w):
        col = dark[:, x].astype(bool)
        run = 0
        best = 0
        for pix in col:
            run = run + 1 if pix else 0
            best = max(best, run)
        col_ok = best >= int(0.72 * h) and 0.03 <= float(np.mean(col)) <= 0.6
        if col_ok and active_start < 0:
            active_start = x
        elif not col_ok and active_start >= 0:
            candidates.append((active_start, x - 1))
            active_start = -1
    if active_start >= 0:
        candidates.append((active_start, w - 1))
    return candidates


def _apply_primary_spines(
    gray: np.ndarray,
    out: np.ndarray,
    removed: np.ndarray,
    counts: dict[str, int],
) -> None:
    h, w = gray.shape
    for x0, x1 in _candidate_columns(gray):
        cx = int((x0 + x1) * 0.5)
        if cx > int(0.27 * w) and cx < int(0.73 * w):
            continue
        line_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(line_mask, (x0, 0), (x1, h - 1), 255, thickness=-1)
        _apply_local_clear(gray, out, removed, line_mask, is_spine=True, counts=counts)


def _apply_gradient_spines(
    gray: np.ndarray,
    blur: np.ndarray,
    out: np.ndarray,
    removed: np.ndarray,
    counts: dict[str, int],
) -> None:
    h, w = gray.shape
    col_strength = np.mean(np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)), axis=0)
    if col_strength.size == 0:
        return
    zone = np.zeros(w, dtype=bool)
    zone[: int(0.3 * w)] = True
    zone[int(0.7 * w) :] = True
    zone_vals = col_strength[zone]
    if zone_vals.size == 0:
        return
    active = col_strength >= max(4.0, float(np.percentile(zone_vals, 92)))
    x = 0
    while x < w:
        if not active[x]:
            x += 1
            continue
        x0 = x
        while x < w and active[x]:
            x += 1
        x1 = x - 1
        cx = int((x0 + x1) * 0.5)
        strip = gray[:, x0 : x1 + 1]
        if x1 - x0 + 1 > 18 or (cx > int(0.3 * w) and cx < int(0.7 * w)):
            continue
        if strip.size == 0 or float(np.mean(strip)) > 238.0:
            continue
        line_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(line_mask, (x0, 0), (x1, h - 1), 255, thickness=-1)
        _apply_local_clear(gray, out, removed, line_mask, is_spine=True, counts=counts)


def _apply_hough_lines(
    gray: np.ndarray,
    lines: np.ndarray | None,
    out: np.ndarray,
    removed: np.ndarray,
    *,
    min_len: int,
    border_band: int,
    counts: dict[str, int],
) -> None:
    if lines is None:
        return
    h, w = gray.shape
    line_points = lines[:, 0, :] if lines.ndim == 3 else lines
    for line_pts in line_points:
        x1, y1, x2, y2 = [int(v) for v in line_pts]
        dx, dy = x2 - x1, y2 - y1
        length = float((dx * dx + dy * dy) ** 0.5)
        angle = abs(np.degrees(np.arctan2(dy, max(1e-6, dx))))
        mx = int((x1 + x2) * 0.5)
        in_zone = mx <= border_band or mx >= (w - border_band)
        in_zone = in_zone or mx <= int(0.23 * w) or mx >= int(0.77 * w)
        if length < min_len or angle < 62.0 or (not in_zone and length < 0.88 * h):
            continue
        line_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=3)
        _apply_local_clear(
            gray,
            out,
            removed,
            line_mask,
            is_spine=mx <= border_band or mx >= (w - border_band),
            counts=counts,
        )


def apply_line_artifact_refine(
    *,
    stem: str,
    pages_dir: Path,
    cleaned_gray_path: Path,
    enable: bool,
    min_length_ratio: float = 0.55,
    border_band_ratio: float = 0.16,
) -> dict[str, object]:
    """
    Suppress long straight structural lines (spine/folds) while protecting body text.
    """
    gray = _load_gray(cleaned_gray_path)
    pages_dir = Path(pages_dir)
    before = pages_dir / f"{stem}.line_refine_before.png"
    after = pages_dir / f"{stem}.line_refine_after.png"
    removed_path = pages_dir / f"{stem}.line_refine_removed_mask.png"
    _save_gray(before, gray)

    meta: dict[str, object] = {
        "line_refine_enabled": bool(enable),
        "line_refine_before_path": str(before),
        "line_refine_after_path": str(after),
        "line_refine_removed_mask_path": str(removed_path),
        "line_refine_applied": False,
        "line_refine_removed_lines": 0,
        "line_refine_removed_spine_lines": 0,
        "line_refine_removed_fold_lines": 0,
        "line_refine_min_length_ratio": float(min_length_ratio),
        "line_refine_border_band_ratio": float(border_band_ratio),
        "line_refine_reason": "disabled" if not enable else "none_removed",
    }
    if not enable:
        _save_gray(after, gray)
        _save_gray(removed_path, np.zeros_like(gray, dtype=np.uint8))
        return meta

    h, w = gray.shape
    min_len = int(max(40, max(h, w) * float(np.clip(min_length_ratio, 0.3, 0.95))))
    bx = max(12, int(w * float(np.clip(border_band_ratio, 0.08, 0.35))))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 135)
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=100,
        minLineLength=min_len,
        maxLineGap=22,
    )

    out = gray.copy()
    removed = np.zeros_like(gray, dtype=np.uint8)
    counts = {"lines": 0, "spine": 0, "fold": 0}
    _apply_primary_spines(gray, out, removed, counts)
    _apply_gradient_spines(gray, blur, out, removed, counts)
    _apply_hough_lines(
        gray,
        lines,
        out,
        removed,
        min_len=min_len,
        border_band=bx,
        counts=counts,
    )

    _save_gray(after, out)
    _save_gray(removed_path, removed)
    _save_gray(Path(cleaned_gray_path), out)
    meta["line_refine_applied"] = counts["lines"] > 0
    meta["line_refine_removed_lines"] = counts["lines"]
    meta["line_refine_removed_spine_lines"] = counts["spine"]
    meta["line_refine_removed_fold_lines"] = counts["fold"]
    meta["line_refine_reason"] = (
        "removed_structural_lines" if counts["lines"] > 0 else "none_removed"
    )
    return meta
