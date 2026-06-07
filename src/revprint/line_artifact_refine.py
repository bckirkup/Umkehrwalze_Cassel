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
    removed_lines = 0
    removed_spine = 0
    removed_fold = 0

    def _apply_local_clear(line_mask: np.ndarray, *, is_spine: bool) -> bool:
        nonlocal removed_lines, removed_spine, removed_fold
        local = cv2.dilate(line_mask, np.ones((7, 7), np.uint8), iterations=1)
        line_dark = float(np.mean((255 - gray)[line_mask > 0])) if np.any(line_mask > 0) else 0.0
        local_dark_cov = (
            float(np.mean((gray[local > 0] < 150).astype(np.float32))) if np.any(local > 0) else 1.0
        )
        cov_limit = 0.52 if is_spine else 0.34
        if line_dark < 16.0 or local_dark_cov > cov_limit:
            return False
        boost = 82 if is_spine else 60
        out[local > 0] = np.minimum(255, out[local > 0].astype(np.int16) + boost).astype(np.uint8)
        removed[local > 0] = 255
        removed_lines += 1
        if is_spine:
            removed_spine += 1
        else:
            removed_fold += 1
        return True

    # Primary spine detector: full-height dark vertical structures.
    dark = (gray < 165).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((1, 23), np.uint8))
    candidate_cols: list[tuple[int, int]] = []
    active_start = -1
    for x in range(w):
        col = dark[:, x].astype(bool)
        run = 0
        best = 0
        for pix in col:
            if pix:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        density = float(np.mean(col))
        col_ok = best >= int(0.72 * h) and 0.03 <= density <= 0.6
        if col_ok:
            if active_start < 0:
                active_start = x
        elif active_start >= 0:
            candidate_cols.append((active_start, x - 1))
            active_start = -1
    if active_start >= 0:
        candidate_cols.append((active_start, w - 1))

    for x0, x1 in candidate_cols:
        cx = int((x0 + x1) * 0.5)
        is_spine = cx <= int(0.27 * w) or cx >= int(0.73 * w)
        if not is_spine:
            continue
        line_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.rectangle(line_mask, (x0, 0), (x1, h - 1), 255, thickness=-1)
        _apply_local_clear(line_mask, is_spine=True)

    # Fallback for low-contrast straight spine lines: use vertical gradient peaks by column.
    gradx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    col_strength = np.mean(gradx, axis=0)
    if col_strength.size > 0:
        zone = np.zeros(w, dtype=bool)
        zone[: int(0.3 * w)] = True
        zone[int(0.7 * w) :] = True
        zone_vals = col_strength[zone]
        if zone_vals.size > 0:
            thr = float(np.percentile(zone_vals, 92))
            active = col_strength >= max(4.0, thr)
            x = 0
            while x < w:
                if not active[x]:
                    x += 1
                    continue
                x0 = x
                while x < w and active[x]:
                    x += 1
                x1 = x - 1
                if x1 - x0 + 1 > 18:
                    continue
                cx = int((x0 + x1) * 0.5)
                if not (cx <= int(0.3 * w) or cx >= int(0.7 * w)):
                    continue
                strip = gray[:, x0 : x1 + 1]
                if strip.size == 0:
                    continue
                if float(np.mean(strip)) > 238.0:
                    continue
                line_mask = np.zeros_like(gray, dtype=np.uint8)
                cv2.rectangle(line_mask, (x0, 0), (x1, h - 1), 255, thickness=-1)
                _apply_local_clear(line_mask, is_spine=True)

    if lines is not None:
        for line_pts in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line_pts]
            dx = x2 - x1
            dy = y2 - y1
            length = float((dx * dx + dy * dy) ** 0.5)
            if length < min_len:
                continue

            # Keep this pass focused on structural near-vertical / fold-like lines.
            angle = abs(np.degrees(np.arctan2(dy, max(1e-6, dx))))
            if angle < 62.0:
                continue

            mx = int((x1 + x2) * 0.5)
            in_border = mx <= bx or mx >= (w - bx)
            in_spine_zone = mx <= int(0.23 * w) or mx >= int(0.77 * w)
            if not (in_border or in_spine_zone or length >= 0.88 * h):
                continue

            line_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=3)
            _apply_local_clear(line_mask, is_spine=in_border)

    _save_gray(after, out)
    _save_gray(removed_path, removed)
    _save_gray(Path(cleaned_gray_path), out)
    meta["line_refine_applied"] = removed_lines > 0
    meta["line_refine_removed_lines"] = removed_lines
    meta["line_refine_removed_spine_lines"] = removed_spine
    meta["line_refine_removed_fold_lines"] = removed_fold
    meta["line_refine_reason"] = "removed_structural_lines" if removed_lines > 0 else "none_removed"
    return meta
