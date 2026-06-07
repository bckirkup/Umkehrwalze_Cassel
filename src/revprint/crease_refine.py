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


def apply_crease_refine(
    *,
    stem: str,
    pages_dir: Path,
    cleaned_gray_path: Path,
    enable: bool,
    darkness_threshold: float = 12.0,
) -> dict[str, object]:
    gray = _load_gray(cleaned_gray_path)
    pages_dir = Path(pages_dir)
    before = pages_dir / f"{stem}.crease_before.png"
    after = pages_dir / f"{stem}.crease_after.png"
    mask_path = pages_dir / f"{stem}.crease_mask.png"
    _save_gray(before, gray)
    meta: dict[str, object] = {
        "crease_refine_enabled": bool(enable),
        "crease_before_path": str(before),
        "crease_after_path": str(after),
        "crease_mask_path": str(mask_path),
        "crease_refine_applied": False,
        "crease_regions_removed": 0,
        "crease_darkness_threshold": float(darkness_threshold),
    }
    if not enable:
        _save_gray(after, gray)
        _save_gray(mask_path, np.zeros_like(gray, dtype=np.uint8))
        return meta

    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=23, sigmaY=23)
    dark_gap = np.clip(blur.astype(np.float32) - gray.astype(np.float32), 0, 255)
    cand = (dark_gap >= float(max(4.0, darkness_threshold))).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    keep = np.zeros_like(cand, dtype=np.uint8)
    out = gray.astype(np.float32)
    count = 0
    left_zone = int(0.22 * w)
    right_zone = int(0.78 * w)
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 40:
            continue
        long_vertical = ch >= int(0.38 * h) and cw <= int(0.08 * w)
        long_horizontal = cw >= int(0.42 * w) and ch <= int(0.08 * h)
        cx = x + cw // 2
        in_spine_or_border = cx <= left_zone or cx >= right_zone
        if not (long_vertical or long_horizontal or in_spine_or_border):
            continue
        comp = labels == i
        lift = np.minimum(54.0, dark_gap * (1.25 if long_vertical else 0.8))
        out[comp] = np.minimum(255.0, out[comp] + lift[comp])
        keep[comp] = 255
        count += 1

    _save_gray(after, out.astype(np.uint8))
    _save_gray(mask_path, keep)
    _save_gray(Path(cleaned_gray_path), out.astype(np.uint8))
    meta["crease_refine_applied"] = count > 0
    meta["crease_regions_removed"] = count
    return meta
