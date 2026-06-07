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


def apply_speckle_refine(
    *,
    stem: str,
    pages_dir: Path,
    cleaned_gray_path: Path,
    enable: bool,
    max_component_area: int = 36,
    border_max_component_area: int = 90,
    border_band_ratio: float = 0.11,
) -> dict[str, object]:
    """
    Remove tiny isolated dark blobs while preserving connected strokes.
    Operates conservatively and writes before/after diagnostics.
    """
    pages_dir = Path(pages_dir)
    gray = _load_gray(cleaned_gray_path)
    before = pages_dir / f"{stem}.speckle_before.png"
    after = pages_dir / f"{stem}.speckle_after.png"
    mask_path = pages_dir / f"{stem}.speckle_removed_mask.png"
    _save_gray(before, gray)

    meta: dict[str, object] = {
        "speckle_refine_enabled": bool(enable),
        "speckle_before_path": str(before),
        "speckle_after_path": str(after),
        "speckle_removed_mask_path": str(mask_path),
        "speckle_refine_applied": False,
        "speckle_removed_components": 0,
        "speckle_removed_components_border": 0,
        "speckle_removed_components_inner": 0,
        "speckle_refine_reason": "disabled" if not enable else "none_removed",
        "speckle_max_component_area": int(max_component_area),
        "speckle_border_max_component_area": int(border_max_component_area),
        "speckle_border_band_ratio": float(border_band_ratio),
    }
    if not enable:
        _save_gray(after, gray)
        _save_gray(mask_path, np.zeros_like(gray, dtype=np.uint8))
        return meta

    # Border-focused cleanup mask: stronger at all outer edges (outside, spine, top, bottom).
    h, w = gray.shape
    bx = max(10, int(w * float(np.clip(border_band_ratio, 0.04, 0.25))))
    by = max(10, int(h * float(np.clip(border_band_ratio, 0.04, 0.25))))
    border_mask = np.zeros_like(gray, dtype=np.uint8)
    border_mask[:, :bx] = 255
    border_mask[:, -bx:] = 255
    border_mask[:by, :] = 255
    border_mask[-by:, :] = 255

    # Dark pixels candidate map
    dark = (gray < 150).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    removed = np.zeros_like(gray, dtype=np.uint8)
    out = gray.copy()
    count = 0
    count_border = 0
    count_inner = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        # keep elongated shapes (likely strokes); remove tiny compact dots
        elongated = max(w, h) / max(1, min(w, h))
        comp = labels == i
        in_border = bool(np.any(border_mask[comp] > 0))
        area_limit = border_max_component_area if in_border else max_component_area
        elongated_limit = 3.4 if in_border else 2.3
        if area <= area_limit and elongated < elongated_limit:
            out[comp] = np.minimum(255, out[comp].astype(np.int16) + 80).astype(np.uint8)
            removed[comp] = 255
            count += 1
            if in_border:
                count_border += 1
            else:
                count_inner += 1

    _save_gray(after, out)
    _save_gray(mask_path, removed)
    _save_gray(Path(cleaned_gray_path), out)
    meta["speckle_refine_applied"] = count > 0
    meta["speckle_removed_components"] = count
    meta["speckle_removed_components_border"] = count_border
    meta["speckle_removed_components_inner"] = count_inner
    meta["speckle_refine_reason"] = "removed_small_components" if count > 0 else "none_removed"
    return meta
