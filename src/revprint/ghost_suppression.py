from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import shift as ndi_shift

from revprint.page_interactions import ink_probability
from revprint.plausibility import build_penstroke_plausibility


def _load_gray_full(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        g = ImageOps.grayscale(ImageOps.exif_transpose(im))
        return np.asarray(g, dtype=np.uint8)


def _resize_mask_to_full(mask_u8: np.ndarray, full_hw: tuple[int, int]) -> np.ndarray:
    fh, fw = full_hw
    m = mask_u8.astype(np.float32) / 255.0
    if m.shape == (fh, fw):
        return m
    return cv2.resize(m, (fw, fh), interpolation=cv2.INTER_LINEAR)


def apply_ghost_suppression(
    cleaned_gray_path: Path,
    neighbor_paths: dict[str, Path],
    interactions: list[dict[str, Any]],
    pages_dir: Path,
    stem: str,
    enable: bool,
    confidence_min: float = 0.18,
    plausibility_min: float = 0.55,
    plausibility_passes: int = 2,
) -> dict[str, Any]:
    """
    When enabled, lighten regions consistent with high-confidence mirrored-neighbor
    ghost candidates. Writes before/after review images; updates cleaned_gray_path in place.
    """
    before_path = pages_dir / f"{stem}.ghost_suppress_before.png"
    after_path = pages_dir / f"{stem}.ghost_suppress_after.png"
    gray = _load_gray_full(cleaned_gray_path)
    Image.fromarray(gray, mode="L").save(before_path)

    meta: dict[str, Any] = {
        "ghost_suppression_enabled": enable,
        "ghost_suppress_before_path": str(before_path),
        "ghost_suppress_after_path": str(after_path),
        "ghost_suppression_applied": False,
        "ghost_suppression_reason": "disabled" if not enable else "no_high_confidence_pairs",
        "plausibility_applied": False,
        "plausibility_threshold": float(plausibility_min),
        "plausibility_exhaustive_passes": int(max(1, plausibility_passes)),
    }
    if not enable:
        Image.fromarray(gray, mode="L").save(after_path)
        return meta

    fh, fw = gray.shape
    front_ink = ink_probability(gray)
    combined_ghost = np.zeros((fh, fw), dtype=np.float32)
    used = 0

    for inter in interactions:
        if not inter.get("registration_applied"):
            continue
        if float(inter.get("registration_confidence", 0.0)) < confidence_min:
            continue
        rel = str(inter.get("relation", ""))
        npath = neighbor_paths.get(rel)
        if npath is None or not Path(npath).is_file():
            continue
        mask_path = Path(str(inter["mask_path"]))
        if not mask_path.is_file():
            continue
        shape_hw = inter["analysis_shape_hw"]
        ah, aw = int(shape_hw[0]), int(shape_hw[1])
        shift = inter["shift_yx"]
        dy = float(shift[0]) * (fh / ah)
        dx = float(shift[1]) * (fw / aw)

        nbr = _load_gray_full(Path(npath))
        nbr = cv2.resize(nbr, (fw, fh), interpolation=cv2.INTER_AREA)
        nbr = np.fliplr(nbr)
        nbr_ink = ink_probability(nbr)
        aligned = ndi_shift(nbr_ink, shift=(dy, dx), order=1, mode="constant", cval=0.0)

        mask_small = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        ghost_small = mask_small.astype(np.float32) / 255.0
        ghost_full = _resize_mask_to_full((ghost_small * 255).astype(np.uint8), (fh, fw))
        ghost_full = np.clip(ghost_full * aligned, 0.0, 1.0)
        combined_ghost = np.clip(combined_ghost + ghost_full * 0.85, 0.0, 1.0)
        used += 1

    if used == 0:
        Image.fromarray(gray, mode="L").save(after_path)
        meta["ghost_suppression_reason"] = "no_eligible_interactions"
        return meta

    plausibility = build_penstroke_plausibility(
        stem=stem,
        pages_dir=pages_dir,
        ghost_candidate=combined_ghost,
        front_ink=front_ink,
        threshold=plausibility_min,
        exhaustive_passes=plausibility_passes,
    )
    meta.update(plausibility)
    meta["plausibility_applied"] = True
    protect_path = Path(str(plausibility["plausibility_protect_mask_path"]))
    protect_mask = np.asarray(Image.open(protect_path).convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    # suppress only non-plausible regions and where front ink confidence is weak
    lift = combined_ghost * (1.0 - protect_mask) * (1.0 - np.clip(front_ink * 2.2, 0.0, 1.0))
    lift_energy = float(np.mean(lift))
    if lift_energy < 1e-4:
        Image.fromarray(gray, mode="L").save(after_path)
        meta["ghost_suppression_applied"] = False
        meta["ghost_suppression_reason"] = "all_candidates_protected_by_plausibility"
        meta["ghost_suppression_neighbors_used"] = used
        return meta
    out = gray.astype(np.float32) * (1.0 - 0.62 * lift) + 255.0 * (0.62 * lift)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out_u8, mode="L").save(cleaned_gray_path)
    Image.fromarray(out_u8, mode="L").save(after_path)
    meta["ghost_suppression_applied"] = True
    meta["ghost_suppression_reason"] = f"merged_{used}_neighbor_masks"
    meta["ghost_suppression_neighbors_used"] = used
    return meta
