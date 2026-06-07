from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def _save_mask(path: Path, mask01: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(mask01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def _save_gray(path: Path, gray_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(gray_u8, 0, 255).astype(np.uint8), mode="L").save(path)


def _has_edge_text_hint(text_evidence: dict[str, Any]) -> bool:
    segments = text_evidence.get("segments")
    if not isinstance(segments, list):
        return False
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if len(text) >= 4:
            return True
    return False


def _text_border_affinity(text_evidence: dict[str, Any], shape_hw: tuple[int, int]) -> float:
    segments = text_evidence.get("segments")
    if not isinstance(segments, list) or not segments:
        return 0.0
    h, w = shape_hw
    score = 0.0
    seen = 0
    for seg in segments:
        bbox = seg.get("bbox_xywh")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x, y, bw, bh = [int(v) for v in bbox]
        if bw <= 0 or bh <= 0:
            continue
        cx = x + bw // 2
        cy = y + bh // 2
        dist_x = min(max(0, cx), max(0, w - 1), abs(w - 1 - cx))
        dist_y = min(max(0, cy), max(0, h - 1), abs(h - 1 - cy))
        near_border = 1.0 if min(dist_x / max(1.0, w), dist_y / max(1.0, h)) < 0.12 else 0.0
        score += near_border
        seen += 1
    if seen == 0:
        return 0.0
    return float(np.clip(score / seen, 0.0, 1.0))


def _estimate_penstroke_mask(gray: np.ndarray) -> np.ndarray:
    # Protect obvious handwritten structures from accidental brightening.
    ink = gray.astype(np.float32) < 150.0
    h, w = ink.shape
    horizontal = np.zeros_like(ink, dtype=bool)
    for k in (5, 9):
        run = np.convolve(np.ones(k, dtype=np.int16), np.ones(k, dtype=np.int16), mode="same")
        for y in range(h):
            row = ink[y].astype(np.int16)
            hit = np.convolve(row, np.ones(k, dtype=np.int16), mode="same") >= max(3, int(0.7 * k))
            horizontal[y] |= hit
        horizontal &= run.max() > 0
    vertical = np.zeros_like(ink, dtype=bool)
    for x in range(w):
        col = ink[:, x].astype(np.int16)
        hit = np.convolve(col, np.ones(3, dtype=np.int16), mode="same") >= 2
        vertical[:, x] = hit
    return np.logical_and(horizontal, vertical)


def build_edge_reconstruction_candidates(
    *,
    stem: str,
    pages_dir: Path,
    cleaned_gray_path: Path,
    edge_inpaint_mask_path: Path,
    text_evidence: dict[str, Any],
) -> dict[str, object]:
    """
    Language-aware edge candidate generation (safe, non-destructive):
    - combine edge inpaint mask with dark-ink signal near borders
    - weight confidence up when text evidence exists on the page
    - emit review artifacts only (no pixel replacement)
    """
    pages_dir = Path(pages_dir)
    gray = _load_gray(cleaned_gray_path)
    edge = _load_gray(edge_inpaint_mask_path)
    h, w = gray.shape
    border = np.zeros((h, w), dtype=np.float32)
    by = max(8, int(h * 0.08))
    bx = max(8, int(w * 0.08))
    border[:by, :] = 1.0
    border[-by:, :] = 1.0
    border[:, :bx] = 1.0
    border[:, -bx:] = 1.0

    ink = np.clip((215.0 - gray.astype(np.float32)) / 215.0, 0.0, 1.0)
    edge01 = (edge.astype(np.float32) / 255.0) * border
    candidates = np.clip(edge01 * ink, 0.0, 1.0)
    penstroke_mask = _estimate_penstroke_mask(gray)
    protect_mask = np.logical_and(penstroke_mask, edge01 > 0.0)
    candidates = np.where(protect_mask, candidates * 0.2, candidates)
    candidates = np.where(candidates > 0.12, candidates, 0.0)
    has_text_hint = _has_edge_text_hint(text_evidence)
    border_affinity = _text_border_affinity(text_evidence, (h, w))
    confidence = float(
        np.clip(
            np.mean(candidates) * 3.2
            + (0.12 if has_text_hint else 0.0)
            + (0.2 * border_affinity)
            - (0.22 * float(np.mean(protect_mask))),
            0.0,
            1.0,
        )
    )

    mask_path = pages_dir / f"{stem}.edge_reconstruct_candidate_mask.png"
    overlay_path = pages_dir / f"{stem}.edge_reconstruct_overlay.png"
    protect_mask_path = pages_dir / f"{stem}.edge_reconstruct_protect_mask.png"
    json_path = pages_dir / f"{stem}.edge_reconstruct_candidates.json"
    _save_mask(mask_path, candidates)
    _save_mask(protect_mask_path, protect_mask.astype(np.float32))

    overlay = np.full((h, w, 3), 255, dtype=np.uint8)
    ink_px = ink > 0.45
    cand_px = candidates > 0.2
    protect_px = protect_mask
    overlay[ink_px] = (15, 15, 15)
    overlay[protect_px] = (0, 140, 220)
    overlay[cand_px] = (220, 0, 180)
    Image.fromarray(overlay, mode="RGB").save(overlay_path)

    payload = {
        "edge_candidate_confidence": confidence,
        "edge_candidate_pixels": int(np.count_nonzero(cand_px)),
        "edge_candidate_protected_pixels": int(np.count_nonzero(protect_px)),
        "edge_candidate_protected_ratio": float(np.mean(protect_px)),
        "language_hint_applied": has_text_hint,
        "text_border_affinity": border_affinity,
        "candidate_keep_ratio": float(np.mean(candidates > 0.0)),
        "candidate_scoring_version": "v2_writer_aware",
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "edge_reconstruct_candidate_mask_path": str(mask_path),
        "edge_reconstruct_overlay_path": str(overlay_path),
        "edge_reconstruct_protect_mask_path": str(protect_mask_path),
        "edge_reconstruct_candidates_json_path": str(json_path),
        **payload,
    }


def apply_edge_reconstruction(
    *,
    stem: str,
    pages_dir: Path,
    cleaned_gray_path: Path,
    candidate_mask_path: Path,
    enable: bool,
    strength: float = 0.58,
) -> dict[str, object]:
    """
    Conservative edge cleanup pass:
    - brightens only candidate edge-noise regions
    - writes before/after artifacts
    - preserves original cleaned image unless enabled
    """
    pages_dir = Path(pages_dir)
    gray = _load_gray(cleaned_gray_path)
    before = pages_dir / f"{stem}.edge_reconstruct_before.png"
    after = pages_dir / f"{stem}.edge_reconstruct_after.png"
    applied_path = pages_dir / f"{stem}.edge_reconstruct_applied.png"
    _save_gray(before, gray)

    meta: dict[str, object] = {
        "edge_reconstruct_enabled": bool(enable),
        "edge_reconstruct_before_path": str(before),
        "edge_reconstruct_after_path": str(after),
        "edge_reconstruct_applied_path": str(applied_path),
        "edge_reconstruct_applied": False,
        "edge_reconstruct_reason": "disabled" if not enable else "no_candidate_mask",
        "edge_reconstruct_strength": float(strength),
    }
    if not enable:
        _save_gray(after, gray)
        return meta
    if not Path(candidate_mask_path).is_file():
        _save_gray(after, gray)
        return meta

    cand = _load_gray(candidate_mask_path).astype(np.float32) / 255.0
    if cand.shape != gray.shape:
        # nearest for mask semantics
        cand = np.asarray(
            Image.fromarray((cand * 255).astype(np.uint8), mode="L").resize(
                (gray.shape[1], gray.shape[0]),
                Image.Resampling.NEAREST,
            ),
            dtype=np.uint8,
        ).astype(np.float32) / 255.0
    if float(np.max(cand)) < 1e-3:
        _save_gray(after, gray)
        meta["edge_reconstruct_reason"] = "empty_candidate_mask"
        return meta

    s = float(np.clip(strength, 0.0, 1.0))
    lift = np.clip(cand * s, 0.0, 1.0)
    out = gray.astype(np.float32) * (1.0 - lift) + 255.0 * lift
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    _save_gray(after, out_u8)
    _save_gray(applied_path, out_u8)
    _save_gray(Path(cleaned_gray_path), out_u8)
    meta["edge_reconstruct_applied"] = True
    meta["edge_reconstruct_reason"] = "applied_candidate_brightening"
    return meta
