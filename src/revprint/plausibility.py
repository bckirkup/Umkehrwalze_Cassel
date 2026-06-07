from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _save_u8(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(path)


def _estimate_text_orientation(front_ink: np.ndarray) -> float:
    """Estimate dominant text angle in degrees (small-angle regime)."""
    u8 = np.clip(front_ink * 255.0, 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, threshold1=24, threshold2=72, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=36,
        minLineLength=max(20, int(0.045 * u8.shape[1])),
        maxLineGap=max(6, int(0.018 * u8.shape[1])),
    )
    if lines is None or len(lines) == 0:
        return 0.0
    angles: list[float] = []
    weights: list[float] = []
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in ln]
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1e-6:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        if abs(ang) > 42.0:
            continue
        length = float(np.hypot(dx, dy))
        angles.append(ang)
        weights.append(max(1e-3, length))
    if not angles:
        return 0.0
    return float(np.average(np.asarray(angles), weights=np.asarray(weights)))


def _component_plausibility(
    binary_u8: np.ndarray,
    front_ink: np.ndarray,
    orientation_deg: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    score_map = np.zeros(binary_u8.shape, dtype=np.float32)
    regions: list[dict[str, Any]] = []
    total_px = float(binary_u8.size)
    dist = cv2.distanceTransform(binary_u8, cv2.DIST_L2, 3)
    gx = cv2.Sobel(front_ink.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(front_ink.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    skel_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        comp = labels[y : y + h, x : x + w] == i
        ys, xs = np.where(comp)
        if ys.size < 6:
            continue
        # region shape cues
        aspect = float(max(w, h) / max(1, min(w, h)))
        fill_ratio = float(area / max(1, w * h))
        coords = np.column_stack([xs, ys]).astype(np.float32)
        cov = np.cov(coords, rowvar=False)
        evals = np.linalg.eigvals(cov)
        evals = np.sort(np.real(evals))
        elong = float(np.sqrt((evals[-1] + 1e-6) / (evals[0] + 1e-6)))
        comp_u8 = (comp.astype(np.uint8) * 255)
        # iterative thinning approximation (expensive but robust)
        eroded = comp_u8.copy()
        opened = np.zeros_like(comp_u8)
        skel = np.zeros_like(comp_u8)
        while cv2.countNonZero(eroded) > 0:
            tmp = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, skel_kernel)
            tmp = cv2.subtract(eroded, tmp)
            skel = cv2.bitwise_or(skel, tmp)
            eroded = cv2.erode(eroded, skel_kernel)
            opened = cv2.bitwise_or(opened, tmp)
            if cv2.countNonZero(opened) > int(1.5 * area):
                break
        skel_count = float(max(1, cv2.countNonZero(skel)))
        branch_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        neigh = cv2.filter2D((skel > 0).astype(np.uint8), cv2.CV_16S, branch_kernel)
        branch_points = float(np.count_nonzero(neigh >= 13))
        branchiness = float(branch_points / skel_count)
        local_dist = dist[y : y + h, x : x + w][comp]
        width_med = float(np.median(local_dist) * 2.0) if local_dist.size else 0.0
        width_std = float(np.std(local_dist) * 2.0) if local_dist.size else 0.0
        width_consistency = float(1.0 - min(1.0, width_std / (width_med + 1e-3)))
        local_grad = grad_mag[y : y + h, x : x + w][comp]
        edge_strength = float(np.mean(local_grad)) if local_grad.size else 0.0
        # ink confidence in component neighborhood
        ink_mean = float(np.mean(front_ink[y : y + h, x : x + w][comp]))
        norm_area = float(min(1.0, area / (0.02 * total_px)))
        # Orientation consistency with dominant page text slant
        vx = float(coords[-1, 0] - coords[0, 0]) if coords.shape[0] > 1 else 0.0
        vy = float(coords[-1, 1] - coords[0, 1]) if coords.shape[0] > 1 else 0.0
        comp_ang = float(np.degrees(np.arctan2(vy, vx))) if abs(vx) + abs(vy) > 1e-6 else 0.0
        delta = abs(comp_ang - orientation_deg)
        delta = min(delta, 180.0 - delta)
        orient_consistency = float(1.0 - min(1.0, delta / 36.0))
        # Weighted plausibility: elongated + moderate fill + darker ink => likely penstroke
        p = (
            0.22 * min(1.0, elong / 4.8)
            + 0.12 * min(1.0, aspect / 3.5)
            + 0.14 * (1.0 - min(1.0, abs(fill_ratio - 0.28) / 0.35))
            + 0.14 * ink_mean
            + 0.14 * width_consistency
            + 0.10 * orient_consistency
            + 0.08 * min(1.0, edge_strength / 0.22)
            + 0.06 * (1.0 - min(1.0, branchiness / 0.33))
        )
        # Penalize large compact blotches
        if fill_ratio > 0.62 and norm_area > 0.1:
            p *= 0.55
        p = float(np.clip(p, 0.0, 1.0))
        score_map[y : y + h, x : x + w][comp] = p
        regions.append(
            {
                "bbox_xywh": [x, y, w, h],
                "area_px": area,
                "elongation": elong,
                "aspect_ratio": aspect,
                "fill_ratio": fill_ratio,
                "ink_mean": ink_mean,
                "width_median_px": width_med,
                "width_std_px": width_std,
                "width_consistency": width_consistency,
                "orientation_consistency": orient_consistency,
                "branchiness": branchiness,
                "plausibility_score": p,
            }
        )
    return score_map, regions


def build_penstroke_plausibility(
    stem: str,
    pages_dir: Path,
    ghost_candidate: np.ndarray,
    front_ink: np.ndarray,
    threshold: float = 0.55,
    exhaustive_passes: int = 1,
    blur_sigmas: tuple[float, ...] = (0.0, 0.7, 1.2),
) -> dict[str, Any]:
    """
    Build plausibility artifacts for ghost candidates.
    Higher score = more likely true penstroke and should be protected.
    """
    pages_dir = Path(pages_dir)
    cand = np.clip(ghost_candidate, 0.0, 1.0)
    cand_u8 = (cand * 255).astype(np.uint8)
    _, bw = cv2.threshold(cand_u8, 1, 255, cv2.THRESH_BINARY)
    base_ink = np.clip(front_ink, 0.0, 1.0).astype(np.float32)
    orient = _estimate_text_orientation(base_ink)
    passes = max(1, int(exhaustive_passes))
    accum = np.zeros_like(base_ink, dtype=np.float32)
    all_regions: list[dict[str, Any]] = []
    for _ in range(passes):
        for sigma in blur_sigmas:
            ink = base_ink
            if sigma > 1e-6:
                ink = cv2.GaussianBlur(ink, (0, 0), sigmaX=sigma, sigmaY=sigma)
            s_map, regions = _component_plausibility(bw, ink, orient)
            accum += s_map
            all_regions.extend(regions)
    denom = float(max(1, passes * len(blur_sigmas)))
    score_map = np.clip(accum / denom, 0.0, 1.0)
    protect_mask = (score_map >= float(threshold)).astype(np.uint8) * 255
    heat = (score_map * 255.0).astype(np.uint8)

    heat_path = pages_dir / f"{stem}.plausibility_map.png"
    protect_path = pages_dir / f"{stem}.plausibility_protect_mask.png"
    regions_path = pages_dir / f"{stem}.plausibility_regions.json"
    _save_u8(heat_path, heat)
    _save_u8(protect_path, protect_mask)
    regions_path.write_text(
        json.dumps(
            {
                "orientation_deg": orient,
                "exhaustive_passes": passes,
                "blur_sigmas": list(blur_sigmas),
                "regions": all_regions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "plausibility_map_path": str(heat_path),
        "plausibility_protect_mask_path": str(protect_path),
        "plausibility_regions_path": str(regions_path),
        "plausibility_region_count": len(all_regions),
        "plausibility_threshold": float(threshold),
        "plausibility_mean": float(np.mean(score_map)) if all_regions else 0.0,
        "plausibility_max": float(np.max(score_map)) if all_regions else 0.0,
        "plausibility_orientation_degrees": float(orient),
        "plausibility_exhaustive_passes": passes,
    }
