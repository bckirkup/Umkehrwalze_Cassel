from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation


@dataclass(frozen=True)
class InteractionArtifact:
    source_path: str
    neighbor_path: str
    relation: str
    """Sub-pixel shift (rows, cols) in analysis (downscaled) coordinates."""
    shift_yx: tuple[float, float]
    """Normalized RMS error from phase correlation; lower is better."""
    registration_error: float
    """Heuristic confidence in [0,1] derived from error and shift plausibility."""
    registration_confidence: float
    registration_applied: bool
    registration_reason: str
    """Mean value of body registration mask on ink channel (higher = more interior ink used)."""
    body_mask_coverage: float
    """Scale from full-res page height to analysis height (for upsampling masks/shifts)."""
    analysis_scale_y: float
    analysis_shape_hw: tuple[int, int]
    mask_path: str
    overlay_path: str

    def to_meta(self) -> dict[str, object]:
        return asdict(self)


def _load_gray(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as im:
        g = ImageOps.grayscale(ImageOps.exif_transpose(im))
        if size is not None:
            g = g.resize(size, Image.Resampling.LANCZOS)
        return np.asarray(g, dtype=np.uint8)


def ink_probability(gray: np.ndarray) -> np.ndarray:
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    norm = cv2.divide(gray, bg, scale=245)
    dark = np.clip((218 - norm).astype(np.float32) / 70.0, 0.0, 1.0)
    gx = cv2.Sobel(norm.astype(np.float32) / 255.0, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm.astype(np.float32) / 255.0, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.clip(np.sqrt(gx * gx + gy * gy) * 10.0, 0.0, 1.0)
    score = np.where((norm < 218) | ((norm < 236) & (grad > 0.32)), dark * 0.82 + grad * 0.18, 0.0)
    return np.clip(score, 0.0, 1.0)


def _registration_signal(gray: np.ndarray) -> np.ndarray:
    """
    Build a stable texture map for registration from grayscale page data.
    This is denser than binary-ish ink maps and behaves better when text is sparse.
    """
    g = gray.astype(np.float32)
    blur_small = cv2.GaussianBlur(g, (0, 0), sigmaX=1.4, sigmaY=1.4)
    blur_large = cv2.GaussianBlur(g, (0, 0), sigmaX=8.0, sigmaY=8.0)
    dog = blur_small - blur_large
    # Keep darker-page texture and written strokes.
    dark = np.clip((220.0 - g) / 110.0, 0.0, 1.0)
    edge = np.clip(np.abs(dog) / 26.0, 0.0, 1.0)
    return np.clip(dark * 0.55 + edge * 0.45, 0.0, 1.0)


def _interior_band_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.ones((h, w), dtype=np.float32)
    y = max(12, h // 12)
    x = max(12, w // 12)
    m[:y, :] = 0
    m[-y:, :] = 0
    m[:, :x] = 0
    m[:, -x:] = 0
    return m


def _body_registration_mask(ink: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Mask out page borders, bottom label band, and strong ink edges so registration
    focuses on interior paper texture + faint ghosts rather than torn edges.
    """
    h, w = ink.shape
    interior = _interior_band_mask(ink.shape)
    # Drop bottom archive-label band (common on Staatsarchiv scans)
    label_band = int(h * 0.12)
    interior[-label_band:, :] = 0.0
    # Drop top title band slightly
    interior[: int(h * 0.04), :] = 0.0
    ink_bin = (ink > 0.35).astype(np.uint8) * 255
    ink_bin = cv2.dilate(ink_bin, np.ones((9, 9), np.uint8), iterations=2)
    body = interior * (1.0 - (ink_bin.astype(np.float32) / 255.0))
    valid = interior > 0.5
    if not np.any(valid):
        return body, 0.0
    # Coverage should reflect how much usable interior ink/texture is present
    # for registration, rather than average intensity over the full frame.
    ink_presence = float(np.mean(ink[valid] > 0.18))
    ink_strength = float(np.mean(ink[valid]))
    coverage = max(ink_presence, ink_strength)
    return body, coverage


def _resize_for_analysis(gray: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    h, w = gray.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return gray
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(mask * 255, 0, 255).astype(np.uint8), mode="L").save(path)


def _save_overlay(path: Path, current: np.ndarray, neighbor: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = current.shape
    overlay = np.full((h, w, 3), 255, dtype=np.uint8)
    current_ink = current > 0.52
    neighbor_ink = neighbor > 0.58
    overlay[current_ink] = (0, 0, 0)
    red = neighbor_ink & ~current_ink
    overlay[red] = (220, 0, 0)
    amber = neighbor_ink & current_ink
    overlay[amber] = (150, 80, 0)
    Image.fromarray(overlay, mode="RGB").save(path)


def _registration_decision(
    error: float,
    shift_yx: np.ndarray,
    h: int,
    w: int,
    body_coverage: float,
) -> tuple[bool, float, str]:
    """Return (applied, confidence, reason)."""
    if not np.isfinite(error):
        return False, 0.0, "invalid_registration_error"
    conf = float(max(0.0, min(1.0, 1.0 - error)))
    max_shift = 0.12 * max(h, w)
    if body_coverage < 0.006:
        return False, conf, "low_body_mask_coverage"
    if error > 0.88:
        return False, conf, "high_registration_error"
    if abs(float(shift_yx[0])) > max_shift or abs(float(shift_yx[1])) > max_shift:
        return False, conf, "shift_out_of_bounds"
    if conf < 0.18:
        return False, conf, "low_confidence"
    return True, conf, "ok"


def _analyze_pair(
    source_path: Path,
    neighbor_path: Path,
    relation: str,
    output_dir: Path,
) -> InteractionArtifact:
    full_src = _load_gray(source_path)
    full_h, full_w = full_src.shape
    src_g = _resize_for_analysis(full_src)
    h, w = src_g.shape
    analysis_scale_y = float(full_h / h)

    nbr_g = _load_gray(neighbor_path, size=(w, h))
    nbr_g = np.fliplr(nbr_g)
    src_ink = ink_probability(src_g)
    nbr_ink = ink_probability(nbr_g)
    src_sig = _registration_signal(src_g)
    nbr_sig = _registration_signal(nbr_g)
    body, coverage = _body_registration_mask(src_ink)
    src_reg = src_sig * body
    nbr_reg = nbr_sig * body

    shift_yx = np.array([0.0, 0.0], dtype=np.float64)
    error = 1.0
    try:
        if float(np.sum(body)) > 1e-3 and float(np.std(src_reg)) > 1e-4 and float(np.std(nbr_reg)) > 1e-4:
            shift_yx, error, _ = phase_cross_correlation(
                src_reg,
                nbr_reg,
                upsample_factor=4,
                normalization="phase",
            )
            # Fallback when phase-normalized registration becomes unstable on
            # very sparse/flat masked data.
            if (not np.isfinite(error)) or float(error) > 0.98:
                shift2, error2, _ = phase_cross_correlation(
                    src_reg,
                    nbr_reg,
                    upsample_factor=4,
                    normalization=None,
                )
                if np.isfinite(error2):
                    shift_yx, error = shift2, float(error2)
    except Exception:
        shift_yx = np.array([0.0, 0.0])
        error = 1.0

    applied, conf, reason = _registration_decision(error, shift_yx, h, w, coverage)
    if not applied:
        shift_yx = np.array([0.0, 0.0])

    aligned = ndi_shift(nbr_ink, shift=tuple(float(v) for v in shift_yx), order=1, mode="constant", cval=0.0)
    ghost = np.clip(aligned * (1.0 - np.clip(src_ink * 1.7, 0.0, 1.0)), 0.0, 1.0)
    ghost = cv2.GaussianBlur(ghost.astype(np.float32), (0, 0), sigmaX=0.8)
    ghost = np.where(ghost > 0.55, ghost, 0.0)

    stem = source_path.stem
    mask_path = output_dir / f"{stem}.interaction_{relation}_mirror_mask.png"
    overlay_path = output_dir / f"{stem}.interaction_{relation}_mirror_overlay.png"
    _save_mask(mask_path, ghost)
    _save_overlay(overlay_path, src_ink, aligned)
    return InteractionArtifact(
        source_path=str(source_path),
        neighbor_path=str(neighbor_path),
        relation=relation,
        shift_yx=(float(shift_yx[0]), float(shift_yx[1])),
        registration_error=float(error),
        registration_confidence=conf,
        registration_applied=applied,
        registration_reason=reason,
        body_mask_coverage=coverage,
        analysis_scale_y=analysis_scale_y,
        analysis_shape_hw=(h, w),
        mask_path=str(mask_path),
        overlay_path=str(overlay_path),
    )


def analyze_interactions_for_source(
    source_path: Path,
    all_files: list[Path],
    output_dir: Path,
) -> list[InteractionArtifact]:
    source_path = Path(source_path).resolve()
    resolved = [Path(p).resolve() for p in all_files]
    if source_path not in resolved:
        return []
    idx = resolved.index(source_path)
    artifacts: list[InteractionArtifact] = []
    if idx > 0:
        artifacts.append(_analyze_pair(source_path, resolved[idx - 1], "previous", output_dir))
    if idx + 1 < len(resolved):
        artifacts.append(_analyze_pair(source_path, resolved[idx + 1], "next", output_dir))
    return artifacts
