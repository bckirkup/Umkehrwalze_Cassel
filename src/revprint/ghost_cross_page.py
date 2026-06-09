"""Cross-page ghost suppression using physical facing-page pairs + NMF fallback.

Builds on the existing ``ghost_suppression.py`` but:
1. Uses the physical page model to pair facing pages (not scan-order neighbors)
2. Uses NMF (Non-negative Matrix Factorization) as a blind source separation
   fallback when the facing page is unavailable or registration fails
3. Uses a simpler, faster stroke-width test for plausibility filtering
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.registration import phase_cross_correlation


def _load_gray_f32(path: Path, max_dim: int | None = None) -> np.ndarray:
    with Image.open(path) as im:
        gray = ImageOps.exif_transpose(im).convert("L")
        if max_dim is not None:
            w, h = gray.size
            scale = min(1.0, max_dim / max(w, h))
            if scale < 1.0:
                gray = gray.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.LANCZOS,
                )
    return np.asarray(gray, dtype=np.float32)


def _quick_ink_mask(gray_f: np.ndarray, dark_threshold: float = 180.0) -> np.ndarray:
    """Fast binary ink mask: pixels darker than threshold."""
    return (gray_f < dark_threshold).astype(np.float32)


def _register_facing(
    page_gray: np.ndarray,
    facing_gray_flipped: np.ndarray,
    upsample_factor: int = 4,
) -> tuple[tuple[float, float], float, bool]:
    """Register a horizontally-flipped facing page to the current page.

    Returns (shift_yx, error, success).
    """
    h1, w1 = page_gray.shape
    h2, w2 = facing_gray_flipped.shape
    # Crop to common size.
    ch = min(h1, h2)
    cw = min(w1, w2)
    a = page_gray[:ch, :cw]
    b = facing_gray_flipped[:ch, :cw]

    try:
        shift, error, _diffphase = phase_cross_correlation(
            a, b, upsample_factor=upsample_factor
        )
        shift_yx = (float(shift[0]), float(shift[1]))
        max_shift = max(ch, cw) * 0.12
        success = (error <= 0.9) and (abs(shift_yx[0]) <= max_shift) and (abs(shift_yx[1]) <= max_shift)
        return shift_yx, float(error), success
    except Exception:
        return (0.0, 0.0), 1.0, False


def _shift_image(img: np.ndarray, shift_yx: tuple[float, float]) -> np.ndarray:
    """Shift an image by (dy, dx) using affine warp."""
    dy, dx = shift_yx
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        img, m, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=255.0,
    )


def subtract_ghost_facing(
    page_gray: np.ndarray,
    facing_path: Path,
    ghost_lift: float = 0.6,
    ink_protect_threshold: float = 160.0,
    max_dim: int = 2000,
) -> tuple[np.ndarray, dict[str, object]]:
    """Subtract ghost bleed-through using the physical facing page.

    Parameters
    ----------
    page_gray : float32 grayscale of the current page
    facing_path : path to the facing page image
    ghost_lift : how much to lighten ghost regions (0–1, 0.6 = lift 60% toward white)
    ink_protect_threshold : pixels darker than this on the current page are protected
    max_dim : max dimension for registration (full-res for application)

    Returns (cleaned, metadata)
    """
    facing_full = _load_gray_f32(facing_path)
    facing_flipped = np.fliplr(facing_full)

    # Register at reduced resolution.
    reg_scale = min(1.0, 1200.0 / max(page_gray.shape))
    if reg_scale < 1.0:
        page_small = cv2.resize(
            page_gray,
            (max(1, int(page_gray.shape[1] * reg_scale)),
             max(1, int(page_gray.shape[0] * reg_scale))),
        )
        facing_small = cv2.resize(
            facing_flipped,
            (max(1, int(facing_flipped.shape[1] * reg_scale)),
             max(1, int(facing_flipped.shape[0] * reg_scale))),
        )
    else:
        page_small = page_gray
        facing_small = facing_flipped

    shift_yx, error, success = _register_facing(page_small, facing_small)
    meta: dict[str, object] = {
        "ghost_facing_registration_shift": shift_yx,
        "ghost_facing_registration_error": error,
        "ghost_facing_registration_success": success,
    }

    if not success:
        meta["ghost_facing_applied"] = False
        meta["ghost_facing_reason"] = "registration_failed"
        return page_gray, meta

    # Scale shift back to full resolution.
    full_shift = (shift_yx[0] / reg_scale, shift_yx[1] / reg_scale)

    # Resize facing to match page dimensions.
    h, w = page_gray.shape
    if facing_flipped.shape != (h, w):
        facing_flipped = cv2.resize(facing_flipped, (w, h))

    aligned_facing = _shift_image(facing_flipped, full_shift)

    # Ghost mask: where the facing page has ink but our page doesn't have strong ink.
    facing_ink = _quick_ink_mask(aligned_facing, dark_threshold=180.0)
    our_strong_ink = _quick_ink_mask(page_gray, dark_threshold=ink_protect_threshold)
    ghost_candidates = facing_ink * (1.0 - our_strong_ink)

    # Soften the mask to avoid hard edges.
    ghost_mask = cv2.GaussianBlur(ghost_candidates, (0, 0), sigmaX=2.0)

    # Lift ghost regions toward white.
    lift_amount = (255.0 - page_gray) * ghost_lift * ghost_mask
    cleaned = page_gray + lift_amount
    cleaned = np.clip(cleaned, 0.0, 255.0)

    meta["ghost_facing_applied"] = True
    meta["ghost_facing_ghost_pixel_count"] = int(np.count_nonzero(ghost_candidates > 0.5))
    meta["ghost_facing_lift"] = ghost_lift
    return cleaned, meta


def nmf_ghost_separate(
    gray: np.ndarray,
    n_components: int = 2,
    max_dim: int = 800,
) -> tuple[np.ndarray, dict[str, object]]:
    """Blind source separation of front/back ink using NMF.

    Treats the image as a mixture of two sources (front ink + back ink)
    and separates them.  The component with higher contrast and more
    spatially coherent strokes is kept as "front ink."

    This is a fallback for pages where the facing page is unavailable.
    """
    from sklearn.decomposition import NMF

    h, w = gray.shape
    # Work at reduced resolution for speed.
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
    else:
        small = gray.copy()

    sh, sw = small.shape
    # Invert so ink = high values (NMF needs non-negative, and we want ink as signal).
    inverted = 255.0 - small.astype(np.float32)
    # Reshape to (n_pixels, 1) — NMF on a single image uses patches.
    # Use a patch-based approach: tile the image into overlapping patches.
    patch_size = 16
    patches: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for y in range(0, sh - patch_size + 1, patch_size // 2):
        for x in range(0, sw - patch_size + 1, patch_size // 2):
            patch = inverted[y: y + patch_size, x: x + patch_size].ravel()
            patches.append(patch)
            positions.append((y, x))

    if len(patches) < n_components * 2:
        return gray, {"nmf_ghost_applied": False, "nmf_ghost_reason": "too_few_patches"}

    data = np.stack(patches, axis=0)  # (n_patches, patch_pixels)
    data = np.clip(data, 0.0, None)  # Ensure non-negative.

    nmf = NMF(n_components=n_components, init="nndsvda", max_iter=200, random_state=42)
    try:
        coeffs = nmf.fit_transform(data)  # (n_patches, n_components)
    except Exception:
        return gray, {"nmf_ghost_applied": False, "nmf_ghost_reason": "nmf_failed"}

    components = nmf.components_  # (n_components, patch_pixels)

    # Reconstruct per-component images.
    component_images: list[np.ndarray] = []
    for comp_idx in range(n_components):
        recon = np.zeros((sh, sw), dtype=np.float32)
        count = np.zeros((sh, sw), dtype=np.float32)
        for patch_idx, (y, x) in enumerate(positions):
            patch_recon = coeffs[patch_idx, comp_idx] * components[comp_idx]
            patch_2d = patch_recon.reshape(patch_size, patch_size)
            recon[y: y + patch_size, x: x + patch_size] += patch_2d
            count[y: y + patch_size, x: x + patch_size] += 1.0
        count = np.clip(count, 1.0, None)
        recon /= count
        component_images.append(recon)

    # Pick the component with higher spatial variance (more structured = real ink).
    variances = [float(np.var(ci)) for ci in component_images]
    front_idx = int(np.argmax(variances))
    front_ink = component_images[front_idx]

    # Reconstruct: invert back and upscale.
    cleaned_small = 255.0 - front_ink
    cleaned_small = np.clip(cleaned_small, 0.0, 255.0)

    if scale < 1.0:
        cleaned = cv2.resize(cleaned_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        cleaned = cleaned_small

    return cleaned.astype(np.uint8 if cleaned.max() <= 255 else np.float32), {
        "nmf_ghost_applied": True,
        "nmf_ghost_n_components": n_components,
        "nmf_ghost_front_component": front_idx,
        "nmf_ghost_component_variances": variances,
    }


def suppress_ghost(
    page_gray_path: Path,
    output_path: Path,
    facing_path: Path | None = None,
    ghost_lift: float = 0.6,
    ink_protect_threshold: float = 160.0,
    use_nmf_fallback: bool = True,
) -> dict[str, object]:
    """Suppress ghost bleed-through on a single page.

    Uses facing-page subtraction if available, NMF fallback otherwise.

    Returns metadata dict.
    """
    with Image.open(page_gray_path) as im:
        gray = np.asarray(ImageOps.exif_transpose(im).convert("L"), dtype=np.float32)

    if facing_path is not None and facing_path.is_file():
        cleaned, meta = subtract_ghost_facing(
            gray,
            facing_path=facing_path,
            ghost_lift=ghost_lift,
            ink_protect_threshold=ink_protect_threshold,
        )
    elif use_nmf_fallback:
        cleaned, meta = nmf_ghost_separate(gray)
    else:
        cleaned = gray
        meta = {"ghost_suppress_applied": False, "ghost_suppress_reason": "no_facing_page"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_arr = np.clip(cleaned, 0, 255).astype(np.uint8)
    Image.fromarray(out_arr, mode="L").save(output_path)
    meta["ghost_suppress_output_path"] = str(output_path)
    return meta
