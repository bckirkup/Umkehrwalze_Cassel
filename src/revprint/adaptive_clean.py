"""Adaptive ink extraction using Sauvola binarization with optional GPU acceleration.

Replaces the fixed-threshold ``_ink_on_white()`` in ``edge_refine.py`` with a
locally-adaptive approach that handles:
- Uneven illumination (spine shadows, staining)
- Faded ink near edges
- Variable paper brightness across the corpus
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import threshold_sauvola

_HAS_GPU = False
try:
    import kornia
    import torch

    _HAS_GPU = torch.cuda.is_available()
except ImportError:
    pass


def _load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        gray = ImageOps.exif_transpose(im).convert("L")
    return np.asarray(gray, dtype=np.uint8)


def _sauvola_ink_mask_cpu(
    gray: np.ndarray,
    window_size: int = 51,
    k: float = 0.15,
) -> np.ndarray:
    """Compute adaptive ink mask using Sauvola binarization (CPU).

    Returns a float32 mask in [0, 1] where 1 = ink.
    """
    thresh = threshold_sauvola(gray, window_size=window_size, k=k)
    binary = gray < thresh
    return binary.astype(np.float32)


def _sauvola_ink_mask_gpu(
    gray: np.ndarray,
    window_size: int = 51,
    k: float = 0.15,
) -> np.ndarray:
    """Compute adaptive ink mask using local mean/std on GPU via kornia.

    Approximates Sauvola: T(x,y) = mean(x,y) * (1 + k * (std(x,y)/R - 1))
    """
    tensor = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()

    kernel = (window_size, window_size)
    local_mean = kornia.filters.box_blur(tensor, kernel)
    local_sq_mean = kornia.filters.box_blur(tensor ** 2, kernel)
    local_std = torch.sqrt(torch.clamp(local_sq_mean - local_mean ** 2, min=0.0))

    r = 128.0  # Sauvola dynamic range
    threshold = local_mean * (1.0 + k * (local_std / r - 1.0))
    mask = (tensor < threshold).float()
    return mask.squeeze().cpu().numpy()


def sauvola_ink_mask(
    gray: np.ndarray,
    window_size: int = 51,
    k: float = 0.15,
    prefer_gpu: bool = True,
) -> np.ndarray:
    """Compute Sauvola ink mask, using GPU if available."""
    if prefer_gpu and _HAS_GPU:
        return _sauvola_ink_mask_gpu(gray, window_size=window_size, k=k)
    return _sauvola_ink_mask_cpu(gray, window_size=window_size, k=k)


def extract_ink_on_white(
    gray: np.ndarray,
    window_size: int = 51,
    k: float = 0.15,
    target_white: float = 252.0,
    ink_preserve_gamma: float = 0.85,
    prefer_gpu: bool = True,
) -> np.ndarray:
    """Produce a clean ink-on-white grayscale image.

    Unlike simple binarization, this preserves pen-pressure variation:
    darker strokes remain darker, lighter strokes remain lighter, but
    all non-ink areas become uniform white.

    Parameters
    ----------
    gray : uint8 grayscale image
    window_size : Sauvola window size (must be odd)
    k : Sauvola sensitivity (lower = more aggressive ink detection)
    target_white : brightness level for paper pixels
    ink_preserve_gamma : gamma correction for ink pixels (<1 = lighter, >1 = darker)
    prefer_gpu : use GPU if available
    """
    mask = sauvola_ink_mask(gray, window_size=window_size, k=k, prefer_gpu=prefer_gpu)

    # Local background estimate via large Gaussian blur.
    bg = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=40.0)
    bg = np.clip(bg, 1.0, 255.0)

    # Normalize ink intensity relative to local background.
    normalized = (gray.astype(np.float32) / bg) * target_white
    normalized = np.clip(normalized, 0.0, 255.0)

    # Apply gamma to ink pixels to control darkness.
    ink_norm = np.clip(normalized / 255.0, 0.0, 1.0)
    ink_corrected = np.power(ink_norm, ink_preserve_gamma) * 255.0

    # Blend: ink regions use corrected values, non-ink regions become white.
    result = np.where(
        mask > 0.5,
        ink_corrected,
        target_white,
    )
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def adaptive_clean_page(
    input_path: Path,
    output_path: Path,
    window_size: int = 51,
    k: float = 0.15,
    target_white: float = 252.0,
    prefer_gpu: bool = True,
) -> dict[str, object]:
    """Clean a single page using adaptive ink extraction.

    Returns metadata dict.
    """
    gray = _load_gray(input_path)
    cleaned = extract_ink_on_white(
        gray,
        window_size=window_size,
        k=k,
        target_white=target_white,
        prefer_gpu=prefer_gpu,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cleaned, mode="L").save(output_path)
    return {
        "adaptive_clean_applied": True,
        "adaptive_clean_window_size": window_size,
        "adaptive_clean_k": k,
        "adaptive_clean_target_white": target_white,
        "adaptive_clean_gpu_used": prefer_gpu and _HAS_GPU,
        "adaptive_clean_output_path": str(output_path),
    }
