from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def _estimate_skew_deg(gray_u8: np.ndarray) -> float:
    """Estimate small page skew from dominant ink orientation (downscaled)."""
    h, w = gray_u8.shape
    if h < 32 or w < 32:
        return 0.0
    scale = min(1.0, 520.0 / max(h, w))
    small = cv2.resize(gray_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.04 * float(bw.size):
        return 0.0
    rect = cv2.minAreaRect(largest)
    angle = float(rect[-1])
    if angle < -45.0:
        skew = -(90.0 + angle)
    else:
        skew = -angle
    if abs(skew) > 14.0:
        return 0.0
    return skew


def dewarp_grayscale_optional(
    cleaned_gray_path: Path,
    pages_dir: Path,
    stem: str,
    enable: bool,
) -> dict[str, object]:
    """
    Optional deskew pass: writes ``{stem}.dewarped_gray.png`` next to the cleaned page.
    Keeps the original cleaned file unchanged; reproduction can prefer dewarped when configured.
    """
    cleaned_gray_path = Path(cleaned_gray_path).resolve()
    pages_dir = Path(pages_dir).resolve()
    dewarped_path = pages_dir / f"{stem}.dewarped_gray.png"
    meta: dict[str, object] = {
        "dewarp_enabled": enable,
        "dewarped_grayscale_path": "",
        "dewarp_skew_degrees": 0.0,
        "dewarp_applied": False,
        "dewarp_reason": "disabled" if not enable else "not_needed",
    }
    if not enable or not cleaned_gray_path.is_file():
        if not enable:
            meta["dewarp_reason"] = "disabled"
        else:
            meta["dewarp_reason"] = "missing_cleaned_gray"
        return meta

    with Image.open(cleaned_gray_path) as im:
        gray = ImageOps.grayscale(ImageOps.exif_transpose(im))
        arr = np.asarray(gray, dtype=np.uint8)

    skew = _estimate_skew_deg(arr)
    meta["dewarp_skew_degrees"] = skew
    if abs(skew) < 0.12:
        meta["dewarp_reason"] = "skew_below_threshold"
        Image.fromarray(arr, mode="L").save(dewarped_path)
        meta["dewarped_grayscale_path"] = str(dewarped_path)
        meta["dewarp_applied"] = False
        return meta

    h, w = arr.shape
    center = (w / 2.0, h / 2.0)
    rot = cv2.getRotationMatrix2D(center, skew, 1.0)
    cos = abs(rot[0, 0])
    sin = abs(rot[0, 1])
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))
    rot[0, 2] += (n_w / 2) - center[0]
    rot[1, 2] += (n_h / 2) - center[1]
    out = cv2.warpAffine(arr, rot, (n_w, n_h), flags=cv2.INTER_LINEAR, borderValue=255)
    Image.fromarray(out, mode="L").save(dewarped_path, optimize=True)
    meta["dewarped_grayscale_path"] = str(dewarped_path)
    meta["dewarp_applied"] = True
    meta["dewarp_reason"] = "rotated"
    return meta
