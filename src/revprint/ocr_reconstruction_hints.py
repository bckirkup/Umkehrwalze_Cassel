from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L").save(path)


def build_ocr_reconstruction_hints(
    *,
    stem: str,
    pages_dir: Path,
    image_size_wh: tuple[int, int],
    words: list[dict[str, object]],
    enable: bool,
    confidence_min: float = 0.38,
) -> dict[str, object]:
    pages_dir = Path(pages_dir)
    mask_path = pages_dir / f"{stem}.ocr_reconstruct_hint_mask.png"
    json_path = pages_dir / f"{stem}.ocr_reconstruct_hint_words.json"
    w, h = image_size_wh
    mask = np.zeros((int(max(1, h)), int(max(1, w))), dtype=np.uint8)
    kept: list[dict[str, object]] = []
    if enable:
        for item in words:
            conf = float(item.get("confidence", 0.0) or 0.0)
            if conf < confidence_min:
                continue
            bbox = item.get("bbox_xywh")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x, y, bw, bh = [int(v) for v in bbox]
            if bw <= 0 or bh <= 0:
                continue
            x0 = max(0, x - 2)
            y0 = max(0, y - 2)
            x1 = min(mask.shape[1], x + bw + 2)
            y1 = min(mask.shape[0], y + bh + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            mask[y0:y1, x0:x1] = 255
            kept.append(item)
    _save_gray(mask_path, mask)
    json_path.write_text(json.dumps({"words": kept}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ocr_reconstruct_hint_enabled": bool(enable),
        "ocr_reconstruct_hint_confidence_min": float(confidence_min),
        "ocr_reconstruct_hint_mask_path": str(mask_path),
        "ocr_reconstruct_hint_words_json_path": str(json_path),
        "ocr_reconstruct_hint_word_count": len(kept),
    }
