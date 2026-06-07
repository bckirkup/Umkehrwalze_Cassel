from __future__ import annotations

import json
from pathlib import Path


def _stem_from_cleaned(path: Path) -> str:
    name = path.name
    suffix = ".cleaned_gray.png"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _load_seed_words(pages_dir: Path, stem: str, limit: int = 24) -> list[dict[str, object]]:
    words_json = pages_dir / f"{stem}.ocr_reconstruct_hint_words.json"
    if not words_json.is_file():
        return []
    try:
        data = json.loads(words_json.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = data.get("words", []) if isinstance(data, dict) else []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, object]] = []
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        bbox = item.get("bbox_xywh")
        if not text:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        out.append(
            {
                "text": text,
                "confidence": item.get("confidence"),
                "bbox_xywh": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "language": str(item.get("language", "de")),
                "script": "kurrent",
                "source": "ocr_seed",
            }
        )
    return out


def scaffold_htr_sidecars(pages_dir: Path, overwrite: bool = False) -> list[Path]:
    pages_dir = Path(pages_dir)
    cleaned = sorted(pages_dir.glob("*.cleaned_gray.png"))
    created: list[Path] = []
    for cp in cleaned:
        stem = _stem_from_cleaned(cp)
        out = pages_dir / f"{stem}.htr.json"
        if out.is_file() and not overwrite:
            continue
        payload = {
            "source_engine": "htr-sidecar",
            "language": "de",
            "script": "kurrent",
            "segments": [],
            "seed_candidates": _load_seed_words(pages_dir, stem),
            "notes": "Fill segments with confirmed line/phrase readings for HTR-first translation.",
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        created.append(out)
    return created
