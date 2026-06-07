from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HTRSegment:
    text: str
    confidence: float | None
    bbox_xywh: tuple[int, int, int, int]
    language: str
    script: str


@dataclass(frozen=True)
class HTREvidence:
    source_engine: str
    language: str
    script: str
    segments: list[HTRSegment]

    @property
    def full_text(self) -> str:
        return "\n".join(s.text for s in self.segments if s.text.strip()).strip()

    def to_meta(self) -> dict[str, object]:
        return {
            "source_engine": self.source_engine,
            "language": self.language,
            "script": self.script,
            "segments": [
                {
                    "text": s.text,
                    "confidence": s.confidence,
                    "bbox_xywh": list(s.bbox_xywh),
                    "language": s.language,
                    "script": s.script,
                }
                for s in self.segments
            ],
        }


def load_htr_sidecar(pages_dir: Path, stem: str, enabled: bool) -> tuple[HTREvidence | None, dict[str, object]]:
    path = Path(pages_dir) / f"{stem}.htr.json"
    meta: dict[str, object] = {
        "htr_enabled": bool(enabled),
        "htr_sidecar_path": str(path),
        "htr_used": False,
    }
    if not enabled or not path.is_file():
        return None, meta
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        meta["htr_error"] = f"read_error: {exc}"
        return None, meta
    if not isinstance(data, dict):
        meta["htr_error"] = "invalid_sidecar_shape"
        return None, meta
    segs_raw = data.get("segments", [])
    if not isinstance(segs_raw, list):
        segs_raw = []
    segs: list[HTRSegment] = []
    for it in segs_raw:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "")).strip()
        if not text:
            continue
        bbox = it.get("bbox_xywh", [0, 0, 0, 0])
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        try:
            conf = float(it["confidence"]) if it.get("confidence") is not None else None
        except Exception:
            conf = None
        segs.append(
            HTRSegment(
                text=text,
                confidence=conf,
                bbox_xywh=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                language=str(it.get("language", data.get("language", "de"))),
                script=str(it.get("script", data.get("script", "kurrent"))),
            )
        )
    if not segs:
        return None, meta
    ev = HTREvidence(
        source_engine=str(data.get("source_engine", "htr-sidecar")),
        language=str(data.get("language", "de")),
        script=str(data.get("script", "kurrent")),
        segments=segs,
    )
    meta["htr_used"] = True
    meta["htr_segment_count"] = len(segs)
    return ev, meta
