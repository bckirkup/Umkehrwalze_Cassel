from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

SourceEngine = Literal["tesseract", "manual", "htr", "none"]


@dataclass(frozen=True)
class TextSegment:
    text: str
    confidence: float | None
    bbox_xywh: tuple[int, int, int, int]
    language: str
    script: str


@dataclass(frozen=True)
class TextEvidence:
    source_engine: SourceEngine
    language: str
    script: str
    segments: list[TextSegment]

    def to_meta(self) -> dict[str, object]:
        return asdict(self)


def _full_image_bbox(size_wh: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size_wh
    return (0, 0, int(max(0, w)), int(max(0, h)))


def _line_bbox_for_index(size_wh: tuple[int, int], idx: int, total: int) -> tuple[int, int, int, int]:
    w, h = size_wh
    line_count = max(1, int(total))
    line_h = max(1, int(round(h / line_count)))
    y0 = max(0, min(h - 1, idx * line_h))
    y1 = max(y0 + 1, min(h, y0 + line_h))
    return (0, y0, int(max(0, w)), int(max(0, y1 - y0)))


def _segmentize_text(
    *,
    text: str,
    size_wh: tuple[int, int],
    language: str,
    script: str,
    confidence: float,
) -> list[TextSegment]:
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = raw_lines if raw_lines else [text.strip()]
    segs: list[TextSegment] = []
    for idx, ln in enumerate(lines[:64]):
        segs.append(
            TextSegment(
                text=ln,
                confidence=confidence,
                bbox_xywh=_line_bbox_for_index(size_wh, idx, len(lines)),
                language=language,
                script=script,
            )
        )
    return segs


def extract_text_evidence(
    image_size_wh: tuple[int, int],
    ocr_text: str = "",
    manual_text: str | None = None,
    copied_english_text: str | None = None,
    htr_text: str | None = None,
) -> TextEvidence:
    """
    Lightweight extraction contract:
    - manual/copy sidecar text => source_engine=manual
    - OCR text => source_engine=tesseract
    - none => source_engine=none with empty segments
    """
    copied = (copied_english_text or "").strip()
    manual = (manual_text or "").strip()
    htr = (htr_text or "").strip()
    ocr = (ocr_text or "").strip()

    if copied:
        return TextEvidence(
            source_engine="manual",
            language="en",
            script="latin",
            segments=_segmentize_text(
                text=copied,
                size_wh=image_size_wh,
                language="en",
                script="latin",
                confidence=1.0,
            ),
        )
    if manual:
        return TextEvidence(
            source_engine="manual",
            language="de",
            script="unknown",
            segments=_segmentize_text(
                text=manual,
                size_wh=image_size_wh,
                language="de",
                script="unknown",
                confidence=1.0,
            ),
        )
    if htr:
        return TextEvidence(
            source_engine="htr",
            language="de",
            script="kurrent",
            segments=_segmentize_text(
                text=htr,
                size_wh=image_size_wh,
                language="de",
                script="kurrent",
                confidence=0.7,
            ),
        )
    if ocr:
        return TextEvidence(
            source_engine="tesseract",
            language="de",
            script="unknown",
            segments=_segmentize_text(
                text=ocr,
                size_wh=image_size_wh,
                language="de",
                script="unknown",
                confidence=0.45,
            ),
        )
    return TextEvidence(
        source_engine="none",
        language="unknown",
        script="unknown",
        segments=[],
    )
