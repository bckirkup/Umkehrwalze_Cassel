from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image, ImageFilter, ImageOps

try:
    import pytesseract
except Exception:  # optional dependency
    pytesseract = None  # type: ignore[misc, assignment]


@dataclass(frozen=True)
class OCRCandidate:
    text: str
    confidence: float
    language: str
    preprocess: str


@dataclass(frozen=True)
class OCRWordHypothesis:
    text: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    language: str
    preprocess: str
    line_id: str


@dataclass(frozen=True)
class OCRPhraseHypothesis:
    text: str
    confidence: float
    language: str
    preprocess: str
    line_id: str


def _preprocess_variants(gray: Image.Image) -> list[tuple[str, Image.Image]]:
    base = ImageOps.autocontrast(gray, cutoff=1.0)
    return [
        ("autocontrast_median", base.filter(ImageFilter.MedianFilter(size=3))),
        ("autocontrast_only", base),
        (
            "high_contrast_bw",
            ImageOps.autocontrast(gray, cutoff=2.0).point(lambda p: 255 if p > 165 else 0, mode="L"),
        ),
    ]


def _ocr_confidence(img: Image.Image, lang: str) -> float:
    if pytesseract is None:
        return 0.0
    try:
        data = pytesseract.image_to_data(
            img,
            lang=lang,
            config="--psm 6",
            output_type=pytesseract.Output.DICT,
        )
        confs = []
        for c in data.get("conf", []):
            try:
                cf = float(c)
            except Exception:
                continue
            if cf >= 0:
                confs.append(cf)
        if not confs:
            return 0.0
        return float(sum(confs) / len(confs) / 100.0)
    except Exception:
        return 0.0


def ocr_candidates(gray: Image.Image) -> list[OCRCandidate]:
    if pytesseract is None:
        return []
    out: list[OCRCandidate] = []
    for tag, img in _preprocess_variants(gray):
        for lang in ("deu", "deu+eng"):
            try:
                text = pytesseract.image_to_string(img, lang=lang, config="--psm 6").strip()
            except Exception:
                text = ""
            if not text:
                continue
            conf = _ocr_confidence(img, lang=lang)
            out.append(OCRCandidate(text=text, confidence=conf, language=lang, preprocess=tag))
    out.sort(key=lambda c: (c.confidence, len(c.text)), reverse=True)
    return out


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def ocr_word_hypotheses(gray: Image.Image) -> tuple[list[OCRWordHypothesis], list[OCRPhraseHypothesis]]:
    if pytesseract is None:
        return [], []
    words: list[OCRWordHypothesis] = []
    phrases: list[OCRPhraseHypothesis] = []
    for tag, img in _preprocess_variants(gray):
        for lang in ("deu", "deu+eng"):
            try:
                data = pytesseract.image_to_data(
                    img,
                    lang=lang,
                    config="--psm 6",
                    output_type=pytesseract.Output.DICT,
                )
            except Exception:
                continue
            by_line: dict[str, list[OCRWordHypothesis]] = {}
            texts = data.get("text", [])
            for i, raw in enumerate(texts):
                text = str(raw).strip()
                if not text:
                    continue
                norm = _normalize_token(text)
                if len(norm) < 2:
                    continue
                try:
                    conf = float(data.get("conf", [])[i]) / 100.0
                except Exception:
                    conf = 0.0
                if conf <= 0.0:
                    continue
                try:
                    x = int(data.get("left", [])[i])
                    y = int(data.get("top", [])[i])
                    w = int(data.get("width", [])[i])
                    h = int(data.get("height", [])[i])
                except Exception:
                    continue
                if w <= 0 or h <= 0:
                    continue
                block = data.get("block_num", [0])[i]
                par = data.get("par_num", [0])[i]
                line = data.get("line_num", [0])[i]
                line_id = f"b{block}_p{par}_l{line}"
                wh = OCRWordHypothesis(
                    text=text,
                    confidence=max(0.0, min(1.0, conf)),
                    bbox_xywh=(x, y, w, h),
                    language=lang,
                    preprocess=tag,
                    line_id=line_id,
                )
                words.append(wh)
                by_line.setdefault(line_id, []).append(wh)
            for line_id, line_words in by_line.items():
                line_words.sort(key=lambda w: w.bbox_xywh[0])
                text = " ".join(w.text for w in line_words).strip()
                if not text:
                    continue
                conf = float(sum(w.confidence for w in line_words) / max(1, len(line_words)))
                phrases.append(
                    OCRPhraseHypothesis(
                        text=text,
                        confidence=conf,
                        language=lang,
                        preprocess=tag,
                        line_id=line_id,
                    )
                )
    words.sort(key=lambda w: w.confidence, reverse=True)
    phrases.sort(key=lambda p: (p.confidence, len(p.text)), reverse=True)
    return words, phrases


def ocr_draft_german_english(gray: Image.Image) -> str:
    """
    Best-effort draft text for Google Translate. Kurrent is poorly recognized
    by Tesseract; requires optional `pytesseract` and Tesseract on PATH.
    """
    candidates = ocr_candidates(gray)
    if not candidates:
        return ""
    return candidates[0].text
