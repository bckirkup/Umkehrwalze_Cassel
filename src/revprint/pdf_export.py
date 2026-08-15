from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from revprint.pdf_fonts import register_unicode_font


def _fit_rect(
    img_size: tuple[int, int], box: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    img_w, img_h = img_size
    left, bottom, width, height = box
    scale = min(width / img_w, height / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    x = left + (width - draw_w) / 2
    y = bottom + (height - draw_h) / 2
    return x, y, draw_w, draw_h


def export_reproduction_pdf(image_paths: list[Path], output_pdf: Path) -> Path:
    output_pdf = Path(output_pdf).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_pdf), pagesize=letter)
    page_w, page_h = letter
    margin = 0.35 * inch
    box = (margin, margin, page_w - 2 * margin, page_h - 2 * margin)

    for image_path in image_paths:
        with Image.open(image_path) as img:
            x, y, w, h = _fit_rect(img.size, box)
        c.drawImage(str(image_path), x, y, width=w, height=h, preserveAspectRatio=True, mask="auto")
        c.setStrokeColor(colors.lightgrey)
        c.rect(margin, margin, page_w - 2 * margin, page_h - 2 * margin, stroke=1, fill=0)
        c.showPage()
    c.save()
    return output_pdf


def _draw_block(
    c: canvas.Canvas,
    text: str,
    margin: float,
    y: float,
    text_width: float,
    page_h: float,
    font: str,
    size: int,
) -> float:
    leading = size * 1.2
    c.setFont(font, size)
    for line in simpleSplit(text, font, size, text_width):
        if y < margin + leading * 2:
            c.showPage()
            y = page_h - margin
            c.setFont(font, size)
        c.drawString(margin, y, line)
        y -= leading
    return y


def _recognition_text(rec: dict[str, Any], ev: dict[str, Any] | None) -> tuple[str, str]:
    ev_engine = str(ev.get("source_engine", "")) if ev else ""
    if ev_engine == "manual" and ev:
        heading = "Source text (German seed / manual):"
    elif ev_engine == "htr":
        heading = "Source text (HTR):"
    else:
        heading = "Draft recognition (HTR/OCR):"
    text = (rec.get("ocr_draft") or "").strip()
    if ev and ev_engine in ("htr", "manual"):
        segs = ev.get("segments", [])
        if isinstance(segs, list):
            joined = "\n".join(str(s.get("text", "")).strip() for s in segs if isinstance(s, dict))
            text = joined.strip() or text
    if not text:
        text = (
            "[No recognition text available. Provide {stem}.htr.json (preferred for handwriting), "
            "install pytesseract+Tesseract, or add manual transcription sidecar.]"
        )
    return heading, text


def _translation_text(rec: dict[str, Any], src_type: str) -> tuple[str, str]:
    if src_type == "gemini_seed":
        heading = "English / commentary (imported seed):"
    elif src_type == "gemini":
        heading = "English / commentary (Gemini API):"
    elif src_type == "manual":
        heading = "English (from German source via Translate API):"
    else:
        heading = "English (Google Cloud Translation):"

    if rec.get("translation_en"):
        text = str(rec["translation_en"]).strip()
        metadata = rec.get("translation_meta") or {}
        if isinstance(metadata, dict) and metadata.get("cached") is True:
            text = "[Loaded from local translation cache]\n" + text
    elif rec.get("translation_error"):
        text = f"(Google Translate) {rec.get('translation_error')}"
    elif (
        isinstance(rec.get("translation_meta"), dict)
        and rec["translation_meta"].get("reason") == "ocr_confidence_below_threshold"
    ):
        metadata = rec["translation_meta"]
        text = (
            "[OCR confidence too low for safe auto-translation. "
            f"Observed={metadata.get('ocr_confidence')}, "
            f"required>={metadata.get('ocr_translation_confidence_min')}. "
            "Add manual transcription sidecar for best results.]"
        )
    elif src_type == "manual":
        text = "[German source present but no English draft yet. Add RPK_GOOGLE_TRANSLATE_API_KEY to draft English.]"
    elif src_type == "copied_en":
        text = "[English sidecar supplied directly in {stem}.translation_en.txt.]"
    else:
        text = (
            "[No translation draft yet. Provide API key for Google Translate, "
            "or continue with manual transcription/review workflow.]"
        )
    return heading, text


def _draw_translation_page(
    c: canvas.Canvas,
    rec: dict[str, Any],
    index: int,
    *,
    font: str,
    margin: float,
    page_w: float,
    page_h: float,
) -> None:
    text_width = page_w - 2 * margin
    c.setFont(font, 11)
    y = page_h - margin
    c.drawString(margin, y, f"Translation proof page {index}")
    y -= 0.28 * inch
    c.setFont(font, 7)
    c.drawString(margin, y, f"Source: {str(rec.get('source_path', ''))[:200]}")
    y -= 0.16 * inch
    src_type = str(rec.get("translation_source_type", "unknown"))
    c.drawString(margin, y, f"Translation source type: {src_type}")
    y -= 0.3 * inch

    ev = rec.get("text_evidence")
    ev_dict = ev if isinstance(ev, dict) else None
    heading, text = _recognition_text(rec, ev_dict)
    c.setFont(font, 9)
    c.drawString(margin, y, heading)
    y -= 0.2 * inch
    y = _draw_block(c, text, margin, y, text_width, page_h, font, 9)
    y -= 0.15 * inch
    if y < margin * 2:
        c.showPage()
        y = page_h - margin

    heading, text = _translation_text(rec, src_type)
    c.setFont(font, 9)
    c.drawString(margin, y, heading)
    y -= 0.2 * inch
    c.setFont(font, 9)
    _draw_block(c, text, margin, y, text_width, page_h, font, 9)
    c.setStrokeColor(colors.lightgrey)
    c.rect(margin, margin, text_width, page_h - 2 * margin, stroke=1, fill=0)
    c.showPage()


def export_translation_pdf(
    page_records: list[dict[str, Any]],
    output_pdf: Path,
) -> Path:
    output_pdf = Path(output_pdf).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    font = register_unicode_font()
    c = canvas.Canvas(str(output_pdf), pagesize=letter)
    page_w, page_h = letter
    margin = 0.55 * inch
    for index, rec in enumerate(page_records, start=1):
        _draw_translation_page(
            c, rec, index, font=font, margin=margin, page_w=page_w, page_h=page_h
        )
    c.save()
    return output_pdf
