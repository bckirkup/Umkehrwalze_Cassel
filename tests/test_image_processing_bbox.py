from __future__ import annotations

from PIL import Image, ImageDraw

from revprint.image_processing import trim_bright_scanner_edges


def _synthetic_scan(width: int, height: int, border: int) -> Image.Image:
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle((border, border, width - border - 1, height - border - 1), fill=(180, 180, 180))
    return img


def test_trim_bright_scanner_edges_accepts_large_margin_crop() -> None:
    img = _synthetic_scan(1000, 1400, border=120)
    trimmed = trim_bright_scanner_edges(img, (0, 0, *img.size))
    left, top, right, bottom = trimmed
    assert left > 0
    assert top > 0
    assert right < img.size[0]
    assert bottom < img.size[1]


def test_trim_bright_scanner_edges_keeps_detected_bbox_unchanged() -> None:
    img = _synthetic_scan(1000, 1400, border=120)
    original_bbox = (100, 120, 900, 1280)
    trimmed = trim_bright_scanner_edges(img, original_bbox)
    assert trimmed == original_bbox
