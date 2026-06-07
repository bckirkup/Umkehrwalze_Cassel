from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from revprint.edge_refine import refine_to_grayscale


@dataclass(frozen=True)
class ProcessedPage:
    source_path: str
    crop_bbox: tuple[int, int, int, int]
    rotation_degrees: float
    cropped_color_path: str
    red_suppressed_path: str
    """8-bit grayscale, edge-inpainted and tone-flattened for print."""
    cleaned_grayscale_path: str
    """Mask of inpainted edge regions (L, 255 = filled)."""
    edge_inpaint_mask_path: str
    debug_overlay_path: str
    warnings: list[str]

    def to_meta(self) -> dict[str, object]:
        return asdict(self)


def _corner_background_rgb(img: Image.Image) -> tuple[int, int, int]:
    rgb = img.convert("RGB")
    w, h = rgb.size
    sample = max(8, min(w, h) // 40)
    boxes = [
        (0, 0, sample, sample),
        (w - sample, 0, w, sample),
        (0, h - sample, sample, h),
        (w - sample, h - sample, w, h),
    ]
    pixels: list[tuple[int, int, int]] = []
    for box in boxes:
        pixels.extend(rgb.crop(box).getdata())
    pixels.sort(key=lambda p: p[0] + p[1] + p[2])
    return pixels[len(pixels) // 2]


def detect_page_bbox(img: Image.Image) -> tuple[int, int, int, int]:
    """Detect a conservative content/page bbox against the scanner bed."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    near_white_xs: list[int] = []
    near_white_ys: list[int] = []
    pixels = rgb.load()
    step = 2 if max(w, h) > 1200 else 1
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = pixels[x, y]
            is_scanner_bed = r > 236 and g > 236 and b > 236 and max(r, g, b) - min(r, g, b) < 18
            if not is_scanner_bed:
                near_white_xs.append(x)
                near_white_ys.append(y)

    if near_white_xs and near_white_ys:
        pad = max(4, min(w, h) // 140)
        bbox = (
            max(0, min(near_white_xs) - pad),
            max(0, min(near_white_ys) - pad),
            min(w, max(near_white_xs) + pad + step),
            min(h, max(near_white_ys) + pad + step),
        )
        if bbox != (0, 0, w, h) and (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > (w * h * 0.35):
            return bbox

    bg = _corner_background_rgb(rgb)
    bg_luma = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]

    xs: list[int] = []
    ys: list[int] = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            r, g, b = pixels[x, y]
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            color_dist = abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2])
            if luma < bg_luma - 18 or color_dist > 42:
                xs.append(x)
                ys.append(y)

    if not xs or not ys:
        return (0, 0, w, h)

    pad = max(6, min(w, h) // 100)
    left = max(0, min(xs) - pad)
    top = max(0, min(ys) - pad)
    right = min(w, max(xs) + pad + step)
    bottom = min(h, max(ys) + pad + step)

    if (right - left) * (bottom - top) < (w * h * 0.25):
        return (0, 0, w, h)
    return (left, top, right, bottom)


def trim_bright_scanner_edges(img: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    w, h = img.size
    if bbox != (0, 0, w, h):
        return bbox

    gray = ImageOps.grayscale(img)
    px = gray.load()
    step = max(1, min(w, h) // 900)

    def col_stats(x: int) -> tuple[float, float]:
        vals = [px[x, y] for y in range(0, h, step)]
        avg = sum(vals) / len(vals)
        dark_fraction = sum(1 for v in vals if v < 180) / len(vals)
        return avg, dark_fraction

    def row_stats(y: int) -> tuple[float, float]:
        vals = [px[x, y] for x in range(0, w, step)]
        avg = sum(vals) / len(vals)
        dark_fraction = sum(1 for v in vals if v < 180) / len(vals)
        return avg, dark_fraction

    left = 0
    top = 0
    right = w
    bottom = h
    max_trim_x = max(4, w // 14)
    max_trim_y = max(4, h // 14)

    for x in range(0, max_trim_x, step):
        avg, dark = col_stats(x)
        if avg < 218 or dark > 0.01:
            break
        left = x

    for x in range(w - 1, max(w - max_trim_x, 0), -step):
        avg, dark = col_stats(x)
        if avg < 218 or dark > 0.01:
            break
        right = x

    for y in range(0, max_trim_y, step):
        avg, dark = row_stats(y)
        if avg < 218 or dark > 0.01:
            break
        top = y

    for y in range(h - 1, max(h - max_trim_y, 0), -step):
        avg, dark = row_stats(y)
        if avg < 218 or dark > 0.01:
            break
        bottom = y

    # Allow stronger border trims on oversized scans while still rejecting
    # pathological over-crops.
    if right - left < w * 0.7 or bottom - top < h * 0.7:
        return bbox
    return (left, top, right, bottom)


def suppress_red_marks(img: Image.Image) -> Image.Image:
    out = img.convert("RGB")
    px = out.load()
    w, h = out.size
    for y in range(h):
        for x in range(w):
            r, g, b = px[x, y]
            warm = r > g + 14 and r > b + 14 and r > 90
            orange = r > 120 and g > b + 10 and r > b + 35
            if warm or orange:
                paper_like = min(245, max(r, g, b) + 35)
                px[x, y] = (paper_like, paper_like, paper_like)
    return out


def process_page(source_path: Path, output_dir: Path) -> ProcessedPage:
    source_path = Path(source_path).resolve()
    page_stem = source_path.stem
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    with Image.open(source_path) as opened:
        img = ImageOps.exif_transpose(opened).convert("RGB")

    bbox = trim_bright_scanner_edges(img, detect_page_bbox(img))
    if bbox == (0, 0, *img.size):
        warnings.append("page_bbox_fallback_full_image")

    debug = img.copy()
    draw = ImageDraw.Draw(debug)
    draw.rectangle(bbox, outline=(255, 0, 0), width=4)

    cropped = img.crop(bbox)
    red_suppressed = suppress_red_marks(cropped)
    cleaned_gray, edge_mask = refine_to_grayscale(red_suppressed)

    cropped_path = output_dir / f"{page_stem}.cropped_color.jpg"
    suppressed_path = output_dir / f"{page_stem}.red_suppressed.jpg"
    gray_path = output_dir / f"{page_stem}.cleaned_gray.png"
    mask_path = output_dir / f"{page_stem}.edge_inpaint_mask.png"
    debug_path = output_dir / f"{page_stem}.debug_overlay.jpg"

    cropped.save(cropped_path, quality=95)
    red_suppressed.save(suppressed_path, quality=95)
    cleaned_gray.save(gray_path, optimize=True)
    edge_mask.save(mask_path)
    debug.save(debug_path, quality=90)

    return ProcessedPage(
        source_path=str(source_path),
        crop_bbox=bbox,
        rotation_degrees=0.0,
        cropped_color_path=str(cropped_path),
        red_suppressed_path=str(suppressed_path),
        cleaned_grayscale_path=str(gray_path),
        edge_inpaint_mask_path=str(mask_path),
        debug_overlay_path=str(debug_path),
        warnings=warnings,
    )
