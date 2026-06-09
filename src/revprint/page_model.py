"""Physical page model: filename parsing, recto/verso pairing, facing-page map.

Hessisches Staatsarchiv Marburg scans follow the naming convention:
    hstam_<fond>_<series>_<item>_NNNN.jpg
where NNNN is a sequential scan number.  In a bound ledger scanned as
consecutive pages, even scan numbers are typically verso (left) pages and odd
numbers are recto (right) pages (or vice-versa, depending on whether scanning
started with the cover).  The facing page of scan N is scan N±1 (the page
physically pressed against it when the book is closed).

This module builds that structural model from the filenames and optional JSON
metadata, allowing downstream stages (ghost suppression, batch QA) to operate
on the correct page pairs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PageInfo:
    """Metadata for a single physical page."""

    path: Path
    scan_number: int
    side: Literal["recto", "verso", "unknown"]
    facing_path: Path | None
    json_entry: dict[str, object] | None


@dataclass
class PageModel:
    """Ordered collection of pages with physical facing-page relationships."""

    pages: list[PageInfo] = field(default_factory=list)
    by_scan_number: dict[int, PageInfo] = field(default_factory=dict)
    by_path: dict[Path, PageInfo] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def facing_page(self, path: Path) -> PageInfo | None:
        info = self.by_path.get(path)
        if info is None or info.facing_path is None:
            return None
        return self.by_path.get(info.facing_path)


_SCAN_NUM_RE = re.compile(r"_(\d{4})\.\w+$")


def _extract_scan_number(filename: str) -> int | None:
    m = _SCAN_NUM_RE.search(filename)
    if m is None:
        return None
    return int(m.group(1))


def _detect_spine_side(path: Path) -> Literal["left", "right", "unknown"]:
    """Detect which side of the scan the spine shadow falls on.

    A quick heuristic: compare mean brightness of the left 15% vs right 15%
    column band.  The darker side is the spine.
    """
    try:
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(path) as im:
            gray = ImageOps.grayscale(ImageOps.exif_transpose(im))
            # Work at reduced resolution for speed.
            scale = min(1.0, 600.0 / max(gray.size))
            if scale < 1.0:
                new_w = max(1, int(gray.size[0] * scale))
                new_h = max(1, int(gray.size[1] * scale))
                gray = gray.resize((new_w, new_h), Image.Resampling.LANCZOS)
            arr = np.asarray(gray, dtype=np.float32)
        h, w = arr.shape
        band = max(4, int(w * 0.15))
        left_mean = float(np.mean(arr[:, :band]))
        right_mean = float(np.mean(arr[:, -band:]))
        if abs(left_mean - right_mean) < 5.0:
            return "unknown"
        return "left" if left_mean < right_mean else "right"
    except Exception:
        return "unknown"


def _infer_side(
    scan_number: int,
    spine_side: Literal["left", "right", "unknown"],
) -> Literal["recto", "verso", "unknown"]:
    """Infer recto/verso from scan number parity and spine position.

    In a codex scanned from the front, verso pages have the spine on the
    right and recto pages have the spine on the left.  If spine detection
    is inconclusive, fall back to even=verso, odd=recto (common convention).
    """
    if spine_side == "left":
        return "recto"
    if spine_side == "right":
        return "verso"
    # Fallback: even=verso, odd=recto
    return "verso" if scan_number % 2 == 0 else "recto"


def _load_json_index(json_path: Path) -> dict[str, dict[str, object]]:
    """Load the archive export JSON, keyed by original_filename."""
    if not json_path.is_file():
        return {}
    try:
        with open(json_path, encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            return {}
        return {str(e["original_filename"]): e for e in entries if "original_filename" in e}
    except Exception:
        return {}


def build_page_model(
    image_paths: list[Path],
    json_path: Path | None = None,
    spine_detect_sample: int = 3,
) -> PageModel:
    """Build a :class:`PageModel` from a list of image paths.

    Parameters
    ----------
    image_paths:
        Paths to all JPEG scans in the corpus (not just the current batch).
    json_path:
        Optional path to the archive export JSON for metadata enrichment.
    spine_detect_sample:
        Number of pages to sample for spine-side auto-detection (0 to skip).
    """
    json_index = _load_json_index(json_path) if json_path else {}

    # Extract scan numbers and sort.
    numbered: list[tuple[int, Path]] = []
    for p in image_paths:
        sn = _extract_scan_number(p.name)
        if sn is not None:
            numbered.append((sn, p))
    numbered.sort(key=lambda t: t[0])

    # Auto-detect spine side from a sample to establish recto/verso convention.
    dominant_spine: Literal["left", "right", "unknown"] = "unknown"
    if spine_detect_sample > 0 and numbered:
        step = max(1, len(numbered) // spine_detect_sample)
        samples = [numbered[i] for i in range(0, len(numbered), step)][:spine_detect_sample]
        votes: dict[str, int] = {"left": 0, "right": 0, "unknown": 0}
        for _, p in samples:
            votes[_detect_spine_side(p)] += 1
        if votes["left"] > votes["right"]:
            dominant_spine = "left"
        elif votes["right"] > votes["left"]:
            dominant_spine = "right"

    # Build scan-number index for facing-page lookup.
    scan_to_path: dict[int, Path] = {sn: p for sn, p in numbered}

    pages: list[PageInfo] = []
    for sn, p in numbered:
        side = _infer_side(sn, dominant_spine)
        # Facing page: in a bound codex, scan N faces scan N-1 (for recto)
        # or scan N+1 (for verso).  More precisely, the facing page is the
        # one whose scan number differs by 1 and has the opposite side.
        facing_sn = sn - 1 if side == "recto" else sn + 1
        facing = scan_to_path.get(facing_sn)
        json_entry = json_index.get(p.name)
        info = PageInfo(
            path=p,
            scan_number=sn,
            side=side,
            facing_path=facing,
            json_entry=json_entry,
        )
        pages.append(info)

    model = PageModel(
        pages=pages,
        by_scan_number={pi.scan_number: pi for pi in pages},
        by_path={pi.path: pi for pi in pages},
    )
    return model
