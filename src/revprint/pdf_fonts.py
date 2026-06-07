from __future__ import annotations

import os
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

_registered = False
_FONT = "Helvetica"


def register_unicode_font() -> str:
    """Register a TTF for German/English text if a common path exists."""
    global _registered, _FONT
    if _registered:
        return _FONT
    candidates: list[Path] = []
    win = os.environ.get("SYSTEMROOT", "C:/Windows")
    for name in (
        "arial.ttf",
        "calibri.ttf",
        "segoeui.ttf",
    ):
        candidates.append(Path(win) / "Fonts" / name)
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        ]
    )
    for p in candidates:
        if p.is_file():
            try:
                pdfmetrics.registerFont(TTFont("RevPrintSans", str(p)))
                _FONT = "RevPrintSans"
                _registered = True
                return _FONT
            except OSError:
                continue
    _registered = True
    return "Helvetica"
