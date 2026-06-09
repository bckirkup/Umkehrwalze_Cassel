from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.ghost_cross_page import suppress_ghost


def _make_page_pair(root: Path) -> tuple[Path, Path]:
    """Create a page with ghost bleed-through and its facing page."""
    h, w = 300, 200
    # The 'real' page: light paper with some ink.
    page = np.full((h, w), 220, dtype=np.uint8)
    page[80:85, 30:170] = 40  # Real ink line.
    # Faint ghost from facing page.
    page[150:154, 40:160] = 190  # Faint dark smudge.

    # The facing page (mirror image source of the ghost).
    facing = np.full((h, w), 220, dtype=np.uint8)
    # The ink that caused the ghost (mirrored = same position).
    facing[150:154, 40:160] = 40  # Strong ink.

    page_path = root / "page.png"
    facing_path = root / "facing.png"
    Image.fromarray(page, mode="L").save(page_path)
    Image.fromarray(facing, mode="L").save(facing_path)
    return page_path, facing_path


def test_suppress_ghost_with_facing() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        page_path, facing_path = _make_page_pair(root)
        out = root / "cleaned.png"

        meta = suppress_ghost(
            page_gray_path=page_path,
            output_path=out,
            facing_path=facing_path,
        )
        assert out.is_file()
        assert "ghost_facing_registration_error" in meta or "ghost_suppress_applied" in meta


def test_suppress_ghost_without_facing() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        page = np.full((200, 150), 220, dtype=np.uint8)
        page[80:85, 30:120] = 40
        page_path = root / "page.png"
        Image.fromarray(page, mode="L").save(page_path)
        out = root / "cleaned.png"

        suppress_ghost(
            page_gray_path=page_path,
            output_path=out,
            facing_path=None,
            use_nmf_fallback=True,
        )
        assert out.is_file()
