from __future__ import annotations

import re
from pathlib import Path

_JPEG_SUFFIXES = {".jpg", ".jpeg"}


def _natural_key(name: str) -> list[str | int]:
    """Sort key: split on digit runs so 2 < 10 < foo_2 < foo_10."""
    parts = re.split(r"(\d+)", name)
    out: list[str | int] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def scan_jpegs(root: Path) -> list[Path]:
    """
    List JPEG files under `root` (non-recursive), sorted in natural filename order.
    Only considers files whose suffix matches .jpg or .jpeg (case-insensitive).
    """
    root = Path(root).resolve()
    if not root.is_dir():
        msg = f"Input root is not a directory: {root}"
        raise NotADirectoryError(msg)

    found: list[Path] = []
    for p in root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _JPEG_SUFFIXES:
            continue
        found.append(p)

    found.sort(key=lambda path: _natural_key(path.name))
    return found
