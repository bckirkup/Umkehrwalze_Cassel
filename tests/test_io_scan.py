from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from revprint.io_scan import scan_jpegs


def test_scan_jpegs_natural_order() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "a_2.jpg").write_bytes(b"\xff\xd8\xff")
        (root / "a_10.jpg").write_bytes(b"\xff\xd8\xff")
        (root / "a_1.jpg").write_bytes(b"\xff\xd8\xff")
        (root / "z.txt").write_text("x")
        out = [p.name for p in scan_jpegs(root)]
        assert out == ["a_1.jpg", "a_2.jpg", "a_10.jpg"]


def test_scan_jpegs_not_a_dir() -> None:
    with pytest.raises(NotADirectoryError):
        scan_jpegs(Path("/nonexistent/scan/root/revprint"))
