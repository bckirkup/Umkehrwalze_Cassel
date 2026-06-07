from __future__ import annotations

from revprint.config import Settings


def test_expand_path_handles_empty_string() -> None:
    assert Settings._expand_path("") is None
