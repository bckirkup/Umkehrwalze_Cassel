from __future__ import annotations

import tempfile
from pathlib import Path

from revprint.translation_google import _cache_key, translate_de_to_en


def test_translate_skips_without_key_or_text() -> None:
    out, meta = translate_de_to_en("Guten Tag", "")
    assert out == ""
    assert meta.get("skipped") is True

    out2, meta2 = translate_de_to_en("", "dummy-key")
    assert out2 == ""
    assert meta2.get("skipped") is True


def test_translate_uses_local_cache_when_present() -> None:
    with tempfile.TemporaryDirectory() as d:
        cache_path = Path(d) / "translation_cache.json"
        key = _cache_key("Guten Tag", "de", "en")
        cache_path.write_text(f'{{"{key}":"Good day"}}', encoding="utf-8")
        out, meta = translate_de_to_en(
            "Guten Tag",
            "dummy-key",
            cache_enabled=True,
            cache_path=cache_path,
        )
        assert out == "Good day"
        assert meta.get("cached") is True
