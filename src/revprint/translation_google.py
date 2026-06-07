from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _cache_key(text: str, source: str, target: str) -> str:
    basis = f"{source}\n{target}\n{text[:100_000]}".encode("utf-8", errors="replace")
    return hashlib.sha256(basis).hexdigest()


def _load_cache(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return {str(k): str(v) for k, v in raw.items()}
    except Exception:
        return {}
    return {}


def _save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def translate_de_to_en(
    text: str,
    api_key: str,
    *,
    cache_path: Path | None = None,
    cache_enabled: bool = True,
) -> tuple[str, dict[str, Any]]:
    """
    Google Cloud Translation API v2 (REST).
    https://cloud.google.com/translate/docs/reference/rest/v2/translate
    Returns (translated_text, meta) where meta may include error or usage.
    """
    if not api_key or not text.strip():
        return "", {"skipped": True}

    src = "de"
    tgt = "en"
    cache_hit = False
    key = _cache_key(text, src, tgt)
    if cache_enabled and cache_path is not None:
        cache = _load_cache(Path(cache_path))
        cached = cache.get(key, "")
        if cached:
            return cached, {"ok": True, "cached": True}
    else:
        cache = {}

    url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
    payload = json.dumps(
        {
            "q": text[:100_000],
            "source": src,
            "target": tgt,
            "format": "text",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return "", {"error": f"HTTP {e.code}", "body": err_body[:2000]}
    except Exception as exc:
        return "", {"error": str(exc)}

    try:
        out = body["data"]["translations"][0]["translatedText"]
        if cache_enabled and cache_path is not None and out.strip():
            cache[key] = out
            _save_cache(Path(cache_path), cache)
        return out, {"ok": True, "cached": cache_hit}
    except (KeyError, IndexError, TypeError):
        return "", {"error": "unexpected_response", "raw": str(body)[:1000]}
