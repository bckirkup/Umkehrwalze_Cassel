from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _cache_key(image_bytes: bytes, mime_type: str, prompt: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(mime_type.encode("utf-8"))
    h.update(prompt.encode("utf-8"))
    h.update(image_bytes)
    return h.hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = {str(kk): str(vv) for kk, vv in v.items() if isinstance(kk, str)}
    return out


def _save_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def gemini_translate_image(
    *,
    image_path: Path,
    api_key: str,
    model: str,
    prompt: str,
    cache_path: Path | None = None,
    cache_enabled: bool = True,
) -> tuple[str, dict[str, Any]]:
    if not api_key:
        return "", {"skipped": True, "reason": "missing_api_key"}
    image_path = Path(image_path)
    if not image_path.is_file():
        return "", {"skipped": True, "reason": "missing_image"}
    image_bytes = image_path.read_bytes()
    mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    key = _cache_key(image_bytes, mime_type, prompt, model)
    cache: dict[str, dict[str, str]] = {}
    if cache_enabled and cache_path is not None:
        cache = _load_cache(Path(cache_path))
        hit = cache.get(key)
        if isinstance(hit, dict):
            return hit.get("text", ""), {"ok": True, "cached": True, "model": model}

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(image_bytes).decode("ascii"),
                        }
                    },
                ]
            }
        ]
    }
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return "", {"error": f"HTTP {e.code}", "body": err_body[:2000], "model": model}
    except Exception as exc:
        return "", {"error": str(exc), "model": model}

    try:
        candidates = body.get("candidates", [])
        first = candidates[0]
        parts = first["content"]["parts"]
        text = "\n".join(str(p.get("text", "")).strip() for p in parts if isinstance(p, dict)).strip()
    except Exception:
        return "", {"error": "unexpected_response", "raw": str(body)[:1000], "model": model}
    if cache_enabled and cache_path is not None and text:
        cache[key] = {"text": text, "model": model}
        _save_cache(Path(cache_path), cache)
    return text, {"ok": True, "cached": False, "model": model}
