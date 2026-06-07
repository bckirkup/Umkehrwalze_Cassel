from __future__ import annotations

import re


def _normalize_phrase(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return re.sub(r"[^a-z0-9 ]+", "", lowered)


def observe_phrases(memory: dict[str, int], phrases: list[str]) -> dict[str, int]:
    out = dict(memory)
    for p in phrases:
        norm = _normalize_phrase(p)
        if len(norm) < 4:
            continue
        out[norm] = int(out.get(norm, 0)) + 1
    return out


def phrase_boost(memory: dict[str, int], phrase: str, max_boost: float = 0.25) -> float:
    norm = _normalize_phrase(phrase)
    seen = int(memory.get(norm, 0))
    if seen <= 0:
        return 0.0
    # Mild saturation curve: first few repeats matter most.
    boost = min(max_boost, 0.07 * seen)
    return float(boost)
