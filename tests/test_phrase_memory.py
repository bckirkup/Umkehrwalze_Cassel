from __future__ import annotations

from revprint.phrase_memory import observe_phrases, phrase_boost


def test_phrase_memory_observe_and_boost() -> None:
    mem: dict[str, int] = {}
    mem = observe_phrases(mem, ["Herr Benjamin", "Herr Benjamin", "Anno 1742"])
    assert phrase_boost(mem, "Herr Benjamin") > 0.0
    assert phrase_boost(mem, "Completely unseen phrase") == 0.0
