from __future__ import annotations

from revprint.language_evidence import extract_text_evidence


def test_extract_text_evidence_none_when_no_text() -> None:
    ev = extract_text_evidence((320, 480))
    meta = ev.to_meta()
    assert meta["source_engine"] == "none"
    assert meta["language"] == "unknown"
    assert meta["script"] == "unknown"
    assert meta["segments"] == []


def test_extract_text_evidence_manual_has_segment() -> None:
    ev = extract_text_evidence((100, 120), manual_text="Archivgut")
    meta = ev.to_meta()
    assert meta["source_engine"] == "manual"
    assert len(meta["segments"]) == 1
    seg = meta["segments"][0]
    assert seg["text"] == "Archivgut"
    assert seg["bbox_xywh"] == (0, 0, 100, 120)
    assert seg["language"] == "de"
