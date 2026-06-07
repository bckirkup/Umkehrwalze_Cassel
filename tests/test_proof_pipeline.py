from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from revprint.config import load_settings
from revprint.image_processing import process_page
from revprint.proof import _apply_profile_overrides, _resolve_translation_pipeline, run_proof


def _make_scan(path: Path, text: str = "Test") -> None:
    img = Image.new("RGB", (420, 620), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((60, 40, 360, 580), fill=(145, 155, 135))
    draw.text((120, 160), text, fill=(20, 20, 20))
    draw.line((100, 300, 310, 360), fill=(190, 80, 55), width=3)
    img.save(path)


def test_process_page_writes_review_assets() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        source = root / "scan_0001.jpg"
        _make_scan(source)
        result = process_page(source, root / "out")
        assert Path(result.cropped_color_path).exists()
        assert Path(result.red_suppressed_path).exists()
        assert Path(result.cleaned_grayscale_path).exists()
        assert Path(result.edge_inpaint_mask_path).exists()
        assert Path(result.debug_overlay_path).exists()


def test_process_page_outputs_ink_on_white() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        source = root / "scan_0001.jpg"
        _make_scan(source)
        result = process_page(source, root / "out")
        with Image.open(result.cleaned_grayscale_path) as img:
            extrema = img.convert("L").getextrema()
            assert extrema[1] == 255
            assert extrema[0] < 120


def test_run_proof_writes_manifest_and_pdfs() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        input_root = root / "input"
        input_root.mkdir()
        _make_scan(input_root / "scan_0001.jpg", "One")
        _make_scan(input_root / "scan_0002.jpg", "Two")
        run = run_proof(
            input_root=input_root,
            job_store_path=root / "jobs.sqlite",
            output_root=root / "outputs",
            limit=2,
            start=0,
        )
        assert Path(run.reproduction_pdf).exists()
        assert Path(run.translation_pdf).exists()
        assert Path(run.cloud_manifest_path).exists()
        assert Path(run.pilot_print_bundle_path).exists()
        manifest = json.loads(Path(run.manifest_path).read_text(encoding="utf-8"))
        assert len(manifest["selected_sources"]) == 2
        assert len(manifest["processed_pages"]) == 2
        assert "quality_summary" in manifest
        assert "edge_apply_rate" in manifest["quality_summary"]
        for page in manifest["processed_pages"]:
            assert "translation_source_type" in page
            assert "ghost_suppression_enabled" in page
            assert "ghost_suppress_before_path" in page
            assert "speckle_before_path" in page
            assert "speckle_after_path" in page
            assert "speckle_refine_applied" in page
            assert "line_refine_applied" in page
            assert "line_refine_removed_lines" in page
            assert "crease_refine_applied" in page
            assert "crease_mask_path" in page
            assert "dewarp_enabled" in page
            assert "plausibility_applied" in page
            assert "plausibility_threshold" in page
            assert "plausibility_exhaustive_passes" in page
            assert "text_evidence" in page
            assert "ocr_reconstruct_hint_mask_path" in page
            assert "ocr_reconstruct_hint_words_json_path" in page
            evidence = page["text_evidence"]
            assert "source_engine" in evidence
            assert "segments" in evidence
            assert "edge_reconstruct_candidate_mask_path" in page
            assert "edge_reconstruct_overlay_path" in page
            assert "edge_reconstruct_protect_mask_path" in page
            assert "edge_reconstruct_candidates_json_path" in page
            assert "edge_reconstruct_before_path" in page
            assert "edge_reconstruct_after_path" in page
            assert "edge_reconstruct_applied" in page
            assert "edge_reconstruct_policy_reason" in page
        assert "cloud_manifest_path" in manifest
        assert "pilot_print_bundle_path" in manifest


def test_resolve_translation_pipeline_manual_sidecar() -> None:
    with tempfile.TemporaryDirectory() as d:
        pages = Path(d)
        stem = "page_0001"
        (pages / f"{stem}.translation_source.txt").write_text(
            "Archivgut", encoding="utf-8"
        )
        gray = Image.new("L", (32, 32), 255)
        _en, src_type, meta = _resolve_translation_pipeline(
            pages, stem, pages / f"{stem}.jpg", gray, load_settings()
        )
        assert src_type == "manual"
        assert meta.get("translation_sidecar_path")


def test_resolve_translation_pipeline_copied_en_sidecar() -> None:
    with tempfile.TemporaryDirectory() as d:
        pages = Path(d)
        stem = "page_0001"
        (pages / f"{stem}.translation_en.txt").write_text("Already English.", encoding="utf-8")
        gray = Image.new("L", (8, 8), 200)
        en, src_type, meta = _resolve_translation_pipeline(
            pages, stem, pages / f"{stem}.jpg", gray, load_settings()
        )
        assert src_type == "copied_en"
        assert en == "Already English."
        assert meta.get("skipped_translate_api") is True


def test_resolve_translation_pipeline_skips_low_confidence_ocr() -> None:
    @dataclass(frozen=True)
    class _Phrase:
        text: str
        confidence: float
        language: str = "deu"
        preprocess: str = "test"
        line_id: str = "b1_p1_l1"

    with tempfile.TemporaryDirectory() as d:
        pages = Path(d)
        stem = "page_0001"
        gray = Image.new("L", (16, 16), 180)
        settings = load_settings()
        settings.ocr_translation_confidence_min = 0.5
        with patch(
            "revprint.proof.ocr_word_hypotheses",
            return_value=([], [_Phrase(text="abc", confidence=0.2)]),
        ):
            tr, src_type, meta = _resolve_translation_pipeline(
                pages, stem, pages / f"{stem}.jpg", gray, settings
            )
        assert src_type == "ocr"
        assert tr == ""
        assert meta.get("reason") == "ocr_confidence_below_threshold"


def test_resolve_translation_pipeline_prefers_gemini_seed_sidecar() -> None:
    with tempfile.TemporaryDirectory() as d:
        pages = Path(d)
        stem = "page_0001"
        source = pages / f"{stem}.jpg"
        source.write_bytes(b"fake")
        (pages / f"{stem} translation and commentary.txt").write_text("Gemini seed text", encoding="utf-8")
        gray = Image.new("L", (16, 16), 180)
        tr, src_type, meta = _resolve_translation_pipeline(pages, stem, source, gray, load_settings())
        assert src_type == "gemini_seed"
        assert tr == "Gemini seed text"
        assert str(meta.get("gemini_seed_path", "")).endswith("translation and commentary.txt")


def test_resolve_translation_pipeline_uses_german_seed_when_translatable() -> None:
    with tempfile.TemporaryDirectory() as d:
        pages = Path(d)
        stem = "page_0001"
        source = pages / f"{stem}.jpg"
        source.write_bytes(b"fake")
        (pages / f"{stem} german.txt").write_text("Anno 1742", encoding="utf-8")
        gray = Image.new("L", (16, 16), 180)
        settings = load_settings()
        with patch("revprint.proof.translate_de_to_en", return_value=("Year 1742", {"ok": True})):
            tr, src_type, meta = _resolve_translation_pipeline(pages, stem, source, gray, settings)
        assert src_type == "manual"
        assert tr == "Year 1742"
        assert str(meta.get("seed_source_type", "")) == "german_seed"


def test_resolve_translation_pipeline_gemini_seed_wins_over_german_seed() -> None:
    with tempfile.TemporaryDirectory() as d:
        pages = Path(d)
        stem = "page_0001"
        source = pages / f"{stem}.jpg"
        source.write_bytes(b"fake")
        (pages / f"{stem} german.txt").write_text("Anno 1742", encoding="utf-8")
        (pages / f"{stem} translation and commentary.txt").write_text("Full commentary", encoding="utf-8")
        gray = Image.new("L", (16, 16), 180)
        settings = load_settings()
        with patch("revprint.proof.translate_de_to_en") as mock_tr:
            tr, src_type, meta = _resolve_translation_pipeline(pages, stem, source, gray, settings)
        mock_tr.assert_not_called()
        assert src_type == "gemini_seed"
        assert tr == "Full commentary"
        assert "german.txt" in str(meta.get("german_seed_path", ""))


def test_profile_forensic_overrides_to_heavy_mode() -> None:
    settings = load_settings()
    settings.ghost_suppression_enabled = False
    settings.dewarp_enabled = False
    settings.ghost_plausibility_exhaustive_passes = 1
    out = _apply_profile_overrides(settings, "forensic")
    assert out.ghost_suppression_enabled is True
    assert out.dewarp_enabled is True
    assert out.ghost_plausibility_exhaustive_passes >= 16
    assert out.line_refine_enabled is True
