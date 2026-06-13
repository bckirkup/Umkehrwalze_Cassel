from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from revprint.cloud_runtime import build_local_cloud_manifest
from revprint.config import Settings, load_settings
from revprint.crease_refine import apply_crease_refine
from revprint.dewarp import dewarp_grayscale_optional
from revprint.edge_reconstruction import (
    apply_edge_reconstruction,
    build_edge_reconstruction_candidates,
)
from revprint.ghost_suppression import apply_ghost_suppression
from revprint.htr_adapter import load_htr_sidecar
from revprint.image_processing import ProcessedPage, process_page
from revprint.io_scan import scan_jpegs
from revprint.job_store import JobState, JobStore
from revprint.language_evidence import extract_text_evidence
from revprint.line_artifact_refine import apply_line_artifact_refine
from revprint.ocr_extract import OCRPhraseHypothesis, ocr_word_hypotheses
from revprint.ocr_reconstruction_hints import build_ocr_reconstruction_hints
from revprint.page_interactions import analyze_interactions_for_source
from revprint.pdf_export import export_reproduction_pdf, export_translation_pdf
from revprint.phrase_memory import observe_phrases, phrase_boost
from revprint.pilot_bundle import write_pilot_print_bundle
from revprint.speckle_refine import apply_speckle_refine
from revprint.translation_gemini import gemini_translate_image
from revprint.translation_google import translate_de_to_en


@dataclass(frozen=True)
class ProofRun:
    run_id: str
    output_dir: str
    selected_sources: list[str]
    processed_pages: list[dict[str, object]]
    reproduction_pdf: str
    translation_pdf: str
    manifest_path: str
    cloud_manifest_path: str
    pilot_print_bundle_path: str
    quality_summary: dict[str, object]


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _apply_profile_overrides(settings: Settings, profile: str) -> Settings:
    p = profile.strip().lower()
    if p == "quick":
        settings.ghost_suppression_enabled = False
        settings.dewarp_enabled = False
        settings.speckle_refine_enabled = True
        settings.speckle_max_component_area = min(settings.speckle_max_component_area, 24)
        settings.ghost_plausibility_exhaustive_passes = 1
        settings.ghost_plausibility_min = 0.5
        return settings
    if p == "forensic":
        settings.ghost_suppression_enabled = True
        settings.dewarp_enabled = True
        settings.speckle_refine_enabled = True
        settings.speckle_max_component_area = max(settings.speckle_max_component_area, 44)
        settings.speckle_border_max_component_area = max(settings.speckle_border_max_component_area, 140)
        settings.speckle_border_band_ratio = max(settings.speckle_border_band_ratio, 0.13)
        settings.line_refine_enabled = True
        settings.line_refine_min_length_ratio = max(settings.line_refine_min_length_ratio, 0.5)
        settings.line_refine_border_band_ratio = max(settings.line_refine_border_band_ratio, 0.15)
        settings.crease_refine_enabled = True
        settings.crease_darkness_threshold = min(settings.crease_darkness_threshold, 10.0)
        settings.htr_enabled = True
        settings.edge_reconstruct_enabled = True
        settings.edge_reconstruct_strength = max(settings.edge_reconstruct_strength, 0.62)
        settings.ghost_confidence_min = min(settings.ghost_confidence_min, 0.12)
        settings.ghost_plausibility_exhaustive_passes = max(settings.ghost_plausibility_exhaustive_passes, 16)
        settings.ghost_plausibility_min = max(settings.ghost_plausibility_min, 0.62)
        return settings
    if p == "training":
        settings.ghost_suppression_enabled = True
        settings.dewarp_enabled = True
        settings.speckle_refine_enabled = True
        settings.edge_reconstruct_enabled = True
        settings.ghost_plausibility_exhaustive_passes = max(settings.ghost_plausibility_exhaustive_passes, 6)
        return settings
    return settings


def select_proof_pages(input_root: Path, limit: int = 4, start: int = 1) -> list[Path]:
    files = scan_jpegs(input_root)
    if start < 0:
        raise ValueError("start must be >= 0")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    return files[start : start + limit]


def _read_optional_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _resolve_translation_pipeline(
    pages_dir: Path,
    stem: str,
    source_path: Path,
    gray: Image.Image,
    settings: Settings,
    phrase_memory_state: dict[str, int] | None = None,
) -> tuple[str, str, dict[str, object]]:
    """
    Returns (english_text, source_type, merged_meta).
    source_type in: copied_en, manual, ocr, none
    """
    pages_dir = Path(pages_dir)
    en_sidecar = pages_dir / f"{stem}.translation_en.txt"
    manual_sidecar = pages_dir / f"{stem}.translation_source.txt"
    german_seed_sidecar = source_path.with_name(f"{stem} german.txt")
    gemini_seed_sidecar = source_path.with_name(f"{stem} translation and commentary.txt")
    htr_evidence, htr_meta = load_htr_sidecar(pages_dir, stem, settings.htr_enabled)

    copied = _read_optional_text(en_sidecar)
    if copied is not None:
        meta: dict[str, object] = {
            "translation_source_type": "copied_en",
            "translation_sidecar_path": str(en_sidecar),
            "skipped_translate_api": True,
        }
        return copied, "copied_en", meta

    manual = _read_optional_text(manual_sidecar)
    if manual is not None:
        key = settings.google_translate_api_key or ""
        tr, tmeta = translate_de_to_en(
            manual,
            key,
            cache_enabled=settings.translation_cache_enabled,
            cache_path=settings.translation_cache_path,
        )
        out_meta: dict[str, object] = {
            "translation_source_type": "manual",
            "translation_sidecar_path": str(manual_sidecar),
            **{k: v for k, v in tmeta.items()},
        }
        return tr, "manual", out_meta

    # Prefer precomputed commentary over {stem} german.txt + Translate when both exist.
    gemini_seed = _read_optional_text(gemini_seed_sidecar)
    if gemini_seed is not None:
        gs_german = _read_optional_text(german_seed_sidecar)
        out_meta = {
            "translation_source_type": "gemini_seed",
            "gemini_seed_path": str(gemini_seed_sidecar),
            "german_seed_path": str(german_seed_sidecar) if gs_german is not None else "",
            **htr_meta,
            "skipped_translate_api": True,
        }
        return gemini_seed, "gemini_seed", out_meta

    german_seed = _read_optional_text(german_seed_sidecar)
    if german_seed is not None:
        key = settings.google_translate_api_key or ""
        tr, tmeta = translate_de_to_en(
            german_seed,
            key,
            cache_enabled=settings.translation_cache_enabled,
            cache_path=settings.translation_cache_path,
        )
        if tr.strip():
            out_meta: dict[str, object] = {
                "translation_source_type": "manual",
                "translation_sidecar_path": str(german_seed_sidecar),
                "seed_source_type": "german_seed",
                **htr_meta,
                **{k: v for k, v in tmeta.items()},
            }
            return tr, "manual", out_meta

    if htr_evidence is not None and htr_evidence.full_text:
        key = settings.google_translate_api_key or ""
        tr, tmeta = translate_de_to_en(
            htr_evidence.full_text,
            key,
            cache_enabled=settings.translation_cache_enabled,
            cache_path=settings.translation_cache_path,
        )
        out_meta = {
            "translation_source_type": "htr",
            "htr_source_engine": htr_evidence.source_engine,
            "htr_text": htr_evidence.full_text,
            **htr_meta,
            **{k: v for k, v in tmeta.items()},
        }
        return tr, "htr", out_meta

    words, phrases = ocr_word_hypotheses(gray)
    top_phrase: OCRPhraseHypothesis | None = phrases[0] if phrases else None
    ocr = top_phrase.text if top_phrase is not None else ""
    ocr_conf = float(top_phrase.confidence) if top_phrase is not None else 0.0
    memory_boost = phrase_boost(phrase_memory_state or {}, ocr) if ocr else 0.0
    boosted_conf = min(1.0, ocr_conf + memory_boost)
    key = settings.google_translate_api_key or ""
    if ocr and boosted_conf >= settings.ocr_translation_confidence_min:
        tr, tmeta = translate_de_to_en(
            ocr,
            key,
            cache_enabled=settings.translation_cache_enabled,
            cache_path=settings.translation_cache_path,
        )
    else:
        tr, tmeta = "", {
            "skipped": True,
            "reason": "ocr_confidence_below_threshold" if ocr else "no_ocr_text",
        }
    if not tr and settings.gemini_enabled:
        gemini_text, gmeta = gemini_translate_image(
            image_path=source_path,
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_model,
            prompt=settings.gemini_prompt,
            cache_enabled=settings.gemini_cache_enabled,
            cache_path=settings.gemini_cache_path,
        )
        if gemini_text.strip():
            gemini_sidecar_path = pages_dir / f"{stem}.gemini_translation.txt"
            gemini_sidecar_path.write_text(gemini_text, encoding="utf-8")
            tr = gemini_text
            src_type = "gemini"
            out_meta: dict[str, object] = {
                "translation_source_type": src_type,
                "gemini_saved_path": str(gemini_sidecar_path),
                **htr_meta,
                **gmeta,
            }
            return tr, src_type, out_meta
        tmeta = {**tmeta, **{f"gemini_{k}": v for k, v in gmeta.items()}}
    ocr_stripped = ocr.strip()
    src_type = "ocr" if ocr_stripped else "none"
    ometa: dict[str, object] = {
        "translation_source_type": src_type,
        "ocr_confidence": ocr_conf,
        "ocr_confidence_boosted": boosted_conf,
        "ocr_phrase_memory_boost": memory_boost,
        "ocr_translation_confidence_min": settings.ocr_translation_confidence_min,
        "ocr_words": [
            {
                "text": w.text,
                "confidence": w.confidence,
                "bbox_xywh": list(w.bbox_xywh),
                "language": w.language,
                "preprocess": w.preprocess,
                "line_id": w.line_id,
            }
            for w in words[:240]
        ],
        "ocr_phrases": [
            {
                "text": p.text,
                "confidence": p.confidence,
                "language": p.language,
                "preprocess": p.preprocess,
                "line_id": p.line_id,
            }
            for p in phrases[:120]
        ],
        **htr_meta,
        **{k: v for k, v in tmeta.items()},
    }
    return tr, src_type, ometa


def _build_page_record(
    page: ProcessedPage,
    settings: Settings,
    interactions: list[dict[str, object]],
    ghost_meta: dict[str, object],
    dewarp_meta: dict[str, object],
    pages_dir: Path,
    phrase_memory_state: dict[str, int],
) -> dict[str, object]:
    stem = Path(page.source_path).stem
    pages_dir = Path(pages_dir)
    with Image.open(page.cleaned_grayscale_path) as im:
        gray = im.convert("L")
    translation_en, source_type, tmeta = _resolve_translation_pipeline(
        pages_dir, stem, Path(page.source_path), gray, settings, phrase_memory_state=phrase_memory_state
    )
    htr_text = ""
    if source_type == "htr":
        htr_text = str((tmeta.get("htr_text") or "")).strip()
    ocr_text = str(tmeta.get("ocr_phrases", [{}])[0].get("text", "")).strip() if tmeta.get("ocr_phrases") else ""
    phrase_memory_next = observe_phrases(
        phrase_memory_state,
        [str(p.get("text", "")) for p in tmeta.get("ocr_phrases", []) if isinstance(p, dict)],
    )
    phrase_memory_state.clear()
    phrase_memory_state.update(phrase_memory_next)
    copied_text = _read_optional_text(pages_dir / f"{stem}.translation_en.txt")
    manual_text = _read_optional_text(pages_dir / f"{stem}.translation_source.txt")
    if manual_text is None:
        manual_text = _read_optional_text(Path(page.source_path).with_name(f"{stem} german.txt"))
    text_evidence = extract_text_evidence(
        image_size_wh=gray.size,
        ocr_text=ocr_text,
        manual_text=manual_text,
        copied_english_text=copied_text,
        htr_text=htr_text,
    )
    edge_meta = build_edge_reconstruction_candidates(
        stem=stem,
        pages_dir=pages_dir,
        cleaned_gray_path=Path(page.cleaned_grayscale_path),
        edge_inpaint_mask_path=Path(page.edge_inpaint_mask_path),
        text_evidence=text_evidence.to_meta(),
    )
    ocr_hint_meta = build_ocr_reconstruction_hints(
        stem=stem,
        pages_dir=pages_dir,
        image_size_wh=gray.size,
        words=[w for w in tmeta.get("ocr_words", []) if isinstance(w, dict)],
        enable=settings.ocr_reconstruct_hint_enabled,
        confidence_min=settings.ocr_reconstruct_hint_confidence_min,
    )

    rec: dict[str, object] = {
        **page.to_meta(),
        "translation_en": translation_en,
        "translation_source_type": source_type,
        "translation_meta": tmeta,
        "text_evidence": text_evidence.to_meta(),
        "interactions": interactions,
        **ghost_meta,
        **dewarp_meta,
        **ocr_hint_meta,
        **edge_meta,
    }
    if tmeta.get("error"):
        rec["translation_error"] = tmeta.get("error")
    # Legacy field used in older PDF copy
    rec["ocr_draft"] = ocr_text
    return rec


def _edge_apply_policy(page_record: dict[str, object], settings: Settings) -> tuple[bool, str]:
    if not settings.edge_reconstruct_enabled:
        return False, "disabled_setting"
    confidence = float(page_record.get("edge_candidate_confidence", 0.0) or 0.0)
    candidate_pixels = int(page_record.get("edge_candidate_pixels", 0) or 0)
    protected_ratio = float(page_record.get("edge_candidate_protected_ratio", 0.0) or 0.0)
    if candidate_pixels <= 0:
        return False, "no_candidates"
    if protected_ratio >= 0.05:
        return False, "protected_structure_ratio_high"
    if confidence < 0.22:
        return False, "confidence_below_apply_threshold"
    return True, "policy_passed"


def _build_quality_summary(page_records: list[dict[str, object]]) -> dict[str, object]:
    total_pages = len(page_records)
    if total_pages == 0:
        return {
            "page_count": 0,
            "edge_apply_rate": 0.0,
            "edge_avg_confidence": 0.0,
            "edge_avg_protected_ratio": 0.0,
            "edge_unresolved_pages": 0,
            "edge_policy_version": "v1_balanced",
        }
    edge_applied = [bool(p.get("edge_reconstruct_applied", False)) for p in page_records]
    confidences = [float(p.get("edge_candidate_confidence", 0.0) or 0.0) for p in page_records]
    protected = [float(p.get("edge_candidate_protected_ratio", 0.0) or 0.0) for p in page_records]
    unresolved = 0
    for p in page_records:
        confidence = float(p.get("edge_candidate_confidence", 0.0) or 0.0)
        candidate_pixels = int(p.get("edge_candidate_pixels", 0) or 0)
        applied = bool(p.get("edge_reconstruct_applied", False))
        if candidate_pixels > 0 and not applied and confidence >= 0.2:
            unresolved += 1
    return {
        "page_count": total_pages,
        "edge_apply_rate": float(sum(edge_applied) / total_pages),
        "edge_avg_confidence": float(sum(confidences) / total_pages),
        "edge_avg_protected_ratio": float(sum(protected) / total_pages),
        "edge_unresolved_pages": unresolved,
        "edge_policy_version": "v1_balanced",
    }


def _reproduction_image_path(page: ProcessedPage, rec: dict[str, object], settings: Settings) -> Path:
    dewarped = rec.get("dewarped_grayscale_path")
    if (
        settings.dewarp_enabled
        and settings.repro_use_dewarped_when_available
        and isinstance(dewarped, str)
        and dewarped
        and Path(dewarped).is_file()
    ):
        return Path(dewarped)
    return Path(page.cleaned_grayscale_path)


def run_proof(
    input_root: Path,
    job_store_path: Path,
    output_root: Path = Path("outputs/proof"),
    limit: int = 4,
    start: int = 1,
    run_id: str | None = None,
    profile: str | None = None,
) -> ProofRun:
    selected = select_proof_pages(input_root, limit=limit, start=start)
    all_files = scan_jpegs(input_root)
    if not selected:
        raise RuntimeError(f"No JPEG files found under {input_root}")

    settings = load_settings()
    settings = _apply_profile_overrides(settings, profile or settings.processing_profile)
    run_id = run_id or make_run_id()
    run_dir = Path(output_root).resolve() / run_id
    pages_dir = run_dir / "pages"
    interactions_dir = run_dir / "interactions"
    pdf_dir = run_dir / "pdf"
    run_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    interactions_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    store = JobStore(job_store_path)
    store.init_schema()

    processed: list[ProcessedPage] = []
    page_records: list[dict[str, object]] = []
    phrase_memory_state: dict[str, int] = {}
    pbar = tqdm(selected, desc="Processing pages", unit="page")
    for source in pbar:
        pbar.set_postfix_str(source.name, refresh=True)
        job_id = store.upsert_file(source, JobState.PENDING)
        store.update_state(job_id, JobState.PROCESSING)
        try:
            page = process_page(source, pages_dir)
            interactions = [
                a.to_meta() for a in analyze_interactions_for_source(source, all_files, interactions_dir)
            ]
            stem = Path(page.source_path).stem
            neighbor_paths = {str(m["relation"]): Path(str(m["neighbor_path"])) for m in interactions}
            ghost_meta = apply_ghost_suppression(
                cleaned_gray_path=Path(page.cleaned_grayscale_path),
                neighbor_paths=neighbor_paths,
                interactions=interactions,
                pages_dir=pages_dir,
                stem=stem,
                enable=settings.ghost_suppression_enabled,
                confidence_min=settings.ghost_confidence_min,
                plausibility_min=settings.ghost_plausibility_min,
                plausibility_passes=settings.ghost_plausibility_exhaustive_passes,
            )
            speckle_meta = apply_speckle_refine(
                stem=stem,
                pages_dir=pages_dir,
                cleaned_gray_path=Path(page.cleaned_grayscale_path),
                enable=settings.speckle_refine_enabled,
                max_component_area=settings.speckle_max_component_area,
                border_max_component_area=settings.speckle_border_max_component_area,
                border_band_ratio=settings.speckle_border_band_ratio,
            )
            line_refine_meta = apply_line_artifact_refine(
                stem=stem,
                pages_dir=pages_dir,
                cleaned_gray_path=Path(page.cleaned_grayscale_path),
                enable=settings.line_refine_enabled,
                min_length_ratio=settings.line_refine_min_length_ratio,
                border_band_ratio=settings.line_refine_border_band_ratio,
            )
            crease_meta = apply_crease_refine(
                stem=stem,
                pages_dir=pages_dir,
                cleaned_gray_path=Path(page.cleaned_grayscale_path),
                enable=settings.crease_refine_enabled,
                darkness_threshold=settings.crease_darkness_threshold,
            )
            dewarp_meta = dewarp_grayscale_optional(
                cleaned_gray_path=Path(page.cleaned_grayscale_path),
                pages_dir=pages_dir,
                stem=stem,
                enable=settings.dewarp_enabled,
            )
            rec = _build_page_record(
                page,
                settings,
                interactions,
                {**ghost_meta, **speckle_meta, **line_refine_meta, **crease_meta},
                dewarp_meta,
                pages_dir,
                phrase_memory_state,
            )
            edge_apply_enable, edge_apply_reason = _edge_apply_policy(rec, settings)
            edge_apply_meta = apply_edge_reconstruction(
                stem=stem,
                pages_dir=pages_dir,
                cleaned_gray_path=Path(page.cleaned_grayscale_path),
                candidate_mask_path=Path(str(rec["edge_reconstruct_candidate_mask_path"])),
                enable=edge_apply_enable,
                strength=settings.edge_reconstruct_strength,
            )
            rec.update(edge_apply_meta)
            rec["edge_reconstruct_policy_reason"] = edge_apply_reason
            page_records.append(rec)
            store.update_state(job_id, JobState.DONE, meta=rec, cost_units=0.0)
            processed.append(page)
        except Exception as exc:
            store.update_state(job_id, JobState.FAILED, error=str(exc), cost_units=0.0)
            raise

    repro_paths = [_reproduction_image_path(page, rec, settings) for page, rec in zip(processed, page_records)]
    reproduction_pdf = export_reproduction_pdf(repro_paths, pdf_dir / "reproduction_proof.pdf")
    translation_pdf = export_translation_pdf(page_records, pdf_dir / "translation_proof.pdf")
    quality_summary = _build_quality_summary(page_records)
    pilot_print = write_pilot_print_bundle(
        run_dir=run_dir,
        page_records=page_records,
        reproduction_pdf=reproduction_pdf,
        translation_pdf=translation_pdf,
        manifest_path=run_dir / "manifest.json",
        quality_summary=quality_summary,
    )

    run = ProofRun(
        run_id=run_id,
        output_dir=str(run_dir),
        selected_sources=[str(p) for p in selected],
        processed_pages=page_records,
        reproduction_pdf=str(reproduction_pdf),
        translation_pdf=str(translation_pdf),
        manifest_path=str(run_dir / "manifest.json"),
        cloud_manifest_path="",
        pilot_print_bundle_path=str(pilot_print),
        quality_summary=quality_summary,
    )
    cloud = build_local_cloud_manifest(
        run_id=run.run_id,
        output_dir=Path(run.output_dir),
        processed_pages=page_records,
        profile=(profile or settings.processing_profile),
    )
    run = ProofRun(
        run_id=run.run_id,
        output_dir=run.output_dir,
        selected_sources=run.selected_sources,
        processed_pages=run.processed_pages,
        reproduction_pdf=run.reproduction_pdf,
        translation_pdf=run.translation_pdf,
        manifest_path=run.manifest_path,
        cloud_manifest_path=str(cloud.path),
        pilot_print_bundle_path=run.pilot_print_bundle_path,
        quality_summary=run.quality_summary,
    )
    Path(run.manifest_path).write_text(json.dumps(asdict(run), indent=2, default=str), encoding="utf-8")
    return run
