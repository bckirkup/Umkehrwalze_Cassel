"""Batch pipeline orchestrator for 600+ page corpus processing.

Ties together all phases:
  Phase 0 — page model
  Phase 1 — corpus stats
  Phase 2 — per-page cleaning (stamp suppress → flatten → adaptive clean → dewarp)
  Phase 3 — cross-page ghost suppression
  Phase 4 — QA metrics

Uses joblib for parallel execution of per-page stages.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from revprint.adaptive_clean import adaptive_clean_page
from revprint.corpus_stats import CorpusStats, compute_corpus_stats
from revprint.ghost_cross_page import suppress_ghost
from revprint.io_scan import scan_jpegs
from revprint.line_dewarp import dewarp_page
from revprint.page_model import build_page_model
from revprint.spine_flatten import flatten_spine_shadow
from revprint.stamp_suppress import suppress_stamps

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """Result for a single page through the batch pipeline."""

    source_path: str
    scan_number: int
    side: str
    stamp_meta: dict[str, object] = field(default_factory=dict)
    flatten_meta: dict[str, object] = field(default_factory=dict)
    clean_meta: dict[str, object] = field(default_factory=dict)
    dewarp_meta: dict[str, object] = field(default_factory=dict)
    ghost_meta: dict[str, object] = field(default_factory=dict)
    qa_metrics: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    elapsed_sec: float = 0.0


@dataclass
class QAMetrics:
    """Quality metrics for a single page."""

    ink_coverage: float = 0.0
    brightness_uniformity: float = 0.0
    ghost_residual: float = 0.0
    edge_completeness: float = 0.0
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)


@dataclass
class BatchResult:
    """Result of a full batch pipeline run."""

    run_id: str
    output_dir: str
    page_count: int
    success_count: int
    error_count: int
    flagged_count: int
    total_elapsed_sec: float
    corpus_stats: dict[str, object] = field(default_factory=dict)
    page_results: list[PageResult] = field(default_factory=list)


def _compute_qa_metrics(
    cleaned_path: Path,
    facing_path: Path | None = None,
) -> QAMetrics:
    """Compute quality metrics for a cleaned page."""
    from PIL import Image, ImageOps

    qa = QAMetrics()
    try:
        with Image.open(cleaned_path) as im:
            gray = np.asarray(ImageOps.exif_transpose(im).convert("L"), dtype=np.float32)
    except Exception:
        qa.flagged = True
        qa.flag_reasons.append("cannot_load_cleaned_image")
        return qa

    h, w = gray.shape
    total_pixels = max(1, h * w)

    # Ink coverage: fraction of pixels below threshold.
    ink_mask = gray < 200.0
    qa.ink_coverage = float(np.count_nonzero(ink_mask)) / total_pixels

    # Brightness uniformity: std of paper-region brightness.
    paper_pixels = gray[gray >= 200.0]
    if paper_pixels.size > 0:
        qa.brightness_uniformity = 1.0 - min(1.0, float(np.std(paper_pixels)) / 30.0)
    else:
        qa.brightness_uniformity = 0.0

    # Edge completeness: fraction of border that is white (no clipped text).
    border_band = max(3, int(min(h, w) * 0.02))
    edges = np.concatenate([
        gray[:border_band, :].ravel(),
        gray[-border_band:, :].ravel(),
        gray[:, :border_band].ravel(),
        gray[:, -border_band:].ravel(),
    ])
    qa.edge_completeness = float(np.mean(edges > 220.0))

    # Flag outliers.
    if qa.ink_coverage > 0.5:
        qa.flagged = True
        qa.flag_reasons.append("ink_coverage_too_high")
    if qa.ink_coverage < 0.01:
        qa.flagged = True
        qa.flag_reasons.append("ink_coverage_too_low")
    if qa.brightness_uniformity < 0.5:
        qa.flagged = True
        qa.flag_reasons.append("poor_brightness_uniformity")
    if qa.edge_completeness < 0.7:
        qa.flagged = True
        qa.flag_reasons.append("possible_text_clipping")

    return qa


def _process_single_page(
    source_path: Path,
    output_dir: Path,
    scan_number: int,
    side: str,
    facing_path: Path | None,
    corpus_stats: CorpusStats,
    prefer_gpu: bool = True,
) -> PageResult:
    """Process a single page through all cleaning stages."""
    t0 = time.monotonic()
    stem = source_path.stem
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    result = PageResult(
        source_path=str(source_path),
        scan_number=scan_number,
        side=side,
    )

    try:
        # Stage 1: Stamp suppression (on original RGB).
        stamp_out = pages_dir / f"{stem}.stamp_cleaned.png"
        result.stamp_meta = suppress_stamps(
            input_path=source_path,
            output_path=stamp_out,
            centroids_lab=corpus_stats.stamp_centroids_lab or None,
            radii=corpus_stats.stamp_centroid_radii or None,
        )
        current_gray = stamp_out

        # Stage 2: Spine/illumination flattening.
        flatten_out = pages_dir / f"{stem}.flattened.png"
        result.flatten_meta = flatten_spine_shadow(
            input_path=current_gray,
            output_path=flatten_out,
        )
        current_gray = flatten_out

        # Stage 3: Adaptive ink extraction.
        clean_out = pages_dir / f"{stem}.adaptive_cleaned.png"
        result.clean_meta = adaptive_clean_page(
            input_path=current_gray,
            output_path=clean_out,
            prefer_gpu=prefer_gpu,
        )
        current_gray = clean_out

        # Stage 4: Line-curvature dewarping.
        dewarp_out = pages_dir / f"{stem}.dewarped.png"
        result.dewarp_meta = dewarp_page(
            input_path=current_gray,
            output_path=dewarp_out,
        )
        current_gray = dewarp_out

        # Stage 5: Ghost suppression (needs facing page).
        ghost_out = pages_dir / f"{stem}.ghost_cleaned.png"
        result.ghost_meta = suppress_ghost(
            page_gray_path=current_gray,
            output_path=ghost_out,
            facing_path=facing_path,
        )
        current_gray = ghost_out

        # Stage 6: QA metrics.
        qa = _compute_qa_metrics(current_gray, facing_path=facing_path)
        result.qa_metrics = {
            "ink_coverage": qa.ink_coverage,
            "brightness_uniformity": qa.brightness_uniformity,
            "ghost_residual": qa.ghost_residual,
            "edge_completeness": qa.edge_completeness,
            "flagged": qa.flagged,
            "flag_reasons": qa.flag_reasons,
        }

    except Exception as exc:
        result.error = str(exc)
        logger.exception("Error processing %s", source_path)

    result.elapsed_sec = time.monotonic() - t0
    return result


def run_batch_pipeline(
    input_root: Path,
    output_root: Path,
    json_path: Path | None = None,
    n_jobs: int = -1,
    prefer_gpu: bool = True,
    run_id: str | None = None,
) -> BatchResult:
    """Run the full batch pipeline on all images in input_root.

    Parameters
    ----------
    input_root : directory containing JPEG scans
    output_root : directory for outputs
    json_path : optional path to archive export JSON
    n_jobs : number of parallel workers (-1 = all CPUs).
             Note: GPU stages are serialized internally.
    prefer_gpu : use GPU for adaptive cleaning if available
    run_id : optional run identifier
    """
    from datetime import datetime

    t0 = time.monotonic()
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover images.
    image_paths = scan_jpegs(input_root)
    if not image_paths:
        raise RuntimeError(f"No JPEG files found under {input_root}")

    logger.info("Discovered %d images in %s", len(image_paths), input_root)

    # Phase 0: Build page model.
    logger.info("Phase 0: Building page model")
    model = build_page_model(image_paths, json_path=json_path)

    # Phase 1: Corpus statistics.
    logger.info("Phase 1: Computing corpus statistics")
    stats = compute_corpus_stats(image_paths)

    # Save corpus stats.
    stats_meta = {
        "corpus_paper_median_l": stats.corpus_paper_median_l,
        "corpus_paper_iqr_l": stats.corpus_paper_iqr_l,
        "corpus_ink_median_l": stats.corpus_ink_median_l,
        "corpus_ink_darkest_5pct_l": stats.corpus_ink_darkest_5pct_l,
        "stamp_centroids_lab": stats.stamp_centroids_lab,
        "stamp_centroid_radii": stats.stamp_centroid_radii,
        "page_count": len(stats.page_stats),
    }
    (output_dir / "corpus_stats.json").write_text(
        json.dumps(stats_meta, indent=2, default=str), encoding="utf-8"
    )

    # Phase 2–3: Per-page processing.
    logger.info("Phases 2–3: Processing %d pages", len(model.pages))
    # When using GPU, limit parallelism to avoid GPU memory contention.
    effective_jobs = 1 if prefer_gpu else n_jobs

    results: list[PageResult] = Parallel(n_jobs=effective_jobs, backend="loky")(
        delayed(_process_single_page)(
            source_path=pi.path,
            output_dir=output_dir,
            scan_number=pi.scan_number,
            side=pi.side,
            facing_path=pi.facing_path,
            corpus_stats=stats,
            prefer_gpu=prefer_gpu,
        )
        for pi in model.pages
    )

    # Phase 4: Aggregate QA.
    success_count = sum(1 for r in results if r.error is None)
    error_count = sum(1 for r in results if r.error is not None)
    flagged_count = sum(
        1 for r in results
        if r.qa_metrics.get("flagged", False)
    )

    total_elapsed = time.monotonic() - t0

    batch = BatchResult(
        run_id=run_id,
        output_dir=str(output_dir),
        page_count=len(results),
        success_count=success_count,
        error_count=error_count,
        flagged_count=flagged_count,
        total_elapsed_sec=total_elapsed,
        corpus_stats=stats_meta,
        page_results=results,
    )

    # Save manifest.
    manifest_path = output_dir / "batch_manifest.json"
    manifest_data = {
        "run_id": batch.run_id,
        "page_count": batch.page_count,
        "success_count": batch.success_count,
        "error_count": batch.error_count,
        "flagged_count": batch.flagged_count,
        "total_elapsed_sec": batch.total_elapsed_sec,
        "corpus_stats": batch.corpus_stats,
        "pages": [
            {
                "source_path": r.source_path,
                "scan_number": r.scan_number,
                "side": r.side,
                "error": r.error,
                "elapsed_sec": r.elapsed_sec,
                "qa_metrics": r.qa_metrics,
            }
            for r in results
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, default=str), encoding="utf-8"
    )

    logger.info(
        "Batch complete: %d/%d succeeded, %d flagged, %.1fs total",
        success_count, len(results), flagged_count, total_elapsed,
    )
    return batch
