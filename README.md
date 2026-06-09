# revprint

Image processing pipeline for historical manuscript digitisation.  Transforms raw archival scans (JPEG) into clean ink-on-white reproductions suitable for fresh printing or OCR.

The primary corpus is 18th-century English/German cursive ledgers from the Hessisches Staatsarchiv Marburg (item 4 h Nr. 4156, Hessian field hospital records, 1777–1778).  The pipeline handles binding curvature, ink bleed-through from facing pages, librarian stamps, spine shadows, ragged edges, and staining.

## Requirements

- Python ≥ 3.10
- System packages: none beyond what pip installs

## Install

```bash
pip install -e ".[dev]"          # CPU (CI / development)
pip install -e ".[dev,gpu]"      # GPU-accelerated (local production with CUDA)
```

The `[gpu]` extra adds PyTorch and kornia for accelerated Sauvola binarisation and tensor-based filtering.  All GPU code paths fall back to CPU automatically when CUDA is unavailable.

## CLI commands

```bash
# Discover scans in a directory
revprint scan --input-root inputs/text_dense_8

# Run a small proof (4 pages) through the legacy proof pipeline
revprint process-proof --input-root inputs/text_dense_8 --output-root outputs/proof

# Run the full batch cleanup pipeline (all phases)
revprint batch --input-root inputs/text_dense_8 --output-root outputs/batch \
    [--json inputs/text_dense_8/hessian_archive_export_2026-04-26.json] \
    [--no-gpu] [--n-jobs 4]

# Job store management
revprint init-jobs --input-root inputs/text_dense_8
revprint status

# Project / volume organisation
revprint project init --name "Marburg 4156" --corpus-root /path/to/corpus
revprint project list
revprint volume add --project marburg-4156 --name "text_dense_8" --folder inputs/text_dense_8
revprint volume list --project marburg-4156

# Review labelling
revprint review add --project ... --volume ... --run-id ... --page-stem ... \
    --artifact-type cleaned_gray --artifact-path path/to/file --decision accept
revprint review list --project ... --volume ...
revprint review export --project ... --volume ... --output labels.jsonl
revprint review rubric --manifest outputs/proof/manifest.json

# HTR sidecar scaffolding
revprint htr scaffold --pages-dir outputs/proof/pages

# Web GUI for proof review
revprint gui [--host 127.0.0.1] [--port 5000]
```

## Batch pipeline stages

The `revprint batch` command runs these phases in order:

| Phase | Module | What it does |
|-------|--------|--------------|
| 0 | `page_model` | Reconstruct physical page ordering, recto/verso sides, and facing-page pairs from filenames + optional JSON metadata |
| 1 | `corpus_stats` | One-pass profiling at reduced resolution: paper/ink brightness distributions, spine detection, stamp colour clustering (KMeans) |
| 2a | `stamp_suppress` | Lab-space colour distance to corpus stamp centroids → mask → inpainting |
| 2b | `spine_flatten` | 2D illumination field estimation → division normalisation for uniform paper white |
| 2c | `adaptive_clean` | Sauvola local adaptive binarisation with GPU acceleration (kornia) or CPU fallback |
| 2d | `line_dewarp` | Baseline tracing → polynomial curvature model → dense displacement map → `cv2.remap` |
| 3 | `ghost_cross_page` | Facing-page registration via phase correlation → ghost subtraction; NMF blind source separation fallback |
| 4 | QA metrics | Ink coverage, brightness uniformity, edge completeness → JSON manifest with outlier flagging |

Per-page stages run in parallel via `joblib` (CPU mode) or sequentially (GPU mode to avoid VRAM contention).

## Processing profiles (proof pipeline)

The legacy `process-proof` command supports named profiles:

| Profile | Ghost suppression | Dewarp | Speckle | Edge reconstruct | Use case |
|---------|-------------------|--------|---------|------------------|----------|
| `quick` | off | off | conservative | off | Fast preview |
| `balanced` | off | off | on | off | Default |
| `forensic` | on | on | aggressive | on | Final production |
| `training` | on | on | on | on | ML dataset prep |

## Configuration

Runtime settings are loaded from environment variables or a `.env` file in the project root.  See `.env.example` for available variables.

Key variables:

| Variable | Purpose |
|----------|---------|
| `RPK_INPUT_ROOT` | Default directory containing manuscript JPEG scans |
| `RPK_JOB_STORE` | Path to SQLite job-tracking database (default: `data/jobs.sqlite`) |
| `RPK_PROJECT_STORE` | Path to SQLite project/volume store (default: `data/projects.sqlite`) |
| `RPK_PROCESSING_PROFILE` | Default processing profile (`quick`/`balanced`/`forensic`/`training`) |
| `RPK_GOOGLE_TRANSLATE_API_KEY` | Optional Google Translate v2 key for draft translation PDF |
| `RPK_GEMINI_API_KEY` | Optional Gemini API key for image-based translation |
| `RPK_GEMINI_ENABLED` | Enable Gemini vision translation (default: false) |

## Project structure

```
src/revprint/
├── cli.py                 # CLI entrypoint and subcommands
├── batch_pipeline.py      # Full corpus batch orchestrator
├── adaptive_clean.py      # Sauvola binarisation (GPU/CPU)
├── line_dewarp.py         # Text-line curvature correction
├── ghost_cross_page.py    # Facing-page ghost suppression + NMF
├── spine_flatten.py       # Illumination normalisation
├── corpus_stats.py        # Corpus-level statistics pass
├── page_model.py          # Physical page model (recto/verso pairing)
├── stamp_suppress.py      # Lab-space stamp detection + inpainting
├── proof.py               # Legacy proof pipeline
├── image_processing.py    # Core ink extraction and bounding box
├── ghost_suppression.py   # Legacy ghost suppression (scan-order)
├── config.py              # Pydantic settings from env / .env
├── web.py                 # Flask GUI for proof review
├── pdf_export.py          # ReportLab PDF generation
└── ...                    # Supporting modules
```

## Development

```bash
# Run tests
pytest

# Lint
ruff check src tests

# Run tests without coverage threshold
pytest --no-cov
```

## Licence

Apache 2.0 — see `LICENSE`.
