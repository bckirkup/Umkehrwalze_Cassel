# AGENTS.md — AI Agent Guidelines for Umkehrwalze_Cassel (revprint)

## Repository Purpose
Image processing pipeline for historical manuscript digitisation. Transforms
raw archival scans (JPEG) into clean ink-on-white reproductions suitable for
fresh printing or OCR. Primary corpus: 18th-century English/German cursive
ledgers from the Hessisches Staatsarchiv Marburg.

## Setup
```bash
pip install -e ".[dev]"          # CPU (CI / development)
pip install -e ".[dev,gpu]"      # GPU-accelerated (local production with CUDA)
```

## Validation Commands
Run these before committing:
```bash
ruff check src tests
python -m pytest --no-cov
```

## Architecture Rules
- **Pipeline stages are composable** — each transform is independent and idempotent
- **GPU optional** — all GPU code paths fall back to CPU automatically when CUDA unavailable
- **Non-destructive** — original input files are never modified
- **Deterministic** — same input + config produces identical output
- **Never modify tests to make them pass** — fix the implementation

## Key CLI Commands
```bash
# Discover scans in a directory
revprint scan --input-root inputs/text_dense_8

# Run a small proof (4 pages)
revprint process-proof --input-root inputs/text_dense_8 --output-root outputs/proof

# Full batch cleanup pipeline
revprint batch --input-root inputs/text_dense_8 --output-root outputs/batch \
    [--json inputs/text_dense_8/hessian_archive_export_2026-04-26.json] \
    [--no-gpu] [--n-jobs 4]

# Job store management
revprint init-jobs --input-root inputs/text_dense_8
revprint status

# Project / volume organisation
revprint project init --name "Marburg 4156" --corpus-root /path/to/corpus
```

## Key Files
| Path | Purpose |
|------|---------|
| `src/revprint/` | Core package |
| `src/revprint/cli.py` | CLI entrypoint and subcommands |
| `src/revprint/pipeline/` | Processing stages (binarisation, dewarping, cleanup) |
| `src/revprint/gpu/` | GPU-accelerated paths (Sauvola, tensor filtering) |
| `tests/` | pytest suite |
| `inputs/` | Sample input scans for testing |

## Processing Challenges
- Binding curvature (dewarping)
- Ink bleed-through from facing pages
- Librarian stamps (removal)
- Spine shadows
- Ragged edges
- Staining and foxing

## Code Conventions
- Python 3.10+
- Ruff for linting
- pytest for testing
- PyTorch + kornia for GPU paths (optional)
- Pydantic for configuration models

## PR Requirements
- All ruff checks pass
- All tests pass
- New processing stages include tests with sample inputs
- GPU paths must have CPU fallback
