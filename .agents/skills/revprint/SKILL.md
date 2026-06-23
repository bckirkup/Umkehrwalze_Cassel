---
name: revprint-development
description: Development workflow for the revprint manuscript digitisation pipeline. Covers setup, testing, CLI usage, and extending processing stages.
---

# revprint Development Skill

## Setup

```bash
pip install -e ".[dev]"          # CPU (CI / development)
pip install -e ".[dev,gpu]"      # GPU-accelerated (requires CUDA)
```

## Validation (run before every commit)

```bash
ruff check src tests
python -m pytest --no-cov
```

## Running the Pipeline

```bash
# Discover scans
revprint scan --input-root inputs/text_dense_8

# Small proof run (4 pages)
revprint process-proof --input-root inputs/text_dense_8 --output-root outputs/proof

# Full batch (all phases)
revprint batch --input-root inputs/text_dense_8 --output-root outputs/batch \
    [--json inputs/text_dense_8/hessian_archive_export_2026-04-26.json] \
    [--no-gpu] [--n-jobs 4]

# Job store
revprint init-jobs --input-root inputs/text_dense_8
revprint status
```

## Testing

```bash
# All tests
python -m pytest --no-cov

# Specific module
python -m pytest tests/test_pipeline.py -v

# With coverage
python -m pytest --cov=revprint
```

## Adding a New Processing Stage

1. Create `src/revprint/pipeline/<stage_name>.py`
2. Implement the transform function (input image → output image)
3. Ensure GPU fallback: if using PyTorch/kornia, wrap with CPU fallback
4. Register in the pipeline orchestrator
5. Add tests in `tests/test_<stage_name>.py` with sample inputs
6. Update CLI if new flags are needed

## GPU Acceleration

GPU paths use PyTorch + kornia for:
- Sauvola binarisation (adaptive thresholding)
- Tensor-based filtering (denoising, edge detection)

All GPU code must fall back to CPU automatically:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The `--no-gpu` CLI flag forces CPU mode even when CUDA is available.

## Key Processing Challenges

| Challenge | Approach |
|-----------|----------|
| Binding curvature | Dewarping via polynomial surface fit |
| Ink bleed-through | Frequency-domain separation |
| Librarian stamps | Color-based masking and inpainting |
| Spine shadows | Gradient-based illumination correction |
| Ragged edges | Contour detection and crop |
| Staining/foxing | Local adaptive thresholding (Sauvola) |

## Project Organisation

```bash
revprint project init --name "Marburg 4156" --corpus-root /path/to/corpus
```

Projects track multiple volumes with shared configuration and progress state.
