# revprint

Manuscript image pipeline: scan discovery, cleanup, job tracking, and print-ready outputs.

Processes historical document scans (JPEG) through adaptive binarization, dewarping,
ghost/bleed-through suppression, stamp removal, and PDF export to produce clean
ink-on-white reproductions suitable for fresh printing.

## Install

```bash
pip install -e ".[dev]"        # CPU (CI / development)
pip install -e ".[dev,gpu]"    # GPU-accelerated (local production)
```

## Quick start

```bash
# Discover scans
revprint scan --input-root inputs/text_dense_8

# Run proof pipeline
revprint process-proof --input-root inputs/text_dense_8 --output-root outputs/proof

# Launch review GUI
revprint gui
```

## Processing profiles

| Profile | Ghost suppression | Dewarp | Speckle | Edge reconstruct | Use case |
|---------|-------------------|--------|---------|------------------|----------|
| `quick` | off | off | conservative | off | Fast preview |
| `balanced` | off | off | on | off | Default |
| `forensic` | on | on | aggressive | on | Final production |
| `training` | on | on | on | on | ML dataset prep |
