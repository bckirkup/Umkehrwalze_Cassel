# Sample Pages

This project processes 18th-century historical manuscript pages from the Hessisches Staatsarchiv Marburg.  The source images are public-domain records.

## Recommended sample inputs

Place scans under your `RPK_INPUT_ROOT` directory (see `.env.example`).  The expected filename pattern is:

```
hstam_4_h_nr_4156_NNNN.jpg
```

where `NNNN` is the four-digit scan number.

## Expected outputs

After running `revprint batch` (or `revprint process-proof`), these key artifacts are generated per page:

- `pages/{stem}.stamp_cleaned.png` — stamp regions inpainted
- `pages/{stem}.flattened.png` — illumination normalised
- `pages/{stem}.adaptive_cleaned.png` — ink-on-white via adaptive binarisation
- `pages/{stem}.dewarped.png` — text-line curvature corrected
- `pages/{stem}.ghost_cleaned.png` — bleed-through suppressed

Plus batch-level outputs:
- `corpus_stats.json` — paper/ink distributions and stamp colour centroids
- `batch_manifest.json` — per-page QA metrics and processing metadata

For the legacy proof pipeline:
- `pages/*.cleaned_gray.png` — ink-on-white
- `pages/*.edge_inpaint_mask.png`
- `pdf/reproduction_proof.pdf`
- `pdf/translation_proof.pdf`

## Suggested review workflow

1. Compare source scan to the final `ghost_cleaned.png`.
2. Check `batch_manifest.json` for flagged pages (high ink coverage, low brightness uniformity).
3. Review stamp masks (`*.stamp_mask.png`) for false positives.
4. Open reproduction PDF and confirm readability.
