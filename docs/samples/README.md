# Sample Pages For GitHub Documentation

This project includes two representative sample pages from the working corpus so contributors can see what the pipeline starts with and what it should produce.

## Public-domain status

The manuscript images used here are 18th-century historical documents and are not under modern copyright protection. In plain terms: these pages are public-domain records, certainly not copyrighted by King George.

We still preserve source attribution and provenance metadata for archival integrity.

## Recommended sample inputs

- `HerrBenjaminKirkup/hstam_4_h_nr_4138_0002.jpg`
- `HerrBenjaminKirkup/hstam_4_h_nr_4138_0003.jpg`

## What users should expect to generate

After running `revprint process-proof` (or GUI proof processing), these key artifacts are expected:

- `pages/*.cleaned_gray.png` (ink-on-white)
- `pages/*.edge_inpaint_mask.png`
- `interactions/*.interaction_*_mirror_mask.png`
- `pages/*.plausibility_map.png`
- `pdf/reproduction_proof.pdf`
- `pdf/translation_proof.pdf`

## Suggested walkthrough for contributors

1. Compare each source sample to `cleaned_gray`.
2. Review `interaction` and `plausibility` artifacts for suppression safety.
3. Open both proof PDFs and confirm readability and layout.
