# revolution-print (scaffold)

Python scaffold for a manuscript **image pipeline**: configurable input root, filesystem-based `scan_jpegs()`, and a SQLite **job store** for per-file progress and future cost tracking.

**Developer note: File discovery uses the filesystem API (e.g. `Path.iterdir` / `scan_jpegs`); do not rely on Cursor or other IDE “project search” to enumerate assets.** Large JPEG trees may not be fully indexed; the app must walk the configured folder on disk.

## Layout

- `RPK_INPUT_ROOT` — root folder containing `*.jpg` / `*.jpeg` (e.g. `HerrBenjaminKirkup\`).
- `RPK_JOB_STORE` — path to the SQLite file (default: `data/jobs.sqlite`).
- `RPK_PROJECT_STORE` — project/volume metadata SQLite path (default: `data/projects.sqlite`).
- `RPK_GOOGLE_TRANSLATE_API_KEY` — optional: [Google Cloud Translation API v2](https://cloud.google.com/translate/docs/reference/rest/v2/translate) key to draft English text on translation PDFs.
- `RPK_GEMINI_ENABLED` — optional `true`/`false` (default `false`): enable Gemini image-based translation fallback.
- `RPK_GEMINI_API_KEY` — optional Gemini API key for `generateContent`.
- `RPK_GEMINI_MODEL` — optional model name (default `gemini-2.5-flash`).
- `RPK_GEMINI_PROMPT` — optional custom prompt for Gemini translation + commentary.
- `RPK_GEMINI_CACHE` — optional `true`/`false` (default `true`): cache Gemini responses locally.
- `RPK_GEMINI_CACHE_PATH` — optional path (default `data/gemini_cache.json`): persistent Gemini cache.
- `RPK_TRANSLATION_CACHE` — optional `true`/`false` (default `true`): reuse prior translation API responses from local cache.
- `RPK_TRANSLATION_CACHE_PATH` — optional path (default `data/translation_cache.json`): persistent translation cache location.
- `RPK_OCR_TRANSLATION_CONFIDENCE_MIN` — optional float (default `0.34`): skip auto-translation when OCR confidence is below threshold.
- `RPK_OCR_RECONSTRUCT_HINT` — optional `true`/`false` (default `true`): generate OCR-based stroke continuation hint masks.
- `RPK_OCR_RECONSTRUCT_HINT_CONFIDENCE_MIN` — optional float (default `0.38`): minimum OCR word confidence for reconstruction hints.
- `RPK_HTR_ENABLED` — optional `true`/`false` (default `true`): enable handwritten-text sidecar ingestion from `{stem}.htr.json`.
- `RPK_CREASE_REFINE` — optional `true`/`false` (default `true`): suppress broad fold/spine shadow artifacts.
- `RPK_CREASE_DARKNESS_THRESHOLD` — optional float (default `12.0`): darkness threshold used for crease candidate selection.
- `RPK_GHOST_SUPPRESSION` — optional `true`/`false` (default `false`): confidence-gated lighten of mirrored-neighbor ghost candidates; writes `*.ghost_suppress_before.png` / `after` next to each cleaned page.
- `RPK_GHOST_CONFIDENCE_MIN` — optional float in `[0,1]` (default `0.18`): minimum per-neighbor `registration_confidence` to use a ghost mask.
- `RPK_GHOST_PLAUSIBILITY_MIN` — optional float in `[0,1]` (default `0.55`): protect candidate marks likely to be true penstrokes from suppression.
- `RPK_GHOST_PLAUSIBILITY_PASSES` — optional int (default `2`): expensive multi-scale plausibility passes per page; raise significantly (e.g. `12-32`) when maximizing quality over runtime.
- `RPK_DEWARP` — optional `true`/`false` (default `false`): emit `*.dewarped_gray.png` deskew variant (reproduction PDF prefers it when `RPK_REPRO_USE_DEWARPED` is true).
- `RPK_REPRO_USE_DEWARPED` — default `true` when dewarp is enabled: use dewarped image in the reproduction PDF if present.
- `RPK_PROCESSING_PROFILE` — one of `quick`, `balanced`, `forensic`, `training` (default `balanced`).
- `RPK_SPECKLE_REFINE` — optional `true`/`false` (default `true`): remove tiny isolated dots before edge apply.
- `RPK_SPECKLE_MAX_COMPONENT_AREA` — optional int (default `36`): max blob area to treat as removable speckle.
- `RPK_SPECKLE_BORDER_MAX_COMPONENT_AREA` — optional int (default `90`): stronger border-zone speckle cleanup area cap.
- `RPK_SPECKLE_BORDER_BAND_RATIO` — optional float (default `0.11`): relative width of top/bottom/left/right border bands for targeted cleanup.
- `RPK_LINE_REFINE` — optional `true`/`false` (default `true`): detect and suppress long straight spine/fold artifact lines.
- `RPK_LINE_REFINE_MIN_LENGTH_RATIO` — optional float (default `0.55`): minimum relative line length considered a structural artifact candidate.
- `RPK_LINE_REFINE_BORDER_BAND_RATIO` — optional float (default `0.16`): left/right border band width used when prioritizing spine-line cleanup.

Per-page translation workflow (optional sidecars in the run’s `pages/` folder, same stem as the source JPEG):

- `{stem}.translation_source.txt` — German (or other `de` source) text used instead of OCR for Google Translate (`translation_source_type`: `manual`).
- `{stem} german.txt` (next to source JPG) — German seed transcription imported automatically; translated when API key is present, otherwise retained as source evidence.
- `{stem}.translation_en.txt` — English text used as-is (`translation_source_type`: `copied_en`).
- `{stem}.htr.json` — handwritten text recognition sidecar (preferred for Kurrent/manuscript lines; `translation_source_type`: `htr` when populated).
- `{stem} translation and commentary.txt` (next to source JPG) — precomputed Gemini output imported directly (`translation_source_type`: `gemini_seed`).

Model / training packaging: see `docs/model_dataset_spec.md`.

## Setup

```bash
cd d:\RevolutionPrintingProject
py -3 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Optional OCR for draft German/English text (requires [Tesseract](https://github.com/tesseract-ocr/tesseract) on `PATH`):

```bash
pip install -e ".[dev,ocr]"
```

Copy `.env.example` to `.env` and set `RPK_INPUT_ROOT` to your scan folder (optional if you pass `--input-root` every time).

## CLI

List JPEGs (natural sort on filenames):

```bash
set RPK_INPUT_ROOT=D:\RevolutionPrintingProject\HerrBenjaminKirkup
py -m revprint scan
```

Or:

```bash
py -m revprint scan --input-root D:\RevolutionPrintingProject\HerrBenjaminKirkup
```

Create the job database and register every file as `pending`:

```bash
py -m revprint init-jobs
```

Status:

```bash
py -m revprint status
```

Process four interior JPGs into approval assets:

```bash
py -m revprint process-proof
```

Use volume metadata (segregated output per project/volume):

```bash
py -m revprint project init --name "Staatsarchiv" --corpus-root D:\RevolutionPrintingProject\HerrBenjaminKirkup
py -m revprint volume add --project staatsarchiv --name "4138" --folder D:\RevolutionPrintingProject\HerrBenjaminKirkup --profile forensic
py -m revprint process-proof --project staatsarchiv --volume 4138 --profile forensic
```

By default, `process-proof` uses `--start 1 --limit 4`. This intentionally skips the first file, treating the cover/title page as a unique reconstruction case rather than a representative page for tuning the normal interior-page cleanup pipeline.

This creates a timestamped folder under `outputs/proof/` containing:

- `pages/*.cropped_color.jpg`
- `pages/*.red_suppressed.jpg`
- `pages/*.cleaned_gray.png` (8-bit grayscale, edge-inpainted for print)
- `pages/*.edge_inpaint_mask.png` (where border inpainting was applied)
- `pages/*.debug_overlay.jpg`
- `interactions/*.interaction_*_mirror_mask.png`
- `interactions/*.interaction_*_mirror_overlay.png`
- `pages/*.ghost_suppress_before.png` / `*.ghost_suppress_after.png` (when ghost suppression is enabled)
- `pages/*.speckle_before.png` / `*.speckle_after.png` / `*.speckle_removed_mask.png`
- `pages/*.plausibility_map.png` / `*.plausibility_protect_mask.png` / `*.plausibility_regions.json` (penstroke plausibility diagnostics when ghost suppression runs)
- `pages/*.dewarped_gray.png` (when `RPK_DEWARP=true`)
- `pdf/reproduction_proof.pdf`
- `pdf/translation_proof.pdf`
- `pilot/PRINTING.md` (paths + checklist for first-pages print workflow)
- `manifest.json`
- `cloud_manifest.local.json` (provider-neutral local-only cloud contract for each page)

## Complex Processing Roadmap

The current proof pipeline is deliberately reviewable and layer-oriented:

- **OpenCV** handles deterministic page/edge cleanup, local tone normalization, and inpainting masks.
- **scikit-image** + **SciPy** provide registration for multi-page interactions, starting with mirrored previous/next-page comparison for offset or ghost ink.
- **Tesseract** is optional for draft OCR, but historical Kurrent accuracy is expected to be poor without a specialized model.
- **Google Cloud Translation** can draft English text only after OCR/manual transcription provides source text.

The next model-based stage should use these artifacts as inputs rather than replacing them: original scan, cleaned ink layer, edge masks, mirrored neighbor candidates, and uncertainty/review masks.

Current planning status and next slice:

- **Pilot phase (active):** finish end-to-end printable outputs for the first interior pages only (same stems you already seeded with German / Gemini commentary). Validate reproduction PDF, translation PDF, and `pilot/PRINTING.md` before touching the rest of the corpus (~430 pages).
- Completed: border/spine cleanup, line+crease suppression, OCR phrase memory scaffolding, HTR sidecar adapter path, Gemini seed import, German seed sidecars, and pilot print bundle per run.
- In progress: tighten translation PDF labels and pilot checklist so “good OCR/translation” pages read cleanly for reviewers and print shops.
- Deferred until pilot passes: distinctive-page Gemini sampling at scale, batch HTR import, and corpus-wide automation.

Gemini-assisted bootstrap plan (new):

- Objective split:
  - Reconstruction objective: line/word hypotheses to support stroke continuation and edge recovery.
  - Reading objective: usable translation with traceable source text and confidence.
- What to ask Gemini for each selected page:
  - Diplomatic German transcription (keep historical spelling/abbreviations).
  - Normalized German reading (optional modernization where needed).
  - English translation.
  - Brief uncertainty notes with guessed line references.
- Why request original German too:
  - Yes, this helps significantly. German source text enables later re-translation, consistency checks, and phrase-memory learning that is not tied to one English rendering.
- How many pages/lines to bootstrap:
  - Pilot: 8-12 pages (~120-220 lines) for fast signal.
  - Working baseline: 25-40 pages (~450-900 lines) for stable recurring phrase/hand style coverage.
  - Strong corpus baseline: 80+ pages for robust project-level consistency.
- How to select pages from a large corpus for Gemini:
  - Use stratified sampling: include clean pages, heavy fold/spine pages, faint pages, dense-text pages, and margin-note pages.
  - Prioritize pages with high artifact burden and low OCR confidence first (highest expected quality gain).
  - Include at least a few pages from each section/date/scribe style to avoid overfitting phrase memory.
- Storage conventions (recommended):
  - Keep existing `{stem} translation and commentary.txt` for immediate import.
  - Add optional structured sidecars over time:
    - `{stem}.gemini_de.txt` (diplomatic German)
    - `{stem}.gemini_en.txt` (English)
    - `{stem}.gemini_notes.txt` (uncertainties/comments)
  - Promote reviewed line-level content into `{stem}.htr.json` segments when possible.
- Human-in-the-loop loop:
  - Start with Gemini output, review uncertain lines only, then feed corrected German lines into HTR sidecars.
  - Re-run proof and compare `translation_source_type` mix + visual reconstruction diagnostics.

Start the lightweight local GUI:

```bash
py -m revprint gui
```

Then open `http://127.0.0.1:5000/`. The GUI shows JPG count, job status, a button to run a four-page interior proof, and links to the latest generated PDFs/images.

## Sample Pages And Rights

See `docs/samples/README.md` for the two sample page references and expected outputs. The sample manuscript material is 18th-century public-domain historical content (with archival attribution retained).

## Package import

```python
from pathlib import Path
from revprint.io_scan import scan_jpegs
from revprint.job_store import JobStore

root = Path(r"D:\RevolutionPrintingProject\HerrBenjaminKirkup")
files = scan_jpegs(root)
store = JobStore(Path("data/jobs.sqlite"))
store.init_schema()
store.register_scan(files)
```

## Next steps (not in this scaffold)

Image cleanup, page crop, Kurrent transcription, reconstruction, two PDFs (reproduction + English), web UI, and API token accounting can build on the job store and `JobState` / `meta_json` / `cost_units` fields.

## Review labels for learning loop

Store reviewer decisions and export training labels:

```bash
py -m revprint review add --project staatsarchiv --volume 4138 --run-id 20260425_120000 --page-stem hstam_4_h_nr_4138_0002 --artifact-type edge_candidate --artifact-path outputs/projects/staatsarchiv/4138/proof/20260425_120000/pages/hstam_4_h_nr_4138_0002.edge_reconstruct_candidate_mask.png --decision accept --notes "Likely clipped stroke"
py -m revprint review list --project staatsarchiv --volume 4138 --run-id 20260425_120000
py -m revprint review export --project staatsarchiv --volume 4138 --output outputs/labels/staatsarchiv_4138.jsonl
py -m revprint review rubric --manifest outputs/proof/20260425_150414/manifest.json
py -m revprint htr scaffold --pages-dir outputs/proof/20260425_170940/pages
```

Rubric command output creates `proof_review_rubric.md` beside the manifest by default, giving a consistent human checklist for each proof run.
