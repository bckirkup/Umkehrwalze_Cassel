# Model-assisted reconstruction — dataset & provenance

This document defines a conservative JSON format for packaging manuscript proof artifacts into training or evaluation data for patch-level restoration (inpainting / denoising) with human review.

## Design goals

- Preserve **provenance**: every derived pixel region must trace back to the original scan crop.
- Carry **uncertainty** when models emit it (optional `uncertainty_png` per patch).
- Keep records **filesystem-friendly** (relative paths inside a dataset root).

## Top-level record (`sample.json` or one row in JSONL)

| Field | Type | Description |
| --- | --- | --- |
| `sample_id` | string | Stable id, e.g. `{run_id}__{page_stem}`. |
| `run_id` | string | Proof run folder name under `outputs/proof/`. |
| `page_stem` | string | Source JPEG stem. |
| `source_jpeg` | string | Relative path to original scan. |
| `crop_bbox` | `[l,t,r,b]` | Integers in source pixel space (same as proof manifest). |
| `cleaned_grayscale` | string | Ink-on-white cleaned L image (post edge-inpaint, post optional ghost suppression). |
| `ghost_suppress_before` | string \| null | Optional review frame before ghost suppression. |
| `ghost_suppress_after` | string \| null | Optional review frame after ghost suppression. |
| `edge_inpaint_mask` | string | 8-bit mask, 255 = inpainted edge band. |
| `dewarped_grayscale` | string \| null | Optional deskew output; may differ in canvas size from cleaned. |
| `interaction_masks` | object | Map `previous` / `next` to mirrored-neighbor ghost candidate masks (PNG, analysis-upscaled paths as stored in manifest). |
| `registration` | object | Per-neighbor: `registration_confidence`, `registration_applied`, `registration_reason`, `body_mask_coverage`, `shift_yx`, `registration_error`. |
| `review_targets` | object | Human-approved targets (optional until labeling exists). |
| `review_targets.ink_rgb` | string \| null | Intended reproduction RGB (future). |
| `review_targets.correction_mask` | string \| null | 8-bit mask of regions where model output may replace cleaned pixels. |
| `model_output` | object \| null | Populated after inference. |
| `model_output.restored_patch_dir` | string \| null | Directory of PNG patches. |
| `model_output.uncertainty_patch_dir` | string \| null | Parallel uncertainty tiles. |
| `tags` | string[] | e.g. `["interior", "high_showthrough"]`. |

## Patch index (optional, for tile training)

`patches.jsonl` — one JSON object per line:

```json
{
  "sample_id": "20260101_120000__page_0003",
  "patch_id": "20260101_120000__page_0003__r012_c048",
  "bbox_xywh": [192, 440, 256, 256],
  "inputs": {
    "gray_crop": "patches/.../input.png",
    "ghost_mask_crop": "patches/.../ghost.png"
  },
  "targets": {
    "approved_gray": "patches/.../target.png"
  },
  "provenance": {
    "derived_from": "cleaned_grayscale",
    "original_bbox_in_cleaned": [192, 440, 256, 256]
  }
}
```

## Integrity

- SHA256 checksums may be added later as `checksums: { "source_jpeg": "..." }` without breaking readers that ignore unknown keys.
- Tools **must not** delete originals when generating model candidates; keep side-by-side under the run directory.

## Versioning

- `schema_version`: use `1` for the structure above; bump when fields are renamed or required sets change.
