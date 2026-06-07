# Cloud Offload Contract (Draft)

This document defines provider-neutral contracts for optionally offloading page jobs to cloud
workers in the future. It is intentionally additive groundwork and does not change current local
execution behavior.

## Goals

- Keep the job description stable across local and remote execution targets.
- Support explicit provenance and audit metadata.
- Require explicit opt-in for any outbound upload targets.
- Avoid provider lock-in by using neutral URI/path fields and mode labels.

## Core Contract Objects

- `PageJobIdentity`
  - `project_id`, `volume_id`, `run_id`, `page_id`
- `InputArtifacts`
  - `source_image_uri` (required)
  - `interaction_mask_uri` (optional)
  - `ocr_hints_uri` (optional)
- `StageToggles`
  - `ghost_suppression`, `dewarp`, `ocr_extract`, `translation`, `pdf_export`
- `Provenance`
  - `code_version`
  - `contract_created_at` (UTC-aware timestamp)
  - `input_checksum`, `output_checksum` (optional placeholders)
- `CloudJobContract`
  - `identity`, `inputs`, `output_prefix`
  - `processing_profile` (`quick`, `balanced`, `forensic`, `training`)
  - `stages`, `provenance`
  - `outbound_upload_opt_in` (default `false`)

## Manifest Draft

`CloudManifest` wraps one or more `CloudJobContract` entries:

- `schema_version` (currently `2026-04-draft`)
- `manifest_id`
- `mode`
- `created_at`
- `jobs`

### Modes

- `local-only`
  - All processing remains on local machine.
  - Contracts may still be generated for reproducibility/testing.
- `cloud-assisted`
  - Selected stages or pages may run remotely while local orchestration remains primary.
- `cloud-batch`
  - A manifest of jobs is prepared for asynchronous remote batch processing.

## Security and Privacy Notes

- No cloud SDK integration is part of this draft.
- No network calls should happen when creating or validating these models.
- Any non-file URI (for example provider object storage URIs) is rejected unless
  `outbound_upload_opt_in=true`.
- Checksum fields are placeholders for future integrity verification and may be left empty.
- Provider credentials, secrets, and tenant identifiers are intentionally excluded from the
  contract schema.
