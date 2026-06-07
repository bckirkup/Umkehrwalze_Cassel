from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from revprint.cloud_contract import (
    CloudJobContract,
    InputArtifacts,
    PageJobIdentity,
    ProcessingProfile,
    Provenance,
    StageToggles,
)
from revprint.cloud_manifest import CloudManifest, CloudOffloadMode


@dataclass(frozen=True)
class CloudManifestBuildResult:
    manifest: CloudManifest
    path: Path


def _profile_enum(name: str) -> ProcessingProfile:
    raw = (name or "balanced").strip().lower()
    for p in ProcessingProfile:
        if p.value == raw:
            return p
    return ProcessingProfile.BALANCED


def _stage_toggles(page: dict[str, Any]) -> StageToggles:
    return StageToggles(
        ghost_suppression=bool(page.get("ghost_suppression_enabled", False)),
        dewarp=bool(page.get("dewarp_enabled", False)),
        ocr_extract=bool((page.get("ocr_draft") or "").strip()),
        translation=bool(page.get("translation_en") or page.get("translation_source_type")),
        pdf_export=True,
    )


def build_local_cloud_manifest(
    *,
    run_id: str,
    output_dir: Path,
    processed_pages: list[dict[str, object]],
    project_id: str = "local-project",
    volume_id: str = "local-volume",
    profile: str = "balanced",
    code_version: str = "revprint-0.1.0",
) -> CloudManifestBuildResult:
    jobs: list[CloudJobContract] = []
    out_dir = Path(output_dir).resolve()
    for idx, page in enumerate(processed_pages, start=1):
        source_path = str(page.get("source_path", ""))
        page_stem = Path(source_path).stem or f"page_{idx:04d}"
        contract = CloudJobContract(
            identity=PageJobIdentity(
                project_id=project_id,
                volume_id=volume_id,
                run_id=run_id,
                page_id=page_stem,
            ),
            inputs=InputArtifacts(
                source_image_uri=f"file:///{Path(source_path).resolve()}",
                interaction_mask_uri=str(page.get("edge_reconstruct_candidate_mask_path") or None),
                ocr_hints_uri=str(page.get("ocr_reconstruct_hint_mask_path") or None),
            ),
            output_prefix=f"file:///{(out_dir / 'pages' / page_stem).as_posix()}",
            processing_profile=_profile_enum(profile),
            stages=_stage_toggles(page),
            provenance=Provenance(
                code_version=code_version,
                contract_created_at=datetime.now(timezone.utc),
                input_checksum=None,
                output_checksum=None,
            ),
            outbound_upload_opt_in=False,
        )
        jobs.append(contract)
    manifest = CloudManifest(
        manifest_id=f"{run_id}-local",
        mode=CloudOffloadMode.LOCAL_ONLY,
        jobs=jobs,
    )
    path = out_dir / "cloud_manifest.local.json"
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return CloudManifestBuildResult(manifest=manifest, path=path)
