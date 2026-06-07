from __future__ import annotations

import pytest

from revprint.cloud_contract import (
    CloudJobContract,
    InputArtifacts,
    PageJobIdentity,
    ProcessingProfile,
    Provenance,
    StageToggles,
)
from revprint.cloud_manifest import CloudManifest, CloudOffloadMode


def _identity() -> PageJobIdentity:
    return PageJobIdentity(
        project_id="proj-a",
        volume_id="vol-1",
        run_id="run-20260425",
        page_id="0002",
    )


def _provenance() -> Provenance:
    return Provenance(code_version="revprint-0.1.0")


def test_cloud_job_contract_roundtrip_json() -> None:
    contract = CloudJobContract(
        identity=_identity(),
        inputs=InputArtifacts(source_image_uri="file:///data/source/0002.jpg"),
        output_prefix="file:///data/output/proj-a/vol-1/run-20260425/page-0002/",
        processing_profile=ProcessingProfile.BALANCED,
        stages=StageToggles(ghost_suppression=True, pdf_export=True),
        provenance=_provenance(),
        outbound_upload_opt_in=False,
    )

    payload = contract.model_dump_json()
    rehydrated = CloudJobContract.model_validate_json(payload)

    assert rehydrated.identity.page_id == "0002"
    assert rehydrated.stages.ghost_suppression is True
    assert rehydrated.processing_profile == ProcessingProfile.BALANCED


def test_remote_uri_requires_explicit_opt_in() -> None:
    with pytest.raises(ValueError):
        CloudJobContract(
            identity=_identity(),
            inputs=InputArtifacts(source_image_uri="s3://bucket/input/0002.jpg"),
            output_prefix="s3://bucket/output/proj-a/",
            provenance=_provenance(),
            outbound_upload_opt_in=False,
        )


def test_remote_uri_allowed_with_opt_in() -> None:
    contract = CloudJobContract(
        identity=_identity(),
        inputs=InputArtifacts(source_image_uri="s3://bucket/input/0002.jpg"),
        output_prefix="s3://bucket/output/proj-a/",
        provenance=_provenance(),
        outbound_upload_opt_in=True,
    )

    assert contract.outbound_upload_opt_in is True


def test_cloud_manifest_serialization() -> None:
    contract = CloudJobContract(
        identity=_identity(),
        inputs=InputArtifacts(source_image_uri="file:///data/source/0002.jpg"),
        output_prefix="file:///data/output/proj-a/",
        provenance=_provenance(),
    )
    manifest = CloudManifest(
        manifest_id="manifest-1",
        mode=CloudOffloadMode.CLOUD_BATCH,
        jobs=[contract],
    )

    encoded = manifest.model_dump_json()
    decoded = CloudManifest.model_validate_json(encoded)

    assert decoded.mode == CloudOffloadMode.CLOUD_BATCH
    assert len(decoded.jobs) == 1
    assert decoded.jobs[0].identity.project_id == "proj-a"
