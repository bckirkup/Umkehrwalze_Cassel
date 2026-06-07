from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ProcessingProfile(str, Enum):
    QUICK = "quick"
    BALANCED = "balanced"
    FORENSIC = "forensic"
    TRAINING = "training"


class PageJobIdentity(BaseModel):
    project_id: str = Field(min_length=1)
    volume_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)


class InputArtifacts(BaseModel):
    source_image_uri: str = Field(min_length=1)
    interaction_mask_uri: str | None = None
    ocr_hints_uri: str | None = None


class StageToggles(BaseModel):
    ghost_suppression: bool = False
    dewarp: bool = False
    ocr_extract: bool = False
    translation: bool = False
    pdf_export: bool = True


class Provenance(BaseModel):
    code_version: str = Field(min_length=1)
    contract_created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    input_checksum: str | None = None
    output_checksum: str | None = None

    @field_validator("contract_created_at")
    @classmethod
    def _ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value


class CloudJobContract(BaseModel):
    identity: PageJobIdentity
    inputs: InputArtifacts
    output_prefix: str = Field(min_length=1)
    processing_profile: ProcessingProfile = ProcessingProfile.BALANCED
    stages: StageToggles = Field(default_factory=StageToggles)
    provenance: Provenance
    outbound_upload_opt_in: bool = False

    @model_validator(mode="after")
    def _require_explicit_upload_opt_in(self) -> CloudJobContract:
        values = [
            self.inputs.source_image_uri,
            self.inputs.interaction_mask_uri,
            self.inputs.ocr_hints_uri,
            self.output_prefix,
        ]
        has_remote_target = any(
            value is not None and "://" in value and not value.startswith("file://")
            for value in values
        )
        if has_remote_target and not self.outbound_upload_opt_in:
            raise ValueError(
                "Remote URIs require outbound_upload_opt_in=True for explicit consent."
            )
        return self
