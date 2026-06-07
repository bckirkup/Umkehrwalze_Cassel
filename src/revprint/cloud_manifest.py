from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from revprint.cloud_contract import CloudJobContract


class CloudOffloadMode(str, Enum):
    LOCAL_ONLY = "local-only"
    CLOUD_ASSISTED = "cloud-assisted"
    CLOUD_BATCH = "cloud-batch"


class CloudManifest(BaseModel):
    schema_version: str = "2026-04-draft"
    manifest_id: str = Field(min_length=1)
    mode: CloudOffloadMode
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    jobs: list[CloudJobContract] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def _ensure_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
