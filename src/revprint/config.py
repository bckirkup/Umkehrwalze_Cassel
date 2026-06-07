from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration. Prefer environment variables; optional `.env` in cwd."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Primary input: folder containing manuscript JPGs (e.g. HerrBenjaminKirkup)
    input_root: Path | None = Field(
        default=None,
        description="Root directory to scan for JPEG assets. Required for scan / init-jobs.",
        validation_alias="RPK_INPUT_ROOT",
    )
    project_store_path: Path = Field(
        default=Path("data/projects.sqlite"),
        description="Path to SQLite project/volume store.",
        validation_alias="RPK_PROJECT_STORE",
    )

    # SQLite database path for job queue / progress
    job_store_path: Path = Field(
        default=Path("data/jobs.sqlite"),
        description="Path to SQLite job store file.",
        validation_alias="RPK_JOB_STORE",
    )

    # Google Cloud Translation API v2 (REST) key for draft English in translation PDF
    google_translate_api_key: str | None = Field(
        default=None,
        validation_alias="RPK_GOOGLE_TRANSLATE_API_KEY",
    )
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias="RPK_GEMINI_API_KEY",
    )
    gemini_enabled: bool = Field(
        default=False,
        description="Enable Gemini image translation as fallback for difficult handwriting.",
        validation_alias="RPK_GEMINI_ENABLED",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name for image translation.",
        validation_alias="RPK_GEMINI_MODEL",
    )
    gemini_prompt: str = Field(
        default=(
            "Read this historical manuscript image and provide: "
            "(1) a faithful English translation, "
            "(2) brief contextual commentary about uncertain words."
        ),
        description="Prompt used for Gemini image translation.",
        validation_alias="RPK_GEMINI_PROMPT",
    )
    gemini_cache_enabled: bool = Field(
        default=True,
        description="Enable local caching for Gemini image translation responses.",
        validation_alias="RPK_GEMINI_CACHE",
    )
    gemini_cache_path: Path = Field(
        default=Path("data/gemini_cache.json"),
        description="Path for Gemini translation cache.",
        validation_alias="RPK_GEMINI_CACHE_PATH",
    )
    translation_cache_enabled: bool = Field(
        default=True,
        description="Enable local caching of translation API responses.",
        validation_alias="RPK_TRANSLATION_CACHE",
    )
    translation_cache_path: Path = Field(
        default=Path("data/translation_cache.json"),
        description="Path to persistent local translation cache file.",
        validation_alias="RPK_TRANSLATION_CACHE_PATH",
    )
    ocr_translation_confidence_min: float = Field(
        default=0.34,
        ge=0.0,
        le=1.0,
        description="Minimum OCR confidence required before automatic translation.",
        validation_alias="RPK_OCR_TRANSLATION_CONFIDENCE_MIN",
    )
    ocr_reconstruct_hint_enabled: bool = Field(
        default=True,
        description="Emit OCR-based reconstruction hint mask for stroke continuation guidance.",
        validation_alias="RPK_OCR_RECONSTRUCT_HINT",
    )
    ocr_reconstruct_hint_confidence_min: float = Field(
        default=0.38,
        ge=0.0,
        le=1.0,
        description="Minimum OCR word confidence to include in reconstruction hint mask.",
        validation_alias="RPK_OCR_RECONSTRUCT_HINT_CONFIDENCE_MIN",
    )
    htr_enabled: bool = Field(
        default=True,
        description="Enable optional handwritten text recognition sidecar ingestion.",
        validation_alias="RPK_HTR_ENABLED",
    )
    crease_refine_enabled: bool = Field(
        default=True,
        description="Enable broad crease/shadow suppression for fold and spine artifacts.",
        validation_alias="RPK_CREASE_REFINE",
    )
    crease_darkness_threshold: float = Field(
        default=12.0,
        ge=4.0,
        le=64.0,
        description="Darkness gap threshold for crease candidate regions.",
        validation_alias="RPK_CREASE_DARKNESS_THRESHOLD",
    )

    ghost_suppression_enabled: bool = Field(
        default=False,
        description="When true, lighten mirrored-neighbor ghost candidates gated by registration confidence.",
        validation_alias="RPK_GHOST_SUPPRESSION",
    )

    ghost_confidence_min: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Minimum registration_confidence on an interaction to use its ghost mask.",
        validation_alias="RPK_GHOST_CONFIDENCE_MIN",
    )
    ghost_plausibility_min: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum penstroke plausibility to protect a candidate region from suppression.",
        validation_alias="RPK_GHOST_PLAUSIBILITY_MIN",
    )
    ghost_plausibility_exhaustive_passes: int = Field(
        default=2,
        ge=1,
        le=64,
        description="Number of expensive multi-scale plausibility passes per page.",
        validation_alias="RPK_GHOST_PLAUSIBILITY_PASSES",
    )

    dewarp_enabled: bool = Field(
        default=False,
        description="When true, emit an optional deskewed grayscale next to the cleaned page.",
        validation_alias="RPK_DEWARP",
    )

    repro_use_dewarped_when_available: bool = Field(
        default=True,
        description="Prefer dewarped_gray.png in reproduction PDF when dewarp is enabled.",
        validation_alias="RPK_REPRO_USE_DEWARPED",
    )
    processing_profile: str = Field(
        default="balanced",
        description="Processing profile: quick, balanced, forensic, training.",
        validation_alias="RPK_PROCESSING_PROFILE",
    )
    edge_reconstruct_enabled: bool = Field(
        default=False,
        description="When true, apply conservative brightening in edge candidate regions.",
        validation_alias="RPK_EDGE_RECONSTRUCT",
    )
    edge_reconstruct_strength: float = Field(
        default=0.58,
        ge=0.0,
        le=1.0,
        description="Blend strength for edge reconstruction brightening.",
        validation_alias="RPK_EDGE_RECONSTRUCT_STRENGTH",
    )
    speckle_refine_enabled: bool = Field(
        default=True,
        description="Enable conservative removal of tiny isolated speckles.",
        validation_alias="RPK_SPECKLE_REFINE",
    )
    speckle_max_component_area: int = Field(
        default=36,
        ge=4,
        le=400,
        description="Largest dark connected-component area considered a removable speckle.",
        validation_alias="RPK_SPECKLE_MAX_COMPONENT_AREA",
    )
    speckle_border_max_component_area: int = Field(
        default=90,
        ge=8,
        le=900,
        description="Largest removable dark component area within border bands.",
        validation_alias="RPK_SPECKLE_BORDER_MAX_COMPONENT_AREA",
    )
    speckle_border_band_ratio: float = Field(
        default=0.11,
        ge=0.04,
        le=0.25,
        description="Relative border band width for stronger border cleanup.",
        validation_alias="RPK_SPECKLE_BORDER_BAND_RATIO",
    )
    line_refine_enabled: bool = Field(
        default=True,
        description="Enable suppression of long straight spine/fold artifact lines.",
        validation_alias="RPK_LINE_REFINE",
    )
    line_refine_min_length_ratio: float = Field(
        default=0.55,
        ge=0.3,
        le=0.95,
        description="Minimum relative line length to treat as structural artifact candidate.",
        validation_alias="RPK_LINE_REFINE_MIN_LENGTH_RATIO",
    )
    line_refine_border_band_ratio: float = Field(
        default=0.16,
        ge=0.08,
        le=0.35,
        description="Relative left/right border width for spine-line targeting.",
        validation_alias="RPK_LINE_REFINE_BORDER_BAND_RATIO",
    )

    @field_validator(
        "input_root",
        "job_store_path",
        "project_store_path",
        "translation_cache_path",
        "gemini_cache_path",
        mode="before",
    )
    @classmethod
    def _expand_path(cls, v: str | Path | None) -> Path | None:
        if v is None or v == "":
            return None
        return Path(v).expanduser().resolve()


def load_settings() -> Settings:
    return Settings()
