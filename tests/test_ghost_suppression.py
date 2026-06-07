from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.ghost_suppression import apply_ghost_suppression


def test_ghost_suppression_disabled_writes_before_after_identical() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "page_0001"
        gray = np.full((40, 50), 200, dtype=np.uint8)
        cleaned = root / f"{stem}.cleaned_gray.png"
        Image.fromarray(gray, mode="L").save(cleaned)
        meta = apply_ghost_suppression(
            cleaned_gray_path=cleaned,
            neighbor_paths={},
            interactions=[],
            pages_dir=root,
            stem=stem,
            enable=False,
        )
        assert meta["ghost_suppression_applied"] is False
        before = np.asarray(Image.open(meta["ghost_suppress_before_path"]).convert("L"))
        after = np.asarray(Image.open(meta["ghost_suppress_after_path"]).convert("L"))
        assert before.shape == after.shape
        assert np.max(np.abs(before.astype(int) - after.astype(int))) == 0


def test_ghost_suppression_gated_by_confidence() -> None:
    """High confidence_min should skip merging even when interaction dict looks eligible."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "page_0001"
        gray = np.full((40, 50), 210, dtype=np.uint8)
        cleaned = root / f"{stem}.cleaned_gray.png"
        Image.fromarray(gray, mode="L").save(cleaned)
        neighbor = root / "neighbor.jpg"
        Image.fromarray(gray, mode="L").convert("RGB").save(neighbor)
        mask = root / f"{stem}.interaction_previous_mirror_mask.png"
        Image.fromarray(np.zeros((20, 25), dtype=np.uint8), mode="L").save(mask)

        interactions = [
            {
                "relation": "previous",
                "neighbor_path": str(neighbor),
                "registration_applied": True,
                "registration_confidence": 0.2,
                "mask_path": str(mask),
                "analysis_shape_hw": (20, 25),
                "shift_yx": (0.0, 0.0),
            }
        ]
        meta = apply_ghost_suppression(
            cleaned_gray_path=cleaned,
            neighbor_paths={"previous": neighbor},
            interactions=interactions,
            pages_dir=root,
            stem=stem,
            enable=True,
            confidence_min=0.99,
        )
        assert meta["ghost_suppression_applied"] is False
        assert "no_eligible" in str(meta.get("ghost_suppression_reason", "")).lower() or meta.get(
            "ghost_suppression_reason"
        ) == "no_eligible_interactions"


def test_ghost_suppression_emits_plausibility_artifacts() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "page_0001"
        gray = np.full((60, 80), 240, dtype=np.uint8)
        gray[22:24, 20:50] = 40
        cleaned = root / f"{stem}.cleaned_gray.png"
        Image.fromarray(gray, mode="L").save(cleaned)
        neighbor = root / "neighbor.jpg"
        n = np.full((60, 80), 240, dtype=np.uint8)
        n[22:24, 20:50] = 40
        Image.fromarray(n, mode="L").convert("RGB").save(neighbor)
        mask = root / f"{stem}.interaction_previous_mirror_mask.png"
        m = np.zeros((30, 40), dtype=np.uint8)
        m[10:14, 10:26] = 255
        Image.fromarray(m, mode="L").save(mask)

        interactions = [
            {
                "relation": "previous",
                "neighbor_path": str(neighbor),
                "registration_applied": True,
                "registration_confidence": 0.9,
                "mask_path": str(mask),
                "analysis_shape_hw": (30, 40),
                "shift_yx": (0.0, 0.0),
            }
        ]
        meta = apply_ghost_suppression(
            cleaned_gray_path=cleaned,
            neighbor_paths={"previous": neighbor},
            interactions=interactions,
            pages_dir=root,
            stem=stem,
            enable=True,
            confidence_min=0.1,
            plausibility_min=0.55,
        )
        assert meta["plausibility_applied"] is True
        assert Path(str(meta["plausibility_map_path"])).is_file()
        assert Path(str(meta["plausibility_protect_mask_path"])).is_file()
        assert Path(str(meta["plausibility_regions_path"])).is_file()
        assert int(meta["plausibility_exhaustive_passes"]) >= 1
