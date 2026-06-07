from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from revprint.page_interactions import analyze_interactions_for_source


def _make_page(path: Path, text_x: int) -> None:
    img = Image.new("RGB", (320, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 300, 400), outline=(210, 210, 210), width=2)
    draw.text((text_x, 120), "Ink", fill=(20, 20, 20))
    draw.line((text_x, 160, text_x + 120, 200), fill=(40, 40, 40), width=2)
    img.save(path)


def test_analyze_interactions_writes_masks_and_overlays() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        p1 = root / "page_0001.jpg"
        p2 = root / "page_0002.jpg"
        p3 = root / "page_0003.jpg"
        _make_page(p1, 60)
        _make_page(p2, 90)
        _make_page(p3, 120)

        artifacts = analyze_interactions_for_source(p2, [p1, p2, p3], root / "out")

        assert len(artifacts) == 2
        for artifact in artifacts:
            assert Path(artifact.mask_path).exists()
            assert Path(artifact.overlay_path).exists()
            assert artifact.relation in {"previous", "next"}
            meta = artifact.to_meta()
            assert "registration_confidence" in meta
            assert "registration_applied" in meta
            assert "registration_reason" in meta
            assert "body_mask_coverage" in meta
            assert "analysis_scale_y" in meta
            assert "analysis_shape_hw" in meta
            assert isinstance(meta["shift_yx"], tuple)
