from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from revprint.edge_reconstruction import (
    apply_edge_reconstruction,
    build_edge_reconstruction_candidates,
)


def test_edge_reconstruction_emits_artifacts() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "page_0001"
        gray = np.full((60, 80), 245, dtype=np.uint8)
        gray[5:10, 4:15] = 25
        edge = np.zeros((60, 80), dtype=np.uint8)
        edge[0:8, :] = 255
        cleaned = root / f"{stem}.cleaned_gray.png"
        edge_mask = root / f"{stem}.edge_inpaint_mask.png"
        Image.fromarray(gray, mode="L").save(cleaned)
        Image.fromarray(edge, mode="L").save(edge_mask)
        text_evidence = {
            "source_engine": "manual",
            "segments": [{"text": "Archiv", "confidence": 1.0, "bbox_xywh": [0, 0, 10, 10]}],
        }
        meta = build_edge_reconstruction_candidates(
            stem=stem,
            pages_dir=root,
            cleaned_gray_path=cleaned,
            edge_inpaint_mask_path=edge_mask,
            text_evidence=text_evidence,
        )
        assert Path(str(meta["edge_reconstruct_candidate_mask_path"])).is_file()
        assert Path(str(meta["edge_reconstruct_overlay_path"])).is_file()
        assert Path(str(meta["edge_reconstruct_protect_mask_path"])).is_file()
        assert Path(str(meta["edge_reconstruct_candidates_json_path"])).is_file()
        assert meta["edge_candidate_confidence"] >= 0.0
        assert meta["candidate_scoring_version"] == "v2_writer_aware"


def test_apply_edge_reconstruction_writes_before_after_and_applied() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        stem = "page_0002"
        gray = np.full((50, 70), 210, dtype=np.uint8)
        cleaned = root / f"{stem}.cleaned_gray.png"
        Image.fromarray(gray, mode="L").save(cleaned)
        cand = np.zeros((50, 70), dtype=np.uint8)
        cand[0:6, :] = 255
        cand_path = root / f"{stem}.edge_reconstruct_candidate_mask.png"
        Image.fromarray(cand, mode="L").save(cand_path)
        meta = apply_edge_reconstruction(
            stem=stem,
            pages_dir=root,
            cleaned_gray_path=cleaned,
            candidate_mask_path=cand_path,
            enable=True,
            strength=0.7,
        )
        assert meta["edge_reconstruct_applied"] is True
        assert Path(str(meta["edge_reconstruct_before_path"])).is_file()
        assert Path(str(meta["edge_reconstruct_after_path"])).is_file()
        assert Path(str(meta["edge_reconstruct_applied_path"])).is_file()
