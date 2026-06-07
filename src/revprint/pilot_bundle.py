from __future__ import annotations

from pathlib import Path


def write_pilot_print_bundle(
    *,
    run_dir: Path,
    page_records: list[dict[str, object]],
    reproduction_pdf: Path,
    translation_pdf: Path,
    manifest_path: Path,
    quality_summary: dict[str, object] | None = None,
) -> Path:
    """
    Write a small operator-facing bundle for the first-pages / pilot workflow:
    printable PDF paths, per-page image paths, and a short print checklist.
    """
    run_dir = Path(run_dir).resolve()
    pilot = run_dir / "pilot"
    pilot.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Pilot print bundle",
        "",
        "Use this folder after a proof run when you are validating the first pages only.",
        "Defer full-corpus sampling until this pilot checklist passes.",
        "",
        "## Outputs (send to print or archive)",
        "",
        f"- Reproduction PDF: `{reproduction_pdf}`",
        f"- Translation PDF: `{translation_pdf}`",
        f"- Manifest: `{manifest_path}`",
        "",
        "## Per-page images (reproduction source)",
        "",
    ]
    for rec in page_records:
        src = str(rec.get("source_path", ""))
        stem = Path(src).stem if src else ""
        cleaned = rec.get("cleaned_grayscale_path")
        dewarped = rec.get("dewarped_grayscale_path")
        lines.append(f"- **{stem}**")
        lines.append(f"  - source: `{src}`")
        if cleaned:
            lines.append(f"  - cleaned_gray: `{cleaned}`")
        if dewarped and str(dewarped).strip():
            lines.append(f"  - dewarped_gray: `{dewarped}`")
        tr_src = str(rec.get("translation_source_type", ""))
        lines.append(f"  - translation_source_type: `{tr_src}`")
        lines.append("")
    lines.extend(
        [
            "## Quality summary",
            "",
        ]
    )
    q = quality_summary or {}
    lines.extend(
        [
            f"- edge_apply_rate: `{q.get('edge_apply_rate', 0.0)}`",
            f"- edge_avg_confidence: `{q.get('edge_avg_confidence', 0.0)}`",
            f"- edge_avg_protected_ratio: `{q.get('edge_avg_protected_ratio', 0.0)}`",
            f"- edge_unresolved_pages: `{q.get('edge_unresolved_pages', 0)}`",
            "",
            "## Pilot checklist (before scaling)",
            "",
            "- [ ] Reproduction pages look correct on screen at 100% zoom.",
            "- [ ] No unintended clipping of text after border/crease passes.",
            "- [ ] Edge apply rate and unresolved pages are acceptable for this run.",
            "- [ ] Underlines/double-underlines remain intact where present.",
            "- [ ] Translation PDF matches expected source (seed / manual / API).",
            "- [ ] German sidecars align with the page images where present.",
            "- [ ] One physical test print (optional): letter-size from reproduction PDF.",
            "",
        ]
    )
    out = pilot / "PRINTING.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
