from __future__ import annotations

import json
from pathlib import Path


def build_proof_review_rubric(markdown_title: str, manifest_path: Path) -> str:
    man_path = Path(manifest_path).resolve()
    payload = json.loads(man_path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id", "unknown"))
    pages = payload.get("processed_pages") or []
    page_count = len(pages) if isinstance(pages, list) else 0
    return (
        f"# {markdown_title}\n\n"
        f"- Run ID: `{run_id}`\n"
        f"- Manifest: `{man_path}`\n"
        f"- Page count: `{page_count}`\n\n"
        "## Reviewer Checklist\n\n"
        "- [ ] Reproduction PDF is legible and does not erase true penstrokes.\n"
        "- [ ] Translation PDF exists and source type metadata looks correct (`ocr`/`manual`/`copied_en`).\n"
        "- [ ] Ghost suppression has before/after artifacts and no obvious over-cleaning.\n"
        "- [ ] Plausibility map/protect mask are present when ghost stage runs.\n"
        "- [ ] Edge reconstruction candidates are plausible and localized near borders.\n"
        "- [ ] Registration confidence/reasons in manifest align with observed overlays.\n"
        "- [ ] Cloud manifest exists and lists expected page jobs.\n"
        "- [ ] Overall output quality is improved over inputs with acceptable residual artifacts.\n\n"
        "## Decision\n\n"
        "- [ ] Approve next iteration\n"
        "- [ ] Needs parameter tuning\n"
        "- [ ] Needs algorithm change\n\n"
        "## Notes\n\n"
        "- Reviewer:\n"
        "- Date:\n"
        "- Key pages/issues:\n"
    )


def write_proof_review_rubric(manifest_path: Path, output_path: Path | None = None) -> Path:
    man = Path(manifest_path).resolve()
    out = Path(output_path).resolve() if output_path is not None else man.parent / "proof_review_rubric.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build_proof_review_rubric("Proof Review Rubric", man)
    out.write_text(text, encoding="utf-8")
    return out
