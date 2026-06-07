from __future__ import annotations

import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from flask import Flask, redirect, request, send_file

from revprint.config import load_settings
from revprint.io_scan import scan_jpegs
from revprint.job_store import JobStore
from revprint.project_store import ProjectStore
from revprint.proof import run_proof


def _settings_paths() -> tuple[Path | None, Path, Path]:
    s = load_settings()
    return s.input_root, s.job_store_path, s.project_store_path


def _latest_run(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    runs = [p for p in output_root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.name)[-1]


def _output_roots() -> list[Path]:
    env_root = Path(os.environ.get("RPK_OUTPUT_ROOT", "outputs/proof")).resolve()
    roots = [env_root, Path("outputs/proof").resolve(), Path("outputs/projects").resolve()]
    out: list[Path] = []
    for p in roots:
        if p not in out:
            out.append(p)
    return out


def _path_allowed(path: Path) -> bool:
    for root in _output_roots():
        if path == root or root in path.parents:
            return True
    return False


def _safe_resolve_run(run_str: str) -> Path | None:
    if not run_str:
        return None
    run = Path(run_str).resolve()
    if not run.is_dir() or not _path_allowed(run):
        return None
    return run


def _link(path: Path, label: str | None = None) -> str:
    label = label or path.name
    return f'<a href="/file?path={quote(str(path))}">{html.escape(label)}</a>'


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        input_root, job_store_path, project_store_path = _settings_paths()
        selected_project = request.args.get("project", "").strip().lower()
        selected_volume = request.args.get("volume", "").strip().lower()
        selected_profile = request.args.get("profile", "").strip().lower() or "balanced"
        project_store = ProjectStore(project_store_path)
        project_store.init_schema()
        projects = project_store.list_projects()
        project_options = ['<option value="">(use RPK_INPUT_ROOT)</option>']
        volume_options = ['<option value="">(none)</option>']
        volume_root: Path | None = None
        for proj in projects:
            sel = " selected" if proj.slug == selected_project else ""
            project_options.append(
                f'<option value="{html.escape(proj.slug)}"{sel}>{html.escape(proj.name)} ({html.escape(proj.slug)})</option>'
            )
        if selected_project:
            proj_map = {p.slug: p.id for p in projects}
            pid = proj_map.get(selected_project)
            if pid:
                volumes = project_store.list_volumes(pid)
                for vol in volumes:
                    sel = " selected" if vol.slug == selected_volume else ""
                    volume_options.append(
                        f'<option value="{html.escape(vol.slug)}"{sel}>{html.escape(vol.name)} ({html.escape(vol.slug)})</option>'
                    )
                    if vol.slug == selected_volume:
                        volume_root = Path(vol.folder_path).resolve()
                        selected_profile = vol.processing_profile
        effective_root = volume_root if volume_root is not None else input_root
        output_root = Path(os.environ.get("RPK_OUTPUT_ROOT", "outputs/proof")).resolve()
        count = 0
        scan_error = ""
        if effective_root is not None:
            try:
                count = len(scan_jpegs(effective_root))
            except Exception as exc:
                scan_error = str(exc)

        store = JobStore(job_store_path)
        store.init_schema()
        counts = store.count_by_state()
        latest = _latest_run(output_root)

        latest_html = "<p>No proof runs yet.</p>"
        if latest is not None:
            pdf_dir = latest / "pdf"
            pages_dir = latest / "pages"
            interactions_dir = latest / "interactions"
            image_links = []
            interaction_links = []
            if pages_dir.exists():
                for image_path in sorted(pages_dir.glob("*.cleaned_gray.png")):
                    image_links.append(f"<li>{_link(image_path)}</li>")
            if interactions_dir.exists():
                for image_path in sorted(interactions_dir.glob("*.png")):
                    interaction_links.append(f"<li>{_link(image_path)}</li>")
            ghost_links: list[str] = []
            if pages_dir.exists():
                for image_path in sorted(pages_dir.glob("*.ghost_suppress_*.png")):
                    ghost_links.append(f"<li>{_link(image_path)}</li>")
            dewarp_links: list[str] = []
            if pages_dir.exists():
                for image_path in sorted(pages_dir.glob("*.dewarped_gray.png")):
                    dewarp_links.append(f"<li>{_link(image_path)}</li>")

            diag_html = ""
            manifest_path = latest / "manifest.json"
            if manifest_path.is_file():
                try:
                    man = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    man = {}
                rows: list[str] = []
                for rec in man.get("processed_pages", []) or []:
                    src = str(rec.get("source_path", ""))
                    stem = html.escape(Path(src).name or src)
                    inters = rec.get("interactions") or []
                    reg_cells: list[str] = []
                    for it in inters:
                        rel = html.escape(str(it.get("relation", "")))
                        conf = it.get("registration_confidence")
                        appl = it.get("registration_applied")
                        reason = html.escape(str(it.get("registration_reason", "")))
                        try:
                            conf_s = f"{float(conf):.3f}" if conf is not None else "—"
                        except (TypeError, ValueError):
                            conf_s = "—"
                        reg_cells.append(f"{rel}: conf={conf_s} applied={appl}<br><small>{reason}</small>")
                    reg_html = "<br>".join(reg_cells) if reg_cells else "—"
                    g_applied = rec.get("ghost_suppression_applied")
                    g_reason = html.escape(str(rec.get("ghost_suppression_reason", "")))
                    g_links: list[str] = []
                    for key in ("ghost_suppress_before_path", "ghost_suppress_after_path"):
                        p = rec.get(key)
                        if isinstance(p, str) and Path(p).is_file():
                            g_links.append(_link(Path(p), Path(p).name))
                    ghost_html = f"applied={g_applied}<br>{g_reason}"
                    if g_links:
                        ghost_html += "<br>" + " · ".join(g_links)
                    p_mean = rec.get("plausibility_mean")
                    p_max = rec.get("plausibility_max")
                    try:
                        p_s = f"plausibility mean={float(p_mean):.3f} max={float(p_max):.3f}"
                    except (TypeError, ValueError):
                        p_s = ""
                    if p_s:
                        ghost_html += "<br><small>" + html.escape(p_s) + "</small>"
                    for key in ("plausibility_map_path", "plausibility_protect_mask_path", "plausibility_regions_path"):
                        p = rec.get(key)
                        if isinstance(p, str) and Path(p).is_file():
                            ghost_html += "<br>" + _link(Path(p), Path(p).name)
                    dew_reason = html.escape(str(rec.get("dewarp_reason", "")))
                    skew = rec.get("dewarp_skew_degrees")
                    try:
                        skew_s = f"{float(skew):.2f}°" if skew is not None else ""
                    except (TypeError, ValueError):
                        skew_s = ""
                    dew_html = dew_reason + (f"<br>skew {skew_s}" if skew_s else "")
                    dp = rec.get("dewarped_grayscale_path")
                    if isinstance(dp, str) and Path(dp).is_file():
                        dew_html += "<br>" + _link(Path(dp), Path(dp).name)
                    tr_src = html.escape(str(rec.get("translation_source_type", "")))
                    rows.append(
                        f"<tr><td>{stem}</td><td>{reg_html}</td><td>{ghost_html}</td>"
                        f"<td>{dew_html}</td><td>{tr_src}</td></tr>"
                    )
                if rows:
                    diag_html = f"""
                    <h3>Manifest diagnostics</h3>
                    <table>
                      <tr><th>Page</th><th>Registration</th><th>Ghost suppression</th><th>Dewarp</th><th>Translation source</th></tr>
                      {''.join(rows)}
                    </table>
                    """

            latest_html = f"""
            <h2>Latest Run: {html.escape(latest.name)}</h2>
            <p>{_link(latest / "manifest.json", "manifest.json")}</p>
            <p>{_link(pdf_dir / "reproduction_proof.pdf", "reproduction proof PDF")}</p>
            <p>{_link(pdf_dir / "translation_proof.pdf", "translation proof PDF")}</p>
            """
            pilot_print = latest / "pilot" / "PRINTING.md"
            if pilot_print.is_file():
                latest_html += f"<p>{_link(pilot_print, 'pilot print bundle (PRINTING.md)')}</p>"
            latest_html += f"""
            <p><a href="/htr?run={quote(str(latest))}">Open HTR Editor for this run</a></p>
            <h3>Cleaned Page Images</h3>
            <ul>{''.join(image_links)}</ul>
            <h3>Ghost suppression review</h3>
            <ul>{''.join(ghost_links) if ghost_links else '<li><em>None yet</em></li>'}</ul>
            <h3>Dewarped previews</h3>
            <ul>{''.join(dewarp_links) if dewarp_links else '<li><em>None yet</em></li>'}</ul>
            <h3>Interaction Analysis</h3>
            <p><small>Red overlays are mirrored neighbor-page candidates, useful for reviewing offset/ghost ink.</small></p>
            <ul>{''.join(interaction_links)}</ul>
            {diag_html}
            """

        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(counts.items()))
        if not rows:
            rows = '<tr><td colspan="2">No jobs yet</td></tr>'

        return f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>RevPrint Proof GUI</title>
          <style>
            body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 920px; }}
            code {{ background: #f3f3f3; padding: 0.1rem 0.25rem; }}
            table {{ border-collapse: collapse; margin: 1rem 0; }}
            td, th {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; }}
            button {{ padding: 0.55rem 0.8rem; }}
            .error {{ color: #9b111e; }}
          </style>
        </head>
        <body>
          <h1>RevPrint Proof GUI</h1>
          <p><strong>Input root:</strong> <code>{html.escape(str(effective_root))}</code></p>
          <p><strong>JPG count:</strong> {count}</p>
          <p><strong>Job store:</strong> <code>{html.escape(str(job_store_path))}</code></p>
          {f'<p class="error">{html.escape(scan_error)}</p>' if scan_error else ''}

          <form method="post" action="/process">
            <label>Project
              <select name="project">{''.join(project_options)}</select>
            </label>
            <label>Volume
              <select name="volume">{''.join(volume_options)}</select>
            </label>
            <label>Profile
              <select name="profile">
                <option value="quick" {"selected" if selected_profile == "quick" else ""}>quick</option>
                <option value="balanced" {"selected" if selected_profile == "balanced" else ""}>balanced</option>
                <option value="forensic" {"selected" if selected_profile == "forensic" else ""}>forensic</option>
                <option value="training" {"selected" if selected_profile == "training" else ""}>training</option>
              </select>
            </label>
            <label>Start index <input name="start" value="1" size="4"></label>
            <label>Limit <input name="limit" value="4" size="4"></label>
            <p><small>Default skips the unique cover/title page and processes four interior pages.</small></p>
            <button type="submit">Process proof run</button>
          </form>

          <h2>Job Status</h2>
          <table><tr><th>State</th><th>Count</th></tr>{rows}</table>

          {latest_html}
        </body>
        </html>
        """

    @app.post("/process")
    def process() -> str:
        input_root, job_store_path, project_store_path = _settings_paths()
        project_slug = request.form.get("project", "").strip().lower()
        volume_slug = request.form.get("volume", "").strip().lower()
        profile = request.form.get("profile", "balanced")
        effective_root = input_root
        output_root = Path(os.environ.get("RPK_OUTPUT_ROOT", "outputs/proof")).resolve()
        if project_slug and volume_slug:
            pstore = ProjectStore(project_store_path)
            pstore.init_schema()
            vol = pstore.get_volume(project_slug=project_slug, volume_slug=volume_slug)
            if vol is None:
                return "Unknown project/volume selection.", 400
            effective_root = Path(vol.folder_path).resolve()
            output_root = Path("outputs/projects") / project_slug / volume_slug / "proof"
            profile = vol.processing_profile or profile
        if effective_root is None:
            return "Set RPK_INPUT_ROOT before processing.", 400
        limit = int(request.form.get("limit", "4"))
        start = int(request.form.get("start", "1"))
        run_proof(
            effective_root,
            job_store_path,
            output_root=output_root,
            limit=limit,
            start=start,
            profile=profile,
        )
        return redirect(f"/?project={quote(project_slug)}&volume={quote(volume_slug)}&profile={quote(profile)}")

    @app.get("/htr")
    def htr_editor() -> str:
        run_param = request.args.get("run", "")
        run = _safe_resolve_run(run_param)
        if run is None:
            latest = _latest_run(Path(os.environ.get("RPK_OUTPUT_ROOT", "outputs/proof")).resolve())
            run = latest
        if run is None:
            return "No proof run available yet.", 404
        pages_dir = run / "pages"
        cleaned = sorted(pages_dir.glob("*.cleaned_gray.png"))
        stems = [p.name.replace(".cleaned_gray.png", "") for p in cleaned]
        if not stems:
            return "No cleaned pages found in run.", 404
        stem = request.args.get("stem", "").strip() or stems[0]
        if stem not in stems:
            stem = stems[0]
        cleaned_path = pages_dir / f"{stem}.cleaned_gray.png"
        htr_path = pages_dir / f"{stem}.htr.json"
        payload: dict[str, object] = {}
        if htr_path.is_file():
            try:
                payload = json.loads(htr_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        source_engine = str(payload.get("source_engine", "htr-sidecar"))
        language = str(payload.get("language", "de"))
        script = str(payload.get("script", "kurrent"))
        segments = payload.get("segments", [])
        if not isinstance(segments, list):
            segments = []
        seed_candidates = payload.get("seed_candidates", [])
        if not isinstance(seed_candidates, list):
            seed_candidates = []
        stem_links = " ".join(
            [
                f'<a href="/htr?run={quote(str(run))}&stem={quote(s)}">{html.escape(s)}</a>'
                if s != stem
                else f"<strong>{html.escape(s)}</strong>"
                for s in stems
            ]
        )
        return f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>HTR Editor</title>
          <style>
            body {{ font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 1100px; }}
            .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: start; }}
            textarea {{ width: 100%; min-height: 430px; font-family: ui-monospace, monospace; }}
            code {{ background: #f3f3f3; padding: 0.1rem 0.25rem; }}
            .small {{ font-size: 0.9rem; color: #444; }}
            .seeds {{ max-height: 220px; overflow: auto; border: 1px solid #ddd; padding: 0.5rem; }}
          </style>
        </head>
        <body>
          <h1>HTR Editor</h1>
          <p><a href="/">Back to GUI</a></p>
          <p><strong>Run:</strong> <code>{html.escape(str(run))}</code></p>
          <p><strong>Page:</strong> {stem_links}</p>
          <div class="cols">
            <div>
              <h3>Page preview</h3>
              <p>{_link(cleaned_path, cleaned_path.name)}</p>
              <img src="/file?path={quote(str(cleaned_path))}" style="max-width:100%;border:1px solid #ddd" />
              <p class="small">Use bbox as <code>[x,y,w,h]</code> in segments.</p>
              <h3>Seed candidates</h3>
              <div class="seeds"><pre>{html.escape(json.dumps(seed_candidates, ensure_ascii=False, indent=2))}</pre></div>
            </div>
            <div>
              <form method="post" action="/htr/save">
                <input type="hidden" name="run" value="{html.escape(str(run))}">
                <input type="hidden" name="stem" value="{html.escape(stem)}">
                <label>source_engine <input name="source_engine" value="{html.escape(source_engine)}"></label><br>
                <label>language <input name="language" value="{html.escape(language)}"></label><br>
                <label>script <input name="script" value="{html.escape(script)}"></label>
                <p><strong>segments JSON (array):</strong></p>
                <textarea name="segments_json">{html.escape(json.dumps(segments, ensure_ascii=False, indent=2))}</textarea>
                <p><button type="submit">Save HTR sidecar</button></p>
              </form>
              <p class="small">Each segment shape: <code>{{"text":"...", "confidence":0.82, "bbox_xywh":[x,y,w,h], "language":"de", "script":"kurrent"}}</code></p>
            </div>
          </div>
        </body>
        </html>
        """

    @app.post("/htr/save")
    def htr_save() -> object:
        run = _safe_resolve_run(request.form.get("run", ""))
        if run is None:
            return "Invalid run path.", 400
        stem = request.form.get("stem", "").strip()
        if not stem:
            return "Missing stem.", 400
        pages_dir = run / "pages"
        cleaned_path = pages_dir / f"{stem}.cleaned_gray.png"
        if not cleaned_path.is_file():
            return "Unknown page stem for this run.", 400
        try:
            segments = json.loads(request.form.get("segments_json", "[]"))
        except Exception as exc:
            return f"Invalid JSON: {exc}", 400
        if not isinstance(segments, list):
            return "segments_json must be a JSON array.", 400
        out: list[dict[str, object]] = []
        for item in segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            bbox = item.get("bbox_xywh", [0, 0, 0, 0])
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                bbox = [0, 0, 0, 0]
            try:
                confidence = float(item["confidence"]) if item.get("confidence") is not None else None
            except Exception:
                confidence = None
            out.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "bbox_xywh": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                    "language": str(item.get("language", request.form.get("language", "de"))),
                    "script": str(item.get("script", request.form.get("script", "kurrent"))),
                }
            )
        htr_path = pages_dir / f"{stem}.htr.json"
        existing: dict[str, object] = {}
        if htr_path.is_file():
            try:
                loaded = json.loads(htr_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}
        payload = {
            **existing,
            "source_engine": request.form.get("source_engine", "htr-sidecar").strip() or "htr-sidecar",
            "language": request.form.get("language", "de").strip() or "de",
            "script": request.form.get("script", "kurrent").strip() or "kurrent",
            "segments": out,
        }
        htr_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return redirect(f"/htr?run={quote(str(run))}&stem={quote(stem)}")

    @app.get("/file")
    def file() -> object:
        path = Path(request.args["path"]).resolve()
        if not _path_allowed(path):
            return "Not allowed", 403
        if not path.exists() or not path.is_file():
            return "Not found", 404
        return send_file(path)

    return app


def run_gui(host: str = "127.0.0.1", port: int = 5000) -> None:
    create_app().run(host=host, port=port, debug=False)
