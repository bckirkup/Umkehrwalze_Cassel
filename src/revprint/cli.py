from __future__ import annotations

import argparse
import sys
from pathlib import Path

from revprint.config import load_settings
from revprint.htr_templates import scaffold_htr_sidecars
from revprint.io_scan import scan_jpegs
from revprint.job_store import JobStore
from revprint.project_store import ProjectStore
from revprint.proof import run_proof
from revprint.proof_review import write_proof_review_rubric
from revprint.review_store import ReviewStore
from revprint.web import run_gui


def _job_store_path(args: argparse.Namespace) -> Path:
    if args.job_store is not None:
        return Path(args.job_store).expanduser().resolve()
    s = load_settings()
    return s.job_store_path


def _project_store_path(args: argparse.Namespace) -> Path:
    if getattr(args, "project_store", None) is not None:
        return Path(args.project_store).expanduser().resolve()
    s = load_settings()
    return s.project_store_path


def _input_root_path(args: argparse.Namespace) -> Path:
    project = getattr(args, "project", None)
    volume = getattr(args, "volume", None)
    if project and volume:
        store = ProjectStore(_project_store_path(args))
        store.init_schema()
        rec = store.get_volume(project_slug=project, volume_slug=volume)
        if rec is None:
            print(f"Unknown project/volume: {project}/{volume}", file=sys.stderr)
            raise SystemExit(2)
        return Path(rec.folder_path).resolve()
    if args.input_root is not None:
        return Path(args.input_root).expanduser().resolve()
    s = load_settings()
    if s.input_root is None:
        print(
            "Set RPK_INPUT_ROOT or pass --input-root (folder with JPGs).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return s.input_root


def _cmd_scan(args: argparse.Namespace) -> int:
    input_root = _input_root_path(args)
    paths = scan_jpegs(input_root)
    for p in paths:
        print(p)
    print(f"count={len(paths)}", file=sys.stderr)
    return 0


def _cmd_init_jobs(args: argparse.Namespace) -> int:
    input_root = _input_root_path(args)
    job_store = _job_store_path(args)
    store = JobStore(job_store)
    store.init_schema()
    files = scan_jpegs(input_root)
    n = store.register_scan(files)
    print(f"registered {n} file(s) at {input_root} -> {job_store}", file=sys.stderr)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    job_store = _job_store_path(args)
    store = JobStore(job_store)
    store.init_schema()
    counts = store.count_by_state()
    for state, c in sorted(counts.items()):
        print(f"{state}: {c}")
    if not counts:
        print("no jobs in database")
    return 0


def _cmd_process_proof(args: argparse.Namespace) -> int:
    input_root = _input_root_path(args)
    job_store = _job_store_path(args)
    output_root = args.output_root
    project = getattr(args, "project", None)
    volume = getattr(args, "volume", None)
    if project and volume:
        output_root = Path("outputs/projects") / project / volume / "proof"
    run = run_proof(
        input_root=input_root,
        job_store_path=job_store,
        output_root=output_root,
        limit=args.limit,
        start=args.start,
        profile=args.profile,
    )
    print(f"run_id={run.run_id}")
    print(f"output_dir={run.output_dir}")
    print(f"reproduction_pdf={run.reproduction_pdf}")
    print(f"translation_pdf={run.translation_pdf}")
    print(f"manifest={run.manifest_path}")
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    run_gui(host=args.host, port=args.port)
    return 0


def _cmd_project_init(args: argparse.Namespace) -> int:
    store = ProjectStore(_project_store_path(args))
    store.init_schema()
    project_id = store.upsert_project(args.name, args.corpus_root, notes=args.notes)
    print(f"project_id={project_id}")
    return 0


def _cmd_project_list(args: argparse.Namespace) -> int:
    store = ProjectStore(_project_store_path(args))
    store.init_schema()
    projects = store.list_projects()
    for proj in projects:
        print(f"{proj.slug}\t{proj.name}\t{proj.corpus_root}")
    if not projects:
        print("No projects yet.")
    return 0


def _cmd_volume_add(args: argparse.Namespace) -> int:
    store = ProjectStore(_project_store_path(args))
    store.init_schema()
    projects = {p.slug: p.id for p in store.list_projects()}
    if args.project not in projects:
        print(f"Unknown project slug: {args.project}", file=sys.stderr)
        raise SystemExit(2)
    volume_id = store.add_volume(
        project_id=projects[args.project],
        name=args.name,
        folder_path=args.folder,
        processing_profile=args.profile,
    )
    print(f"volume_id={volume_id}")
    return 0


def _cmd_volume_list(args: argparse.Namespace) -> int:
    store = ProjectStore(_project_store_path(args))
    store.init_schema()
    projects = {p.slug: p.id for p in store.list_projects()}
    if args.project not in projects:
        print(f"Unknown project slug: {args.project}", file=sys.stderr)
        raise SystemExit(2)
    volumes = store.list_volumes(projects[args.project])
    for vol in volumes:
        print(f"{vol.slug}\t{vol.name}\t{vol.folder_path}\tprofile={vol.processing_profile}")
    if not volumes:
        print("No volumes yet.")
    return 0


def _cmd_review_add(args: argparse.Namespace) -> int:
    store = ReviewStore(Path(args.review_store).expanduser().resolve())
    store.init_schema()
    rid = store.add_decision(
        project_slug=args.project,
        volume_slug=args.volume,
        run_id=args.run_id,
        page_stem=args.page_stem,
        artifact_type=args.artifact_type,
        artifact_path=str(Path(args.artifact_path).expanduser().resolve()),
        decision=args.decision,
        notes=args.notes,
    )
    print(f"review_id={rid}")
    return 0


def _cmd_review_list(args: argparse.Namespace) -> int:
    store = ReviewStore(Path(args.review_store).expanduser().resolve())
    store.init_schema()
    rows = store.list_decisions(project_slug=args.project, volume_slug=args.volume, run_id=args.run_id)
    for row in rows:
        print(f"{row.run_id}\t{row.page_stem}\t{row.artifact_type}\t{row.decision}\t{row.artifact_path}")
    if not rows:
        print("No review decisions.")
    return 0


def _cmd_review_export(args: argparse.Namespace) -> int:
    store = ReviewStore(Path(args.review_store).expanduser().resolve())
    store.init_schema()
    out = store.export_jsonl(
        Path(args.output),
        project_slug=args.project,
        volume_slug=args.volume,
        run_id=args.run_id,
    )
    print(f"exported={out}")
    return 0


def _cmd_review_rubric(args: argparse.Namespace) -> int:
    out = write_proof_review_rubric(
        manifest_path=Path(args.manifest).expanduser().resolve(),
        output_path=(Path(args.output).expanduser().resolve() if args.output else None),
    )
    print(f"rubric={out}")
    return 0


def _cmd_htr_scaffold(args: argparse.Namespace) -> int:
    pages_dir = Path(args.pages_dir).expanduser().resolve()
    created = scaffold_htr_sidecars(pages_dir=pages_dir, overwrite=bool(args.overwrite))
    for path in created:
        print(path)
    print(f"created={len(created)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="revprint",
        description="Manuscript assets: list JPEGs and manage SQLite job store.",
    )
    p.add_argument(
        "--input-root",
        type=Path,
        help="Override RPK_INPUT_ROOT: directory containing .jpg / .jpeg files.",
    )
    p.add_argument(
        "--job-store",
        type=Path,
        help="Override RPK_JOB_STORE: path to jobs.sqlite .",
    )
    p.add_argument(
        "--project-store",
        type=Path,
        help="Override RPK_PROJECT_STORE: path to project metadata sqlite.",
    )
    p.add_argument(
        "--review-store",
        type=Path,
        default=Path("data/reviews.sqlite"),
        help="Path to review decisions sqlite (default: data/reviews.sqlite).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s_scan = sub.add_parser("scan", help="List JPEGs under input root (natural sort).")
    s_scan.set_defaults(func=_cmd_scan)

    s_init = sub.add_parser("init-jobs", help="Create DB and register all JPEGs as pending jobs.")
    s_init.set_defaults(func=_cmd_init_jobs)

    s_stat = sub.add_parser("status", help="Show job counts by state.")
    s_stat.set_defaults(func=_cmd_status)

    s_proof = sub.add_parser("process-proof", help="Process a small proof run into images and PDFs.")
    s_proof.add_argument("--limit", type=int, default=4, help="Number of pages to process (default: 4).")
    s_proof.add_argument(
        "--start",
        type=int,
        default=1,
        help="Zero-based start index in natural sort order (default: 1, skipping the cover).",
    )
    s_proof.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/proof"),
        help="Root directory for proof run outputs.",
    )
    s_proof.add_argument("--project", type=str, help="Project slug from project store.")
    s_proof.add_argument("--volume", type=str, help="Volume slug from project store.")
    s_proof.add_argument(
        "--profile",
        type=str,
        default="balanced",
        choices=["quick", "balanced", "forensic", "training"],
        help="Processing profile overrides.",
    )
    s_proof.set_defaults(func=_cmd_process_proof)

    s_project = sub.add_parser("project", help="Project metadata operations.")
    project_sub = s_project.add_subparsers(dest="project_command", required=True)
    s_project_init = project_sub.add_parser("init", help="Create or update a project.")
    s_project_init.add_argument("--name", required=True)
    s_project_init.add_argument("--corpus-root", type=Path, required=True)
    s_project_init.add_argument("--notes", default="")
    s_project_init.set_defaults(func=_cmd_project_init)
    s_project_list = project_sub.add_parser("list", help="List projects.")
    s_project_list.set_defaults(func=_cmd_project_list)

    s_volume = sub.add_parser("volume", help="Volume metadata operations.")
    volume_sub = s_volume.add_subparsers(dest="volume_command", required=True)
    s_volume_add = volume_sub.add_parser("add", help="Add a volume to a project.")
    s_volume_add.add_argument("--project", required=True, help="Project slug.")
    s_volume_add.add_argument("--name", required=True)
    s_volume_add.add_argument("--folder", type=Path, required=True)
    s_volume_add.add_argument(
        "--profile",
        default="balanced",
        choices=["quick", "balanced", "forensic", "training"],
    )
    s_volume_add.set_defaults(func=_cmd_volume_add)
    s_volume_list = volume_sub.add_parser("list", help="List project volumes.")
    s_volume_list.add_argument("--project", required=True, help="Project slug.")
    s_volume_list.set_defaults(func=_cmd_volume_list)

    s_review = sub.add_parser("review", help="Review-label store operations.")
    review_sub = s_review.add_subparsers(dest="review_command", required=True)
    s_review_add = review_sub.add_parser("add", help="Add a review decision.")
    s_review_add.add_argument("--project", required=True)
    s_review_add.add_argument("--volume", required=True)
    s_review_add.add_argument("--run-id", required=True)
    s_review_add.add_argument("--page-stem", required=True)
    s_review_add.add_argument("--artifact-type", required=True)
    s_review_add.add_argument("--artifact-path", required=True)
    s_review_add.add_argument("--decision", required=True)
    s_review_add.add_argument("--notes", default="")
    s_review_add.set_defaults(func=_cmd_review_add)
    s_review_list = review_sub.add_parser("list", help="List review decisions.")
    s_review_list.add_argument("--project", required=True)
    s_review_list.add_argument("--volume", required=True)
    s_review_list.add_argument("--run-id", default=None)
    s_review_list.set_defaults(func=_cmd_review_list)
    s_review_export = review_sub.add_parser("export", help="Export review decisions to JSONL.")
    s_review_export.add_argument("--project", required=True)
    s_review_export.add_argument("--volume", required=True)
    s_review_export.add_argument("--run-id", default=None)
    s_review_export.add_argument("--output", type=Path, required=True)
    s_review_export.set_defaults(func=_cmd_review_export)
    s_review_rubric = review_sub.add_parser("rubric", help="Create proof review rubric from manifest.")
    s_review_rubric.add_argument("--manifest", type=Path, required=True)
    s_review_rubric.add_argument("--output", type=Path, default=None)
    s_review_rubric.set_defaults(func=_cmd_review_rubric)

    s_htr = sub.add_parser("htr", help="HTR helper operations.")
    htr_sub = s_htr.add_subparsers(dest="htr_command", required=True)
    s_htr_scaffold = htr_sub.add_parser(
        "scaffold",
        help="Create {stem}.htr.json sidecar templates from pages/*.cleaned_gray.png.",
    )
    s_htr_scaffold.add_argument("--pages-dir", type=Path, required=True)
    s_htr_scaffold.add_argument("--overwrite", action="store_true")
    s_htr_scaffold.set_defaults(func=_cmd_htr_scaffold)

    s_gui = sub.add_parser("gui", help="Start the lightweight local proof web GUI.")
    s_gui.add_argument("--host", default="127.0.0.1")
    s_gui.add_argument("--port", type=int, default=5000)
    s_gui.set_defaults(func=_cmd_gui)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    fn = args.func
    return int(fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
