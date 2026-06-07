from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pytest

import revprint.cli as cli
from revprint.cli import _input_root_path, build_parser
from revprint.project_store import ProjectStore


def test_build_parser_accepts_project_volume_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--project-store",
            "x.sqlite",
            "process-proof",
            "--project",
            "archive-a",
            "--volume",
            "vol-1",
            "--profile",
            "forensic",
        ]
    )
    assert args.project == "archive-a"
    assert args.volume == "vol-1"
    assert args.profile == "forensic"


def test_input_root_path_resolves_from_project_volume() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        corpus = root / "corpus"
        corpus.mkdir()
        db = root / "projects.sqlite"
        store = ProjectStore(db)
        store.init_schema()
        pid = store.upsert_project("Archive A", corpus)
        store.add_volume(pid, "Vol 1", corpus)
        args = argparse.Namespace(
            project="archive-a",
            volume="vol-1",
            project_store=db,
            input_root=None,
        )
        got = _input_root_path(args)
        assert got == corpus.resolve()


def test_project_init_calls_store_and_prints_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class FakeStore:
        def __init__(self, _path: Path) -> None:
            pass

        def init_schema(self) -> None:
            calls["init_schema"] = True

        def upsert_project(self, name: str, corpus_root: Path, notes: str = "") -> str:
            calls["name"] = name
            calls["corpus_root"] = corpus_root
            calls["notes"] = notes
            return "project-123"

    monkeypatch.setattr(cli, "ProjectStore", FakeStore)
    args = argparse.Namespace(project_store=tmp_path / "projects.sqlite", name="Archive A", corpus_root=tmp_path, notes="n")

    rc = cli._cmd_project_init(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert calls["init_schema"] is True
    assert calls["name"] == "Archive A"
    assert calls["corpus_root"] == tmp_path
    assert calls["notes"] == "n"
    assert "project_id=project-123" in out


def test_project_list_prints_projects(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    project = type("Project", (), {"slug": "archive-a", "name": "Archive A", "corpus_root": str(tmp_path / "corpus")})()

    class FakeStore:
        def __init__(self, _path: Path) -> None:
            pass

        def init_schema(self) -> None:
            return None

        def list_projects(self) -> list[object]:
            return [project]

    monkeypatch.setattr(cli, "ProjectStore", FakeStore)
    args = argparse.Namespace(project_store=tmp_path / "projects.sqlite")

    rc = cli._cmd_project_list(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert "archive-a\tArchive A\t" in out


def test_volume_add_calls_store_with_parsed_project(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    calls: dict[str, object] = {}
    project = type("Project", (), {"id": "p-1", "slug": "archive-a"})()

    class FakeStore:
        def __init__(self, _path: Path) -> None:
            pass

        def init_schema(self) -> None:
            return None

        def list_projects(self) -> list[object]:
            return [project]

        def add_volume(self, project_id: str, name: str, folder_path: Path, processing_profile: str) -> str:
            calls["project_id"] = project_id
            calls["name"] = name
            calls["folder_path"] = folder_path
            calls["processing_profile"] = processing_profile
            return "vol-123"

    monkeypatch.setattr(cli, "ProjectStore", FakeStore)
    args = argparse.Namespace(
        project_store=tmp_path / "projects.sqlite",
        project="archive-a",
        name="Vol 1",
        folder=tmp_path / "vol1",
        profile="forensic",
    )

    rc = cli._cmd_volume_add(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert calls == {
        "project_id": "p-1",
        "name": "Vol 1",
        "folder_path": tmp_path / "vol1",
        "processing_profile": "forensic",
    }
    assert "volume_id=vol-123" in out


def test_volume_list_prints_project_volumes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    project = type("Project", (), {"id": "p-1", "slug": "archive-a"})()
    volume = type(
        "Volume",
        (),
        {"slug": "vol-1", "name": "Vol 1", "folder_path": str(tmp_path / "vol1"), "processing_profile": "balanced"},
    )()

    class FakeStore:
        def __init__(self, _path: Path) -> None:
            pass

        def init_schema(self) -> None:
            return None

        def list_projects(self) -> list[object]:
            return [project]

        def list_volumes(self, _project_id: str) -> list[object]:
            return [volume]

    monkeypatch.setattr(cli, "ProjectStore", FakeStore)
    args = argparse.Namespace(project_store=tmp_path / "projects.sqlite", project="archive-a")

    rc = cli._cmd_volume_list(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert "vol-1\tVol 1\t" in out
    assert "profile=balanced" in out


def test_process_proof_with_project_volume_uses_segregated_output_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called: dict[str, object] = {}

    def fake_input_root_path(_args: argparse.Namespace) -> Path:
        return tmp_path / "selected-input"

    def fake_job_store_path(_args: argparse.Namespace) -> Path:
        return tmp_path / "jobs.sqlite"

    class FakeRun:
        run_id = "r-1"
        output_dir = str(tmp_path / "output")
        reproduction_pdf = str(tmp_path / "rep.pdf")
        translation_pdf = str(tmp_path / "tr.pdf")
        manifest_path = str(tmp_path / "manifest.json")

    def fake_run_proof(
        *,
        input_root: Path,
        job_store_path: Path,
        output_root: Path,
        limit: int,
        start: int,
        profile: str,
    ) -> FakeRun:
        called["input_root"] = input_root
        called["job_store_path"] = job_store_path
        called["output_root"] = output_root
        called["limit"] = limit
        called["start"] = start
        called["profile"] = profile
        return FakeRun()

    monkeypatch.setattr(cli, "_input_root_path", fake_input_root_path)
    monkeypatch.setattr(cli, "_job_store_path", fake_job_store_path)
    monkeypatch.setattr(cli, "run_proof", fake_run_proof)
    args = argparse.Namespace(
        project="archive-a",
        volume="vol-1",
        output_root=tmp_path / "unused",
        limit=7,
        start=2,
        profile="quick",
    )

    rc = cli._cmd_process_proof(args)

    assert rc == 0
    assert called["input_root"] == tmp_path / "selected-input"
    assert called["job_store_path"] == tmp_path / "jobs.sqlite"
    assert called["output_root"] == Path("outputs/projects") / "archive-a" / "vol-1" / "proof"
    assert called["limit"] == 7
    assert called["start"] == 2
    assert called["profile"] == "quick"


def test_htr_scaffold_creates_templates(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "scan_0001.cleaned_gray.png").write_bytes(b"placeholder")
    args = argparse.Namespace(pages_dir=pages, overwrite=False)
    rc = cli._cmd_htr_scaffold(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert (pages / "scan_0001.htr.json").is_file()
    assert "created=1" in out
