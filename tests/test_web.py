from __future__ import annotations

from pathlib import Path

import pytest

import revprint.web as web


def test_index_renders_project_volume_profile_controls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeProjectStore:
        def __init__(self, _path: Path) -> None:
            pass

        def init_schema(self) -> None:
            return None

        def list_projects(self) -> list[object]:
            return [type("Project", (), {"id": "p1", "slug": "archive-a", "name": "Archive A"})()]

        def list_volumes(self, _project_id: str) -> list[object]:
            return [
                type(
                    "Volume",
                    (),
                    {
                        "slug": "vol-1",
                        "name": "Vol 1",
                        "folder_path": str(tmp_path / "vol1"),
                        "processing_profile": "forensic",
                    },
                )()
            ]

    class FakeJobStore:
        def __init__(self, _path: Path) -> None:
            pass

        def init_schema(self) -> None:
            return None

        def count_by_state(self) -> dict[str, int]:
            return {}

    monkeypatch.setattr(web, "_settings_paths", lambda: (tmp_path / "input", tmp_path / "jobs.sqlite", tmp_path / "projects.sqlite"))
    monkeypatch.setattr(web, "ProjectStore", FakeProjectStore)
    monkeypatch.setattr(web, "JobStore", FakeJobStore)
    monkeypatch.setattr(web, "scan_jpegs", lambda _root: [])
    app = web.create_app()
    client = app.test_client()

    resp = client.get("/?project=archive-a&volume=vol-1")

    body = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert '<select name="project">' in body
    assert '<select name="volume">' in body
    assert '<select name="profile">' in body
    assert "archive-a" in body
    assert "vol-1" in body


def test_process_rejects_unknown_project_volume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeProjectStore:
        def __init__(self, _path: Path) -> None:
            pass

        def init_schema(self) -> None:
            return None

        def get_volume(self, project_slug: str, volume_slug: str) -> object | None:
            assert project_slug == "archive-a"
            assert volume_slug == "missing-vol"
            return None

    monkeypatch.setattr(web, "_settings_paths", lambda: (tmp_path / "input", tmp_path / "jobs.sqlite", tmp_path / "projects.sqlite"))
    monkeypatch.setattr(web, "ProjectStore", FakeProjectStore)
    app = web.create_app()
    client = app.test_client()

    resp = client.post("/process", data={"project": "archive-a", "volume": "missing-vol", "profile": "balanced"})

    assert resp.status_code == 400
    assert "Unknown project/volume selection." in resp.get_data(as_text=True)


def test_process_known_project_volume_calls_run_proof_with_expected_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    class FakeProjectStore:
        def __init__(self, _path: Path) -> None:
            pass

        def init_schema(self) -> None:
            return None

        def get_volume(self, project_slug: str, volume_slug: str) -> object | None:
            assert project_slug == "archive-a"
            assert volume_slug == "vol-1"
            return type(
                "Volume",
                (),
                {"folder_path": str(tmp_path / "vol1"), "processing_profile": "forensic"},
            )()

    def fake_run_proof(
        input_root: Path,
        job_store_path: Path,
        *,
        output_root: Path,
        limit: int,
        start: int,
        profile: str,
    ) -> None:
        calls["input_root"] = input_root
        calls["job_store_path"] = job_store_path
        calls["output_root"] = output_root
        calls["limit"] = limit
        calls["start"] = start
        calls["profile"] = profile

    monkeypatch.setattr(web, "_settings_paths", lambda: (tmp_path / "input", tmp_path / "jobs.sqlite", tmp_path / "projects.sqlite"))
    monkeypatch.setattr(web, "ProjectStore", FakeProjectStore)
    monkeypatch.setattr(web, "run_proof", fake_run_proof)
    app = web.create_app()
    client = app.test_client()

    resp = client.post(
        "/process",
        data={"project": "archive-a", "volume": "vol-1", "profile": "quick", "start": "3", "limit": "9"},
    )

    assert resp.status_code == 302
    assert calls["input_root"] == (tmp_path / "vol1").resolve()
    assert calls["job_store_path"] == tmp_path / "jobs.sqlite"
    assert calls["output_root"] == Path("outputs/projects") / "archive-a" / "vol-1" / "proof"
    assert calls["limit"] == 9
    assert calls["start"] == 3
    assert calls["profile"] == "forensic"


def test_htr_editor_and_save_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run = tmp_path / "outputs" / "proof" / "run1"
    pages = run / "pages"
    pages.mkdir(parents=True)
    (pages / "scan_0001.cleaned_gray.png").write_bytes(b"img")
    monkeypatch.setenv("RPK_OUTPUT_ROOT", str((tmp_path / "outputs" / "proof").resolve()))
    app = web.create_app()
    client = app.test_client()

    resp = client.get(f"/htr?run={run}")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "HTR Editor" in body
    assert "scan_0001" in body

    resp2 = client.post(
        "/htr/save",
        data={
            "run": str(run),
            "stem": "scan_0001",
            "source_engine": "htr-sidecar",
            "language": "de",
            "script": "kurrent",
            "segments_json": '[{"text":"Anno 1742","confidence":0.81,"bbox_xywh":[1,2,3,4]}]',
        },
    )
    assert resp2.status_code == 302
    saved = pages / "scan_0001.htr.json"
    assert saved.is_file()
    text = saved.read_text(encoding="utf-8")
    assert "Anno 1742" in text
