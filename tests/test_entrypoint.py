from __future__ import annotations

import runpy

import pytest


def test_module_entrypoint_invokes_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    def _fake_main() -> int:
        called["n"] += 1
        return 0

    monkeypatch.setattr("revprint.cli.main", _fake_main)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("revprint.__main__", run_name="__main__")
    assert int(exc.value.code) == 0
    assert called["n"] == 1
