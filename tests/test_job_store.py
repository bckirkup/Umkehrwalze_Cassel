from __future__ import annotations

import tempfile
from pathlib import Path

from revprint.job_store import JobState, JobStore


def test_job_store_register_and_status() -> None:
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "t.sqlite"
        store = JobStore(db)
        store.init_schema()
        files = [Path(d) / "a.jpg", Path(d) / "b.jpeg"]
        for f in files:
            f.write_bytes(b"\xff\xd8\xff")
        n = store.register_scan(files)
        assert n == 2
        assert store.count_by_state()["pending"] == 2
        jobs = store.list_all()
        store.update_state(jobs[0].id, JobState.DONE, meta={"tokens": 100}, cost_units=0.01)
        done = store.list_by_state(JobState.DONE)
        assert len(done) == 1
        assert done[0].meta.get("tokens") == 100
