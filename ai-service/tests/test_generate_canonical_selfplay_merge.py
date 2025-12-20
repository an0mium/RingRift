from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure app.* and scripts.* imports resolve when running pytest from repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import scripts.generate_canonical_selfplay as gcs


def test_merge_distributed_dbs_archives_existing_dest_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dest_db = tmp_path / "canonical_square8.db"
    dest_db.write_text("old_db")

    src1 = tmp_path / "host1.db"
    src2 = tmp_path / "host2.db"
    src1.write_text("a")
    src2.write_text("b")

    captured: dict[str, object] = {}

    def fake_run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        # Simulate successful merge by writing a new dest DB.
        dest_db.write_text("merged_db")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(gcs, "_run_cmd", fake_run_cmd)

    result = gcs.merge_distributed_dbs(
        source_dbs=[src1, src2],
        dest_db=dest_db,
        reset_db=True,
    )

    assert result["returncode"] == 0
    archived = result.get("archived_previous_db")
    assert isinstance(archived, str) and archived

    archived_path = Path(archived)
    assert archived_path.exists()
    assert archived_path.read_text() == "old_db"
    assert dest_db.exists()
    assert dest_db.read_text() == "merged_db"

    cmd = captured.get("cmd")
    assert isinstance(cmd, list)
    assert "scripts/merge_game_dbs.py" in cmd
    assert str(src1) in cmd and str(src2) in cmd

