"""Tests for the baseline Square-8 2-player NN training script."""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import scripts.run_nn_training_baseline as nn_baseline

# Keep demo runs fast to avoid heavy work in CI.
TEST_TIMEOUT_SECONDS = 60


@pytest.mark.timeout(TEST_TIMEOUT_SECONDS)
def test_run_nn_training_baseline_demo_writes_report(tmp_path) -> None:
    """Smoke test that the baseline NN script writes a demo report and
    artefacts."""
    run_dir = tmp_path / "nn_demo"
    argv = [
        "--board",
        "square8",
        "--num-players",
        "2",
        "--run-dir",
        str(run_dir),
        "--demo",
        "--seed",
        "123",
    ]

    rc = nn_baseline.main(argv)
    assert rc == 0

    report_path = run_dir / "nn_training_report.json"
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["board"] == "square8"
    assert report["num_players"] == 2
    assert report["mode"] == "demo"
    assert isinstance(report.get("model_id"), str)

    # Check training_params structure (new format)
    training_params = report["training_params"]
    assert training_params["board_type"] == "SQUARE8"
    assert training_params["epochs_per_iter"] == 1
    assert training_params["seed"] == 123

    # Demo run should specify a data_path (dummy file that triggers synthetic data)
    assert "data_path" in report
    assert isinstance(report["data_path"], str)

    # Metrics should record epochs run
    metrics = report["metrics"]
    assert metrics["train_epochs_run"] == 1


