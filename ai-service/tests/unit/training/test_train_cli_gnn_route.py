"""Smoke tests for GNN routing in train_cli."""

from __future__ import annotations

import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

from app.training import train_cli


def test_train_cli_routes_gnn(monkeypatch, tmp_path):
    """Ensure --model-type gnn routes to train_gnn_policy."""
    dummy_train = types.ModuleType("app.training.train")
    dummy_train.run_cmaes_heuristic_optimization = Mock()
    dummy_train.train_model = Mock()
    monkeypatch.setitem(sys.modules, "app.training.train", dummy_train)

    calls = []

    def fake_run(cmd, check=False, **kwargs):
        calls.append({"cmd": cmd, "check": check, "kwargs": kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    args = SimpleNamespace(
        enable_pipeline_auto_trigger=False,
        config=None,
        curriculum=False,
        data_path=str(tmp_path / "data.npz"),
        epochs=3,
        batch_size=4,
        learning_rate=0.01,
        seed=123,
        policy_label_smoothing=0.0,
        weight_decay=None,
        label_smoothing=0.0,
        feature_version=None,
        filter_empty_policies=False,
        board_type="hex8",
        checkpoint_dir=None,
        save_path=str(tmp_path / "model.pth"),
        model_version=None,
        model_type="gnn",
        num_players=2,
    )

    monkeypatch.setattr(train_cli, "parse_args", lambda: args)

    train_cli.main()

    assert calls, "Expected train_cli to invoke subprocess.run for GNN training."
    cmd = calls[0]["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "app.training.train_gnn_policy"]
    assert "--board-type" in cmd
    assert "hex8" in cmd
    assert "--data-path" in cmd
    assert str(tmp_path / "data.npz") in cmd
    assert "--output-dir" in cmd
    assert str(tmp_path) in cmd

    dummy_train.train_model.assert_not_called()
    dummy_train.run_cmaes_heuristic_optimization.assert_not_called()
