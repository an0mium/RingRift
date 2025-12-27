import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from app.coordination.pipeline_triggers import PipelineTrigger, TriggerConfig


@pytest.mark.asyncio
async def test_check_databases_exist_empty(monkeypatch) -> None:
    class FakeDiscovery:
        def find_databases_for_config(self, _board_type: str, _num_players: int):
            return []

    monkeypatch.setattr("app.utils.game_discovery.GameDiscovery", FakeDiscovery)

    trigger = PipelineTrigger()
    result = await trigger.check_databases_exist("square8", 2)

    assert not result.passed
    assert result.details["databases_found"] == 0


@pytest.mark.asyncio
async def test_check_databases_exist_counts_games(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "games.db"
    db_path.write_text("placeholder")

    class FakeDiscovery:
        def find_databases_for_config(self, _board_type: str, _num_players: int):
            return [SimpleNamespace(game_count=7, path=db_path)]

    monkeypatch.setattr("app.utils.game_discovery.GameDiscovery", FakeDiscovery)

    trigger = PipelineTrigger()
    result = await trigger.check_databases_exist("square8", 2)

    assert result.passed
    assert result.details["total_games"] == 7


@pytest.mark.asyncio
async def test_check_npz_exists_prefers_largest(monkeypatch, tmp_path) -> None:
    training_dir = tmp_path / "data" / "training"
    training_dir.mkdir(parents=True)

    np.savez(training_dir / "square8_2p_v1.npz", features=np.zeros((2, 3)))
    np.savez(training_dir / "square8_2p_iter2.npz", features=np.zeros((5, 3)))

    config = TriggerConfig(
        ai_service_root=tmp_path,
        min_samples_for_training=1,
        min_npz_size_bytes=0,
    )
    trigger = PipelineTrigger(config)
    result = await trigger.check_npz_exists("square8", 2)

    assert result.passed
    assert result.details["samples"] == 5


@pytest.mark.asyncio
async def test_check_model_exists_with_explicit_path(tmp_path) -> None:
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"model")

    trigger = PipelineTrigger(TriggerConfig(ai_service_root=tmp_path))
    result = await trigger.check_model_exists("square8", 2, model_path=str(model_path))

    assert result.passed
    assert result.details["model_path"] == str(model_path)


@pytest.mark.asyncio
async def test_check_no_training_running(monkeypatch) -> None:
    def fake_run_running(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["pgrep"], returncode=0, stdout="123\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run_running)

    trigger = PipelineTrigger()
    result = await trigger.check_no_training_running("square8", 2)

    assert not result.passed
    assert "123" in result.message


@pytest.mark.asyncio
async def test_check_no_training_running_when_clear(monkeypatch) -> None:
    def fake_run_clear(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["pgrep"], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run_clear)

    trigger = PipelineTrigger()
    result = await trigger.check_no_training_running("square8", 2)

    assert result.passed
