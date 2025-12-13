from __future__ import annotations

from pathlib import Path

import pytest
import torch

from app.ai import neural_net as neural_net_mod
from app.ai.neural_net import NeuralNetAI, RingRiftCNN_v2, RingRiftCNN_v3
from app.models import AIConfig, BoardType
from app.training.model_versioning import ModelVersionManager


def _write_versioned_checkpoint(
    models_dir: Path,
    filename: str,
    *,
    board_size: int,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    manager = ModelVersionManager()

    model = RingRiftCNN_v2(
        board_size=board_size,
        in_channels=14,
        global_features=20,
        num_res_blocks=2,
        num_filters=24,
        history_length=3,
        policy_size=123,
    )
    metadata = manager.create_metadata(model)
    manager.save_checkpoint(model, metadata, str(models_dir / filename))


def _write_v3_versioned_checkpoint(
    models_dir: Path,
    filename: str,
    *,
    board_size: int,
    num_players: int,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    manager = ModelVersionManager()

    model = RingRiftCNN_v3(
        board_size=board_size,
        in_channels=14,
        global_features=20,
        num_res_blocks=2,
        num_filters=24,
        history_length=3,
        num_players=num_players,
    )
    metadata = manager.create_metadata(model)
    manager.save_checkpoint(model, metadata, str(models_dir / filename))


def test_neural_netai_defaults_to_v5_sq8_model_id_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When nn_model_id is unset, square8 should prefer v5/v3-family checkpoints."""
    monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")

    base_dir = tmp_path / "ai-service"
    models_dir = base_dir / "models"

    _write_v3_versioned_checkpoint(
        models_dir,
        "ringrift_v5_sq8_2p_2xh100.pth",
        board_size=8,
        num_players=2,
    )

    neural_net_mod._MODEL_CACHE.clear()

    cfg = AIConfig(
        difficulty=6,
        rng_seed=1,
        use_neural_net=True,
        allow_fresh_weights=False,
    )
    nn = NeuralNetAI(player_number=1, config=cfg)
    nn._base_dir = str(base_dir)
    nn._ensure_model_initialized(BoardType.SQUARE8)

    assert nn.model is not None
    model = getattr(nn.model, "_orig_mod", nn.model)
    assert isinstance(model, RingRiftCNN_v3)


def test_neural_netai_does_not_append_square19_suffix_when_model_id_includes_board_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Square19 defaults should not double-suffix ids like *_sq19_2p_19x19."""
    monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")

    base_dir = tmp_path / "ai-service"
    models_dir = base_dir / "models"

    _write_versioned_checkpoint(
        models_dir,
        "ringrift_v4_sq19_2p.pth",
        board_size=19,
    )

    neural_net_mod._MODEL_CACHE.clear()

    cfg = AIConfig(
        difficulty=6,
        rng_seed=1,
        use_neural_net=True,
        allow_fresh_weights=False,
    )
    nn = NeuralNetAI(player_number=1, config=cfg)
    nn._base_dir = str(base_dir)
    nn._ensure_model_initialized(BoardType.SQUARE19)

    assert nn.model is not None
    model = getattr(nn.model, "_orig_mod", nn.model)
    assert isinstance(model, RingRiftCNN_v2)


def test_neural_netai_infers_architecture_from_bare_state_dict_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Legacy checkpoints without metadata should still load via shape inference."""
    monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")

    base_dir = tmp_path / "ai-service"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Use non-default shapes so the test fails unless NeuralNetAI infers the
    # correct num_filters/policy_size from the state_dict.
    legacy_model = RingRiftCNN_v2(
        board_size=8,
        in_channels=14,
        global_features=20,
        num_res_blocks=2,
        num_filters=24,
        history_length=3,
        policy_size=123,
    )
    torch.save(legacy_model.state_dict(), models_dir / "legacy_bare_state_dict.pth")

    neural_net_mod._MODEL_CACHE.clear()

    cfg = AIConfig(
        difficulty=6,
        rng_seed=1,
        use_neural_net=True,
        allow_fresh_weights=False,
        nn_model_id="legacy_bare_state_dict",
    )
    nn = NeuralNetAI(player_number=1, config=cfg)
    nn._base_dir = str(base_dir)
    nn._ensure_model_initialized(BoardType.SQUARE8)

    assert nn.model is not None
    model = getattr(nn.model, "_orig_mod", nn.model)
    assert isinstance(model, RingRiftCNN_v2)
    assert model.num_filters == 24
    assert model.policy_size == 123


def test_neural_netai_repairs_value_fc1_shape_mismatch_by_rebuilding_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """NeuralNetAI should self-heal when instantiated with stale hyperparams.

    This regression test covers the failure mode observed in tournaments:
    a checkpoint trained with 192 filters (value_fc1 in_features=212) loaded
    into a runtime model built with 128 filters (expected in_features=148),
    which caused neural tiers to silently fall back to heuristic rollouts.
    """
    monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")

    base_dir = tmp_path / "ai-service"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save a small versioned checkpoint with non-default shapes so the test is fast
    # but still exercises the repair logic.
    manager = ModelVersionManager()
    checkpoint_filters = 24
    checkpoint_policy_size = 123
    checkpoint_model = RingRiftCNN_v2(
        board_size=8,
        in_channels=14,
        global_features=20,
        num_res_blocks=2,
        num_filters=checkpoint_filters,
        history_length=3,
        policy_size=checkpoint_policy_size,
    )
    metadata = manager.create_metadata(checkpoint_model)
    ckpt_path = models_dir / "repair_test.pth"
    manager.save_checkpoint(checkpoint_model, metadata, str(ckpt_path))

    neural_net_mod._MODEL_CACHE.clear()

    cfg = AIConfig(
        difficulty=6,
        rng_seed=1,
        use_neural_net=True,
        allow_fresh_weights=False,
        nn_model_id="repair_test",
    )
    nn = NeuralNetAI(player_number=1, config=cfg)
    nn._base_dir = str(base_dir)

    # Force an incorrect model shape (simulates stale defaults).
    nn.model = RingRiftCNN_v2(
        board_size=8,
        in_channels=14,
        global_features=20,
        num_res_blocks=2,
        num_filters=16,  # wrong on purpose
        history_length=3,
        policy_size=checkpoint_policy_size,
    )
    nn.model.to(nn.device)

    nn._load_model_checkpoint(str(ckpt_path))

    assert nn.model is not None
    assert nn.model.num_filters == checkpoint_filters
    assert nn.model.value_fc1.in_features == checkpoint_filters + 20


def test_neural_netai_infers_num_players_for_v3_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """V3 checkpoints may be trained with num_players=2/3/4; infer from weights."""
    monkeypatch.setenv("RINGRIFT_FORCE_CPU", "1")

    base_dir = tmp_path / "ai-service"
    models_dir = base_dir / "models"

    _write_v3_versioned_checkpoint(
        models_dir,
        "v3_num_players_2.pth",
        board_size=8,
        num_players=2,
    )

    neural_net_mod._MODEL_CACHE.clear()

    cfg = AIConfig(
        difficulty=6,
        rng_seed=1,
        use_neural_net=True,
        allow_fresh_weights=False,
        nn_model_id="v3_num_players_2",
    )
    nn = NeuralNetAI(player_number=1, config=cfg)
    nn._base_dir = str(base_dir)
    nn._ensure_model_initialized(BoardType.SQUARE8)

    assert nn.model is not None
    model = getattr(nn.model, "_orig_mod", nn.model)
    assert isinstance(model, RingRiftCNN_v3)
    assert model.num_players == 2
    assert model.num_filters == 24
