from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest

from app.ai.async_nn_eval import AsyncNeuralBatcher
from app.models import BoardType


class _DummyNeuralNet:
    def __init__(self, *, model: object, call_sizes: List[int]) -> None:
        self.device = "cuda"
        self.model = model
        self.history_length = 3
        self.game_history: dict[str, Any] = {}
        self._call_sizes = call_sizes

    def _ensure_model_initialized(self, board_type: BoardType) -> None:
        # Model is injected; nothing to do.
        _ = board_type

    def evaluate_batch(self, game_states):  # type: ignore[no-untyped-def]
        self._call_sizes.append(len(game_states))
        values = [float(getattr(s, "tag")) for s in game_states]
        policies = np.asarray([[float(getattr(s, "tag"))] for s in game_states], dtype=np.float32)
        return values, policies


def _state(tag: int, board_type: BoardType = BoardType.SQUARE8, board_size: int = 8) -> Any:
    board = type("Board", (), {"type": board_type, "size": board_size})()
    return type("State", (), {"board": board, "tag": tag})()


def test_async_neural_batcher_microbatches_across_instances_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RINGRIFT_NN_EVAL_QUEUE", "1")
    monkeypatch.setenv("RINGRIFT_NN_EVAL_BATCH_TIMEOUT_MS", "50")
    monkeypatch.setenv("RINGRIFT_NN_EVAL_MAX_BATCH", "256")

    calls: list[int] = []
    shared_model = object()

    nn1 = _DummyNeuralNet(model=shared_model, call_sizes=calls)
    nn2 = _DummyNeuralNet(model=shared_model, call_sizes=calls)

    batcher1 = AsyncNeuralBatcher(nn1)  # type: ignore[arg-type]
    batcher2 = AsyncNeuralBatcher(nn2)  # type: ignore[arg-type]
    try:
        fut1 = batcher1.submit([_state(1), _state(2)])
        fut2 = batcher2.submit([_state(3)])

        values1, pol1 = fut1.result(timeout=2)
        values2, pol2 = fut2.result(timeout=2)

        assert values1 == [1.0, 2.0]
        assert values2 == [3.0]
        assert pol1.shape == (2, 1)
        assert pol2.shape == (1, 1)

        # Both submits should be coalesced into a single underlying eval call.
        assert calls == [3]
    finally:
        batcher1.shutdown()
        batcher2.shutdown()

