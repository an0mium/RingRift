from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from app.ai.neural_net import NeuralNetAI
from app.models import BoardType
from app.models.core import AIConfig, BoardState, Move, Position


def _make_ai() -> NeuralNetAI:
    cfg = AIConfig(
        difficulty=6,
        think_time=1,
        randomness=0.0,
        rngSeed=1,
        use_neural_net=True,
        nn_model_id="sq8_2p_nn_baseline",
        allow_fresh_weights=True,  # We will inject a dummy model instead of loading weights.
    )
    return NeuralNetAI(1, cfg)


def _make_move_stack(from_xy: tuple[int, int], to_xy: tuple[int, int]) -> Move:
    return Move(
        id="m",
        type="move_stack",
        player=1,
        from_pos=Position(x=from_xy[0], y=from_xy[1]),
        to=Position(x=to_xy[0], y=to_xy[1]),
        timestamp=datetime.now(),
        thinkTime=0,
        moveNumber=0,
    )


def test_square8_uses_board_specific_encoding_when_policy_size_matches() -> None:
    ai = _make_ai()
    ai.model = SimpleNamespace(policy_size=7000)  # square8 board-specific head

    board = BoardState(type=BoardType.SQUARE8, size=8, stacks={}, markers={}, collapsedSpaces={})
    move = _make_move_stack((0, 0), (1, 0))

    idx = ai.encode_move(move, board)
    assert idx == 220  # board-specific square8 encoding
    assert 0 <= idx < ai.model.policy_size


def test_square8_falls_back_to_legacy_maxn_encoding_when_policy_size_mismatch() -> None:
    ai = _make_ai()
    ai.model = SimpleNamespace(policy_size=54875)  # legacy MAX_N head size

    board = BoardState(type=BoardType.SQUARE8, size=8, stacks={}, markers={}, collapsedSpaces={})
    move = _make_move_stack((0, 0), (1, 0))

    idx = ai.encode_move(move, board)
    assert idx == 1155  # legacy MAX_N=19 encoding
    assert 0 <= idx < ai.model.policy_size

