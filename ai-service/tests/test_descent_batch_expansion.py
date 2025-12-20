from unittest.mock import MagicMock, patch

import numpy as np

from app.ai.descent_ai import DescentAI
from app.models import AIConfig, BoardType


def test_descent_batches_child_evaluations_in_legacy_expansion() -> None:
    """DescentAI should evaluate expanded children with one NN batch call."""
    config = AIConfig(difficulty=5, think_time=0)

    nn_instance = MagicMock()
    nn_instance.encode_move.return_value = 0

    def fake_eval_batch(game_states, value_head=None):
        batch = len(game_states)
        policy = np.zeros((batch, 8), dtype=np.float32)
        if batch == 1:
            return [0.0], policy
        return [0.1] * batch, policy

    nn_instance.evaluate_batch.side_effect = fake_eval_batch

    with patch("app.ai.neural_net.NeuralNetAI", return_value=nn_instance):
        ai = DescentAI(player_number=1, config=config)

    # Force legacy (immutable) path.
    ai.use_incremental_search = False

    # Avoid real TT behavior in this unit test.
    ai.transposition_table = MagicMock()
    ai.transposition_table.get.return_value = None
    ai.transposition_table.put.return_value = None

    rules_engine = MagicMock()
    ai.rules_engine = rules_engine

    move1, move2, move3 = MagicMock(), MagicMock(), MagicMock()
    rules_engine.get_valid_moves.return_value = [move1, move2, move3]

    def apply_move(state, move):
        child = MagicMock()
        child.game_status = "active"
        child.winner = None
        child.players = state.players
        child.current_player = 2
        child.board = state.board
        child.zobrist_hash = None
        return child

    rules_engine.apply_move.side_effect = apply_move

    root = MagicMock()
    root.game_status = "active"
    root.current_player = 1
    root.players = [MagicMock(), MagicMock()]
    root.board = MagicMock()
    root.board.type = BoardType.SQUARE8
    root.zobrist_hash = 123

    val = ai._descent_iteration(root)

    batch_sizes = [
        len(call.args[0]) for call in nn_instance.evaluate_batch.call_args_list
    ]
    assert sum(1 for s in batch_sizes if s > 1) == 1
    assert val == -0.1

