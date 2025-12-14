import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


from app.game_engine import GameEngine
from app.models import BoardType, GamePhase, MoveType
from app.training.env import RingRiftEnv


def test_game_engine_capture_phase_surfaces_skip_capture_without_attacker_pos() -> None:
    env = RingRiftEnv(board_type=BoardType.SQUARE8, num_players=2)
    state = env.reset(seed=123)

    # Synthetic "corrupt/legacy" state: CAPTURE without a landing position.
    state = state.model_copy(update={"current_phase": GamePhase.CAPTURE, "move_history": []})

    moves = GameEngine.get_valid_moves(state, state.current_player)
    assert [m.type for m in moves] == [MoveType.SKIP_CAPTURE]


def test_ringrift_env_legal_moves_does_not_mutate_capture_phase() -> None:
    env = RingRiftEnv(board_type=BoardType.SQUARE8, num_players=2)
    state = env.reset(seed=456)

    state = state.model_copy(update={"current_phase": GamePhase.CAPTURE, "move_history": []})
    env._state = state

    _ = env.legal_moves()
    assert env.state.current_phase == GamePhase.CAPTURE

