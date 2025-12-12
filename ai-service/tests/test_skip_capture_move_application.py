import pytest

from app.game_engine import GameEngine
from app.models import GamePhase, MoveType, Position


def test_apply_move_accepts_skip_capture_and_enters_line_processing(
    game_state_factory,
    move_factory,
):
    """
    `skip_capture` is a canonical capture-phase move (RR-CANON-R073).

    Python must accept it during replay/parity, and phase logic must advance
    to line_processing (mirroring TS TurnOrchestrator).
    """
    prior_move = move_factory(
        player=1,
        x=2,
        y=2,
        from_pos=Position(x=1, y=1),
        move_type=MoveType.MOVE_STACK,
        move_number=1,
    )
    state = game_state_factory(
        current_phase=GamePhase.CAPTURE,
        current_player=1,
        move_history=[prior_move],
    )

    skip_move = move_factory(
        player=1,
        x=0,
        y=0,
        move_type=MoveType.SKIP_CAPTURE,
        move_number=2,
    )

    new_state = GameEngine.apply_move(state, skip_move, trace_mode=True)
    assert new_state.current_phase == GamePhase.LINE_PROCESSING

