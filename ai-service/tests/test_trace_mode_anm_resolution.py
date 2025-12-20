from datetime import datetime

from app.game_engine import GameEngine
from app.models import Move, MoveType
from app.training.generate_data import create_initial_state


def _make_no_placement_move(player: int) -> Move:
    return Move(
        id="noop-1",
        type=MoveType.NO_PLACEMENT_ACTION,
        player=player,
        timestamp=datetime.now(),
    )


def test_trace_mode_skips_anm_resolution(monkeypatch):
    calls = {"count": 0}

    def _spy(_state):
        calls["count"] += 1

    monkeypatch.setattr(GameEngine, "_resolve_anm_for_current_player", _spy)

    state = create_initial_state()
    move = _make_no_placement_move(state.current_player)
    GameEngine.apply_move(state, move, trace_mode=True)

    assert calls["count"] == 0


def test_non_trace_mode_runs_anm_resolution(monkeypatch):
    calls = {"count": 0}

    def _spy(_state):
        calls["count"] += 1

    monkeypatch.setattr(GameEngine, "_resolve_anm_for_current_player", _spy)

    state = create_initial_state()
    move = _make_no_placement_move(state.current_player)
    GameEngine.apply_move(state, move, trace_mode=False)

    assert calls["count"] == 1
