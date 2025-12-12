import app.ai.heuristic_ai as heuristic_ai
from app.ai.heuristic_ai import HeuristicAI
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType
from app.training.generate_data import create_initial_state


def _advance_p1_first_turn(state):
    """Advance through Player 1's first turn using first-legal moves.

    We intentionally use the host-level phase requirement helpers because
    GameEngine.get_valid_moves does not auto-synthesize no_*_action moves
    (RR-CANON-R076).
    """
    steps = 0
    while state.current_player == 1 and steps < 32:
        legal = GameEngine.get_valid_moves(state, 1)
        if not legal:
            requirement = GameEngine.get_phase_requirement(state, 1)
            assert requirement is not None
            move = GameEngine.synthesize_bookkeeping_move(requirement, state)
        else:
            move = legal[0]

        state = GameEngine.apply_move(state, move, trace_mode=True)
        steps += 1

    return state


def test_heuristic_batch_eval_handles_swap_sides(monkeypatch):
    """Regression: batch eval path must not crash on SWAP_SIDES."""
    monkeypatch.setattr(heuristic_ai, "USE_MAKE_UNMAKE", True)
    monkeypatch.setattr(heuristic_ai, "USE_BATCH_EVAL", True)
    monkeypatch.setattr(heuristic_ai, "BATCH_EVAL_THRESHOLD", 1)

    state = create_initial_state(BoardType.SQUARE8, num_players=2)
    state = _advance_p1_first_turn(state)

    assert state.current_player == 2
    assert state.current_phase.value == "ring_placement"

    legal = GameEngine.get_valid_moves(state, 2)
    assert any(m.type.value == "swap_sides" for m in legal)

    ai = HeuristicAI(2, AIConfig(difficulty=2, rng_seed=2, think_time=0))
    move = ai.select_move(state)
    assert move is not None
    assert ai.player_number == 2

    # Validate that the move matches a legal move surface (ignore move ID).
    assert any(
        m.type == move.type
        and m.player == move.player
        and getattr(m, "to", None) == getattr(move, "to", None)
        and getattr(m, "from_pos", None) == getattr(move, "from_pos", None)
        for m in legal
    )

