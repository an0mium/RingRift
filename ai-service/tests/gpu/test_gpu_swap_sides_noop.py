import pytest
import torch


try:
    from app.ai.gpu_parallel_games import (
        BatchGameState,
        GamePhase,
        GameStatus,
        ParallelGameRunner,
    )

    GPU_MODULES_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import guard
    GPU_MODULES_AVAILABLE = False
    GPU_IMPORT_ERROR = str(exc)


pytestmark = pytest.mark.skipif(
    not GPU_MODULES_AVAILABLE,
    reason=f"GPU modules not available: {GPU_IMPORT_ERROR if not GPU_MODULES_AVAILABLE else ''}",
)


def test_gpu_swap_sides_is_noop_until_explicitly_recorded() -> None:
    """GPU runner must not silently apply swap_sides without recording it."""
    device = torch.device("cpu")
    runner = ParallelGameRunner(
        batch_size=1,
        board_size=8,
        num_players=2,
        device=device,
        shadow_validation=False,
        state_validation=False,
        swap_enabled=True,
    )
    state: BatchGameState = runner.state

    g = 0
    p1 = 1
    p2 = 2

    # Set asymmetric per-seat counters so a silent "swap" would be observable.
    state.rings_in_hand[g, p1] = 3
    state.rings_in_hand[g, p2] = 7
    state.territory_count[g, p1] = 2
    state.territory_count[g, p2] = 5

    # Give Player 1 a stack so (historically) the naive swap heuristic would trigger.
    state.stack_owner[g, 0, 0] = p1
    state.stack_height[g, 0, 0] = 1
    state.cap_height[g, 0, 0] = 1

    state.current_player[g] = p1
    state.current_phase[g] = GamePhase.END_TURN
    state.game_status[g] = GameStatus.ACTIVE
    state.swap_offered[g] = False

    weights = [runner._default_weights()]
    runner._step_games(weights)

    assert state.current_player[g].item() == p2
    assert state.current_phase[g].item() == GamePhase.RING_PLACEMENT
    assert bool(state.swap_offered[g].item()) is True

    # Ensure no hidden ownership/counter swap was applied.
    assert state.stack_owner[g, 0, 0].item() == p1
    assert state.rings_in_hand[g, p1].item() == 3
    assert state.rings_in_hand[g, p2].item() == 7
    assert state.territory_count[g, p1].item() == 2
    assert state.territory_count[g, p2].item() == 5


def test_gpu_hex_embedding_defaults_to_canonical_ring_supply() -> None:
    """Hex GPU kernels use a 25×25 embedding; default rings must still be canonical."""
    device = torch.device("cpu")
    runner = ParallelGameRunner(
        batch_size=1,
        board_size=25,
        num_players=2,
        device=device,
        shadow_validation=False,
        state_validation=False,
    )
    state: BatchGameState = runner.state

    g = 0
    assert state.rings_in_hand[g, 1].item() == 96
    assert state.rings_in_hand[g, 2].item() == 96
