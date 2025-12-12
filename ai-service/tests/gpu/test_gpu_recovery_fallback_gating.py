import pytest
import torch


try:
    from app.ai.gpu_parallel_games import (
        BatchGameState,
        generate_recovery_moves_batch,
    )

    GPU_MODULES_AVAILABLE = True
except Exception as exc:  # pragma: no cover - import guard
    GPU_MODULES_AVAILABLE = False
    GPU_IMPORT_ERROR = str(exc)


pytestmark = pytest.mark.skipif(
    not GPU_MODULES_AVAILABLE,
    reason=f"GPU modules not available: {GPU_IMPORT_ERROR if not GPU_MODULES_AVAILABLE else ''}",
)


def test_gpu_recovery_fallback_gating_prefers_line_forming_slides() -> None:
    """When any line-forming recovery slide exists, fallback-class moves are illegal."""
    device = torch.device("cpu")
    state = BatchGameState.create_batch(
        batch_size=1,
        board_size=8,
        num_players=2,
        device=device,
        max_history_moves=50,
    )

    g = 0
    player = 1
    opponent = 2

    # Recovery-eligible: no controlled stacks, at least one marker, at least one buried ring.
    state.current_player[g] = player
    state.buried_rings[g, player] = 1

    # Build a 3-marker line and a marker that can slide to complete a 4-line.
    # Existing markers at (3,1), (3,2), (3,3); slide marker from (2,4) -> (3,4).
    state.marker_owner[g, 3, 1] = player
    state.marker_owner[g, 3, 2] = player
    state.marker_owner[g, 3, 3] = player
    state.marker_owner[g, 2, 4] = player

    # Add an adjacent opponent stack that would be stack-strike eligible under fallback,
    # but must be excluded because a line-forming recovery exists.
    state.stack_owner[g, 2, 5] = opponent
    state.stack_height[g, 2, 5] = 2
    state.cap_height[g, 2, 5] = 2

    active_mask = torch.tensor([True], dtype=torch.bool, device=device)
    moves = generate_recovery_moves_batch(state, active_mask)

    assert moves.total_moves == 1
    assert moves.from_y[0].item() == 2
    assert moves.from_x[0].item() == 4
    assert moves.to_y[0].item() == 3
    assert moves.to_x[0].item() == 4

