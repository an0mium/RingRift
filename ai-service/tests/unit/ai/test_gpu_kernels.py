"""Unit tests for GPU kernel functions.

Tests for app/ai/gpu_kernels.py - vectorized GPU operations for RingRift.
"""

from __future__ import annotations

import pytest
import torch

from app.ai.gpu_kernels import (
    DIRECTIONS_X,
    DIRECTIONS_Y,
    apply_capture_batch,
    apply_movement_batch,
    apply_placement_batch,
    check_victory_conditions_kernel,
    count_controlled_stacks,
    detect_lines_kernel,
    evaluate_positions_kernel,
    generate_capture_moves_vectorized,
    generate_normal_moves_vectorized,
    generate_placement_mask_kernel,
    generate_placement_moves_vectorized,
    get_device,
    get_directions,
    get_mobility,
    is_cuda_available,
    is_mps_available,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def device() -> torch.device:
    """Get a test device (CPU for deterministic tests)."""
    return torch.device("cpu")


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 4


@pytest.fixture
def board_size() -> int:
    """Default board size (8x8 square)."""
    return 8


@pytest.fixture
def num_players() -> int:
    """Default number of players."""
    return 2


@pytest.fixture
def empty_board_state(device: torch.device, batch_size: int, board_size: int, num_players: int):
    """Create empty board state tensors."""
    return {
        "stack_owner": torch.zeros(batch_size, board_size, board_size, dtype=torch.int32, device=device),
        "stack_height": torch.zeros(batch_size, board_size, board_size, dtype=torch.int32, device=device),
        "marker_owner": torch.zeros(batch_size, board_size, board_size, dtype=torch.int32, device=device),
        "territory_owner": torch.zeros(batch_size, board_size, board_size, dtype=torch.int32, device=device),
        "rings_in_hand": torch.full((batch_size, num_players + 1), 18, dtype=torch.int32, device=device),
        "territory_count": torch.zeros(batch_size, num_players + 1, dtype=torch.int32, device=device),
        "eliminated_rings": torch.zeros(batch_size, num_players + 1, dtype=torch.int32, device=device),
        "buried_rings": torch.zeros(batch_size, num_players + 1, dtype=torch.int32, device=device),
        "current_player": torch.ones(batch_size, dtype=torch.int32, device=device),
        "active_mask": torch.ones(batch_size, dtype=torch.bool, device=device),
        "game_status": torch.zeros(batch_size, dtype=torch.int32, device=device),
    }


# =============================================================================
# Device Detection Tests
# =============================================================================


class TestDeviceDetection:
    """Tests for device detection functions."""

    def test_get_device_returns_torch_device(self):
        """get_device should return a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_is_cuda_available_returns_bool(self):
        """is_cuda_available should return a boolean."""
        result = is_cuda_available()
        assert isinstance(result, bool)
        # Should match torch directly
        assert result == torch.cuda.is_available()

    def test_is_mps_available_returns_bool(self):
        """is_mps_available should return a boolean."""
        result = is_mps_available()
        assert isinstance(result, bool)


# =============================================================================
# Direction Constants Tests
# =============================================================================


class TestDirectionConstants:
    """Tests for direction constant tensors."""

    def test_directions_have_8_elements(self):
        """Direction tensors should have 8 elements (N, NE, E, SE, S, SW, W, NW)."""
        assert len(DIRECTIONS_Y) == 8
        assert len(DIRECTIONS_X) == 8

    def test_directions_are_correct_values(self):
        """Direction vectors should be correct for 8 compass directions."""
        # Expected: N, NE, E, SE, S, SW, W, NW
        expected_y = torch.tensor([-1, -1, 0, 1, 1, 1, 0, -1], dtype=torch.int32)
        expected_x = torch.tensor([0, 1, 1, 1, 0, -1, -1, -1], dtype=torch.int32)

        assert torch.equal(DIRECTIONS_Y, expected_y)
        assert torch.equal(DIRECTIONS_X, expected_x)

    def test_opposite_directions_cancel(self):
        """Opposite directions should sum to zero."""
        for i in range(4):
            # N+S, NE+SW, E+W, SE+NW
            opposite_idx = i + 4
            assert DIRECTIONS_Y[i] + DIRECTIONS_Y[opposite_idx] == 0
            assert DIRECTIONS_X[i] + DIRECTIONS_X[opposite_idx] == 0

    def test_get_directions_moves_to_device(self, device: torch.device):
        """get_directions should move tensors to specified device."""
        dir_y, dir_x = get_directions(device)
        assert dir_y.device == device
        assert dir_x.device == device
        assert torch.equal(dir_y, DIRECTIONS_Y.to(device))
        assert torch.equal(dir_x, DIRECTIONS_X.to(device))


# =============================================================================
# Placement Mask Generation Tests
# =============================================================================


class TestPlacementMaskKernel:
    """Tests for generate_placement_mask_kernel."""

    def test_empty_board_all_valid(self, empty_board_state: dict):
        """All positions valid on empty board with rings in hand."""
        state = empty_board_state
        mask = generate_placement_mask_kernel(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
            state["marker_owner"],
            torch.zeros_like(state["stack_owner"], dtype=torch.bool),  # is_collapsed
        )

        # All 64 positions should be valid
        assert mask.all()

    def test_no_rings_no_placements(self, empty_board_state: dict):
        """No valid placements when player has no rings."""
        state = empty_board_state
        state["rings_in_hand"][:, 1] = 0  # Player 1 has no rings

        mask = generate_placement_mask_kernel(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
            state["marker_owner"],
            torch.zeros_like(state["stack_owner"], dtype=torch.bool),
        )

        # No positions should be valid for player 1
        assert not mask.any()

    def test_markers_block_placement(self, empty_board_state: dict):
        """Positions with markers should be invalid for placement."""
        state = empty_board_state
        # Place a marker at (0, 0)
        state["marker_owner"][:, 0, 0] = 2

        mask = generate_placement_mask_kernel(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
            state["marker_owner"],
            torch.zeros_like(state["stack_owner"], dtype=torch.bool),
        )

        # (0, 0) should be invalid
        assert not mask[:, 0, 0].any()
        # Other positions should still be valid
        assert mask[:, 1, 1].all()

    def test_collapsed_spaces_block_placement(self, empty_board_state: dict):
        """Collapsed spaces should be invalid for placement."""
        state = empty_board_state
        is_collapsed = torch.zeros_like(state["stack_owner"], dtype=torch.bool)
        is_collapsed[:, 3, 3] = True

        mask = generate_placement_mask_kernel(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
            state["marker_owner"],
            is_collapsed,
        )

        # (3, 3) should be invalid
        assert not mask[:, 3, 3].any()

    def test_inactive_games_no_placements(self, empty_board_state: dict):
        """Inactive games should have no valid placements."""
        state = empty_board_state
        state["active_mask"][0] = False

        mask = generate_placement_mask_kernel(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
            state["marker_owner"],
            torch.zeros_like(state["stack_owner"], dtype=torch.bool),
        )

        # Game 0 should have no valid placements
        assert not mask[0].any()
        # Other games should still have valid placements
        assert mask[1:].any()


class TestGeneratePlacementMovesVectorized:
    """Tests for generate_placement_moves_vectorized."""

    def test_returns_correct_tuple_structure(self, empty_board_state: dict):
        """Should return (game_idx, to_y, to_x, num_moves)."""
        state = empty_board_state
        result = generate_placement_moves_vectorized(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
        )

        assert len(result) == 4
        game_idx, to_y, to_x, num_moves = result
        assert game_idx.dtype == torch.int32
        assert to_y.dtype == torch.int32
        assert to_x.dtype == torch.int32
        assert num_moves.dtype == torch.int32

    def test_counts_all_positions_on_empty_board(self, empty_board_state: dict, batch_size: int, board_size: int):
        """Should count 64 moves per game on empty 8x8 board."""
        state = empty_board_state
        _, _, _, num_moves = generate_placement_moves_vectorized(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
        )

        expected_moves = board_size * board_size
        assert num_moves.tolist() == [expected_moves] * batch_size

    def test_empty_result_when_no_rings(self, empty_board_state: dict):
        """Should return empty tensors when no player has rings."""
        state = empty_board_state
        state["rings_in_hand"][:] = 0

        game_idx, to_y, to_x, num_moves = generate_placement_moves_vectorized(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
        )

        assert len(game_idx) == 0
        assert len(to_y) == 0
        assert len(to_x) == 0
        assert (num_moves == 0).all()


# =============================================================================
# Normal Move Generation Tests
# =============================================================================


class TestGenerateNormalMovesVectorized:
    """Tests for generate_normal_moves_vectorized."""

    def test_empty_board_no_moves(self, empty_board_state: dict):
        """No normal moves possible on empty board (no stacks)."""
        state = empty_board_state
        _, _, _, _, _, num_moves = generate_normal_moves_vectorized(
            state["stack_owner"],
            state["stack_height"],
            state["current_player"],
            state["active_mask"],
        )

        assert (num_moves == 0).all()

    def test_single_stack_generates_moves(self, empty_board_state: dict):
        """A single stack should generate moves in all open directions."""
        state = empty_board_state
        # Place player 1's stack at center (3, 3) with height 1
        state["stack_owner"][0, 3, 3] = 1
        state["stack_height"][0, 3, 3] = 1

        game_idx, from_y, from_x, to_y, to_x, num_moves = generate_normal_moves_vectorized(
            state["stack_owner"],
            state["stack_height"],
            state["current_player"],
            state["active_mask"],
        )

        # Should have moves from (3, 3)
        assert num_moves[0] > 0
        assert (from_y == 3).any()
        assert (from_x == 3).any()

    def test_height_determines_min_distance(self, empty_board_state: dict):
        """Stack height determines minimum move distance."""
        state = empty_board_state
        # Place stack with height 3 at (3, 3)
        state["stack_owner"][0, 3, 3] = 1
        state["stack_height"][0, 3, 3] = 3

        _, from_y, from_x, to_y, to_x, _ = generate_normal_moves_vectorized(
            state["stack_owner"],
            state["stack_height"],
            state["current_player"],
            state["active_mask"],
        )

        # All moves should be at least distance 3
        mask = from_y >= 0  # All valid moves
        distances = torch.abs(to_y - from_y) + torch.abs(to_x - from_x)
        # For diagonal moves, distance is max(dy, dx)
        for i in range(len(from_y)):
            dy = abs(to_y[i].item() - from_y[i].item())
            dx = abs(to_x[i].item() - from_x[i].item())
            # For diagonal moves, effective distance is max(dy, dx)
            effective_dist = max(dy, dx)
            assert effective_dist >= 3

    def test_inactive_game_no_moves(self, empty_board_state: dict):
        """Inactive games should generate no moves."""
        state = empty_board_state
        state["stack_owner"][0, 3, 3] = 1
        state["stack_height"][0, 3, 3] = 1
        state["active_mask"][0] = False

        _, _, _, _, _, num_moves = generate_normal_moves_vectorized(
            state["stack_owner"],
            state["stack_height"],
            state["current_player"],
            state["active_mask"],
        )

        assert num_moves[0] == 0


# =============================================================================
# Capture Move Generation Tests
# =============================================================================


class TestGenerateCaptureMovesVectorized:
    """Tests for generate_capture_moves_vectorized."""

    def test_empty_board_no_captures(self, empty_board_state: dict):
        """No captures possible on empty board."""
        state = empty_board_state
        _, _, _, _, _, num_moves = generate_capture_moves_vectorized(
            state["stack_owner"],
            state["stack_height"],
            state["current_player"],
            state["active_mask"],
        )

        assert (num_moves == 0).all()

    def test_capture_possible_when_taller(self, empty_board_state: dict):
        """Capture should be possible when attacker is taller or equal."""
        state = empty_board_state
        # Player 1's stack at (3, 3), height 2
        state["stack_owner"][0, 3, 3] = 1
        state["stack_height"][0, 3, 3] = 2
        # Player 2's stack at (5, 3), height 1 (capturable)
        state["stack_owner"][0, 5, 3] = 2
        state["stack_height"][0, 5, 3] = 1

        game_idx, from_y, from_x, to_y, to_x, num_moves = generate_capture_moves_vectorized(
            state["stack_owner"],
            state["stack_height"],
            state["current_player"],
            state["active_mask"],
        )

        # Should have at least one capture
        assert num_moves[0] > 0
        # Check that the capture target is at (5, 3)
        game_0_mask = game_idx == 0
        assert (to_y[game_0_mask] == 5).any()
        assert (to_x[game_0_mask] == 3).any()

    def test_no_capture_when_shorter(self, empty_board_state: dict):
        """No capture when attacker is shorter than defender."""
        state = empty_board_state
        # Player 1's stack at (3, 3), height 1
        state["stack_owner"][0, 3, 3] = 1
        state["stack_height"][0, 3, 3] = 1
        # Player 2's stack at (4, 3), height 2 (not capturable)
        state["stack_owner"][0, 4, 3] = 2
        state["stack_height"][0, 4, 3] = 2

        _, _, _, _, _, num_moves = generate_capture_moves_vectorized(
            state["stack_owner"],
            state["stack_height"],
            state["current_player"],
            state["active_mask"],
        )

        # No captures should be possible
        assert num_moves[0] == 0


# =============================================================================
# Heuristic Evaluation Tests
# =============================================================================


class TestEvaluatePositionsKernel:
    """Tests for evaluate_positions_kernel."""

    def test_returns_correct_shape(self, empty_board_state: dict, batch_size: int, num_players: int, board_size: int):
        """Should return (batch, num_players+1) tensor."""
        state = empty_board_state
        weights = {"material_weight": 1.0}

        scores = evaluate_positions_kernel(
            state["stack_owner"],
            state["stack_height"],
            state["marker_owner"],
            state["territory_owner"],
            state["rings_in_hand"],
            state["territory_count"],
            state["eliminated_rings"],
            state["buried_rings"],
            state["current_player"],
            state["active_mask"],
            weights,
            board_size,
            num_players,
        )

        assert scores.shape == (batch_size, num_players + 1)

    def test_more_stacks_higher_score(self, empty_board_state: dict, board_size: int, num_players: int):
        """Player with more stacks should have higher score."""
        state = empty_board_state
        # Player 1 has 3 stacks
        state["stack_owner"][0, 0, 0] = 1
        state["stack_owner"][0, 1, 1] = 1
        state["stack_owner"][0, 2, 2] = 1
        state["stack_height"][0, 0, 0] = 1
        state["stack_height"][0, 1, 1] = 1
        state["stack_height"][0, 2, 2] = 1
        # Player 2 has 1 stack
        state["stack_owner"][0, 5, 5] = 2
        state["stack_height"][0, 5, 5] = 1

        weights = {"material_weight": 1.0}

        scores = evaluate_positions_kernel(
            state["stack_owner"],
            state["stack_height"],
            state["marker_owner"],
            state["territory_owner"],
            state["rings_in_hand"],
            state["territory_count"],
            state["eliminated_rings"],
            state["buried_rings"],
            state["current_player"],
            state["active_mask"],
            weights,
            board_size,
            num_players,
        )

        # Player 1 should have higher score
        assert scores[0, 1] > scores[0, 2]

    def test_center_control_bonus(self, empty_board_state: dict, board_size: int, num_players: int):
        """Center positions should contribute more to score."""
        state = empty_board_state
        center = board_size // 2

        # Player 1 has stack at center
        state["stack_owner"][0, center, center] = 1
        state["stack_height"][0, center, center] = 1

        # Player 2 has stack at corner
        state["stack_owner"][0, 0, 0] = 2
        state["stack_height"][0, 0, 0] = 1

        weights = {"center_control_weight": 1.0}

        scores = evaluate_positions_kernel(
            state["stack_owner"],
            state["stack_height"],
            state["marker_owner"],
            state["territory_owner"],
            state["rings_in_hand"],
            state["territory_count"],
            state["eliminated_rings"],
            state["buried_rings"],
            state["current_player"],
            state["active_mask"],
            weights,
            board_size,
            num_players,
        )

        # Player 1 (center) should have higher center control score
        assert scores[0, 1] > scores[0, 2]

    def test_eliminated_rings_bonus(self, empty_board_state: dict, board_size: int, num_players: int):
        """Eliminated rings should contribute to score."""
        state = empty_board_state
        state["eliminated_rings"][0, 1] = 5

        weights = {"WEIGHT_ELIMINATED_RINGS": 1.0}

        scores = evaluate_positions_kernel(
            state["stack_owner"],
            state["stack_height"],
            state["marker_owner"],
            state["territory_owner"],
            state["rings_in_hand"],
            state["territory_count"],
            state["eliminated_rings"],
            state["buried_rings"],
            state["current_player"],
            state["active_mask"],
            weights,
            board_size,
            num_players,
        )

        assert scores[0, 1] > 0


# =============================================================================
# Line Detection Tests
# =============================================================================


class TestDetectLinesKernel:
    """Tests for detect_lines_kernel."""

    def test_empty_board_no_lines(self, empty_board_state: dict, board_size: int):
        """No lines on empty board."""
        state = empty_board_state
        game_idx, player, start_y, start_x = detect_lines_kernel(
            state["marker_owner"],
            board_size,
            min_line_length=4,
        )

        assert len(game_idx) == 0

    def test_detects_horizontal_line(self, empty_board_state: dict, board_size: int):
        """Should detect horizontal line of 4+ markers."""
        state = empty_board_state
        # Create horizontal line of 4 markers at row 3
        for x in range(4):
            state["marker_owner"][0, 3, x] = 1

        game_idx, player, start_y, start_x = detect_lines_kernel(
            state["marker_owner"],
            board_size,
            min_line_length=4,
        )

        # Should detect at least one line
        assert len(game_idx) > 0
        # All detections should be for player 1
        assert (player == 1).all()

    def test_detects_vertical_line(self, empty_board_state: dict, board_size: int):
        """Should detect vertical line of 4+ markers."""
        state = empty_board_state
        # Create vertical line of 4 markers at column 2
        for y in range(4):
            state["marker_owner"][0, y, 2] = 2

        game_idx, player, start_y, start_x = detect_lines_kernel(
            state["marker_owner"],
            board_size,
            min_line_length=4,
        )

        assert len(game_idx) > 0
        assert (player == 2).all()

    def test_detects_diagonal_line(self, empty_board_state: dict, board_size: int):
        """Should detect diagonal line of 4+ markers."""
        state = empty_board_state
        # Create diagonal line
        for i in range(4):
            state["marker_owner"][0, i, i] = 1

        game_idx, player, start_y, start_x = detect_lines_kernel(
            state["marker_owner"],
            board_size,
            min_line_length=4,
        )

        assert len(game_idx) > 0

    def test_short_line_not_detected(self, empty_board_state: dict, board_size: int):
        """Lines shorter than min_line_length should not be detected."""
        state = empty_board_state
        # Create horizontal line of only 3 markers
        for x in range(3):
            state["marker_owner"][0, 3, x] = 1

        game_idx, _, _, _ = detect_lines_kernel(
            state["marker_owner"],
            board_size,
            min_line_length=4,
        )

        assert len(game_idx) == 0


# =============================================================================
# Victory Condition Tests
# =============================================================================


class TestCheckVictoryConditionsKernel:
    """Tests for check_victory_conditions_kernel."""

    def test_no_victory_initially(self, empty_board_state: dict, num_players: int, board_size: int):
        """No victory on fresh game state."""
        state = empty_board_state
        winner, victory_type, status = check_victory_conditions_kernel(
            state["eliminated_rings"],
            state["rings_in_hand"],
            state["stack_owner"],
            state["buried_rings"],
            state["game_status"],
            num_players,
            board_size,
        )

        assert (winner == 0).all()
        assert (victory_type == 0).all()

    def test_ring_elimination_victory(self, empty_board_state: dict, num_players: int, board_size: int):
        """Victory when player eliminates enough rings."""
        state = empty_board_state
        # Victory threshold for 8x8 2p = round(18 * (2/3 + 1/3 * 1)) = 18
        state["eliminated_rings"][0, 1] = 18

        winner, victory_type, status = check_victory_conditions_kernel(
            state["eliminated_rings"],
            state["rings_in_hand"],
            state["stack_owner"],
            state["buried_rings"],
            state["game_status"],
            num_players,
            board_size,
        )

        assert winner[0] == 1
        assert victory_type[0] == 1  # ring_elimination

    def test_last_standing_victory(self, empty_board_state: dict, num_players: int, board_size: int):
        """Victory when only one player has rings remaining."""
        state = empty_board_state
        # Player 2 has no rings anywhere
        state["rings_in_hand"][0, 2] = 0
        state["buried_rings"][0, 2] = 0
        # Make sure no stacks belong to player 2
        # Player 1 still has rings
        state["rings_in_hand"][0, 1] = 10

        winner, victory_type, status = check_victory_conditions_kernel(
            state["eliminated_rings"],
            state["rings_in_hand"],
            state["stack_owner"],
            state["buried_rings"],
            state["game_status"],
            num_players,
            board_size,
        )

        assert winner[0] == 1
        assert victory_type[0] == 2  # last_standing


# =============================================================================
# Batch Move Application Tests
# =============================================================================


class TestApplyPlacementBatch:
    """Tests for apply_placement_batch."""

    def test_placement_updates_board(self, empty_board_state: dict, device: torch.device):
        """Placement should update stack owner and height."""
        state = empty_board_state
        game_idx = torch.tensor([0], dtype=torch.int32, device=device)
        to_y = torch.tensor([3], dtype=torch.int32, device=device)
        to_x = torch.tensor([4], dtype=torch.int32, device=device)

        new_owner, new_height, new_rings = apply_placement_batch(
            state["stack_owner"],
            state["stack_height"],
            state["rings_in_hand"],
            game_idx,
            to_y,
            to_x,
            state["current_player"],
        )

        assert new_owner[0, 3, 4] == 1  # Player 1 owns the stack
        assert new_height[0, 3, 4] == 1  # Height is 1
        assert new_rings[0, 1] == 17  # Player 1 lost a ring

    def test_placement_decrements_rings(self, empty_board_state: dict, device: torch.device):
        """Placement should decrease rings in hand."""
        state = empty_board_state
        initial_rings = state["rings_in_hand"][0, 1].item()

        game_idx = torch.tensor([0], dtype=torch.int32, device=device)
        to_y = torch.tensor([0], dtype=torch.int32, device=device)
        to_x = torch.tensor([0], dtype=torch.int32, device=device)

        _, _, new_rings = apply_placement_batch(
            state["stack_owner"],
            state["stack_height"],
            state["rings_in_hand"],
            game_idx,
            to_y,
            to_x,
            state["current_player"],
        )

        assert new_rings[0, 1] == initial_rings - 1


class TestApplyMovementBatch:
    """Tests for apply_movement_batch."""

    def test_movement_clears_source(self, empty_board_state: dict, device: torch.device):
        """Movement should clear the source position."""
        state = empty_board_state
        state["stack_owner"][0, 2, 2] = 1
        state["stack_height"][0, 2, 2] = 2

        game_idx = torch.tensor([0], dtype=torch.int32, device=device)
        from_y = torch.tensor([2], dtype=torch.int32, device=device)
        from_x = torch.tensor([2], dtype=torch.int32, device=device)
        to_y = torch.tensor([5], dtype=torch.int32, device=device)
        to_x = torch.tensor([5], dtype=torch.int32, device=device)

        new_owner, new_height = apply_movement_batch(
            state["stack_owner"],
            state["stack_height"],
            game_idx,
            from_y,
            from_x,
            to_y,
            to_x,
            state["current_player"],
        )

        assert new_owner[0, 2, 2] == 0  # Source cleared
        assert new_height[0, 2, 2] == 0
        assert new_owner[0, 5, 5] == 1  # Destination has stack
        assert new_height[0, 5, 5] == 2  # Height preserved

    def test_movement_preserves_height(self, empty_board_state: dict, device: torch.device):
        """Movement should preserve stack height."""
        state = empty_board_state
        state["stack_owner"][0, 1, 1] = 1
        state["stack_height"][0, 1, 1] = 5

        game_idx = torch.tensor([0], dtype=torch.int32, device=device)
        from_y = torch.tensor([1], dtype=torch.int32, device=device)
        from_x = torch.tensor([1], dtype=torch.int32, device=device)
        to_y = torch.tensor([6], dtype=torch.int32, device=device)
        to_x = torch.tensor([6], dtype=torch.int32, device=device)

        _, new_height = apply_movement_batch(
            state["stack_owner"],
            state["stack_height"],
            game_idx,
            from_y,
            from_x,
            to_y,
            to_x,
            state["current_player"],
        )

        assert new_height[0, 6, 6] == 5


class TestApplyCaptureBatch:
    """Tests for apply_capture_batch."""

    def test_capture_merges_stacks(self, empty_board_state: dict, device: torch.device):
        """Capture should merge attacker and defender stacks."""
        state = empty_board_state
        # Attacker at (2, 2) with height 3
        state["stack_owner"][0, 2, 2] = 1
        state["stack_height"][0, 2, 2] = 3
        # Defender at (4, 4) with height 2
        state["stack_owner"][0, 4, 4] = 2
        state["stack_height"][0, 4, 4] = 2

        game_idx = torch.tensor([0], dtype=torch.int32, device=device)
        from_y = torch.tensor([2], dtype=torch.int32, device=device)
        from_x = torch.tensor([2], dtype=torch.int32, device=device)
        to_y = torch.tensor([4], dtype=torch.int32, device=device)
        to_x = torch.tensor([4], dtype=torch.int32, device=device)

        new_owner, new_height, new_marker, new_elim, new_buried = apply_capture_batch(
            state["stack_owner"],
            state["stack_height"],
            state["marker_owner"],
            state["eliminated_rings"],
            state["buried_rings"],
            game_idx,
            from_y,
            from_x,
            to_y,
            to_x,
            state["current_player"],
        )

        # Source should be cleared
        assert new_owner[0, 2, 2] == 0
        # Destination owned by attacker
        assert new_owner[0, 4, 4] == 1
        # Height should be 3 + 2 - 1 = 4
        assert new_height[0, 4, 4] == 4
        # Marker placed
        assert new_marker[0, 4, 4] == 1
        # Eliminated ring tracked
        assert new_elim[0, 1] == 1

    def test_capture_places_marker(self, empty_board_state: dict, device: torch.device):
        """Capture should place attacker's marker at capture location."""
        state = empty_board_state
        state["stack_owner"][0, 0, 0] = 1
        state["stack_height"][0, 0, 0] = 2
        state["stack_owner"][0, 2, 0] = 2
        state["stack_height"][0, 2, 0] = 1

        game_idx = torch.tensor([0], dtype=torch.int32, device=device)
        from_y = torch.tensor([0], dtype=torch.int32, device=device)
        from_x = torch.tensor([0], dtype=torch.int32, device=device)
        to_y = torch.tensor([2], dtype=torch.int32, device=device)
        to_x = torch.tensor([0], dtype=torch.int32, device=device)

        _, _, new_marker, _, _ = apply_capture_batch(
            state["stack_owner"],
            state["stack_height"],
            state["marker_owner"],
            state["eliminated_rings"],
            state["buried_rings"],
            game_idx,
            from_y,
            from_x,
            to_y,
            to_x,
            state["current_player"],
        )

        assert new_marker[0, 2, 0] == 1  # Player 1's marker


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestCountControlledStacks:
    """Tests for count_controlled_stacks."""

    def test_empty_board_zero_counts(self, empty_board_state: dict, num_players: int):
        """Empty board should have zero stack counts."""
        state = empty_board_state
        counts = count_controlled_stacks(state["stack_owner"], num_players)

        assert (counts == 0).all()

    def test_counts_correct_stacks(self, empty_board_state: dict, num_players: int):
        """Should correctly count stacks per player."""
        state = empty_board_state
        # Player 1 has 3 stacks
        state["stack_owner"][0, 0, 0] = 1
        state["stack_owner"][0, 1, 1] = 1
        state["stack_owner"][0, 2, 2] = 1
        # Player 2 has 2 stacks
        state["stack_owner"][0, 5, 5] = 2
        state["stack_owner"][0, 6, 6] = 2

        counts = count_controlled_stacks(state["stack_owner"], num_players)

        assert counts[0, 1] == 3  # Player 1
        assert counts[0, 2] == 2  # Player 2


class TestGetMobility:
    """Tests for get_mobility."""

    def test_empty_board_zero_mobility(self, empty_board_state: dict):
        """Empty board should have zero mobility."""
        state = empty_board_state
        mobility = get_mobility(
            state["stack_owner"],
            state["stack_height"],
            state["current_player"],
            state["active_mask"],
        )

        assert (mobility == 0).all()

    def test_single_stack_positive_mobility(self, empty_board_state: dict):
        """Board with one stack should have positive mobility."""
        state = empty_board_state
        state["stack_owner"][0, 3, 3] = 1
        state["stack_height"][0, 3, 3] = 1

        mobility = get_mobility(
            state["stack_owner"],
            state["stack_height"],
            state["current_player"],
            state["active_mask"],
        )

        assert mobility[0] > 0


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for gpu_kernels."""

    def test_large_batch_size(self, device: torch.device, board_size: int, num_players: int):
        """Should handle large batch sizes."""
        large_batch = 100
        state = {
            "stack_owner": torch.zeros(large_batch, board_size, board_size, dtype=torch.int32, device=device),
            "stack_height": torch.zeros(large_batch, board_size, board_size, dtype=torch.int32, device=device),
            "rings_in_hand": torch.full((large_batch, num_players + 1), 18, dtype=torch.int32, device=device),
            "current_player": torch.ones(large_batch, dtype=torch.int32, device=device),
            "active_mask": torch.ones(large_batch, dtype=torch.bool, device=device),
            "marker_owner": torch.zeros(large_batch, board_size, board_size, dtype=torch.int32, device=device),
        }

        # Should not crash
        _, _, _, num_moves = generate_placement_moves_vectorized(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
        )

        assert len(num_moves) == large_batch
        assert (num_moves == 64).all()  # 8x8 board

    def test_all_inactive_games(self, empty_board_state: dict):
        """All inactive games should produce no moves."""
        state = empty_board_state
        state["active_mask"][:] = False

        _, _, _, num_moves = generate_placement_moves_vectorized(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
        )

        assert (num_moves == 0).all()

    def test_mixed_player_turns(self, empty_board_state: dict, device: torch.device):
        """Different games can have different current players."""
        state = empty_board_state
        state["current_player"][0] = 1
        state["current_player"][1] = 2
        state["current_player"][2] = 1
        state["current_player"][3] = 2

        # Give each player rings
        state["rings_in_hand"][:, 1] = 10
        state["rings_in_hand"][:, 2] = 10

        _, _, _, num_moves = generate_placement_moves_vectorized(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
        )

        # All should have placements
        assert (num_moves > 0).all()

    def test_single_game_batch(self, device: torch.device, board_size: int, num_players: int):
        """Should work with batch size of 1."""
        state = {
            "stack_owner": torch.zeros(1, board_size, board_size, dtype=torch.int32, device=device),
            "stack_height": torch.zeros(1, board_size, board_size, dtype=torch.int32, device=device),
            "rings_in_hand": torch.full((1, num_players + 1), 18, dtype=torch.int32, device=device),
            "current_player": torch.ones(1, dtype=torch.int32, device=device),
            "active_mask": torch.ones(1, dtype=torch.bool, device=device),
            "marker_owner": torch.zeros(1, board_size, board_size, dtype=torch.int32, device=device),
        }

        _, _, _, num_moves = generate_placement_moves_vectorized(
            state["stack_owner"],
            state["rings_in_hand"],
            state["current_player"],
            state["active_mask"],
        )

        assert len(num_moves) == 1
        assert num_moves[0] == 64
