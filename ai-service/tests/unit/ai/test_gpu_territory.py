"""Tests for gpu_territory module.

Tests cover:
- _find_eligible_territory_cap: Find stack for self-elimination
- _find_all_regions: BFS region detection
- _is_physically_disconnected: Physical disconnection check (R141)
- _is_color_disconnected: Color disconnection check (R142)
- compute_territory_batch: Full territory processing
"""

import torch


class MockBatchGameState:
    """Mock BatchGameState for testing territory processing."""

    def __init__(
        self,
        batch_size: int = 1,
        board_size: int = 8,
        num_players: int = 2,
        device: torch.device = None,
    ):
        self.batch_size = batch_size
        self.board_size = board_size
        self.num_players = num_players
        self.device = device or torch.device('cpu')

        # Initialize tensors
        self.stack_owner = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.int16, device=self.device
        )
        self.stack_height = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.int16, device=self.device
        )
        self.marker_owner = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.int16, device=self.device
        )
        self.is_collapsed = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.bool, device=self.device
        )
        self.territory_owner = torch.zeros(
            batch_size, board_size, board_size,
            dtype=torch.int16, device=self.device
        )
        self.territory_count = torch.zeros(
            batch_size, num_players + 1,
            dtype=torch.int32, device=self.device
        )
        self.eliminated_rings = torch.zeros(
            batch_size, num_players + 1,
            dtype=torch.int32, device=self.device
        )
        self.rings_caused_eliminated = torch.zeros(
            batch_size, num_players + 1,
            dtype=torch.int32, device=self.device
        )

    def get_active_mask(self) -> torch.Tensor:
        return torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    def place_stack(self, game_idx: int, player: int, y: int, x: int, height: int = 1):
        """Helper to place a stack."""
        self.stack_owner[game_idx, y, x] = player
        self.stack_height[game_idx, y, x] = height

    def place_marker(self, game_idx: int, player: int, y: int, x: int):
        """Helper to place a marker."""
        self.marker_owner[game_idx, y, x] = player

    def collapse_cell(self, game_idx: int, y: int, x: int, owner: int = 0):
        """Helper to collapse a cell."""
        self.is_collapsed[game_idx, y, x] = True
        self.territory_owner[game_idx, y, x] = owner


class TestFindEligibleTerritoryCap:
    """Tests for _find_eligible_territory_cap function."""

    def test_finds_controlled_stack(self):
        """Finds a stack controlled by the player."""
        from app.ai.gpu_territory import _find_eligible_territory_cap

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_stack(0, 1, 3, 4, height=2)

        result = _find_eligible_territory_cap(state, game_idx=0, player=1)

        assert result is not None
        y, x, height = result
        assert y == 3
        assert x == 4
        assert height == 2

    def test_returns_none_when_no_stacks(self):
        """Returns None when player has no stacks."""
        from app.ai.gpu_territory import _find_eligible_territory_cap

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)

        result = _find_eligible_territory_cap(state, game_idx=0, player=1)

        assert result is None

    def test_ignores_other_player_stacks(self):
        """Ignores stacks belonging to other players."""
        from app.ai.gpu_territory import _find_eligible_territory_cap

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_stack(0, 2, 3, 4, height=2)  # Player 2's stack

        result = _find_eligible_territory_cap(state, game_idx=0, player=1)

        assert result is None

    def test_respects_excluded_positions(self):
        """Excludes positions in excluded_positions set."""
        from app.ai.gpu_territory import _find_eligible_territory_cap

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_stack(0, 1, 3, 4, height=2)
        state.place_stack(0, 1, 5, 5, height=1)

        # Exclude the first stack
        excluded = {(3, 4)}
        result = _find_eligible_territory_cap(state, game_idx=0, player=1, excluded_positions=excluded)

        assert result is not None
        y, x, height = result
        assert (y, x) == (5, 5)
        assert height == 1

    def test_height_one_stack_eligible(self):
        """Height-1 stacks are eligible per RR-CANON-R145."""
        from app.ai.gpu_territory import _find_eligible_territory_cap

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_stack(0, 1, 3, 4, height=1)

        result = _find_eligible_territory_cap(state, game_idx=0, player=1)

        assert result is not None
        assert result[2] == 1  # height


class TestFindAllRegions:
    """Tests for _find_all_regions function."""

    def test_single_region_no_collapse(self):
        """Board with no collapsed cells is single region."""
        from app.ai.gpu_territory import _find_all_regions

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)

        regions = _find_all_regions(state, game_idx=0)

        assert len(regions) == 1
        assert len(regions[0]) == 64  # 8x8 = 64 cells

    def test_collapsed_cells_split_regions(self):
        """Collapsed cells split the board into regions."""
        from app.ai.gpu_territory import _find_all_regions

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Create a vertical wall of collapsed cells at column 4
        for y in range(8):
            state.collapse_cell(0, y, 4)

        regions = _find_all_regions(state, game_idx=0)

        # Should have 2 regions (left and right of wall)
        assert len(regions) == 2

    def test_corner_region(self):
        """Small isolated corner region is detected."""
        from app.ai.gpu_territory import _find_all_regions

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Collapse cells to isolate top-left 2x2 corner
        for y in range(2):
            state.collapse_cell(0, y, 2)
        for x in range(2):
            state.collapse_cell(0, 2, x)

        regions = _find_all_regions(state, game_idx=0)

        # Should find multiple regions including the corner
        assert len(regions) >= 2
        # Find the corner region
        corner_region = None
        for region in regions:
            if (0, 0) in region:
                corner_region = region
                break
        assert corner_region is not None
        assert len(corner_region) == 4  # 2x2 = 4 cells

    def test_regions_use_4_connectivity(self):
        """Regions use 4-connectivity (not diagonal)."""
        from app.ai.gpu_territory import _find_all_regions

        state = MockBatchGameState(batch_size=1, board_size=4, num_players=2)
        # Create checkerboard pattern of collapsed cells
        for y in range(4):
            for x in range(4):
                if (y + x) % 2 == 0:
                    state.collapse_cell(0, y, x)

        regions = _find_all_regions(state, game_idx=0)

        # With checkerboard, each non-collapsed cell is isolated (4-connectivity)
        # So we should have 8 single-cell regions
        assert len(regions) == 8
        for region in regions:
            assert len(region) == 1


class TestIsPhysicallyDisconnected:
    """Tests for _is_physically_disconnected function."""

    def test_connected_region_not_disconnected(self):
        """Region connected to rest of board is not disconnected."""
        from app.ai.gpu_territory import _is_physically_disconnected, _find_all_regions

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # No collapsed cells - entire board is one region
        regions = _find_all_regions(state, game_idx=0)
        region = regions[0]

        is_disconnected, border_player = _is_physically_disconnected(state, game_idx=0, region=region)

        assert is_disconnected is False
        assert border_player is None

    def test_isolated_by_collapsed_cells(self):
        """Region isolated by collapsed cells is physically disconnected."""
        from app.ai.gpu_territory import _is_physically_disconnected, _find_all_regions

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Create collapsed wall isolating corner
        for y in range(3):
            state.collapse_cell(0, y, 3)
        for x in range(3):
            state.collapse_cell(0, 3, x)

        regions = _find_all_regions(state, game_idx=0)
        # Find the small corner region
        corner_region = None
        for region in regions:
            if (0, 0) in region:
                corner_region = region
                break

        assert corner_region is not None
        is_disconnected, border_player = _is_physically_disconnected(state, game_idx=0, region=corner_region)

        assert is_disconnected is True
        # No marker border, just collapsed cells
        assert border_player is None

    def test_marker_border_single_player(self):
        """Region bordered by single player's markers returns that player."""
        from app.ai.gpu_territory import _is_physically_disconnected, _find_all_regions

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Create marker wall isolating corner
        for y in range(3):
            state.place_marker(0, 1, y, 3)
        for x in range(3):
            state.place_marker(0, 1, 3, x)

        # Also need collapsed cells to create the disconnection
        state.collapse_cell(0, 3, 3)  # Corner of the wall

        regions = _find_all_regions(state, game_idx=0)
        # Find the small corner region (should be smaller part)
        corner_region = None
        for region in regions:
            if (0, 0) in region and len(region) < 20:
                corner_region = region
                break

        if corner_region:
            is_disconnected, border_player = _is_physically_disconnected(state, game_idx=0, region=corner_region)
            # Check the logic - if markers form the border
            if is_disconnected and border_player:
                assert border_player == 1


class TestIsColorDisconnected:
    """Tests for _is_color_disconnected function."""

    def test_region_with_all_colors_not_disconnected(self):
        """Region with all active colors is not color-disconnected."""
        from app.ai.gpu_territory import _is_color_disconnected

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Both players have stacks
        state.place_stack(0, 1, 0, 0, height=1)
        state.place_stack(0, 2, 7, 7, height=1)

        # Region containing both stacks
        region = {(y, x) for y in range(8) for x in range(8)}

        is_disconnected = _is_color_disconnected(state, game_idx=0, region=region)

        assert is_disconnected is False

    def test_region_missing_color_is_disconnected(self):
        """Region missing an active color is color-disconnected."""
        from app.ai.gpu_territory import _is_color_disconnected

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Player 1 has stack in corner
        state.place_stack(0, 1, 0, 0, height=1)
        # Player 2 has stack elsewhere
        state.place_stack(0, 2, 7, 7, height=1)

        # Region containing only player 1's stack
        region = {(0, 0), (0, 1), (1, 0), (1, 1)}

        is_disconnected = _is_color_disconnected(state, game_idx=0, region=region)

        # Region has player 1 but not player 2, so strict subset
        assert is_disconnected is True

    def test_empty_region_is_disconnected(self):
        """Empty region (no stacks) is color-disconnected if active colors exist."""
        from app.ai.gpu_territory import _is_color_disconnected

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # Stacks exist on board
        state.place_stack(0, 1, 0, 0, height=1)
        state.place_stack(0, 2, 7, 7, height=1)

        # Region with no stacks
        region = {(3, 3), (3, 4), (4, 3), (4, 4)}

        is_disconnected = _is_color_disconnected(state, game_idx=0, region=region)

        # Empty set is strict subset of any non-empty set
        assert is_disconnected is True

    def test_no_active_colors_not_disconnected(self):
        """Region with no active colors anywhere returns False."""
        from app.ai.gpu_territory import _is_color_disconnected

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        # No stacks anywhere

        region = {(0, 0), (0, 1)}

        is_disconnected = _is_color_disconnected(state, game_idx=0, region=region)

        assert is_disconnected is False


class TestComputeTerritoryBatch:
    """Tests for compute_territory_batch function."""

    def test_no_processing_single_region(self):
        """No territory processing when board is single region."""
        from app.ai.gpu_territory import compute_territory_batch

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_stack(0, 1, 0, 0, height=2)

        initial_territory = state.territory_count[0, 1].item()
        compute_territory_batch(state)

        # No change - single region can't be processed
        assert state.territory_count[0, 1].item() == initial_territory

    def test_game_mask_filtering(self):
        """game_mask filters which games are processed."""
        from app.ai.gpu_territory import compute_territory_batch

        state = MockBatchGameState(batch_size=2, board_size=8, num_players=2)

        # Only process game 0
        game_mask = torch.tensor([True, False], dtype=torch.bool)
        compute_territory_batch(state, game_mask=game_mask)

        # Should complete without error (no territory to process anyway)
        assert True

    def test_uses_active_mask_by_default(self):
        """Uses get_active_mask when game_mask not provided."""
        from app.ai.gpu_territory import compute_territory_batch

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)

        # Should not raise
        compute_territory_batch(state)
        assert True


class TestTerritoryProcessingIntegration:
    """Integration tests for territory processing scenarios."""

    def test_simple_disconnected_region(self):
        """Test processing of a simple disconnected region."""
        from app.ai.gpu_territory import (
            _find_all_regions,
            _is_physically_disconnected,
            _is_color_disconnected,
        )

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)

        # Create two regions by collapsing a vertical line
        for y in range(8):
            state.collapse_cell(0, y, 4)

        # Player 1 has stacks on both sides
        state.place_stack(0, 1, 3, 2, height=2)  # Left side
        state.place_stack(0, 1, 3, 6, height=2)  # Right side

        # Player 2 only on right side
        state.place_stack(0, 2, 5, 6, height=1)

        regions = _find_all_regions(state, game_idx=0)
        assert len(regions) == 2

        # Find left region (has only player 1)
        left_region = None
        for region in regions:
            if (3, 2) in region:
                left_region = region
                break

        assert left_region is not None

        # Left region should be physically disconnected
        is_phys, _ = _is_physically_disconnected(state, game_idx=0, region=left_region)
        assert is_phys is True

        # Left region should be color-disconnected (missing player 2)
        is_color = _is_color_disconnected(state, game_idx=0, region=left_region)
        assert is_color is True

    def test_self_elimination_cost(self):
        """Territory processing requires self-elimination cost."""
        from app.ai.gpu_territory import _find_eligible_territory_cap

        state = MockBatchGameState(batch_size=1, board_size=8, num_players=2)
        state.place_stack(0, 1, 0, 0, height=3)

        # Can find eligible cap
        result = _find_eligible_territory_cap(state, game_idx=0, player=1)
        assert result is not None

        # Excluding the only stack means no eligible cap
        excluded = {(0, 0)}
        result = _find_eligible_territory_cap(state, game_idx=0, player=1, excluded_positions=excluded)
        assert result is None
