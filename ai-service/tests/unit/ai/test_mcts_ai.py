"""Tests for mcts_ai.py - Monte Carlo Tree Search AI implementation.

This module tests the core MCTS components including:
- MCTSNode (legacy immutable search node)
- MCTSNodeLite (lightweight incremental search node)
- DynamicBatchSizer (memory-aware batch sizing)
- MCTSAI (main MCTS AI class)
- Helper functions (_pos_key, _move_key, _moves_match, etc.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.ai.mcts_ai import (
    DynamicBatchSizer,
    MCTSNode,
    MCTSNodeLite,
    _move_key,
    _moves_match,
    _pos_key,
    _pos_seq_key,
)
from app.models import AIConfig, BoardType, GamePhase, GameState, Move, MoveType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_game_state():
    """Create a minimal game state for testing."""
    mock_state = MagicMock(spec=GameState)
    mock_state.current_player = 1
    mock_state.phase = GamePhase.RING_PLACEMENT
    mock_state.game_over = False
    mock_state.winner = None
    mock_state.board = MagicMock()
    mock_state.board.type = BoardType.HEX8
    mock_state.players = [MagicMock(), MagicMock()]
    return mock_state


@pytest.fixture
def sample_move():
    """Create a sample move for testing."""
    move = MagicMock(spec=Move)
    move.type = MoveType.PLACE_RING
    move.player = 1
    move.from_pos = None
    move.to = MagicMock()
    move.to.x = 4
    move.to.y = 4
    move.to.z = None
    move.to.to_key = MagicMock(return_value="4,4")
    move.capture_target = None
    move.placement_count = 1
    move.placed_on_stack = None
    move.line_index = None
    move.collapsed_markers = None
    move.collapse_positions = None
    move.extraction_stacks = None
    move.recovery_option = None
    move.recovery_mode = None
    move.elimination_context = None
    move.capture_chain = None
    move.overtaken_rings = None
    return move


@pytest.fixture
def sample_moves(sample_move):
    """Create a list of sample moves."""
    moves = []
    for i in range(3):
        move = MagicMock(spec=Move)
        move.type = MoveType.PLACE_RING
        move.player = 1
        move.from_pos = None
        move.to = MagicMock()
        move.to.x = i
        move.to.y = i
        move.to.z = None
        move.to.to_key = MagicMock(return_value=f"{i},{i}")
        move.capture_target = None
        move.placement_count = 1
        move.placed_on_stack = None
        move.line_index = None
        move.collapsed_markers = None
        move.collapse_positions = None
        move.extraction_stacks = None
        move.recovery_option = None
        move.recovery_mode = None
        move.elimination_context = None
        move.capture_chain = None
        move.overtaken_rings = None
        moves.append(move)
    return moves


@pytest.fixture
def ai_config():
    """Create a minimal AI config for testing."""
    config = MagicMock(spec=AIConfig)
    config.difficulty = 5
    config.board_type = BoardType.HEX8
    config.num_players = 2
    config.think_time = 1000
    config.use_incremental_search = True
    config.use_neural_net = False
    return config


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestPosKey:
    """Tests for _pos_key helper function."""

    def test_pos_key_with_none(self):
        """Test that None position returns None."""
        assert _pos_key(None) is None

    def test_pos_key_with_to_key_method(self):
        """Test position with to_key() method."""
        pos = MagicMock()
        pos.to_key = MagicMock(return_value="5,3")
        assert _pos_key(pos) == "5,3"

    def test_pos_key_with_xy_attributes(self):
        """Test position with x,y attributes."""
        pos = MagicMock(spec=[])  # Empty spec to avoid to_key
        pos.x = 4
        pos.y = 7
        pos.z = None
        # Remove to_key if present
        if hasattr(pos, 'to_key'):
            delattr(pos, 'to_key')
        assert _pos_key(pos) == "4,7"

    def test_pos_key_with_xyz_attributes(self):
        """Test position with x,y,z attributes."""
        @dataclass
        class Pos3D:
            x: int
            y: int
            z: int

        pos = Pos3D(x=1, y=2, z=3)
        assert _pos_key(pos) == "1,2,3"

    def test_pos_key_with_missing_coords(self):
        """Test position with missing coordinates returns None."""
        pos = MagicMock(spec=[])
        pos.x = 5
        # No y attribute
        assert _pos_key(pos) is None


class TestPosSeqKey:
    """Tests for _pos_seq_key helper function."""

    def test_pos_seq_key_with_none(self):
        """Test None sequence returns None."""
        assert _pos_seq_key(None) is None

    def test_pos_seq_key_with_empty_tuple(self):
        """Test empty tuple returns None."""
        assert _pos_seq_key(()) is None

    def test_pos_seq_key_with_positions(self):
        """Test sequence of positions."""
        pos1 = MagicMock()
        pos1.to_key = MagicMock(return_value="1,1")
        pos2 = MagicMock()
        pos2.to_key = MagicMock(return_value="2,2")

        result = _pos_seq_key((pos1, pos2))
        assert result == ("1,1", "2,2")

    def test_pos_seq_key_filters_none(self):
        """Test that None positions are filtered."""
        pos1 = MagicMock()
        pos1.to_key = MagicMock(return_value="1,1")

        result = _pos_seq_key((pos1, None))
        assert result == ("1,1",)


class TestMoveKey:
    """Tests for _move_key helper function."""

    def test_move_key_basic(self, sample_move):
        """Test basic move key generation."""
        key = _move_key(sample_move)
        assert isinstance(key, tuple)
        # Should contain move type, player, and various position keys
        assert key[0] == MoveType.PLACE_RING.value
        assert key[1] == 1  # player

    def test_move_key_different_moves_differ(self, sample_move):
        """Test that different moves have different keys."""
        key1 = _move_key(sample_move)

        # Create a different move
        other_move = MagicMock(spec=Move)
        other_move.type = MoveType.MOVE_RING
        other_move.player = 2
        other_move.from_pos = None
        other_move.to = None
        other_move.capture_target = None
        other_move.placement_count = None
        other_move.placed_on_stack = None
        other_move.line_index = None
        other_move.collapsed_markers = None
        other_move.collapse_positions = None
        other_move.extraction_stacks = None
        other_move.recovery_option = None
        other_move.recovery_mode = None
        other_move.elimination_context = None
        other_move.capture_chain = None
        other_move.overtaken_rings = None

        key2 = _move_key(other_move)
        assert key1 != key2


class TestMovesMatch:
    """Tests for _moves_match helper function."""

    def test_moves_match_identical(self, sample_move):
        """Test that identical moves match."""
        # Create two identical moves
        m1 = sample_move
        m2 = MagicMock(spec=Move)
        m2.type = m1.type
        m2.player = m1.player
        m2.from_pos = m1.from_pos
        m2.to = m1.to
        m2.capture_target = m1.capture_target
        m2.placement_count = m1.placement_count
        m2.placed_on_stack = m1.placed_on_stack
        m2.line_index = m1.line_index
        m2.collapsed_markers = m1.collapsed_markers
        m2.collapse_positions = m1.collapse_positions
        m2.extraction_stacks = m1.extraction_stacks
        m2.recovery_option = m1.recovery_option
        m2.recovery_mode = m1.recovery_mode
        m2.elimination_context = m1.elimination_context
        m2.capture_chain = m1.capture_chain
        m2.overtaken_rings = m1.overtaken_rings

        assert _moves_match(m1, m2)

    def test_moves_match_different_type(self, sample_move):
        """Test that moves with different types don't match."""
        m1 = sample_move
        m2 = MagicMock(spec=Move)
        m2.type = MoveType.MOVE_RING  # Different type
        m2.player = m1.player

        assert not _moves_match(m1, m2)

    def test_moves_match_different_player(self, sample_move):
        """Test that moves with different players don't match."""
        m1 = sample_move
        m2 = MagicMock(spec=Move)
        m2.type = m1.type
        m2.player = 2  # Different player

        assert not _moves_match(m1, m2)


# =============================================================================
# Test MCTSNode (Legacy)
# =============================================================================


class TestMCTSNode:
    """Tests for MCTSNode (legacy immutable search node)."""

    def test_init_default(self, minimal_game_state):
        """Test default initialization."""
        node = MCTSNode(minimal_game_state)

        assert node.game_state == minimal_game_state
        assert node.parent is None
        assert node.move is None
        assert node.children == []
        assert node.wins == 0
        assert node.visits == 0
        assert node.amaf_wins == 0
        assert node.amaf_visits == 0
        assert node.untried_moves == []
        assert node.prior == 0.0
        assert node.to_move_is_root is True

    def test_init_with_parent_and_move(self, minimal_game_state, sample_move):
        """Test initialization with parent and move."""
        parent = MCTSNode(minimal_game_state)
        child = MCTSNode(minimal_game_state, parent=parent, move=sample_move)

        assert child.parent == parent
        assert child.move == sample_move

    def test_add_child(self, minimal_game_state, sample_move):
        """Test adding a child node."""
        parent = MCTSNode(minimal_game_state)
        parent.untried_moves = [sample_move]

        child = parent.add_child(sample_move, minimal_game_state, prior=0.5)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent
        assert child.move == sample_move
        assert child.prior == 0.5
        assert sample_move not in parent.untried_moves

    def test_update(self, minimal_game_state):
        """Test updating node stats."""
        node = MCTSNode(minimal_game_state)

        node.update(1.0)

        assert node.visits == 1
        assert node.wins == 1.0

        node.update(0.5)
        assert node.visits == 2
        assert node.wins == 1.5

    def test_update_with_amaf(self, minimal_game_state, sample_move):
        """Test AMAF updates when move was played."""
        parent = MCTSNode(minimal_game_state)
        parent.untried_moves = [sample_move]
        child = parent.add_child(sample_move, minimal_game_state)

        # Update with the child's move in played_moves
        child.update(1.0, played_moves=[sample_move])

        assert child.amaf_visits == 1
        assert child.amaf_wins == 1.0

    def test_uct_select_child_basic(self, minimal_game_state, sample_moves):
        """Test UCT child selection."""
        parent = MCTSNode(minimal_game_state)
        parent.visits = 10

        # Add children with different visit/win counts
        for i, move in enumerate(sample_moves):
            parent.untried_moves = [move]
            child = parent.add_child(move, minimal_game_state, prior=0.3)
            child.visits = i + 1
            child.wins = (i + 1) * 0.5

        # Should select based on PUCT formula
        selected = parent.uct_select_child(c_puct=1.0, rave_k=0.0)
        assert selected in parent.children


# =============================================================================
# Test MCTSNodeLite (Incremental)
# =============================================================================


class TestMCTSNodeLite:
    """Tests for MCTSNodeLite (lightweight incremental search node)."""

    def test_init_default(self):
        """Test default initialization."""
        node = MCTSNodeLite()

        assert node.parent is None
        assert node.move is None
        assert node.children == []
        assert node.wins == 0.0
        assert node.visits == 0
        assert node.amaf_wins == 0.0
        assert node.amaf_visits == 0
        assert node.untried_moves == []
        assert node.prior == 0.0
        assert node.to_move_is_root is True

    def test_init_with_parent_and_move(self, sample_move):
        """Test initialization with parent and move."""
        parent = MCTSNodeLite()
        child = MCTSNodeLite(parent=parent, move=sample_move, to_move_is_root=False)

        assert child.parent == parent
        assert child.move == sample_move
        assert child.to_move_is_root is False

    def test_is_leaf(self):
        """Test is_leaf property."""
        node = MCTSNodeLite()
        assert node.is_leaf() is True

        node.children.append(MCTSNodeLite(parent=node))
        assert node.is_leaf() is False

    def test_is_fully_expanded(self, sample_moves):
        """Test is_fully_expanded property."""
        node = MCTSNodeLite()
        node.untried_moves = list(sample_moves)

        assert node.is_fully_expanded() is False

        node.untried_moves = []
        assert node.is_fully_expanded() is True

    def test_add_child(self, sample_move):
        """Test adding a child node."""
        parent = MCTSNodeLite()
        parent.untried_moves = [sample_move]

        child = parent.add_child(sample_move, prior=0.8, to_move_is_root=False)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent
        assert child.move == sample_move
        assert child.prior == 0.8
        assert child.to_move_is_root is False
        assert sample_move not in parent.untried_moves

    def test_update(self):
        """Test updating node stats."""
        node = MCTSNodeLite()

        node.update(1.0)

        assert node.visits == 1
        assert node.wins == 1.0

        node.update(-0.5)
        assert node.visits == 2
        assert node.wins == 0.5

    def test_uct_select_child_basic(self, sample_moves):
        """Test UCT child selection."""
        parent = MCTSNodeLite()
        parent.visits = 10

        # Add children with different visit/win counts
        for i, move in enumerate(sample_moves):
            parent.untried_moves = [move]
            child = parent.add_child(move, prior=0.3)
            child.visits = i + 1
            child.wins = (i + 1) * 0.5

        # Should select based on PUCT formula
        selected = parent.uct_select_child(c_puct=1.0, rave_k=0.0)
        assert selected in parent.children

    def test_uct_select_child_with_rave(self, sample_moves):
        """Test UCT child selection with RAVE enabled."""
        parent = MCTSNodeLite()
        parent.visits = 50

        # Add children with AMAF stats
        for i, move in enumerate(sample_moves):
            parent.untried_moves = [move]
            child = parent.add_child(move, prior=0.3)
            child.visits = 5
            child.wins = 2.5
            child.amaf_visits = 10
            child.amaf_wins = i + 1  # Different AMAF scores

        # With RAVE enabled, should consider AMAF values
        selected = parent.uct_select_child(c_puct=1.0, rave_k=1000.0)
        assert selected in parent.children

    def test_uct_select_child_with_fpu(self, sample_moves):
        """Test UCT child selection with FPU reduction."""
        parent = MCTSNodeLite()
        parent.visits = 10
        parent.wins = 5.0  # 50% win rate

        # Add children, one unexplored
        for i, move in enumerate(sample_moves):
            parent.untried_moves = [move]
            child = parent.add_child(move, prior=0.3)
            if i > 0:
                child.visits = 3
                child.wins = 1.5
            # First child has 0 visits

        # With FPU reduction, unexplored child should have reduced value
        selected = parent.uct_select_child(c_puct=1.0, rave_k=0.0, fpu_reduction=0.1)
        assert selected in parent.children


# =============================================================================
# Test DynamicBatchSizer
# =============================================================================


class TestDynamicBatchSizer:
    """Tests for DynamicBatchSizer."""

    def test_init_default(self):
        """Test default initialization."""
        from app.utils.memory_config import MemoryConfig

        config = MemoryConfig()
        sizer = DynamicBatchSizer(memory_config=config)

        # Uses actual attribute names from the class
        assert sizer.batch_size_min == 100
        assert sizer.batch_size_max == 1600
        assert sizer.memory_safety_margin == 0.8

    def test_init_custom(self):
        """Test custom initialization."""
        from app.utils.memory_config import MemoryConfig

        config = MemoryConfig()
        sizer = DynamicBatchSizer(
            memory_config=config,
            batch_size_min=16,
            batch_size_max=512,
            memory_safety_margin=0.90,
        )

        assert sizer.batch_size_min == 16
        assert sizer.batch_size_max == 512
        assert sizer.memory_safety_margin == 0.90

    def test_get_optimal_batch_size_returns_valid_range(self):
        """Test that get_optimal_batch_size returns value in valid range."""
        from app.utils.memory_config import MemoryConfig

        config = MemoryConfig()
        sizer = DynamicBatchSizer(
            memory_config=config,
            batch_size_min=8,
            batch_size_max=256,
        )

        batch_size = sizer.get_optimal_batch_size()
        assert sizer.batch_size_min <= batch_size <= sizer.batch_size_max

    def test_get_optimal_batch_size_with_current_nodes(self):
        """Test batch size calculation with existing node count."""
        from app.utils.memory_config import MemoryConfig

        config = MemoryConfig()
        sizer = DynamicBatchSizer(
            memory_config=config,
            batch_size_min=8,
            batch_size_max=256,
        )

        # With current nodes, should still return valid range
        batch_size = sizer.get_optimal_batch_size(current_node_count=1000)
        assert batch_size >= sizer.batch_size_min


# =============================================================================
# Test MCTSAI
# =============================================================================


class TestMCTSAI:
    """Tests for MCTSAI main class."""

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_init_basic(self, mock_heuristic_init, ai_config):
        """Test basic initialization without neural network."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)

        assert ai.use_incremental_search is True
        assert ai.neural_net is None  # Not loaded due to difficulty < 6

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_init_with_high_difficulty(self, mock_heuristic_init, ai_config):
        """Test initialization with high difficulty (neural MCTS)."""
        mock_heuristic_init.return_value = None
        ai_config.difficulty = 6  # Neural MCTS tier

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        ai.config = ai_config  # Set config manually after mock init
        # At D6+, MCTS should attempt to use neural network guidance
        # (may fall back to heuristic if model not available)

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_normalized_entropy_empty(self, mock_heuristic_init, ai_config):
        """Test normalized entropy with empty list."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        assert ai._normalized_entropy([]) == 0.0

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_normalized_entropy_single(self, mock_heuristic_init, ai_config):
        """Test normalized entropy with single value."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        assert ai._normalized_entropy([1.0]) == 0.0

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_normalized_entropy_uniform(self, mock_heuristic_init, ai_config):
        """Test normalized entropy with uniform distribution."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        # Uniform distribution should have maximum entropy = 1.0
        entropy = ai._normalized_entropy([0.25, 0.25, 0.25, 0.25])
        assert entropy == pytest.approx(1.0, rel=1e-3)

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_normalized_entropy_peaked(self, mock_heuristic_init, ai_config):
        """Test normalized entropy with peaked distribution."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        # Highly peaked distribution should have low entropy
        entropy = ai._normalized_entropy([0.99, 0.005, 0.005])
        assert entropy < 0.3

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_dynamic_c_puct(self, mock_heuristic_init, ai_config):
        """Test dynamic PUCT calculation."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)

        # With 0 visits and uniform priors
        c_puct = ai._dynamic_c_puct(0, [0.25, 0.25, 0.25, 0.25])
        assert 0.25 <= c_puct <= 4.0

        # With more visits
        c_puct_high = ai._dynamic_c_puct(100, [0.25, 0.25, 0.25, 0.25])
        assert 0.25 <= c_puct_high <= 4.0

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_rave_k_for_node(self, mock_heuristic_init, ai_config):
        """Test RAVE k calculation."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        ai.config = ai_config  # Set config manually after mock init

        rave_k = ai._rave_k_for_node(0, [0.25, 0.25, 0.25, 0.25])
        assert rave_k >= 0.0

        # Higher visits should reduce RAVE influence
        rave_k_high = ai._rave_k_for_node(500, [0.25, 0.25, 0.25, 0.25])
        assert rave_k_high <= rave_k

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_fpu_reduction_for_phase(self, mock_heuristic_init, ai_config):
        """Test FPU reduction values per phase."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)

        # Each phase should have a specific FPU reduction
        assert ai._fpu_reduction_for_phase(GamePhase.RING_PLACEMENT) == 0.05
        assert ai._fpu_reduction_for_phase(GamePhase.MOVEMENT) == 0.10
        assert ai._fpu_reduction_for_phase(GamePhase.CAPTURE) == 0.12
        assert ai._fpu_reduction_for_phase(GamePhase.LINE_PROCESSING) == 0.16
        assert ai._fpu_reduction_for_phase(GamePhase.TERRITORY_PROCESSING) == 0.20

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_get_visit_distribution_no_search(self, mock_heuristic_init, ai_config):
        """Test visit distribution with no search performed."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        ai._training_root = None
        ai._training_root_lite = None

        moves, probs = ai.get_visit_distribution()
        assert moves == []
        assert probs == []

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_get_visit_distribution_legacy(self, mock_heuristic_init, ai_config, minimal_game_state, sample_moves):
        """Test visit distribution from legacy root."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        ai._training_root_lite = None

        # Create a legacy root with children
        root = MCTSNode(minimal_game_state)
        root.visits = 30

        for i, move in enumerate(sample_moves):
            root.untried_moves = [move]
            child = root.add_child(move, minimal_game_state)
            child.visits = (i + 1) * 5  # 5, 10, 15 visits

        ai._training_root = root

        moves, probs = ai.get_visit_distribution()
        assert len(moves) == 3
        assert len(probs) == 3
        assert sum(probs) == pytest.approx(1.0)

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_get_visit_distribution_lite(self, mock_heuristic_init, ai_config, sample_moves):
        """Test visit distribution from incremental root."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        ai._training_root = None

        # Create a lite root with children
        root = MCTSNodeLite()
        root.visits = 30

        for i, move in enumerate(sample_moves):
            root.untried_moves = [move]
            child = root.add_child(move)
            child.visits = (i + 1) * 5  # 5, 10, 15 visits

        ai._training_root_lite = root

        moves, probs = ai.get_visit_distribution()
        assert len(moves) == 3
        assert len(probs) == 3
        assert sum(probs) == pytest.approx(1.0)

    @patch("app.ai.mcts_ai.HeuristicAI.__init__")
    def test_puct_params_for_node(self, mock_heuristic_init, ai_config):
        """Test PUCT parameter calculation for a node."""
        mock_heuristic_init.return_value = None

        from app.ai.mcts_ai import MCTSAI

        ai = MCTSAI(player_number=1, config=ai_config)
        ai.config = ai_config  # Set config manually after mock init

        # Create a node with children
        node = MCTSNodeLite()
        node.visits = 50
        for _ in range(3):
            child = MCTSNodeLite(parent=node)
            child.prior = 0.3
            node.children.append(child)

        c_puct, rave_k, fpu_reduction = ai._puct_params_for_node(
            node, GamePhase.MOVEMENT
        )

        assert 0.25 <= c_puct <= 4.0
        assert rave_k >= 0.0
        assert fpu_reduction == 0.10  # MOVEMENT phase


# =============================================================================
# Test Edge Cases and Integration
# =============================================================================


class TestMCTSEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_node_with_all_losses(self, minimal_game_state, sample_moves):
        """Test node behavior when all simulations are losses."""
        root = MCTSNode(minimal_game_state)
        root.visits = 30
        root.wins = 0  # All losses

        for move in sample_moves:
            root.untried_moves = [move]
            child = root.add_child(move, minimal_game_state)
            child.visits = 10
            child.wins = 0  # All losses

        # Should still be able to select a child
        selected = root.uct_select_child(c_puct=1.0, rave_k=0.0)
        assert selected in root.children

    def test_node_with_single_child(self, minimal_game_state, sample_move):
        """Test node with only one child."""
        root = MCTSNode(minimal_game_state)
        root.visits = 10
        root.untried_moves = [sample_move]

        child = root.add_child(sample_move, minimal_game_state)
        child.visits = 5
        child.wins = 3.0

        # Should return the only child
        selected = root.uct_select_child(c_puct=1.0, rave_k=0.0)
        assert selected == child

    def test_lite_node_memory_efficiency(self, sample_moves):
        """Test that MCTSNodeLite uses __slots__ for memory efficiency."""
        # __slots__ prevents adding arbitrary attributes
        node = MCTSNodeLite()

        # These should work (defined in __slots__)
        node.visits = 10
        node.wins = 5.0
        node.prior = 0.5

        # This should raise AttributeError due to __slots__
        with pytest.raises(AttributeError):
            node.arbitrary_attribute = "test"

    def test_node_backpropagation(self, minimal_game_state, sample_moves):
        """Test backpropagation of values through the tree."""
        # Build a small tree
        root = MCTSNode(minimal_game_state)

        root.untried_moves = sample_moves.copy()
        child = root.add_child(sample_moves[0], minimal_game_state)

        child.untried_moves = sample_moves[1:].copy()
        grandchild = child.add_child(sample_moves[1], minimal_game_state)

        # Simulate backpropagation
        result = 1.0
        node = grandchild
        while node is not None:
            node.update(result)
            node = node.parent
            result = -result  # Alternate perspective

        # Check values propagated
        assert grandchild.visits == 1
        assert grandchild.wins == 1.0
        assert child.visits == 1
        assert child.wins == -1.0  # Opponent's perspective
        assert root.visits == 1
        assert root.wins == 1.0  # Our perspective


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
