"""Tests for app.ai.tensor_gumbel_tree module.

Tests the GPU-accelerated Gumbel MCTS tree implementation:
- SearchStats dataclass
- TensorGumbelTree class (Structure of Arrays for GPU)
- add_dirichlet_noise function
- GPUGumbelMCTSConfig and GPUGumbelMCTS
- MultiTreeMCTSConfig and MultiTreeMCTS
- compute_gumbel_completed_q function

Created Dec 2025 as part of comprehensive test coverage improvement.
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.ai.tensor_gumbel_tree import (
    GPUGumbelMCTS,
    GPUGumbelMCTSConfig,
    MultiTreeMCTS,
    MultiTreeMCTSConfig,
    SearchStats,
    TensorGumbelTree,
    add_dirichlet_noise,
    compute_gumbel_completed_q,
)


# =============================================================================
# SearchStats Tests
# =============================================================================


class TestSearchStats:
    """Tests for SearchStats dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating SearchStats with all required fields."""
        stats = SearchStats(
            q_values={"move1": 0.5, "move2": -0.3},
            visit_counts={"move1": 10, "move2": 5},
            search_depth=3,
            uncertainty=0.2,
            prior_policy={"move1": 0.6, "move2": 0.4},
            root_value=0.1,
            nodes_explored=15,
            total_simulations=100,
        )
        assert stats.q_values == {"move1": 0.5, "move2": -0.3}
        assert stats.visit_counts == {"move1": 10, "move2": 5}
        assert stats.search_depth == 3
        assert stats.uncertainty == 0.2
        assert stats.prior_policy == {"move1": 0.6, "move2": 0.4}
        assert stats.root_value == 0.1
        assert stats.nodes_explored == 15
        assert stats.total_simulations == 100

    def test_to_json_dict(self):
        """Test converting SearchStats to JSON-serializable dict."""
        stats = SearchStats(
            q_values={"a": 0.5},
            visit_counts={"a": 10},
            search_depth=2,
            uncertainty=0.1,
            prior_policy={"a": 0.8},
            root_value=0.0,
            nodes_explored=5,
            total_simulations=50,
        )
        result = stats.to_json_dict()

        assert isinstance(result, dict)
        assert result["q_values"] == {"a": 0.5}
        assert result["visit_counts"] == {"a": 10}
        assert result["search_depth"] == 2
        assert result["uncertainty"] == 0.1
        assert result["prior_policy"] == {"a": 0.8}
        assert result["root_value"] == 0.0
        assert result["nodes_explored"] == 5
        assert result["total_simulations"] == 50

    def test_to_json_dict_with_empty_dicts(self):
        """Test to_json_dict handles empty dictionaries."""
        stats = SearchStats(
            q_values={},
            visit_counts={},
            search_depth=0,
            uncertainty=0.0,
            prior_policy={},
            root_value=0.0,
            nodes_explored=1,
            total_simulations=0,
        )
        result = stats.to_json_dict()
        assert result["q_values"] == {}
        assert result["visit_counts"] == {}
        assert result["prior_policy"] == {}


# =============================================================================
# TensorGumbelTree Tests
# =============================================================================


class TestTensorGumbelTree:
    """Tests for TensorGumbelTree class."""

    @pytest.fixture
    def device(self):
        """Use CPU device for tests."""
        return torch.device("cpu")

    @pytest.fixture
    def small_tree(self, device):
        """Create a small tree for testing."""
        return TensorGumbelTree.create(
            batch_size=1,
            max_nodes=16,
            num_actions=8,
            board_height=4,
            board_width=4,
            device=device,
        )

    def test_create_basic(self, device):
        """Test basic tree creation."""
        tree = TensorGumbelTree.create(
            batch_size=1,
            max_nodes=32,
            num_actions=16,
            board_height=8,
            board_width=8,
            device=device,
        )
        assert tree.batch_size == 1
        assert tree.max_nodes == 32
        assert tree.num_actions == 16
        assert tree.board_height == 8
        assert tree.board_width == 8
        assert tree.device == device

    def test_create_with_defaults(self):
        """Test tree creation with default values."""
        tree = TensorGumbelTree.create(device="cpu")
        assert tree.batch_size == 1
        assert tree.max_nodes == 1024
        assert tree.num_actions == 256
        assert tree.board_height == 8
        assert tree.board_width == 8

    def test_create_batch_tree(self, device):
        """Test creating tree for batch of games."""
        tree = TensorGumbelTree.create(
            batch_size=4,
            max_nodes=64,
            num_actions=32,
            device=device,
        )
        assert tree.batch_size == 4
        assert tree.visit_counts.shape == (4, 64, 32)
        assert tree.total_values.shape == (4, 64, 32)
        assert tree.prior_logits.shape == (4, 64, 32)

    def test_tensor_shapes_after_init(self, small_tree):
        """Test that all tensors have correct shapes after initialization."""
        tree = small_tree

        # Tree structure tensors
        assert tree.visit_counts.shape == (1, 16, 8)
        assert tree.total_values.shape == (1, 16, 8)
        assert tree.prior_logits.shape == (1, 16, 8)
        assert tree.parent_idx.shape == (1, 16)
        assert tree.children_idx.shape == (1, 16, 8)
        assert tree.node_depth.shape == (1, 16)
        assert tree.is_expanded.shape == (1, 16)

        # Gumbel-specific
        assert tree.gumbel_noise.shape == (1, 8)
        assert tree.remaining_mask.shape == (1, 8)

        # Sequential Halving state
        assert tree.action_values.shape == (1, 8)
        assert tree.action_visits.shape == (1, 8)

    def test_tensor_dtypes(self, small_tree):
        """Test that tensors have correct dtypes."""
        tree = small_tree

        assert tree.visit_counts.dtype == torch.int32
        assert tree.total_values.dtype == torch.float32
        assert tree.prior_logits.dtype == torch.float32
        assert tree.parent_idx.dtype == torch.int32
        assert tree.children_idx.dtype == torch.int32
        assert tree.node_depth.dtype == torch.int32
        assert tree.is_expanded.dtype == torch.bool
        assert tree.gumbel_noise.dtype == torch.float32
        assert tree.remaining_mask.dtype == torch.bool
        assert tree.action_values.dtype == torch.float32
        assert tree.action_visits.dtype == torch.int32

    def test_initial_values(self, small_tree):
        """Test that tensors are initialized correctly."""
        tree = small_tree

        assert torch.all(tree.visit_counts == 0)
        assert torch.all(tree.total_values == 0)
        assert torch.all(tree.prior_logits == 0)
        assert torch.all(tree.parent_idx == -1)
        assert torch.all(tree.children_idx == -1)
        assert torch.all(tree.node_depth == 0)
        assert torch.all(tree.is_expanded == False)
        assert torch.all(tree.remaining_mask == True)
        assert tree.current_phase == 0
        assert torch.all(tree.next_node_idx == 1)

    def test_reset(self, small_tree):
        """Test resetting tree state."""
        tree = small_tree

        # Modify some values
        tree.visit_counts[0, 0, 0] = 10
        tree.total_values[0, 0, 0] = 5.0
        tree.remaining_mask[0, 0] = False
        tree.current_phase = 3
        tree.next_node_idx[0] = 5

        # Reset
        tree.reset()

        # Verify reset state
        assert torch.all(tree.visit_counts == 0)
        assert torch.all(tree.total_values == 0)
        assert torch.all(tree.remaining_mask == True)
        assert tree.current_phase == 0
        assert torch.all(tree.next_node_idx == 1)

    def test_initialize_root_basic(self, small_tree, device):
        """Test basic root initialization."""
        tree = small_tree
        prior_logits = torch.randn(1, 8, device=device)

        top_k_indices = tree.initialize_root(
            prior_logits=prior_logits,
            num_sampled_actions=4,
            gumbel_scale=1.0,
            use_root_noise=False,
        )

        # Check return shape
        assert top_k_indices.shape == (1, 4)

        # Check root is expanded
        assert tree.is_expanded[0, 0].item() == True

        # Check remaining mask has exactly 4 True values
        assert tree.remaining_mask[0].sum().item() == 4

    def test_initialize_root_with_dirichlet(self, small_tree, device):
        """Test root initialization with Dirichlet noise."""
        tree = small_tree
        prior_logits = torch.randn(1, 8, device=device)

        top_k_indices = tree.initialize_root(
            prior_logits=prior_logits,
            num_sampled_actions=4,
            gumbel_scale=1.0,
            use_root_noise=True,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=0.3,
        )

        # Should still work with noise
        assert top_k_indices.shape == (1, 4)
        assert tree.remaining_mask[0].sum().item() == 4

    def test_initialize_root_respects_num_sampled(self, small_tree, device):
        """Test that num_sampled_actions is respected."""
        tree = small_tree
        prior_logits = torch.randn(1, 8, device=device)

        for k in [2, 4, 6, 8]:
            tree.reset()
            top_k = tree.initialize_root(
                prior_logits=prior_logits,
                num_sampled_actions=k,
                use_root_noise=False,
            )
            assert top_k.shape[1] == k
            assert tree.remaining_mask[0].sum().item() == k

    def test_compute_sequential_halving_budget_single_action(self, small_tree):
        """Test budget computation with single action."""
        budget = small_tree.compute_sequential_halving_budget(100, 1)
        assert budget == [(1, 100)]

    def test_compute_sequential_halving_budget_two_actions(self, small_tree):
        """Test budget computation with two actions."""
        budget = small_tree.compute_sequential_halving_budget(100, 2)
        assert len(budget) == 1
        assert budget[0][0] == 2  # 2 actions in phase 0

    def test_compute_sequential_halving_budget_multiple_phases(self, small_tree):
        """Test budget computation results in halving pattern."""
        budget = small_tree.compute_sequential_halving_budget(160, 16)

        # Should have ~4 phases for 16 actions
        assert len(budget) >= 3

        # Actions should decrease (roughly halve) each phase
        prev_actions = budget[0][0]
        for num_actions, _ in budget[1:]:
            assert num_actions <= prev_actions
            prev_actions = num_actions

    def test_get_remaining_action_indices(self, small_tree, device):
        """Test getting remaining action indices."""
        tree = small_tree
        prior_logits = torch.randn(1, 8, device=device)

        tree.initialize_root(
            prior_logits=prior_logits,
            num_sampled_actions=4,
            use_root_noise=False,
        )

        remaining = tree.get_remaining_action_indices(tree_idx=0)
        assert len(remaining) == 4
        assert remaining.dtype == torch.int64

    def test_update_action_values_single(self, small_tree, device):
        """Test updating action values with single values."""
        tree = small_tree
        action_indices = torch.tensor([0, 2, 5], device=device)
        values = torch.tensor([0.5, -0.3, 0.8], device=device)

        tree.update_action_values(action_indices, values, tree_idx=0)

        assert tree.action_values[0, 0].item() == pytest.approx(0.5)
        assert tree.action_values[0, 2].item() == pytest.approx(-0.3)
        assert tree.action_values[0, 5].item() == pytest.approx(0.8)
        assert tree.action_visits[0, 0].item() == 1
        assert tree.action_visits[0, 2].item() == 1
        assert tree.action_visits[0, 5].item() == 1

    def test_update_action_values_multiple(self, small_tree, device):
        """Test updating action values with multiple values per action."""
        tree = small_tree
        action_indices = torch.tensor([0, 1], device=device)
        values = torch.tensor([[0.5, 0.3], [0.2, -0.1]], device=device)

        tree.update_action_values(action_indices, values, tree_idx=0)

        # Action 0: sum = 0.5 + 0.3 = 0.8, visits = 2
        assert tree.action_values[0, 0].item() == pytest.approx(0.8)
        assert tree.action_visits[0, 0].item() == 2

        # Action 1: sum = 0.2 + (-0.1) = 0.1, visits = 2
        assert tree.action_values[0, 1].item() == pytest.approx(0.1)
        assert tree.action_visits[0, 1].item() == 2

    def test_prune_actions(self, small_tree, device):
        """Test pruning bottom half of actions."""
        tree = small_tree
        prior_logits = torch.randn(1, 8, device=device)

        tree.initialize_root(
            prior_logits=prior_logits,
            num_sampled_actions=4,
            use_root_noise=False,
        )

        # Get remaining actions
        remaining = tree.get_remaining_action_indices(tree_idx=0)

        # Add different values for each action
        for i, idx in enumerate(remaining):
            tree.action_values[0, idx] = float(i)
            tree.action_visits[0, idx] = 1

        initial_count = len(remaining)

        # Prune
        tree.prune_actions(tree_idx=0)

        # Should have roughly half remaining
        new_remaining = tree.get_remaining_action_indices(tree_idx=0)
        assert len(new_remaining) <= initial_count
        assert len(new_remaining) >= 1

    def test_prune_actions_keeps_best(self, small_tree, device):
        """Test that pruning keeps highest-value actions."""
        tree = small_tree

        # Manually set up remaining mask
        tree.remaining_mask.zero_()
        tree.remaining_mask[0, :4] = True  # Actions 0, 1, 2, 3

        # Set values: 3 > 2 > 1 > 0
        tree.action_values[0, 0] = 0.0
        tree.action_values[0, 1] = 1.0
        tree.action_values[0, 2] = 2.0
        tree.action_values[0, 3] = 3.0
        tree.action_visits[0, :4] = 1

        tree.prune_actions(tree_idx=0)

        # Actions 2 and 3 should remain (highest values)
        remaining = tree.get_remaining_action_indices(tree_idx=0)
        assert 3 in remaining.tolist()
        assert 2 in remaining.tolist()

    def test_get_best_action_single_remaining(self, small_tree, device):
        """Test getting best action when only one remains."""
        tree = small_tree

        tree.remaining_mask.zero_()
        tree.remaining_mask[0, 5] = True
        tree.action_values[0, 5] = 0.5
        tree.action_visits[0, 5] = 1

        best = tree.get_best_action(tree_idx=0)
        assert best == 5

    def test_get_best_action_with_visits(self, small_tree, device):
        """Test getting best action based on mean values."""
        tree = small_tree

        tree.remaining_mask.zero_()
        tree.remaining_mask[0, 0] = True
        tree.remaining_mask[0, 1] = True

        # Action 0: total=2.0, visits=4 -> mean=0.5
        tree.action_values[0, 0] = 2.0
        tree.action_visits[0, 0] = 4

        # Action 1: total=3.0, visits=3 -> mean=1.0 (higher)
        tree.action_values[0, 1] = 3.0
        tree.action_visits[0, 1] = 3

        # Set up gumbel and prior for tie-breaking
        tree.gumbel_noise[0, 0] = 0.0
        tree.gumbel_noise[0, 1] = 0.0
        tree.prior_logits[0, 0, 0] = 0.0
        tree.prior_logits[0, 0, 1] = 0.0

        best = tree.get_best_action(tree_idx=0)
        assert best == 1  # Higher mean value

    def test_get_best_action_empty_returns_zero(self, small_tree):
        """Test that empty remaining returns 0 with warning."""
        tree = small_tree
        tree.remaining_mask.zero_()

        best = tree.get_best_action(tree_idx=0)
        assert best == 0

    def test_get_policy_distribution(self, small_tree, device):
        """Test getting policy distribution from visit counts."""
        tree = small_tree

        # Set up some visits
        tree.action_visits[0, 0] = 10
        tree.action_visits[0, 1] = 20
        tree.action_visits[0, 2] = 30
        tree.action_visits[0, 3] = 40

        policy = tree.get_policy_distribution(tree_idx=0)

        assert policy.shape == (8,)
        assert policy.sum().item() == pytest.approx(1.0)
        assert policy[0].item() == pytest.approx(0.1)  # 10/100
        assert policy[1].item() == pytest.approx(0.2)  # 20/100
        assert policy[2].item() == pytest.approx(0.3)  # 30/100
        assert policy[3].item() == pytest.approx(0.4)  # 40/100

    def test_get_policy_distribution_uniform_when_no_visits(self, small_tree):
        """Test uniform policy when no visits."""
        tree = small_tree

        # All remaining (default), no visits
        policy = tree.get_policy_distribution(tree_idx=0)

        assert policy.shape == (8,)
        assert policy.sum().item() == pytest.approx(1.0)
        # Should be uniform
        expected = 1.0 / 8
        for i in range(8):
            assert policy[i].item() == pytest.approx(expected, rel=1e-5)


# =============================================================================
# add_dirichlet_noise Tests
# =============================================================================


class TestAddDirichletNoise:
    """Tests for add_dirichlet_noise function."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_output_shape(self, device):
        """Test that output has same shape as input."""
        policy = torch.ones(4, 16, device=device) / 16
        noised = add_dirichlet_noise(policy, epsilon=0.25, alpha=0.3)
        assert noised.shape == (4, 16)

    def test_output_is_probability_distribution(self, device):
        """Test that output sums to 1."""
        policy = torch.ones(1, 8, device=device) / 8
        noised = add_dirichlet_noise(policy, epsilon=0.25, alpha=0.3)
        assert noised.sum().item() == pytest.approx(1.0, rel=1e-5)

    def test_epsilon_zero_returns_original(self, device):
        """Test that epsilon=0 returns original policy."""
        policy = torch.tensor([[0.1, 0.2, 0.3, 0.4]], device=device)
        noised = add_dirichlet_noise(policy, epsilon=0.0, alpha=0.3)
        assert torch.allclose(noised, policy, atol=1e-5)

    def test_epsilon_one_returns_noise(self, device):
        """Test that epsilon=1 uses only Dirichlet noise."""
        policy = torch.ones(1, 8, device=device) / 8
        noised = add_dirichlet_noise(policy, epsilon=1.0, alpha=0.3)

        # Should still be valid probability
        assert noised.sum().item() == pytest.approx(1.0, rel=1e-5)
        # Should be different from uniform
        # (statistically very unlikely to be exactly uniform)

    def test_different_alphas(self, device):
        """Test that different alpha values work."""
        policy = torch.ones(1, 10, device=device) / 10

        for alpha in [0.03, 0.1, 0.3, 1.0, 3.0]:
            noised = add_dirichlet_noise(policy, epsilon=0.25, alpha=alpha)
            assert noised.shape == (1, 10)
            assert noised.sum().item() == pytest.approx(1.0, rel=1e-5)

    def test_batch_processing(self, device):
        """Test processing multiple policies at once."""
        batch_size = 8
        num_actions = 16
        policy = torch.ones(batch_size, num_actions, device=device) / num_actions

        noised = add_dirichlet_noise(policy, epsilon=0.25, alpha=0.3)

        assert noised.shape == (batch_size, num_actions)
        for i in range(batch_size):
            assert noised[i].sum().item() == pytest.approx(1.0, rel=1e-5)


# =============================================================================
# GPUGumbelMCTSConfig Tests
# =============================================================================


class TestGPUGumbelMCTSConfig:
    """Tests for GPUGumbelMCTSConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GPUGumbelMCTSConfig()
        assert config.num_sampled_actions == 16
        assert config.gumbel_scale == 1.0
        assert config.simulation_budget == 800
        assert config.c_visit == 50.0
        assert config.c_scale == 1.0
        assert config.max_nodes == 1024
        assert config.max_actions == 256
        assert config.max_rollout_depth == 10
        assert config.eval_mode == "heuristic"
        assert config.use_nn_rollout == False
        assert config.use_root_noise == True
        assert config.dirichlet_epsilon == 0.25
        assert config.dirichlet_alpha == 0.3
        assert config.device == "cuda"

    def test_legacy_use_nn_rollout_flag(self):
        """Test that legacy use_nn_rollout flag is converted."""
        config = GPUGumbelMCTSConfig(use_nn_rollout=True)
        assert config.eval_mode == "nn"

    def test_legacy_flag_doesnt_override_explicit_mode(self):
        """Test that explicit eval_mode takes precedence over legacy flag."""
        config = GPUGumbelMCTSConfig(eval_mode="hybrid", use_nn_rollout=True)
        assert config.eval_mode == "hybrid"

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = GPUGumbelMCTSConfig(
            num_sampled_actions=8,
            simulation_budget=100,
            eval_mode="hybrid",
            device="cpu",
        )
        assert config.num_sampled_actions == 8
        assert config.simulation_budget == 100
        assert config.eval_mode == "hybrid"
        assert config.device == "cpu"


# =============================================================================
# compute_gumbel_completed_q Tests
# =============================================================================


class TestComputeGumbelCompletedQ:
    """Tests for compute_gumbel_completed_q function."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_basic_computation(self, device):
        """Test basic Q-value computation."""
        values = torch.tensor([[2.0, 3.0, 1.0]], device=device)
        visits = torch.tensor([[2, 3, 1]], device=device)
        gumbel = torch.zeros(1, 3, device=device)
        prior = torch.zeros(1, 3, device=device)

        result = compute_gumbel_completed_q(values, visits, gumbel, prior)

        assert result.shape == (1, 3)

    def test_zero_visits_handling(self, device):
        """Test handling of zero visits."""
        values = torch.tensor([[0.0, 0.0]], device=device)
        visits = torch.tensor([[0, 1]], device=device)
        gumbel = torch.zeros(1, 2, device=device)
        prior = torch.zeros(1, 2, device=device)

        result = compute_gumbel_completed_q(values, visits, gumbel, prior)

        # Should not crash with zero visits
        assert result.shape == (1, 2)
        assert torch.isfinite(result).all()

    def test_gumbel_noise_affects_result(self, device):
        """Test that Gumbel noise affects the result."""
        values = torch.tensor([[1.0, 1.0]], device=device)
        visits = torch.tensor([[1, 1]], device=device)
        prior = torch.zeros(1, 2, device=device)

        gumbel1 = torch.tensor([[0.5, 0.0]], device=device)
        gumbel2 = torch.tensor([[0.0, 0.5]], device=device)

        result1 = compute_gumbel_completed_q(values, visits, gumbel1, prior)
        result2 = compute_gumbel_completed_q(values, visits, gumbel2, prior)

        # Different Gumbel noise should give different rankings
        assert result1[0, 0] > result1[0, 1]  # First action higher with positive gumbel
        assert result2[0, 1] > result2[0, 0]  # Second action higher

    def test_batch_processing(self, device):
        """Test batch processing."""
        batch_size = 4
        num_actions = 8

        values = torch.randn(batch_size, num_actions, device=device)
        visits = torch.randint(1, 10, (batch_size, num_actions), device=device)
        gumbel = torch.randn(batch_size, num_actions, device=device)
        prior = torch.randn(batch_size, num_actions, device=device)

        result = compute_gumbel_completed_q(values, visits, gumbel, prior)

        assert result.shape == (batch_size, num_actions)


# =============================================================================
# MultiTreeMCTSConfig Tests
# =============================================================================


class TestMultiTreeMCTSConfig:
    """Tests for MultiTreeMCTSConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MultiTreeMCTSConfig()
        assert config.num_sampled_actions == 16
        assert config.gumbel_scale == 1.0
        assert config.simulation_budget == 64
        assert config.c_visit == 50.0
        assert config.c_scale == 1.0
        assert config.max_nodes == 256
        assert config.max_actions == 256
        assert config.max_rollout_depth == 10
        assert config.eval_mode == "heuristic"
        assert config.use_root_noise == True
        assert config.dirichlet_epsilon == 0.25
        assert config.dirichlet_alpha == 0.3
        assert config.device == "cuda"

    def test_lower_budget_than_single(self):
        """Verify MultiTree uses lower budget (designed for batch efficiency)."""
        single = GPUGumbelMCTSConfig()
        multi = MultiTreeMCTSConfig()
        assert multi.simulation_budget < single.simulation_budget

    def test_smaller_tree_size(self):
        """Verify MultiTree uses smaller trees."""
        single = GPUGumbelMCTSConfig()
        multi = MultiTreeMCTSConfig()
        assert multi.max_nodes < single.max_nodes


# =============================================================================
# GPUGumbelMCTS Tests
# =============================================================================


class TestGPUGumbelMCTS:
    """Tests for GPUGumbelMCTS class."""

    @pytest.fixture
    def config(self):
        """Create CPU config for testing."""
        return GPUGumbelMCTSConfig(
            device="cpu",
            simulation_budget=4,
            num_sampled_actions=4,
            max_nodes=32,
        )

    @pytest.fixture
    def mcts(self, config):
        """Create MCTS searcher for testing."""
        return GPUGumbelMCTS(config)

    def test_initialization(self, config):
        """Test MCTS initialization."""
        mcts = GPUGumbelMCTS(config)
        assert mcts.config == config
        assert mcts.device == torch.device("cpu")
        assert mcts.tree is None

    def test_move_to_key_with_position(self, mcts):
        """Test move to key conversion."""
        # Create mock move with positions
        move = MagicMock()
        move.type = MagicMock()
        move.type.value = "move_ring"
        move.from_pos = MagicMock()
        move.from_pos.x = 2
        move.from_pos.y = 3
        move.to = MagicMock()
        move.to.x = 4
        move.to.y = 5
        move.placement_count = None

        key = mcts._move_to_key(move)
        assert key == "move_ring_2,3_4,5"

    def test_move_to_key_without_from(self, mcts):
        """Test move to key with no from position."""
        move = MagicMock()
        move.type = MagicMock()
        move.type.value = "place_ring"
        move.from_pos = None
        move.to = MagicMock()
        move.to.x = 4
        move.to.y = 5
        move.placement_count = 3

        key = mcts._move_to_key(move)
        assert key == "place_ring_none_4,5_3"

    def test_move_to_key_with_id(self, mcts):
        """Test move to key with move ID."""
        # Create a mock that doesn't have 'type' or 'to' attributes
        # so it falls through to the ID branch
        move = MagicMock(spec=["id"])
        move.id = "unique_move_id"

        key = mcts._move_to_key(move)
        assert key == "unique_move_id"

    def test_move_to_key_simulated(self, mcts):
        """Test move to key with simulated ID falls through to str()."""
        # Simulated moves have id='simulated' which should be skipped
        move = MagicMock(spec=["id"])
        move.id = "simulated"

        key = mcts._move_to_key(move)
        # Should call str(move) since 'simulated' ID is skipped
        assert "MagicMock" in key or "simulated" not in key or key != "simulated"

    def test_move_to_key_fallback_to_str(self, mcts):
        """Test move to key falls back to str() for unknown moves."""
        # Move with no recognized attributes
        move = MagicMock(spec=[])

        key = mcts._move_to_key(move)
        # Should be string representation
        assert isinstance(key, str)

    def test_find_phase_valid_move(self, mcts):
        """Test finding phase-valid fallback move."""
        from app.models import GamePhase, MoveType

        # Create mock game state
        game_state = MagicMock()
        game_state.current_phase = GamePhase.RING_PLACEMENT

        # Create moves with different types
        move1 = MagicMock()
        move1.type = MoveType.MOVE_RING  # Invalid for RING_PLACEMENT

        move2 = MagicMock()
        move2.type = MoveType.PLACE_RING  # Valid for RING_PLACEMENT

        moves = [move1, move2]

        result = mcts._find_phase_valid_move(game_state, moves)
        assert result == move2

    def test_find_phase_valid_move_returns_none_when_no_valid(self, mcts):
        """Test returns None when no phase-valid move exists."""
        from app.models import GamePhase, MoveType

        game_state = MagicMock()
        game_state.current_phase = GamePhase.RING_PLACEMENT

        move = MagicMock()
        move.type = MoveType.MOVE_RING  # Invalid for RING_PLACEMENT

        result = mcts._find_phase_valid_move(game_state, [move])
        assert result is None


# =============================================================================
# MultiTreeMCTS Tests
# =============================================================================


class TestMultiTreeMCTS:
    """Tests for MultiTreeMCTS class."""

    @pytest.fixture
    def config(self):
        """Create CPU config for testing."""
        return MultiTreeMCTSConfig(
            device="cpu",
            simulation_budget=4,
            num_sampled_actions=4,
            max_nodes=16,
        )

    @pytest.fixture
    def mcts(self, config):
        """Create MultiTreeMCTS searcher for testing."""
        return MultiTreeMCTS(config)

    def test_initialization(self, config):
        """Test MultiTreeMCTS initialization."""
        mcts = MultiTreeMCTS(config)
        assert mcts.config == config
        assert mcts.device == torch.device("cpu")
        assert mcts.tree is None

    def test_search_batch_empty_states(self, mcts):
        """Test search_batch with empty list."""
        moves, policies = mcts.search_batch([], None)
        assert moves == []
        assert policies == []

    def test_move_to_key(self, mcts):
        """Test move to key conversion."""
        move = MagicMock()
        move.type = MagicMock()
        move.type.value = "move_stack"
        move.from_pos = MagicMock()
        move.from_pos.x = 1
        move.from_pos.y = 2
        move.to = MagicMock()
        move.to.x = 3
        move.to.y = 4
        move.placement_count = None

        key = mcts._move_to_key(move)
        assert key == "move_stack_1,2_3,4"


# =============================================================================
# Integration-style Tests (with mocking)
# =============================================================================


class TestIntegration:
    """Integration-style tests with mocked dependencies."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_full_tree_lifecycle(self, device):
        """Test complete tree usage lifecycle."""
        tree = TensorGumbelTree.create(
            batch_size=1,
            max_nodes=32,
            num_actions=8,
            device=device,
        )

        # Initialize root
        prior = torch.randn(1, 8, device=device)
        top_k = tree.initialize_root(prior, num_sampled_actions=4, use_root_noise=False)

        assert tree.is_expanded[0, 0].item()
        assert len(top_k[0]) == 4

        # Simulate some action values
        remaining = tree.get_remaining_action_indices()
        for idx in remaining:
            tree.action_values[0, idx] = torch.rand(1).item()
            tree.action_visits[0, idx] = 1

        # Prune
        tree.prune_actions()
        new_remaining = tree.get_remaining_action_indices()
        assert len(new_remaining) <= len(remaining)

        # Get best action
        best = tree.get_best_action()
        assert best in new_remaining.tolist()

        # Get policy
        policy = tree.get_policy_distribution()
        assert policy.sum().item() == pytest.approx(1.0, rel=1e-5)

    def test_sequential_halving_budget_coverage(self, device):
        """Test that sequential halving covers all simulations."""
        tree = TensorGumbelTree.create(device=device)

        for num_actions in [2, 4, 8, 16, 32]:
            for budget in [16, 64, 100, 800]:
                phases = tree.compute_sequential_halving_budget(budget, num_actions)

                # Verify phase structure
                assert len(phases) >= 1
                assert phases[0][0] == num_actions  # First phase has all actions

                # Verify last phase has at least 1 action
                assert phases[-1][0] >= 1
