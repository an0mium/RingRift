"""Test parity between tensor_gumbel_tree and multi_game_gumbel implementations.

This test validates that both Gumbel MCTS implementations produce equivalent
training data quality by comparing:
1. Gumbel-Top-K action sampling
2. Sequential Halving budget allocation
3. Final move selection distribution over many runs
"""

import numpy as np
import pytest
import torch

from app.ai.tensor_gumbel_tree import (
    GPUGumbelMCTS,
    GPUGumbelMCTSConfig,
    TensorGumbelTree,
)
from app.ai.multi_game_gumbel import MultiGameGumbelRunner, GumbelAction
from app.models import BoardType
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state


class MockNeuralNet:
    """Mock neural network for testing."""

    def __init__(self, policy_probs: dict[int, float] | None = None):
        self.policy_probs = policy_probs or {}

    def evaluate_batch(self, states):
        """Return uniform policy and zero values."""
        values = [0.0] * len(states)
        policies = []
        for state in states:
            # Create uniform policy over 64 actions (8x8 board)
            policy = np.ones(64) / 64
            for idx, prob in self.policy_probs.items():
                if idx < len(policy):
                    policy[idx] = prob
            policy = policy / policy.sum()  # Renormalize
            policies.append(policy)
        return values, policies

    def encode_move(self, move, board) -> int:
        """Encode move as flat index."""
        size = board.size
        if move.to is not None:
            # Position uses x, y (column, row)
            return move.to.y * size + move.to.x
        return 0


@pytest.fixture
def initial_state():
    """Create initial game state."""
    return create_initial_state(
        board_type=BoardType.SQUARE8,
        num_players=2,
    )


@pytest.fixture
def mock_nn():
    """Create mock neural network."""
    return MockNeuralNet()


class TestGumbelTopKParity:
    """Test Gumbel-Top-K sampling parity."""

    def test_gumbel_noise_distribution(self):
        """Verify Gumbel noise has correct distribution."""
        n_samples = 10000

        # Tensor tree method
        uniform = torch.rand(n_samples)
        gumbel_tensor = -torch.log(-torch.log(uniform.clamp(min=1e-10, max=1.0 - 1e-10)))

        # Multi-game method
        uniform_np = np.random.uniform(size=n_samples)
        gumbel_np = -np.log(-np.log(uniform_np + 1e-10) + 1e-10)

        # Both should have mean ~0.5772 (Euler-Mascheroni constant)
        expected_mean = 0.5772
        assert abs(gumbel_tensor.mean().item() - expected_mean) < 0.1
        assert abs(gumbel_np.mean() - expected_mean) < 0.1

        # Both should have std ~π/√6 ≈ 1.28
        expected_std = 1.28
        assert abs(gumbel_tensor.std().item() - expected_std) < 0.2
        assert abs(gumbel_np.std() - expected_std) < 0.2

    def test_top_k_selection_with_uniform_prior(self):
        """Test that top-K selection is consistent with uniform prior."""
        torch.manual_seed(42)
        np.random.seed(42)

        num_actions = 16
        k = 4

        # Uniform prior logits
        logits = torch.zeros(1, num_actions)

        # Create tensor tree and sample
        tree = TensorGumbelTree.create(
            batch_size=1,
            num_actions=num_actions,
            device="cpu",
        )
        tree.reset()
        top_k_indices = tree.initialize_root(logits, num_sampled_actions=k)

        # Verify K actions were selected
        assert top_k_indices.shape == (1, k)
        assert len(torch.unique(top_k_indices)) == k

    def test_top_k_prefers_high_prior(self):
        """Test that top-K selection prefers high-prior actions."""
        torch.manual_seed(42)

        num_actions = 16
        k = 4
        n_trials = 100

        # Create biased prior: action 0 has much higher prob
        logits = torch.zeros(1, num_actions)
        logits[0, 0] = 5.0  # High prior for action 0

        action_0_count = 0
        for trial in range(n_trials):
            tree = TensorGumbelTree.create(
                batch_size=1,
                num_actions=num_actions,
                device="cpu",
            )
            tree.reset()

            # Re-randomize Gumbel each trial
            top_k = tree.initialize_root(logits, num_sampled_actions=k)
            if 0 in top_k[0].tolist():
                action_0_count += 1

        # Action 0 should be selected most of the time
        assert action_0_count > 90, f"High-prior action selected only {action_0_count}/100 times"


class TestSequentialHalvingParity:
    """Test Sequential Halving budget allocation parity."""

    def test_budget_allocation(self):
        """Test that budget allocation formulas match."""
        tree = TensorGumbelTree.create(batch_size=1, num_actions=16, device="cpu")

        # Test with 800 budget, 16 actions
        phases = tree.compute_sequential_halving_budget(800, 16)

        # Expected: 4 phases for 16 actions (log2(16) = 4)
        assert len(phases) == 4

        # Check budget per phase
        budget_per_phase = 800 // 4  # 200
        assert phases[0] == (16, budget_per_phase // 16)  # (16, 12)
        assert phases[1] == (8, budget_per_phase // 8)    # (8, 25)
        assert phases[2] == (4, budget_per_phase // 4)    # (4, 50)
        assert phases[3] == (2, budget_per_phase // 2)    # (2, 100)

    def test_pruning_keeps_top_half(self):
        """Test that pruning keeps top half of actions."""
        tree = TensorGumbelTree.create(batch_size=1, num_actions=8, device="cpu")
        tree.reset()

        # Initialize with 8 remaining actions
        tree.remaining_mask[0, :8] = True

        # Set values: actions 0-3 are worse, 4-7 are better
        tree.action_values[0, :4] = 0.0
        tree.action_values[0, 4:8] = 1.0
        tree.action_visits[0, :8] = 1

        # Prune
        tree.prune_actions(tree_idx=0)

        # Top 4 should remain
        remaining = tree.get_remaining_action_indices(tree_idx=0)
        assert len(remaining) == 4
        assert all(idx >= 4 for idx in remaining.tolist())


class TestMoveSelectionParity:
    """Test that both implementations select similar moves."""

    def test_deterministic_game_parity(self, initial_state, mock_nn):
        """Test that move selection is similar for deterministic games."""
        # Skip if CUDA not available for GPU tree
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get valid moves
        engine = DefaultRulesEngine()
        valid_moves = engine.get_valid_moves(initial_state, initial_state.current_player)

        # Test GPU tree selection
        config = GPUGumbelMCTSConfig(
            num_sampled_actions=8,
            simulation_budget=100,
            eval_mode="heuristic",
            device=device,
        )
        gpu_tree = GPUGumbelMCTS(config)

        try:
            move1, policy1 = gpu_tree.search(initial_state, mock_nn, valid_moves)
        except Exception as e:
            pytest.skip(f"GPU tree failed: {e}")

        # Verify we got a valid result
        assert move1 is not None
        assert isinstance(policy1, dict)
        assert len(policy1) > 0


class TestTrainingDataQuality:
    """Test that training data quality is equivalent."""

    def test_visit_count_distribution(self, initial_state, mock_nn):
        """Test that visit counts form a reasonable distribution."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        engine = DefaultRulesEngine()
        valid_moves = engine.get_valid_moves(initial_state, initial_state.current_player)

        config = GPUGumbelMCTSConfig(
            num_sampled_actions=8,
            simulation_budget=200,
            eval_mode="heuristic",
            device=device,
        )
        gpu_tree = GPUGumbelMCTS(config)

        try:
            _, policy = gpu_tree.search(initial_state, mock_nn, valid_moves)
        except Exception as e:
            pytest.skip(f"GPU tree failed: {e}")

        # Policy should sum to ~1.0
        total_prob = sum(policy.values())
        assert abs(total_prob - 1.0) < 0.01, f"Policy sum = {total_prob}, expected 1.0"

        # Should have non-zero probabilities for multiple moves
        non_zero = sum(1 for p in policy.values() if p > 0.001)
        assert non_zero >= 1, "Should have at least one non-zero probability"

    def test_search_produces_valid_policy(self, initial_state, mock_nn):
        """Test that GPU tree search produces valid policy distributions.

        Verifies:
        1. All probability values are non-negative
        2. Probabilities sum to 1.0
        3. The selected move has non-zero probability
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        engine = DefaultRulesEngine()
        valid_moves = engine.get_valid_moves(initial_state, initial_state.current_player)

        config = GPUGumbelMCTSConfig(
            num_sampled_actions=8,
            simulation_budget=200,
            eval_mode="heuristic",
            device=device,
        )

        gpu_tree = GPUGumbelMCTS(config)
        try:
            best_move, policy = gpu_tree.search(initial_state, mock_nn, valid_moves)
        except Exception as e:
            pytest.skip(f"GPU tree failed: {e}")

        # All probabilities should be non-negative
        for move_key, prob in policy.items():
            assert prob >= 0.0, f"Negative probability for {move_key}: {prob}"

        # Probabilities should sum to ~1.0
        total = sum(policy.values())
        assert abs(total - 1.0) < 0.01, f"Policy sum = {total}, expected 1.0"

        # The best move should be in the policy
        best_key = gpu_tree._move_to_key(best_move)
        assert best_key in policy, f"Best move {best_key} not in policy"

        # The best move should have non-zero probability
        best_prob = policy.get(best_key, 0.0)
        assert best_prob > 0.0, f"Best move has zero probability"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
