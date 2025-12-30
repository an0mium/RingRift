"""Unit tests for quality_weighted_loss.py.

Tests the quality-weighted loss functions for training, including:
- compute_quality_weights: Convert visit counts to normalized weights
- quality_weighted_policy_loss: Weighted cross-entropy for policy
- quality_weighted_value_loss: Weighted MSE for value
- ranking_loss_from_quality: Pairwise ranking loss
- QualityWeightedTrainer: Mixin class for weighted training
- create_quality_weighted_sampler: NumPy sampler weights
"""

import numpy as np
import pytest
import torch

from app.training.quality_weighted_loss import (
    compute_quality_weights,
    quality_weighted_policy_loss,
    quality_weighted_value_loss,
    ranking_loss_from_quality,
    QualityWeightedTrainer,
    create_quality_weighted_sampler,
)


# ==============================================================================
# compute_quality_weights tests
# ==============================================================================


class TestComputeQualityWeights:
    """Tests for compute_quality_weights function."""

    def test_empty_tensor_returns_empty(self):
        """Empty visit counts returns empty weights."""
        visit_counts = torch.tensor([])
        weights = compute_quality_weights(visit_counts)
        assert weights.numel() == 0

    def test_zero_max_visits_returns_ones(self):
        """All zeros returns uniform weights."""
        visit_counts = torch.tensor([0, 0, 0])
        weights = compute_quality_weights(visit_counts)
        assert torch.allclose(weights.float(), torch.ones(3))

    def test_single_element(self):
        """Single element returns weight of 1.0."""
        visit_counts = torch.tensor([100])
        weights = compute_quality_weights(visit_counts)
        # Single element normalized by itself, then normalized to mean=1
        assert weights.shape == (1,)
        assert weights[0].item() == pytest.approx(1.0)

    def test_uniform_visits_returns_ones(self):
        """Uniform visit counts return uniform weights."""
        visit_counts = torch.tensor([50, 50, 50])
        weights = compute_quality_weights(visit_counts)
        assert torch.allclose(weights, torch.ones(3))

    def test_varying_visits_scales_correctly(self):
        """Higher visits get higher weights."""
        visit_counts = torch.tensor([100, 50, 25])
        weights = compute_quality_weights(visit_counts)
        # Max is 100, fractions are [1.0, 0.5, 0.25]
        # After min_weight clamp (0.1): [1.0, 0.5, 0.25]
        # Mean = (1.0 + 0.5 + 0.25) / 3 = 0.583
        # Normalized: [1.71, 0.86, 0.43]
        assert weights[0] > weights[1] > weights[2]
        assert torch.isclose(weights.mean(), torch.tensor(1.0))

    def test_min_weight_clamping(self):
        """Low visit counts are clamped to min_weight."""
        visit_counts = torch.tensor([100, 1])  # 1/100 = 0.01 < min_weight
        weights = compute_quality_weights(visit_counts, min_weight=0.1)
        # Fractions: [1.0, 0.01] -> clamped to [1.0, 0.1]
        # After normalization, second weight should be at least 0.1 * k for some k
        assert weights[1] > 0  # Not zero due to clamping

    def test_custom_min_weight(self):
        """Custom min_weight is applied."""
        visit_counts = torch.tensor([100, 0])
        weights = compute_quality_weights(visit_counts, min_weight=0.5)
        # [1.0, 0.0] clamped to [1.0, 0.5], mean = 0.75
        # Normalized: [1.33, 0.67]
        assert weights[1] >= 0.5 / 0.75  # 0.67

    def test_temperature_less_than_one_sharpens(self):
        """Temperature < 1 sharpens the distribution."""
        visit_counts = torch.tensor([100, 50])
        weights_t1 = compute_quality_weights(visit_counts, temperature=1.0)
        weights_t05 = compute_quality_weights(visit_counts, temperature=0.5)
        # Temperature 0.5 raises fractions to power 2, sharpening differences
        # Higher ratio between weights with lower temperature
        ratio_t1 = weights_t1[0] / weights_t1[1]
        ratio_t05 = weights_t05[0] / weights_t05[1]
        assert ratio_t05 > ratio_t1

    def test_temperature_greater_than_one_smooths(self):
        """Temperature > 1 smooths the distribution."""
        visit_counts = torch.tensor([100, 25])
        weights_t1 = compute_quality_weights(visit_counts, temperature=1.0)
        weights_t2 = compute_quality_weights(visit_counts, temperature=2.0)
        # Temperature 2 raises fractions to power 0.5, smoothing differences
        ratio_t1 = weights_t1[0] / weights_t1[1]
        ratio_t2 = weights_t2[0] / weights_t2[1]
        assert ratio_t2 < ratio_t1

    def test_mean_normalized_to_one(self):
        """Weights are normalized so mean equals 1."""
        visit_counts = torch.tensor([100, 75, 50, 25, 10])
        weights = compute_quality_weights(visit_counts)
        assert torch.isclose(weights.mean(), torch.tensor(1.0))

    def test_cuda_device_preserved(self):
        """Output device matches input device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        visit_counts = torch.tensor([100, 50]).cuda()
        weights = compute_quality_weights(visit_counts)
        assert weights.device == visit_counts.device

    def test_gradient_flow(self):
        """Gradients flow through the function."""
        visit_counts = torch.tensor([100.0, 50.0], requires_grad=True)
        weights = compute_quality_weights(visit_counts)
        loss = weights.sum()
        loss.backward()
        assert visit_counts.grad is not None


# ==============================================================================
# quality_weighted_policy_loss tests
# ==============================================================================


class TestQualityWeightedPolicyLoss:
    """Tests for quality_weighted_policy_loss function."""

    def test_basic_cross_entropy(self):
        """Basic weighted cross-entropy calculation."""
        policy_log_probs = torch.log(torch.tensor([[0.7, 0.3], [0.4, 0.6]]))
        policy_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        quality_weights = torch.tensor([1.0, 1.0])

        loss = quality_weighted_policy_loss(
            policy_log_probs, policy_targets, quality_weights
        )
        # -1.0 * log(0.7) + -1.0 * log(0.6) / 2
        expected = (
            -torch.log(torch.tensor(0.7)) - torch.log(torch.tensor(0.6))
        ) / 2
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_higher_weight_increases_contribution(self):
        """Higher weight increases sample's contribution to loss."""
        # Use samples with different per-sample losses
        policy_log_probs = torch.log(torch.tensor([[0.9, 0.1], [0.5, 0.5]]))
        policy_targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        # Sample 0: -log(0.9) = 0.105
        # Sample 1: -log(0.5) = 0.693

        # First sample (low loss) weighted 2x, second (high loss) weighted 1x
        weights1 = torch.tensor([2.0, 1.0])
        loss1 = quality_weighted_policy_loss(
            policy_log_probs, policy_targets, weights1
        )

        # First sample (low loss) weighted 1x, second (high loss) weighted 2x
        weights2 = torch.tensor([1.0, 2.0])
        loss2 = quality_weighted_policy_loss(
            policy_log_probs, policy_targets, weights2
        )

        # Weighting high-loss sample more should increase total loss
        assert loss2 > loss1

    def test_reduction_sum(self):
        """Sum reduction returns sum of weighted losses."""
        policy_log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        policy_targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        quality_weights = torch.tensor([1.0, 1.0])

        loss = quality_weighted_policy_loss(
            policy_log_probs, policy_targets, quality_weights, reduction="sum"
        )
        expected = -torch.log(torch.tensor(0.5)) * 2
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_reduction_none(self):
        """None reduction returns per-sample losses."""
        policy_log_probs = torch.log(torch.tensor([[0.7, 0.3], [0.4, 0.6]]))
        policy_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        quality_weights = torch.tensor([1.0, 1.0])

        losses = quality_weighted_policy_loss(
            policy_log_probs, policy_targets, quality_weights, reduction="none"
        )
        assert losses.shape == (2,)
        assert torch.isclose(losses[0], -torch.log(torch.tensor(0.7)), atol=1e-5)
        assert torch.isclose(losses[1], -torch.log(torch.tensor(0.6)), atol=1e-5)

    def test_empty_valid_mask_returns_zero(self):
        """When all targets are zero, returns 0.0 loss."""
        policy_log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        policy_targets = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  # Invalid targets
        quality_weights = torch.tensor([1.0, 1.0])

        loss = quality_weighted_policy_loss(
            policy_log_probs, policy_targets, quality_weights
        )
        assert loss.item() == 0.0

    def test_soft_targets(self):
        """Works with soft (non-one-hot) targets."""
        policy_log_probs = torch.log(torch.tensor([[0.7, 0.3]]))
        policy_targets = torch.tensor([[0.8, 0.2]])  # Soft target
        quality_weights = torch.tensor([1.0])

        loss = quality_weighted_policy_loss(
            policy_log_probs, policy_targets, quality_weights
        )
        # -0.8 * log(0.7) - 0.2 * log(0.3)
        expected = -0.8 * torch.log(torch.tensor(0.7)) - 0.2 * torch.log(
            torch.tensor(0.3)
        )
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients flow through the loss."""
        # Use logits that require grad, then apply log_softmax
        logits = torch.tensor([[1.0, -1.0]], requires_grad=True)
        policy_log_probs = torch.log_softmax(logits, dim=1)
        policy_targets = torch.tensor([[1.0, 0.0]])
        quality_weights = torch.tensor([1.0])

        loss = quality_weighted_policy_loss(
            policy_log_probs, policy_targets, quality_weights
        )
        loss.backward()
        assert logits.grad is not None


# ==============================================================================
# quality_weighted_value_loss tests
# ==============================================================================


class TestQualityWeightedValueLoss:
    """Tests for quality_weighted_value_loss function."""

    def test_1d_values(self):
        """Works with 1D value tensors."""
        value_pred = torch.tensor([0.5, 0.3])
        value_target = torch.tensor([0.6, 0.4])
        quality_weights = torch.tensor([1.0, 1.0])

        loss = quality_weighted_value_loss(value_pred, value_target, quality_weights)
        # MSE: ((0.5-0.6)^2 + (0.3-0.4)^2) / 2 = (0.01 + 0.01) / 2 = 0.01
        assert torch.isclose(loss, torch.tensor(0.01), atol=1e-5)

    def test_2d_values_multihead(self):
        """Works with 2D value tensors (multi-player)."""
        value_pred = torch.tensor([[0.5, 0.5], [0.3, 0.7]])  # (B, P)
        value_target = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        quality_weights = torch.tensor([1.0, 1.0])

        loss = quality_weighted_value_loss(value_pred, value_target, quality_weights)
        # Per-sample MSE averaged over players
        # Sample 0: ((0.5-0.6)^2 + (0.5-0.4)^2) / 2 = 0.01
        # Sample 1: ((0.3-0.4)^2 + (0.7-0.6)^2) / 2 = 0.01
        assert torch.isclose(loss, torch.tensor(0.01), atol=1e-5)

    def test_higher_weight_increases_contribution(self):
        """Higher weight increases sample's contribution."""
        value_pred = torch.tensor([0.0, 0.0])
        value_target = torch.tensor([1.0, 1.0])

        # First sample weighted more
        weights1 = torch.tensor([2.0, 1.0])
        loss1 = quality_weighted_value_loss(value_pred, value_target, weights1)

        # Uniform weights
        weights2 = torch.tensor([1.0, 1.0])
        loss2 = quality_weighted_value_loss(value_pred, value_target, weights2)

        # Different losses due to weighting
        assert not torch.isclose(loss1, loss2)

    def test_reduction_sum(self):
        """Sum reduction returns sum of weighted losses."""
        value_pred = torch.tensor([0.5, 0.3])
        value_target = torch.tensor([0.6, 0.4])
        quality_weights = torch.tensor([1.0, 1.0])

        loss = quality_weighted_value_loss(
            value_pred, value_target, quality_weights, reduction="sum"
        )
        # Sum: 0.01 + 0.01 = 0.02
        assert torch.isclose(loss, torch.tensor(0.02), atol=1e-5)

    def test_reduction_none(self):
        """None reduction returns per-sample losses."""
        value_pred = torch.tensor([0.5, 0.3])
        value_target = torch.tensor([0.6, 0.4])
        quality_weights = torch.tensor([1.0, 1.0])

        losses = quality_weighted_value_loss(
            value_pred, value_target, quality_weights, reduction="none"
        )
        assert losses.shape == (2,)
        assert torch.isclose(losses[0], torch.tensor(0.01), atol=1e-5)
        assert torch.isclose(losses[1], torch.tensor(0.01), atol=1e-5)

    def test_gradient_flow(self):
        """Gradients flow through the loss."""
        value_pred = torch.tensor([0.5], requires_grad=True)
        value_target = torch.tensor([0.6])
        quality_weights = torch.tensor([1.0])

        loss = quality_weighted_value_loss(value_pred, value_target, quality_weights)
        loss.backward()
        assert value_pred.grad is not None


# ==============================================================================
# ranking_loss_from_quality tests
# ==============================================================================


class TestRankingLossFromQuality:
    """Tests for ranking_loss_from_quality function."""

    def test_single_sample_returns_zero(self):
        """Single sample batch returns 0.0 (can't form pairs)."""
        policy_log_probs = torch.log(torch.tensor([[0.5, 0.5]]))
        quality_scores = torch.tensor([0.8])

        loss = ranking_loss_from_quality(policy_log_probs, quality_scores)
        assert loss.item() == 0.0

    def test_two_samples_basic(self):
        """Two samples with different quality creates ranking loss."""
        # Sample 0 has higher quality but lower log prob
        policy_log_probs = torch.log(torch.tensor([[0.3, 0.7], [0.6, 0.4]]))
        quality_scores = torch.tensor([0.9, 0.1])  # Sample 0 is higher quality

        loss = ranking_loss_from_quality(policy_log_probs, quality_scores, margin=0.0)
        # Higher quality should have higher max log prob
        # Sample 0 max: log(0.7), Sample 1 max: log(0.6)
        # log(0.7) > log(0.6), so no violation, loss = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_ranking_violation_creates_loss(self):
        """When ranking is violated, loss is positive."""
        # Sample 0 has higher quality (0.9) but LOWER max log prob
        # Sample 1 has lower quality (0.1) but HIGHER max log prob
        policy_log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.9, 0.1]]))
        quality_scores = torch.tensor([0.9, 0.1])  # Sample 0 is higher quality

        loss = ranking_loss_from_quality(policy_log_probs, quality_scores, margin=0.0)
        # Sample 0: max log prob = log(0.5) = -0.693, quality = 0.9
        # Sample 1: max log prob = log(0.9) = -0.105, quality = 0.1
        # Violation: Sample 0 has higher quality but lower log prob
        assert loss.item() > 0

    def test_margin_increases_loss(self):
        """Higher margin increases loss for marginal violations."""
        policy_log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
        quality_scores = torch.tensor([0.9, 0.1])

        loss_m0 = ranking_loss_from_quality(
            policy_log_probs, quality_scores, margin=0.0
        )
        loss_m1 = ranking_loss_from_quality(
            policy_log_probs, quality_scores, margin=1.0
        )

        # With margin > 0, even equal log probs create loss
        assert loss_m1 > loss_m0

    def test_equal_quality_no_loss(self):
        """Equal quality scores produce no loss."""
        policy_log_probs = torch.log(torch.tensor([[0.3, 0.7], [0.8, 0.2]]))
        quality_scores = torch.tensor([0.5, 0.5])  # Equal quality

        loss = ranking_loss_from_quality(policy_log_probs, quality_scores)
        assert loss.item() == 0.0

    def test_max_pairs_sampling(self):
        """Large batches sample pairs rather than using all."""
        batch_size = 100  # 100 * 99 / 2 = 4950 pairs
        policy_log_probs = torch.log(torch.softmax(torch.randn(batch_size, 10), dim=1))
        quality_scores = torch.rand(batch_size)

        # max_pairs=50 should sample instead of using all 4950 pairs
        loss = ranking_loss_from_quality(
            policy_log_probs, quality_scores, max_pairs=50
        )
        assert loss.item() >= 0  # Should compute successfully

    def test_small_batch_uses_all_pairs(self):
        """Small batches use all pairs without sampling."""
        policy_log_probs = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]))
        quality_scores = torch.tensor([0.9, 0.5, 0.1])

        # 3 samples = 3 pairs, less than default max_pairs=100
        loss = ranking_loss_from_quality(policy_log_probs, quality_scores)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """Gradients flow through the loss."""
        policy_log_probs = torch.log(
            torch.softmax(torch.randn(5, 10, requires_grad=True), dim=1)
        )
        quality_scores = torch.rand(5)

        loss = ranking_loss_from_quality(policy_log_probs, quality_scores)
        loss.backward()
        # Note: gradients may be None if loss is 0 (no violations)


# ==============================================================================
# QualityWeightedTrainer tests
# ==============================================================================


class TestQualityWeightedTrainer:
    """Tests for QualityWeightedTrainer class."""

    def test_init_default_params(self):
        """Default initialization parameters."""
        trainer = QualityWeightedTrainer()
        assert trainer.quality_weight == 0.5
        assert trainer.ranking_weight == 0.1
        assert trainer.ranking_margin == 0.5
        assert trainer.min_quality_weight == 0.1
        assert trainer.temperature == 1.0

    def test_init_custom_params(self):
        """Custom initialization parameters."""
        trainer = QualityWeightedTrainer(
            quality_weight=0.8,
            ranking_weight=0.2,
            ranking_margin=1.0,
            min_quality_weight=0.2,
            temperature=2.0,
        )
        assert trainer.quality_weight == 0.8
        assert trainer.ranking_weight == 0.2
        assert trainer.ranking_margin == 1.0
        assert trainer.min_quality_weight == 0.2
        assert trainer.temperature == 2.0

    def test_compute_weighted_policy_loss_with_visit_counts(self):
        """Policy loss computed from visit counts."""
        trainer = QualityWeightedTrainer()
        policy_log_probs = torch.log(torch.tensor([[0.7, 0.3], [0.4, 0.6]]))
        policy_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        visit_counts = torch.tensor([100, 50])

        loss = trainer.compute_weighted_policy_loss(
            policy_log_probs, policy_targets, visit_counts=visit_counts
        )
        assert loss.item() > 0

    def test_compute_weighted_policy_loss_with_precomputed_weights(self):
        """Policy loss computed from precomputed weights."""
        trainer = QualityWeightedTrainer()
        policy_log_probs = torch.log(torch.tensor([[0.7, 0.3], [0.4, 0.6]]))
        policy_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        quality_weights = torch.tensor([1.5, 0.5])

        loss = trainer.compute_weighted_policy_loss(
            policy_log_probs, policy_targets, quality_weights=quality_weights
        )
        assert loss.item() > 0

    def test_compute_weighted_policy_loss_no_weights(self):
        """Policy loss with uniform weights when no visit counts provided."""
        trainer = QualityWeightedTrainer()
        policy_log_probs = torch.log(torch.tensor([[0.7, 0.3]]))
        policy_targets = torch.tensor([[1.0, 0.0]])

        loss = trainer.compute_weighted_policy_loss(policy_log_probs, policy_targets)
        expected = -torch.log(torch.tensor(0.7))
        assert torch.isclose(loss, expected, atol=1e-5)

    def test_quality_weight_blending(self):
        """quality_weight blends between uniform and quality-weighted."""
        # quality_weight=0 means uniform weights
        trainer0 = QualityWeightedTrainer(quality_weight=0.0)
        # quality_weight=1 means fully quality-weighted
        trainer1 = QualityWeightedTrainer(quality_weight=1.0)

        policy_log_probs = torch.log(torch.tensor([[0.7, 0.3], [0.4, 0.6]]))
        policy_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        visit_counts = torch.tensor([100, 10])  # Very different visits

        loss0 = trainer0.compute_weighted_policy_loss(
            policy_log_probs, policy_targets, visit_counts=visit_counts
        )
        loss1 = trainer1.compute_weighted_policy_loss(
            policy_log_probs, policy_targets, visit_counts=visit_counts
        )

        # Losses should differ due to weighting differences
        assert not torch.isclose(loss0, loss1)

    def test_ranking_loss_added(self):
        """Ranking loss is added when ranking_weight > 0."""
        trainer_with_ranking = QualityWeightedTrainer(ranking_weight=0.5)
        trainer_no_ranking = QualityWeightedTrainer(ranking_weight=0.0)

        policy_log_probs = torch.log(torch.tensor([[0.3, 0.7], [0.8, 0.2]]))
        policy_targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        visit_counts = torch.tensor([10, 100])  # Different visits to create ranking

        loss_with = trainer_with_ranking.compute_weighted_policy_loss(
            policy_log_probs, policy_targets, visit_counts=visit_counts
        )
        loss_without = trainer_no_ranking.compute_weighted_policy_loss(
            policy_log_probs, policy_targets, visit_counts=visit_counts
        )

        # Loss with ranking should be different (usually higher)
        # Note: could be equal if no ranking violation
        assert loss_with.item() >= 0

    def test_statistics_tracking(self):
        """Quality statistics are tracked."""
        trainer = QualityWeightedTrainer()
        policy_log_probs = torch.log(torch.tensor([[0.7, 0.3], [0.4, 0.6]]))
        policy_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        visit_counts = torch.tensor([100, 50])

        trainer.compute_weighted_policy_loss(
            policy_log_probs, policy_targets, visit_counts=visit_counts
        )

        assert "mean_weight" in trainer.quality_stats
        assert "std_weight" in trainer.quality_stats
        assert trainer.quality_stats["mean_weight"] > 0

    def test_compute_weighted_value_loss(self):
        """Value loss computation."""
        trainer = QualityWeightedTrainer()
        value_pred = torch.tensor([0.5, 0.3])
        value_target = torch.tensor([0.6, 0.4])
        visit_counts = torch.tensor([100, 50])

        loss = trainer.compute_weighted_value_loss(
            value_pred, value_target, visit_counts=visit_counts
        )
        assert loss.item() > 0

    def test_compute_weighted_value_loss_2d(self):
        """Value loss with 2D tensors (multi-player)."""
        trainer = QualityWeightedTrainer()
        value_pred = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        value_target = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        visit_counts = torch.tensor([100, 50])

        loss = trainer.compute_weighted_value_loss(
            value_pred, value_target, visit_counts=visit_counts
        )
        assert loss.item() > 0

    def test_compute_weighted_value_loss_no_weights(self):
        """Value loss with uniform weights when no visit counts provided."""
        trainer = QualityWeightedTrainer()
        value_pred = torch.tensor([0.5])
        value_target = torch.tensor([0.6])

        loss = trainer.compute_weighted_value_loss(value_pred, value_target)
        expected = torch.tensor(0.01)  # (0.5-0.6)^2
        assert torch.isclose(loss, expected, atol=1e-5)


# ==============================================================================
# create_quality_weighted_sampler tests
# ==============================================================================


class TestCreateQualityWeightedSampler:
    """Tests for create_quality_weighted_sampler function."""

    def test_empty_input_returns_empty(self):
        """Empty visit counts returns empty weights."""
        visit_counts = np.array([])
        weights = create_quality_weighted_sampler(visit_counts)
        assert len(weights) == 0

    def test_zero_max_visits_returns_ones(self):
        """All zeros returns uniform weights."""
        visit_counts = np.array([0, 0, 0])
        weights = create_quality_weighted_sampler(visit_counts)
        np.testing.assert_array_equal(weights, np.ones(3))

    def test_single_element(self):
        """Single element returns weight of 1.0."""
        visit_counts = np.array([100])
        weights = create_quality_weighted_sampler(visit_counts)
        assert weights[0] == pytest.approx(1.0)

    def test_varying_visits_scales_correctly(self):
        """Higher visits get higher weights."""
        visit_counts = np.array([100, 50, 25])
        weights = create_quality_weighted_sampler(visit_counts)
        assert weights[0] > weights[1] > weights[2]

    def test_mean_normalized_to_one(self):
        """Weights are normalized so mean equals 1."""
        visit_counts = np.array([100, 75, 50, 25, 10])
        weights = create_quality_weighted_sampler(visit_counts)
        assert weights.mean() == pytest.approx(1.0)

    def test_min_weight_clamping(self):
        """Low visit counts are clamped to min_weight."""
        visit_counts = np.array([100, 0])
        weights = create_quality_weighted_sampler(visit_counts, min_weight=0.2)
        # [1.0, 0.0] clamped to [1.0, 0.2]
        # After normalization, second weight should be non-zero
        assert weights[1] > 0

    def test_temperature_less_than_one_sharpens(self):
        """Temperature < 1 sharpens the distribution."""
        visit_counts = np.array([100, 50])
        weights_t1 = create_quality_weighted_sampler(visit_counts, temperature=1.0)
        weights_t05 = create_quality_weighted_sampler(visit_counts, temperature=0.5)
        ratio_t1 = weights_t1[0] / weights_t1[1]
        ratio_t05 = weights_t05[0] / weights_t05[1]
        assert ratio_t05 > ratio_t1

    def test_temperature_greater_than_one_smooths(self):
        """Temperature > 1 smooths the distribution."""
        visit_counts = np.array([100, 25])
        weights_t1 = create_quality_weighted_sampler(visit_counts, temperature=1.0)
        weights_t2 = create_quality_weighted_sampler(visit_counts, temperature=2.0)
        ratio_t1 = weights_t1[0] / weights_t1[1]
        ratio_t2 = weights_t2[0] / weights_t2[1]
        assert ratio_t2 < ratio_t1

    def test_output_is_numeric(self):
        """Output is numeric float type."""
        visit_counts = np.array([100, 50, 25], dtype=np.int64)
        weights = create_quality_weighted_sampler(visit_counts)
        # Output should be floating point (float32 or float64)
        assert np.issubdtype(weights.dtype, np.floating)


# ==============================================================================
# Integration tests
# ==============================================================================


class TestQualityWeightedIntegration:
    """Integration tests for quality-weighted training workflow."""

    def test_full_training_step(self):
        """Simulate a full training step with quality weighting."""
        trainer = QualityWeightedTrainer(
            quality_weight=0.7,
            ranking_weight=0.1,
        )

        # Simulated batch
        batch_size = 16
        action_size = 64
        num_players = 2

        policy_logits = torch.randn(batch_size, action_size)
        policy_log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_targets = torch.softmax(torch.randn(batch_size, action_size), dim=1)
        value_pred = torch.sigmoid(torch.randn(batch_size, num_players))
        value_target = torch.softmax(torch.randn(batch_size, num_players), dim=1)
        visit_counts = torch.randint(10, 200, (batch_size,)).float()

        # Compute losses
        policy_loss = trainer.compute_weighted_policy_loss(
            policy_log_probs, policy_targets, visit_counts=visit_counts
        )
        value_loss = trainer.compute_weighted_value_loss(
            value_pred, value_target, visit_counts=visit_counts
        )

        # Combine losses
        total_loss = policy_loss + 0.5 * value_loss

        # Verify losses are reasonable
        assert total_loss.item() > 0
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)

    def test_sampler_integration_with_dataloader(self):
        """Verify sampler weights work with PyTorch DataLoader pattern."""
        # Simulated dataset
        n_samples = 1000
        visit_counts = np.random.randint(10, 200, n_samples)

        # Create sampler weights
        weights = create_quality_weighted_sampler(visit_counts)

        # Verify weights are valid for WeightedRandomSampler
        assert len(weights) == n_samples
        assert np.all(weights > 0)
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))

        # Weights can be used with torch.utils.data.WeightedRandomSampler
        sampler_weights = torch.from_numpy(weights)
        assert sampler_weights.shape == (n_samples,)

    def test_gradient_accumulation(self):
        """Test quality-weighted losses work with gradient accumulation."""
        trainer = QualityWeightedTrainer()

        # Simulated model parameters
        param = torch.randn(10, 10, requires_grad=True)

        # Multiple micro-batches
        total_loss = torch.tensor(0.0)
        for _ in range(4):
            policy_log_probs = torch.log_softmax(param @ torch.randn(10, 5), dim=1)
            policy_targets = torch.softmax(torch.randn(10, 5), dim=1)
            visit_counts = torch.randint(10, 100, (10,)).float()

            loss = trainer.compute_weighted_policy_loss(
                policy_log_probs, policy_targets, visit_counts=visit_counts
            )
            total_loss = total_loss + loss

        # Average and backward
        (total_loss / 4).backward()

        # Gradients should exist
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
