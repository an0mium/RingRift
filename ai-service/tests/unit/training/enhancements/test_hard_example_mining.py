"""
Tests for hard example mining functionality.

December 2025: Added comprehensive test coverage for app/training/enhancements/hard_example_mining.py
"""

import numpy as np
import pytest
import torch

from app.training.enhancements.hard_example_mining import (
    HardExample,
    HardExampleMiner,
)


class TestHardExample:
    """Tests for HardExample dataclass."""

    def test_creation_defaults(self):
        """Test creating HardExample with defaults."""
        ex = HardExample(index=0, loss=0.5, uncertainty=0.3)
        assert ex.index == 0
        assert ex.loss == 0.5
        assert ex.uncertainty == 0.3
        assert ex.times_sampled == 1
        assert ex.last_seen_step == 0

    def test_creation_with_values(self):
        """Test creating HardExample with all values."""
        ex = HardExample(
            index=42,
            loss=1.5,
            uncertainty=0.8,
            times_sampled=5,
            last_seen_step=100,
        )
        assert ex.index == 42
        assert ex.loss == 1.5
        assert ex.uncertainty == 0.8
        assert ex.times_sampled == 5
        assert ex.last_seen_step == 100


class TestHardExampleMinerInit:
    """Tests for HardExampleMiner initialization."""

    def test_default_init(self):
        """Test default initialization."""
        miner = HardExampleMiner()
        assert miner.buffer_size == 10000
        assert miner.hard_fraction == 0.3
        assert miner.loss_threshold_percentile == 80.0
        assert miner.uncertainty_weight == 0.3
        assert miner.decay_rate == 0.99
        assert miner.min_samples_before_mining == 1000
        assert miner.max_times_sampled == 10

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        miner = HardExampleMiner(
            buffer_size=5000,
            hard_fraction=0.5,
            loss_threshold_percentile=90.0,
            uncertainty_weight=0.5,
            decay_rate=0.95,
            min_samples_before_mining=500,
            max_times_sampled=5,
        )
        assert miner.buffer_size == 5000
        assert miner.hard_fraction == 0.5
        assert miner.loss_threshold_percentile == 90.0
        assert miner.uncertainty_weight == 0.5
        assert miner.decay_rate == 0.95
        assert miner.min_samples_before_mining == 500
        assert miner.max_times_sampled == 5


class TestRecordBatch:
    """Tests for record_batch functionality."""

    def test_record_single_batch(self):
        """Test recording a single batch."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        indices = [0, 1, 2]
        losses = [0.5, 1.0, 0.3]

        miner.record_batch(indices, losses)

        assert miner._total_samples_seen == 3
        assert len(miner._examples) == 3
        assert miner._examples[0].loss == 0.5
        assert miner._examples[1].loss == 1.0
        assert miner._examples[2].loss == 0.3

    def test_record_with_numpy(self):
        """Test recording with numpy arrays."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        indices = np.array([0, 1, 2])
        losses = np.array([0.5, 1.0, 0.3])

        miner.record_batch(indices, losses)

        assert miner._total_samples_seen == 3
        assert len(miner._examples) == 3

    def test_record_with_torch(self):
        """Test recording with torch tensors."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        indices = torch.tensor([0, 1, 2])
        losses = torch.tensor([0.5, 1.0, 0.3])

        miner.record_batch(indices, losses)

        assert miner._total_samples_seen == 3
        assert len(miner._examples) == 3

    def test_record_with_uncertainties(self):
        """Test recording with uncertainties."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        indices = [0, 1]
        losses = [0.5, 1.0]
        uncertainties = [0.2, 0.8]

        miner.record_batch(indices, losses, uncertainties)

        assert miner._examples[0].uncertainty == 0.2
        assert miner._examples[1].uncertainty == 0.8

    def test_record_updates_existing(self):
        """Test that recording updates existing examples with EMA."""
        miner = HardExampleMiner(min_samples_before_mining=0)

        # Record initial
        miner.record_batch([0], [1.0])
        assert miner._examples[0].loss == 1.0
        assert miner._examples[0].times_sampled == 1

        # Record update - EMA: 0.7 * 1.0 + 0.3 * 0.5 = 0.85
        miner.record_batch([0], [0.5])
        assert miner._examples[0].loss == pytest.approx(0.85, rel=0.01)
        assert miner._examples[0].times_sampled == 2

    def test_buffer_pruning(self):
        """Test that buffer is pruned when exceeding size."""
        miner = HardExampleMiner(
            buffer_size=5,
            min_samples_before_mining=0,
        )

        # Record 10 examples with varying losses
        for i in range(10):
            miner.record_batch([i], [float(i)])

        # Buffer should be pruned to 5
        assert len(miner._examples) == 5

        # Should keep hardest examples (highest losses)
        kept_indices = set(miner._examples.keys())
        assert 9 in kept_indices  # Highest loss
        assert 8 in kept_indices


class TestGetHardIndices:
    """Tests for get_hard_indices functionality."""

    def test_empty_before_min_samples(self):
        """Test returns empty before minimum samples."""
        miner = HardExampleMiner(min_samples_before_mining=100)
        miner.record_batch([0, 1, 2], [1.0, 2.0, 3.0])

        result = miner.get_hard_indices(5)
        assert len(result) == 0

    def test_empty_with_no_examples(self):
        """Test returns empty with no examples."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        result = miner.get_hard_indices(5)
        assert len(result) == 0

    def test_returns_hard_examples(self):
        """Test returns hard examples after sufficient data."""
        miner = HardExampleMiner(
            min_samples_before_mining=0,
            loss_threshold_percentile=50,
        )

        # Record examples with varying losses
        indices = list(range(100))
        losses = [i / 100 for i in range(100)]  # 0.0 to 0.99
        miner.record_batch(indices, losses)

        # Get hard examples
        result = miner.get_hard_indices(10)
        assert len(result) == 10

        # Should be from high-loss examples
        for idx in result:
            assert miner._examples[idx].loss >= 0.5

    def test_return_scores(self):
        """Test returning scores with indices."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        miner.record_batch(list(range(100)), [i / 100 for i in range(100)])

        indices, scores = miner.get_hard_indices(5, return_scores=True)
        assert len(indices) == 5
        assert len(scores) == 5
        assert all(s > 0 for s in scores)


class TestComputeHardness:
    """Tests for hardness computation."""

    def test_hardness_decay(self):
        """Test that hardness decays with staleness."""
        miner = HardExampleMiner(decay_rate=0.9)

        ex = HardExample(index=0, loss=1.0, uncertainty=0.0, last_seen_step=0)
        miner._current_step = 0
        h1 = miner._compute_hardness(ex)

        miner._current_step = 10
        h2 = miner._compute_hardness(ex)

        assert h2 < h1  # Should decay

    def test_hardness_sample_penalty(self):
        """Test that over-sampled examples are penalized."""
        miner = HardExampleMiner(max_times_sampled=10)
        miner._current_step = 0

        # Low times sampled
        ex1 = HardExample(index=0, loss=1.0, uncertainty=0.0, times_sampled=1)
        h1 = miner._compute_hardness(ex1)

        # High times sampled
        ex2 = HardExample(index=1, loss=1.0, uncertainty=0.0, times_sampled=10)
        h2 = miner._compute_hardness(ex2)

        assert h2 < h1  # Higher samples = lower hardness

    def test_hardness_uncertainty_weight(self):
        """Test uncertainty weight in hardness calculation."""
        miner = HardExampleMiner(uncertainty_weight=0.5)
        miner._current_step = 0

        # Same loss, different uncertainty
        ex1 = HardExample(index=0, loss=1.0, uncertainty=0.0)
        ex2 = HardExample(index=1, loss=1.0, uncertainty=1.0)

        h1 = miner._compute_hardness(ex1)
        h2 = miner._compute_hardness(ex2)

        assert h2 > h1  # Higher uncertainty = higher hardness


class TestGetSampleWeights:
    """Tests for get_sample_weights functionality."""

    def test_before_mining_active(self):
        """Test returns base weights before mining active."""
        miner = HardExampleMiner(min_samples_before_mining=100)
        miner.record_batch([0, 1, 2], [1.0, 2.0, 3.0])

        weights = miner.get_sample_weights([0, 1, 2])
        assert all(w == 1.0 for w in weights)

    def test_upweights_hard_examples(self):
        """Test that hard examples get higher weights."""
        miner = HardExampleMiner(
            min_samples_before_mining=0,
            loss_threshold_percentile=50,
        )

        # Record varied losses
        for i in range(200):
            miner.record_batch([i], [i / 100])

        # Get weights
        weights = miner.get_sample_weights(
            [10, 150],  # Low loss and high loss
            base_weight=1.0,
            hard_weight=2.0,
        )

        # Low loss should have base weight
        assert weights[0] == 1.0
        # High loss should have hard weight
        assert weights[1] == 2.0


class TestCreateMixedBatchIndices:
    """Tests for create_mixed_batch_indices functionality."""

    def test_mixed_batch_creation(self):
        """Test creating mixed batch with hard and random examples."""
        miner = HardExampleMiner(
            min_samples_before_mining=0,
            hard_fraction=0.3,
        )

        # Record examples
        for i in range(100):
            miner.record_batch([i], [i / 100])

        all_indices = np.arange(100)
        batch = miner.create_mixed_batch_indices(
            batch_size=10,
            all_indices=all_indices,
        )

        assert len(batch) == 10
        assert len(set(batch)) == 10  # All unique

    def test_handles_insufficient_hard_examples(self):
        """Test handles case with no hard examples."""
        miner = HardExampleMiner(min_samples_before_mining=1000)

        # Not enough samples for mining
        miner.record_batch([0, 1, 2], [0.5, 0.5, 0.5])

        all_indices = np.arange(10)
        batch = miner.create_mixed_batch_indices(
            batch_size=5,
            all_indices=all_indices,
        )

        # Should still work with random examples only
        assert len(batch) <= 5


class TestStatistics:
    """Tests for statistics functionality."""

    def test_empty_statistics(self):
        """Test statistics with no data."""
        miner = HardExampleMiner()
        stats = miner.get_statistics()

        assert stats['total_samples_seen'] == 0
        assert stats['tracked_examples'] == 0
        assert stats['mining_active'] is False

    def test_statistics_after_recording(self):
        """Test statistics after recording data."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        miner.record_batch(list(range(10)), [i * 0.1 for i in range(10)])

        stats = miner.get_statistics()

        assert stats['total_samples_seen'] == 10
        assert stats['tracked_examples'] == 10
        assert stats['mining_active'] is True
        assert stats['mean_loss'] == pytest.approx(0.45, rel=0.01)
        assert stats['max_loss'] == pytest.approx(0.9, rel=0.01)


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        miner = HardExampleMiner()
        miner.record_batch(list(range(100)), [i / 100 for i in range(100)])

        miner.reset()

        assert miner._total_samples_seen == 0
        assert len(miner._examples) == 0
        assert miner._current_step == 0
        assert len(miner._loss_history) == 0


class TestBackwardCompatibility:
    """Tests for backward compatibility methods."""

    def test_update_errors_alias(self):
        """Test update_errors is an alias for record_batch."""
        miner = HardExampleMiner(min_samples_before_mining=0)

        miner.update_errors([0, 1, 2], [0.5, 1.0, 0.3])

        assert len(miner._examples) == 3
        assert miner._examples[1].loss == 1.0

    def test_get_all_sample_weights(self):
        """Test get_all_sample_weights for full dataset."""
        miner = HardExampleMiner(
            min_samples_before_mining=0,
            loss_threshold_percentile=50,
        )

        # Record examples for half the dataset
        for i in range(50, 100):
            miner.record_batch([i], [i / 100])

        # Get weights for full dataset
        weights = miner.get_all_sample_weights(
            dataset_size=100,
            min_weight=0.5,
            max_weight=3.0,
        )

        assert len(weights) == 100
        assert weights.dtype == np.float32

    def test_get_all_sample_weights_before_mining(self):
        """Test get_all_sample_weights returns ones before mining."""
        miner = HardExampleMiner(min_samples_before_mining=1000)
        miner.record_batch([0, 1], [0.5, 1.0])

        weights = miner.get_all_sample_weights(dataset_size=10)
        assert all(w == 1.0 for w in weights)

    def test_get_stats_compatibility(self):
        """Test get_stats returns expected format."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        miner.record_batch(list(range(10)), [i * 0.1 for i in range(10)])

        stats = miner.get_stats()

        assert 'seen_samples' in stats
        assert 'seen_ratio' in stats
        assert 'mean_error' in stats
        assert 'max_error' in stats
        assert 'mining_active' in stats
        assert stats['seen_samples'] == 10


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_example(self):
        """Test with single example."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        miner.record_batch([0], [0.5])

        assert len(miner._examples) == 1
        stats = miner.get_statistics()
        assert stats['tracked_examples'] == 1

    def test_large_batch(self):
        """Test with large batch."""
        miner = HardExampleMiner(
            buffer_size=10000,
            min_samples_before_mining=0,
        )

        indices = list(range(5000))
        losses = [i / 5000 for i in range(5000)]
        miner.record_batch(indices, losses)

        assert miner._total_samples_seen == 5000
        assert len(miner._examples) == 5000

    def test_identical_losses(self):
        """Test handling of identical losses."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        miner.record_batch([0, 1, 2, 3], [0.5, 0.5, 0.5, 0.5])

        # Should still work
        result = miner.get_hard_indices(2)
        assert len(result) <= 2

    def test_zero_losses(self):
        """Test handling of zero losses."""
        miner = HardExampleMiner(min_samples_before_mining=0)
        miner.record_batch([0, 1, 2], [0.0, 0.0, 0.0])

        stats = miner.get_statistics()
        assert stats['mean_loss'] == 0.0
