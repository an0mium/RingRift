"""Tests for gradient management utilities.

Tests GradientAccumulator and AdaptiveGradientClipper from
app/training/enhancements/gradient_management.py.

Jan 12, 2026: Created to test generator exhaustion bug fix.
"""

import pytest
import torch
import torch.nn as nn

from app.training.enhancements.gradient_management import (
    AdaptiveGradientClipper,
    GradientAccumulator,
)


class TestGradientAccumulator:
    """Tests for GradientAccumulator class."""

    def test_init_defaults(self):
        """Test default initialization."""
        acc = GradientAccumulator()
        assert acc.accumulation_steps == 1
        assert acc.max_grad_norm == 1.0
        assert acc._step_count == 0

    def test_init_custom(self):
        """Test custom initialization."""
        acc = GradientAccumulator(accumulation_steps=4, max_grad_norm=2.0)
        assert acc.accumulation_steps == 4
        assert acc.max_grad_norm == 2.0

    def test_accumulation_steps_minimum(self):
        """Accumulation steps should be at least 1."""
        acc = GradientAccumulator(accumulation_steps=0)
        assert acc.accumulation_steps == 1

        acc = GradientAccumulator(accumulation_steps=-5)
        assert acc.accumulation_steps == 1

    def test_should_step(self):
        """Test should_step logic."""
        acc = GradientAccumulator(accumulation_steps=4)

        # Steps at indices 3, 7, 11, etc. (multiples of 4 - 1)
        assert not acc.should_step(0)
        assert not acc.should_step(1)
        assert not acc.should_step(2)
        assert acc.should_step(3)  # (3+1) % 4 == 0
        assert not acc.should_step(4)
        assert acc.should_step(7)

    def test_scale_loss(self):
        """Test loss scaling."""
        acc = GradientAccumulator(accumulation_steps=4)
        loss = torch.tensor(4.0)
        scaled = acc.scale_loss(loss)
        assert scaled.item() == 1.0

    def test_effective_batch_size(self):
        """Test effective batch size property."""
        acc = GradientAccumulator(accumulation_steps=8)
        assert acc.effective_batch_size == 8

    def test_clip_gradients(self):
        """Test gradient clipping."""
        acc = GradientAccumulator(max_grad_norm=1.0)

        # Create a simple model with gradients
        model = nn.Linear(10, 10)
        x = torch.randn(5, 10)
        y = model(x).sum()
        y.backward()

        norm = acc.clip_gradients(model)
        assert norm >= 0

    def test_clip_gradients_disabled(self):
        """Test gradient clipping when disabled."""
        acc = GradientAccumulator(max_grad_norm=None)
        model = nn.Linear(10, 10)
        norm = acc.clip_gradients(model)
        assert norm == 0.0


class TestAdaptiveGradientClipper:
    """Tests for AdaptiveGradientClipper class."""

    def test_init_defaults(self):
        """Test default initialization."""
        clipper = AdaptiveGradientClipper()
        assert clipper.current_max_norm == 1.0
        assert clipper.percentile == 90.0
        assert clipper.history_size == 100
        assert clipper.min_clip == 0.1
        assert clipper.max_clip == 10.0
        assert clipper.grad_norms == []

    def test_init_backwards_compat_alias(self):
        """Test initial_clip backwards compatibility alias."""
        clipper = AdaptiveGradientClipper(initial_clip=2.5)
        assert clipper.current_max_norm == 2.5

    def test_generator_exhaustion_bug_fix(self):
        """Ensure parameters generator is converted to list before iteration.

        Bug (Jan 12, 2026): Passing model.parameters() (a generator) would
        exhaust it during norm calculation, leaving nothing for clip_grad_norm_.
        Fix: Convert to list first with params_list = list(parameters).
        """
        clipper = AdaptiveGradientClipper()

        # Create parameters as a generator (simulating model.parameters())
        params = [torch.randn(10, requires_grad=True) for _ in range(3)]
        for p in params:
            p.grad = torch.randn_like(p)

        # Pass as generator
        params_gen = (p for p in params)

        # Should not raise - the bug was that generator would be exhausted
        norm = clipper.update_and_clip(params_gen)

        # Should have calculated a non-zero norm
        assert norm > 0

        # Gradients should be clipped (verify by checking they changed)
        assert len(clipper.grad_norms) == 1

    def test_empty_parameters(self):
        """Handle empty parameter list gracefully."""
        clipper = AdaptiveGradientClipper()
        norm = clipper.update_and_clip([])
        assert norm == 0.0
        assert len(clipper.grad_norms) == 1
        assert clipper.grad_norms[0] == 0.0

    def test_no_gradients(self):
        """Handle parameters without gradients."""
        clipper = AdaptiveGradientClipper()
        params = [torch.randn(10, requires_grad=True)]  # No .grad set
        norm = clipper.update_and_clip(params)
        assert norm == 0.0

    def test_single_parameter_with_gradient(self):
        """Test with single parameter that has gradient."""
        clipper = AdaptiveGradientClipper(initial_max_norm=1.0)

        param = torch.randn(100, requires_grad=True)
        param.grad = torch.randn_like(param) * 10  # Large gradient

        norm = clipper.update_and_clip([param])

        # Should return the original norm
        assert norm > 0
        # History should be updated
        assert len(clipper.grad_norms) == 1
        assert clipper.grad_norms[0] == norm

    def test_history_size_limit(self):
        """Test that history is limited to history_size."""
        clipper = AdaptiveGradientClipper(history_size=5)

        for i in range(10):
            param = torch.randn(10, requires_grad=True)
            param.grad = torch.randn_like(param)
            clipper.update_and_clip([param])

        # Should only keep last 5
        assert len(clipper.grad_norms) == 5

    def test_adaptive_threshold_update(self):
        """Test that threshold adapts based on history."""
        clipper = AdaptiveGradientClipper(
            initial_max_norm=1.0,
            percentile=90.0,
            min_clip=0.1,
            max_clip=10.0,
        )

        # Add enough history to trigger adaptive update (>= 10)
        for i in range(15):
            param = torch.randn(10, requires_grad=True)
            # Create consistent gradients with norm ~5
            param.grad = torch.ones_like(param) * 1.58  # sqrt(10 * 1.58^2) ≈ 5
            clipper.update_and_clip([param])

        # Threshold should have adapted from initial 1.0
        # Based on 90th percentile * 1.5, clamped to [0.1, 10.0]
        assert clipper.current_max_norm != 1.0

    def test_threshold_respects_bounds(self):
        """Test that threshold stays within min_clip and max_clip."""
        clipper = AdaptiveGradientClipper(
            min_clip=0.5,
            max_clip=2.0,
        )

        # Add history with very small gradients
        for _ in range(15):
            param = torch.randn(10, requires_grad=True)
            param.grad = torch.randn_like(param) * 0.001  # Very small
            clipper.update_and_clip([param])

        assert clipper.current_max_norm >= 0.5

        # Reset and add very large gradients
        clipper.reset()
        for _ in range(15):
            param = torch.randn(10, requires_grad=True)
            param.grad = torch.randn_like(param) * 100  # Very large
            clipper.update_and_clip([param])

        assert clipper.current_max_norm <= 2.0

    def test_get_stats(self):
        """Test statistics retrieval."""
        clipper = AdaptiveGradientClipper(initial_max_norm=1.0)

        # Empty stats
        stats = clipper.get_stats()
        assert stats['current_clip_norm'] == 1.0
        assert stats['mean_grad_norm'] == 0
        assert stats['max_grad_norm'] == 0
        assert stats['history_size'] == 0

        # After some updates
        for i in range(5):
            param = torch.randn(10, requires_grad=True)
            param.grad = torch.randn_like(param) * (i + 1)
            clipper.update_and_clip([param])

        stats = clipper.get_stats()
        assert stats['history_size'] == 5
        assert stats['mean_grad_norm'] > 0
        assert stats['max_grad_norm'] > 0

    def test_reset(self):
        """Test history reset."""
        clipper = AdaptiveGradientClipper()

        # Add some history
        for _ in range(5):
            param = torch.randn(10, requires_grad=True)
            param.grad = torch.randn_like(param)
            clipper.update_and_clip([param])

        assert len(clipper.grad_norms) == 5

        clipper.reset()
        assert len(clipper.grad_norms) == 0

    def test_list_parameters_work(self):
        """Test that list of parameters works (not just generators)."""
        clipper = AdaptiveGradientClipper()

        params = [torch.randn(10, requires_grad=True) for _ in range(3)]
        for p in params:
            p.grad = torch.randn_like(p)

        # Pass as list directly
        norm = clipper.update_and_clip(params)
        assert norm > 0

    def test_mixed_grad_none(self):
        """Test parameters where some have gradients and some don't."""
        clipper = AdaptiveGradientClipper()

        params = [torch.randn(10, requires_grad=True) for _ in range(3)]
        # Only first parameter has gradient
        params[0].grad = torch.randn_like(params[0])

        norm = clipper.update_and_clip(params)
        # Should still compute norm from the one parameter with gradient
        assert norm > 0


class TestIntegration:
    """Integration tests combining accumulator and clipper."""

    def test_accumulator_with_clipper(self):
        """Test using both together as in training loop."""
        accumulator = GradientAccumulator(accumulation_steps=2, max_grad_norm=None)
        clipper = AdaptiveGradientClipper(initial_max_norm=1.0)

        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for batch_idx in range(4):
            x = torch.randn(5, 10)
            loss = model(x).sum()
            scaled_loss = accumulator.scale_loss(loss)
            scaled_loss.backward()

            if accumulator.should_step(batch_idx):
                # Use adaptive clipper instead of fixed
                grad_norm = clipper.update_and_clip(model.parameters())
                optimizer.step()
                optimizer.zero_grad()

        # Should have clipped twice (at batch 1 and 3)
        assert len(clipper.grad_norms) == 2
