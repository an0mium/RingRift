"""Learning Rate Finder for RingRift AI.

Extracted from advanced_training.py (December 2025).

Implements the technique from "Cyclical Learning Rates for Training
Neural Networks" (Smith, 2017). Gradually increases LR from min to max
and records the loss at each step.

Usage:
    from app.training.lr_finder import LRFinder, LRFinderResult

    finder = LRFinder(model, optimizer, criterion)
    result = finder.range_test(train_loader, min_lr=1e-7, max_lr=10)

    # Use suggested LR
    suggested_lr = result.suggested_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = suggested_lr
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class LRFinderResult:
    """Results from learning rate finder."""
    lrs: list[float]
    losses: list[float]
    suggested_lr: float
    min_lr: float
    max_lr: float
    best_lr: float  # LR at minimum loss
    steepest_lr: float  # LR at steepest gradient


class LRFinder:
    """
    Learning Rate Finder for optimal LR range detection.

    Implements the technique from "Cyclical Learning Rates for Training
    Neural Networks" (Smith, 2017). Gradually increases LR from min to max
    and records the loss at each step.

    Usage:
        finder = LRFinder(model, optimizer, criterion)
        result = finder.range_test(train_loader, min_lr=1e-7, max_lr=10)

        # Use suggested LR
        suggested_lr = result.suggested_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = suggested_lr
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device | None = None,
    ):
        """
        Args:
            model: Model to find LR for
            optimizer: Optimizer to use
            criterion: Loss function
            device: Device for training
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or next(model.parameters()).device

        # Save initial state for restoration
        self._initial_model_state = copy.deepcopy(model.state_dict())
        self._initial_optimizer_state = copy.deepcopy(optimizer.state_dict())

    def range_test(
        self,
        train_loader: DataLoader,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_iter: int | None = None,
        step_mode: str = "exp",
        smooth_factor: float = 0.05,
        diverge_threshold: float = 5.0,
    ) -> LRFinderResult:
        """
        Run LR range test.

        Args:
            train_loader: Training data loader
            min_lr: Minimum learning rate to test
            max_lr: Maximum learning rate to test
            num_iter: Number of iterations (default: len(train_loader))
            step_mode: "exp" for exponential, "linear" for linear increase
            smooth_factor: Smoothing factor for loss (0-1)
            diverge_threshold: Stop if loss exceeds this multiple of min loss

        Returns:
            LRFinderResult with LRs, losses, and suggestions
        """
        # Set up
        num_iter = num_iter or len(train_loader)
        self.model.train()

        # Initialize tracking
        lrs = []
        losses = []
        best_loss = float('inf')
        smoothed_loss = 0.0

        # Calculate LR multiplier
        if step_mode == "exp":
            lr_mult = (max_lr / min_lr) ** (1 / num_iter)
        else:
            lr_step = (max_lr - min_lr) / num_iter

        # Set initial LR
        current_lr = min_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        # Iterate through data
        data_iter = iter(train_loader)
        for i in range(num_iter):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Forward pass
            loss = self._train_step(batch)

            # Smooth loss
            if i == 0:
                smoothed_loss = loss
            else:
                smoothed_loss = smooth_factor * loss + (1 - smooth_factor) * smoothed_loss

            # Record
            lrs.append(current_lr)
            losses.append(smoothed_loss)

            # Track best
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            # Check for divergence
            if smoothed_loss > diverge_threshold * best_loss:
                logger.info(f"LR finder stopped early at iter {i} (loss diverged)")
                break

            # Update LR
            if step_mode == "exp":
                current_lr *= lr_mult
            else:
                current_lr += lr_step

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

        # Restore initial state
        self.model.load_state_dict(self._initial_model_state)
        self.optimizer.load_state_dict(self._initial_optimizer_state)

        # Analyze results
        result = self._analyze_results(lrs, losses, min_lr, max_lr)
        return result

    def _train_step(self, batch: Any) -> float:
        """Execute a single training step."""
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs = batch[0]
                targets = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            inputs = batch.get('features', batch.get('input'))
            targets = batch.get('targets', batch.get('labels'))
        else:
            inputs = batch
            targets = None

        # Move to device
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(self.device)
        if targets is not None and isinstance(targets, torch.Tensor):
            targets = targets.to(self.device)

        # Forward
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        # Handle tuple outputs (policy, value)
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Use policy for LR finding

        # Compute loss
        if targets is not None:
            loss = self.criterion(outputs, targets)
        else:
            # Self-supervised or unsupervised loss
            loss = outputs.mean() if hasattr(outputs, 'mean') else outputs

        # Backward
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _analyze_results(
        self,
        lrs: list[float],
        losses: list[float],
        min_lr: float,
        max_lr: float,
    ) -> LRFinderResult:
        """Analyze LR finder results to suggest optimal LR."""
        if not lrs or not losses:
            return LRFinderResult(
                lrs=[], losses=[], suggested_lr=1e-3,
                min_lr=min_lr, max_lr=max_lr,
                best_lr=1e-3, steepest_lr=1e-3,
            )

        # Find LR at minimum loss
        min_loss_idx = np.argmin(losses)
        best_lr = lrs[min_loss_idx]

        # Find steepest gradient (maximum loss decrease rate)
        # Use gradient of smoothed loss
        if len(losses) > 5:
            log_lrs = np.log10(lrs)
            gradients = np.gradient(losses, log_lrs)

            # Find minimum gradient (steepest descent)
            # Only consider first 80% to avoid divergence region
            cutoff = int(len(gradients) * 0.8)
            steepest_idx = np.argmin(gradients[:cutoff])
            steepest_lr = lrs[steepest_idx]
        else:
            steepest_lr = best_lr

        # Suggested LR: one order of magnitude below steepest point
        # This is a safe default that usually works well
        suggested_lr = steepest_lr / 10

        # Clamp to valid range
        suggested_lr = max(min_lr, min(suggested_lr, best_lr))

        logger.info(
            f"LR Finder: best_lr={best_lr:.2e}, steepest_lr={steepest_lr:.2e}, "
            f"suggested_lr={suggested_lr:.2e}"
        )

        return LRFinderResult(
            lrs=lrs,
            losses=losses,
            suggested_lr=suggested_lr,
            min_lr=min_lr,
            max_lr=max_lr,
            best_lr=best_lr,
            steepest_lr=steepest_lr,
        )

    def plot(self, result: LRFinderResult, save_path: str | None = None):
        """Plot LR finder results."""
        try:
            import matplotlib.pyplot as plt

            _fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(result.lrs, result.losses, 'b-', linewidth=2)
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Loss')
            ax.set_title('Learning Rate Finder')

            # Mark suggested LR
            ax.axvline(x=result.suggested_lr, color='r', linestyle='--',
                      label=f'Suggested LR: {result.suggested_lr:.2e}')
            ax.axvline(x=result.best_lr, color='g', linestyle=':',
                      label=f'Best LR: {result.best_lr:.2e}')

            ax.legend()
            ax.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Saved LR finder plot to {save_path}")

            plt.close()

        except ImportError:
            logger.warning("matplotlib not available for plotting")


__all__ = [
    "LRFinder",
    "LRFinderResult",
]
