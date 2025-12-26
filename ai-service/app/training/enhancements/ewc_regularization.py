"""
EWC (Elastic Weight Consolidation) for Continual Learning.

Prevents catastrophic forgetting when training on new data by
penalizing changes to important parameters.

Extracted from training_enhancements.py (December 2025).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data

from app.utils.torch_utils import safe_load_checkpoint

logger = logging.getLogger(__name__)

__all__ = ["EWCRegularizer"]


class EWCRegularizer:
    """
    Elastic Weight Consolidation for continual learning.

    Prevents catastrophic forgetting when training on new data by
    penalizing changes to important parameters.

    Usage:
        ewc = EWCRegularizer(model)

        # After training on task 1
        ewc.compute_fisher(dataloader_task1)

        # When training on task 2
        loss = task_loss + ewc.penalty(model)
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        normalize_fisher: bool = True,
    ):
        """
        Args:
            model: Model to apply EWC to
            lambda_ewc: Importance weight for EWC penalty
            normalize_fisher: Normalize Fisher information matrix
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.normalize_fisher = normalize_fisher

        self._fisher: dict[str, torch.Tensor] = {}
        self._optimal_params: dict[str, torch.Tensor] = {}
        self._computed = False

    def compute_fisher(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module | None = None,
        num_samples: int = 1000,
        device: torch.device | None = None,
    ) -> None:
        """
        Compute Fisher information matrix from dataloader.

        Args:
            dataloader: DataLoader for computing Fisher
            criterion: Loss function (default: cross entropy)
            num_samples: Number of samples to use
            device: Device for computation
        """
        if device is None:
            device = next(self.model.parameters()).device

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Store optimal parameters
        self._optimal_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # Initialize Fisher to zero
        self._fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        samples_seen = 0

        for inputs, targets in dataloader:
            if samples_seen >= num_samples:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            # Use log-softmax for computing Fisher
            log_probs = torch.log_softmax(outputs, dim=-1)

            # Sample from output distribution
            labels = torch.distributions.Categorical(logits=outputs).sample()
            loss = -log_probs[range(len(labels)), labels].mean()
            loss.backward()

            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self._fisher[name] += param.grad.pow(2)

            samples_seen += len(inputs)

        # Normalize Fisher
        for name in self._fisher:
            self._fisher[name] /= samples_seen

            if self.normalize_fisher:
                max_val = self._fisher[name].max()
                if max_val > 0:
                    self._fisher[name] /= max_val

        self._computed = True
        logger.info(f"Computed Fisher information from {samples_seen} samples")

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty for current model parameters.

        Args:
            model: Model to compute penalty for

        Returns:
            EWC penalty term
        """
        if not self._computed:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(model.parameters()).device)

        for name, param in model.named_parameters():
            if name in self._fisher and param.requires_grad:
                diff = param - self._optimal_params[name]
                penalty += (self._fisher[name] * diff.pow(2)).sum()

        return 0.5 * self.lambda_ewc * penalty

    def save_state(self, path: str | Path) -> None:
        """Save EWC state to file."""
        state = {
            'fisher': self._fisher,
            'optimal_params': self._optimal_params,
            'lambda_ewc': self.lambda_ewc,
            'computed': self._computed,
        }
        torch.save(state, path)

    def load_state(self, path: str | Path) -> None:
        """Load EWC state from file."""
        state = safe_load_checkpoint(path, warn_on_unsafe=False)
        self._fisher = state['fisher']
        self._optimal_params = state['optimal_params']
        self.lambda_ewc = state['lambda_ewc']
        self._computed = state['computed']
