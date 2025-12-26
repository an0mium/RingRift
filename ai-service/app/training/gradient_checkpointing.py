"""Gradient Checkpointing for Memory-Efficient Training.

Extracted from advanced_training.py (December 2025).

Trades compute for memory by not storing intermediate activations
and recomputing them during backward pass.

Usage:
    from app.training.gradient_checkpointing import (
        GradientCheckpointing,
        estimate_memory_savings,
    )

    checkpointing = GradientCheckpointing(model)
    checkpointing.enable()

    # During training, model forward/backward will use checkpointing
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GradientCheckpointing:
    """
    Memory-efficient training via gradient checkpointing.

    Trades compute for memory by not storing intermediate activations
    and recomputing them during backward pass.

    Usage:
        checkpointing = GradientCheckpointing(model)
        checkpointing.enable()

        # During training
        output = checkpointing.checkpoint_forward(model.layer, input)
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_layers: list[str] | None = None,
    ):
        """
        Args:
            model: Model to apply checkpointing to
            checkpoint_layers: Names of layers to checkpoint (None = auto-detect)
        """
        self.model = model
        self.checkpoint_layers = checkpoint_layers
        self._enabled = False
        self._original_forward = {}

    def enable(self) -> None:
        """Enable gradient checkpointing."""
        if self._enabled:
            return

        # Find layers to checkpoint
        layers = self._find_checkpoint_layers()

        for name, module in layers:
            # Store original forward
            self._original_forward[name] = module.forward

            # Replace with checkpointed version
            module.forward = self._make_checkpointed_forward(module)

            logger.debug(f"Enabled checkpointing for {name}")

        self._enabled = True
        logger.info(f"Gradient checkpointing enabled for {len(layers)} layers")

    def disable(self) -> None:
        """Disable gradient checkpointing."""
        if not self._enabled:
            return

        # Restore original forwards
        for name, original_forward in self._original_forward.items():
            # Find module by name
            module = dict(self.model.named_modules()).get(name)
            if module:
                module.forward = original_forward

        self._original_forward.clear()
        self._enabled = False
        logger.info("Gradient checkpointing disabled")

    def _find_checkpoint_layers(self) -> list[tuple[str, nn.Module]]:
        """Find layers suitable for checkpointing."""
        if self.checkpoint_layers:
            # Use specified layers
            modules = dict(self.model.named_modules())
            return [(name, modules[name]) for name in self.checkpoint_layers
                    if name in modules]

        # Auto-detect: checkpoint transformer blocks, conv blocks, etc.
        layers = []
        for name, module in self.model.named_modules():
            # Skip container modules
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                continue

            # Checkpoint large layers
            if any(keyword in name.lower() for keyword in
                   ['block', 'layer', 'encoder', 'decoder', 'transformer']) or isinstance(module, (nn.TransformerEncoderLayer,
                                    nn.TransformerDecoderLayer)):
                layers.append((name, module))

        # If no blocks found, checkpoint every N layers
        if not layers:
            all_modules = list(self.model.named_modules())
            # Skip first (model itself) and last few modules
            step = max(1, len(all_modules) // 4)
            for i in range(step, len(all_modules) - 2, step):
                name, module = all_modules[i]
                if hasattr(module, 'forward'):
                    layers.append((name, module))

        return layers

    def _make_checkpointed_forward(
        self,
        module: nn.Module,
    ) -> Callable:
        """Create a checkpointed forward function."""
        original_forward = module.forward

        def checkpointed_forward(*args, **kwargs):
            # Only use checkpointing during training
            if self.model.training:
                # Use PyTorch's checkpoint utility
                return torch.utils.checkpoint.checkpoint(
                    original_forward,
                    *args,
                    use_reentrant=False,
                    **kwargs,
                )
            else:
                return original_forward(*args, **kwargs)

        return checkpointed_forward

    @staticmethod
    def checkpoint_sequential(
        functions: list[nn.Module],
        segments: int,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Checkpoint a sequential list of functions.

        Args:
            functions: List of modules to run sequentially
            segments: Number of checkpoint segments
            input: Input tensor

        Returns:
            Output after all functions
        """
        return torch.utils.checkpoint.checkpoint_sequential(
            functions, segments, input, use_reentrant=False
        )

    @property
    def is_enabled(self) -> bool:
        """Check if checkpointing is enabled."""
        return self._enabled


def estimate_memory_savings(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
) -> dict[str, float]:
    """
    Estimate memory savings from gradient checkpointing.

    Args:
        model: Model to analyze
        input_shape: Input tensor shape (without batch)
        device: Device to use

    Returns:
        Dictionary with memory estimates
    """
    import gc

    torch.cuda.empty_cache() if device.type == 'cuda' else None
    gc.collect()

    # Measure without checkpointing
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()

    dummy_input = torch.randn(1, *input_shape, device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Forward + backward without checkpointing
    output = model_copy(dummy_input)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.sum()
    loss.backward()

    if device.type == 'cuda':
        normal_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        normal_memory = 0

    del model_copy, dummy_input, output, loss
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    gc.collect()

    # Measure with checkpointing
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()
    checkpointing = GradientCheckpointing(model_copy)
    checkpointing.enable()

    dummy_input = torch.randn(1, *input_shape, device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    output = model_copy(dummy_input)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.sum()
    loss.backward()

    if device.type == 'cuda':
        checkpoint_memory = torch.cuda.max_memory_allocated() / 1024**2
    else:
        checkpoint_memory = 0

    savings_pct = (1 - checkpoint_memory / normal_memory) * 100 if normal_memory > 0 else 0

    return {
        'normal_memory_mb': normal_memory,
        'checkpoint_memory_mb': checkpoint_memory,
        'savings_mb': normal_memory - checkpoint_memory,
        'savings_percent': savings_pct,
    }


__all__ = [
    "GradientCheckpointing",
    "estimate_memory_savings",
]
