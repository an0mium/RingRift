"""
Model Ensemble for Self-Play.

Ensemble of models for diverse self-play opponents.
Using an ensemble provides more diverse training data and
prevents overfitting to a single opponent's weaknesses.

Extracted from training_enhancements.py (December 2025).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from app.utils.torch_utils import safe_load_checkpoint

logger = logging.getLogger(__name__)

__all__ = ["ModelEnsemble"]


class ModelEnsemble:
    """
    Ensemble of models for diverse self-play opponents.

    Using an ensemble for self-play provides more diverse training data
    and prevents overfitting to a single opponent's weaknesses.

    Usage:
        ensemble = ModelEnsemble(model_class=RingRiftCNN_v2)

        # Add models
        ensemble.add_model(best_model, weight=0.5)
        ensemble.add_model(previous_model, weight=0.3)
        ensemble.add_model(random_model, weight=0.2)

        # Sample opponent for self-play game
        opponent = ensemble.sample_model()
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ):
        """
        Args:
            model_class: Class to instantiate models from
            model_kwargs: Arguments for model constructor
            device: Device for models
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.device = device or torch.device('cpu')

        self._models: list[nn.Module] = []
        self._weights: list[float] = []
        self._names: list[str] = []

    def add_model(
        self,
        model_or_path: nn.Module | str | Path,
        weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        """
        Add a model to the ensemble.

        Args:
            model_or_path: Model instance or path to checkpoint
            weight: Sampling weight (higher = more likely to be chosen)
            name: Optional name for the model
        """
        if isinstance(model_or_path, (str, Path)):
            # Load from checkpoint
            model = self.model_class(**self.model_kwargs).to(self.device)
            ckpt = safe_load_checkpoint(model_or_path, map_location=self.device, warn_on_unsafe=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state_dict)
            model.eval()
            model_name = name or Path(model_or_path).stem
        else:
            model = model_or_path.to(self.device)
            model.eval()
            model_name = name or f"model_{len(self._models)}"

        self._models.append(model)
        self._weights.append(weight)
        self._names.append(model_name)

        logger.info(f"Added model '{model_name}' to ensemble with weight {weight}")

    def sample_model(self) -> tuple[nn.Module, str]:
        """
        Sample a model from the ensemble based on weights.

        Returns:
            Tuple of (model, model_name)
        """
        if not self._models:
            raise ValueError("No models in ensemble")

        # Normalize weights
        total = sum(self._weights)
        probs = [w / total for w in self._weights]

        idx = np.random.choice(len(self._models), p=probs)
        return self._models[idx], self._names[idx]

    def get_model(self, name: str) -> nn.Module | None:
        """Get a specific model by name."""
        for i, n in enumerate(self._names):
            if n == name:
                return self._models[i]
        return None

    def update_weight(self, name: str, weight: float) -> None:
        """Update the weight of a specific model."""
        for i, n in enumerate(self._names):
            if n == name:
                self._weights[i] = weight
                return

    @property
    def num_models(self) -> int:
        """Number of models in ensemble."""
        return len(self._models)

    @property
    def model_names(self) -> list[str]:
        """Names of all models in ensemble."""
        return list(self._names)
