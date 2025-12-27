"""
Evaluation Feedback Handler for Training.

Adjusts training hyperparameters based on evaluation feedback.
Subscribes to EVALUATION_COMPLETED events and dynamically adjusts
learning rate based on Elo trend.

Extracted from training_enhancements.py (December 2025).
"""

from __future__ import annotations

import logging
from typing import Any

import torch.optim as optim

logger = logging.getLogger(__name__)

__all__ = [
    "EvaluationFeedbackHandler",
    "create_evaluation_feedback_handler",
]


class EvaluationFeedbackHandler:
    """Adjusts training hyperparameters based on evaluation feedback.

    Subscribes to EVALUATION_COMPLETED events and dynamically adjusts
    learning rate based on Elo trend:
    - Rising Elo → keep or slightly increase LR
    - Flat Elo → reduce LR to fine-tune
    - Falling Elo → significantly reduce LR (potential overtraining)

    Usage:
        handler = EvaluationFeedbackHandler(optimizer, config_key="hex8_2p")
        handler.subscribe()

        # During training loop:
        if handler.should_adjust_lr():
            handler.apply_lr_adjustment()
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        config_key: str,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        elo_history_window: int = 5,
        lr_increase_factor: float = 1.1,
        lr_decrease_factor: float = 0.8,
        lr_plateau_factor: float = 0.95,
        elo_rising_threshold: float = 10.0,
        elo_falling_threshold: float = -10.0,
    ):
        """Initialize the evaluation feedback handler.

        Args:
            optimizer: The PyTorch optimizer to adjust
            config_key: Board configuration key (e.g., "hex8_2p")
            min_lr: Minimum learning rate floor
            max_lr: Maximum learning rate ceiling
            elo_history_window: Number of Elo readings to track
            lr_increase_factor: LR multiplier when Elo is rising
            lr_decrease_factor: LR multiplier when Elo is falling
            lr_plateau_factor: LR multiplier when Elo is flat
            elo_rising_threshold: Elo change to consider "rising"
            elo_falling_threshold: Elo change to consider "falling"
        """
        self.optimizer = optimizer
        self.config_key = config_key
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.elo_history_window = elo_history_window
        self.lr_increase_factor = lr_increase_factor
        self.lr_decrease_factor = lr_decrease_factor
        self.lr_plateau_factor = lr_plateau_factor
        self.elo_rising_threshold = elo_rising_threshold
        self.elo_falling_threshold = elo_falling_threshold

        # State
        self._elo_history: list[float] = []
        self._pending_adjustment: float | None = None
        self._last_adjustment_epoch: int = -1
        self._subscribed = False

        logger.debug(
            f"[EvaluationFeedbackHandler] Initialized for {config_key} "
            f"(LR range: {min_lr:.1e} - {max_lr:.1e})"
        )

    def subscribe(self) -> bool:
        """Subscribe to EVALUATION_COMPLETED events.

        Returns:
            True if subscription succeeded, False otherwise.
        """
        if self._subscribed:
            return True

        try:
            from app.coordination.event_router import get_router, subscribe, DataEventType

            router = get_router()
            if router is None:
                logger.debug("[EvaluationFeedbackHandler] Event router not available")
                return False

            def on_evaluation_completed(event):
                """Handle EVALUATION_COMPLETED events."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config", "")

                # Only respond to our config's events
                if event_config != self.config_key:
                    return

                elo = payload.get("elo", 0.0)
                self._record_elo(elo)

            subscribe(DataEventType.EVALUATION_COMPLETED, on_evaluation_completed)

            # Also subscribe to HYPERPARAMETER_UPDATED for runtime LR updates (December 2025)
            # This closes the feedback loop: GauntletFeedbackController -> HYPERPARAMETER_UPDATED -> runtime LR change
            def on_hyperparameter_updated(event):
                """Handle HYPERPARAMETER_UPDATED for runtime hyperparameter changes."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config", "")

                # Only respond to our config's events
                if event_config != self.config_key:
                    return

                parameter = payload.get("parameter", "")
                new_value = payload.get("new_value", None)
                reason = payload.get("reason", "unknown")

                if parameter == "learning_rate" and new_value is not None:
                    try:
                        new_lr = float(new_value)
                        # Clamp to valid range
                        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
                        old_lr = self.optimizer.param_groups[0]["lr"]

                        # Apply immediately to optimizer
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr

                        logger.info(
                            f"[EvaluationFeedbackHandler] Runtime LR update for {self.config_key}: "
                            f"{old_lr:.2e} -> {new_lr:.2e} (reason: {reason})"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[EvaluationFeedbackHandler] Invalid LR value: {new_value} ({e})")

                # P0.2 (Dec 2025): Also handle lr_multiplier for relative LR adjustments
                elif parameter == "lr_multiplier" and new_value is not None:
                    try:
                        multiplier = float(new_value)
                        old_lr = self.optimizer.param_groups[0]["lr"]
                        new_lr = old_lr * multiplier
                        # Clamp to valid range
                        new_lr = max(self.min_lr, min(self.max_lr, new_lr))

                        # Apply immediately to optimizer
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr

                        logger.info(
                            f"[EvaluationFeedbackHandler] Runtime LR multiplier for {self.config_key}: "
                            f"{old_lr:.2e} * {multiplier:.2f} = {new_lr:.2e} (reason: {reason})"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[EvaluationFeedbackHandler] Invalid LR multiplier: {new_value} ({e})")

            subscribe(DataEventType.HYPERPARAMETER_UPDATED, on_hyperparameter_updated)

            # December 2025: Subscribe to ADAPTIVE_PARAMS_CHANGED for Elo-based training param adjustments
            def on_adaptive_params_changed(event):
                """Handle ADAPTIVE_PARAMS_CHANGED - training params adjusted based on Elo trends."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config", payload.get("config_key", ""))

                # Only respond to our config's events
                if event_config != self.config_key:
                    return

                # Extract parameter changes
                parameter = payload.get("parameter", "")
                old_value = payload.get("old_value")
                new_value = payload.get("new_value")
                reason = payload.get("reason", "elo_adaptive")
                elo_delta = payload.get("elo_delta", 0)

                if parameter == "learning_rate" and new_value is not None:
                    try:
                        new_lr = float(new_value)
                        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
                        old_lr = self.optimizer.param_groups[0]["lr"]

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr

                        logger.info(
                            f"[EvaluationFeedbackHandler] Adaptive LR for {self.config_key}: "
                            f"{old_lr:.2e} -> {new_lr:.2e} (elo_delta={elo_delta:+.1f}, reason={reason})"
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"[EvaluationFeedbackHandler] Invalid adaptive LR: {new_value} ({e})")

                elif parameter in ("batch_size", "weight_decay", "gradient_clip"):
                    # Log other adaptive parameter changes for visibility
                    logger.info(
                        f"[EvaluationFeedbackHandler] Adaptive param for {self.config_key}: "
                        f"{parameter} {old_value} -> {new_value} (elo_delta={elo_delta:+.1f})"
                    )

            subscribe(DataEventType.ADAPTIVE_PARAMS_CHANGED, on_adaptive_params_changed)
            self._subscribed = True

            logger.info(
                f"[EvaluationFeedbackHandler] Subscribed to EVALUATION_COMPLETED + HYPERPARAMETER_UPDATED "
                f"+ ADAPTIVE_PARAMS_CHANGED for {self.config_key}"
            )
            return True

        except ImportError:
            logger.debug("[EvaluationFeedbackHandler] Event system not available")
            return False
        except Exception as e:
            logger.debug(f"[EvaluationFeedbackHandler] Failed to subscribe: {e}")
            return False

    def _record_elo(self, elo: float) -> None:
        """Record a new Elo rating and compute LR adjustment."""
        self._elo_history.append(elo)

        # Keep only the last N readings
        if len(self._elo_history) > self.elo_history_window:
            self._elo_history = self._elo_history[-self.elo_history_window:]

        # Need at least 2 readings to compute trend
        if len(self._elo_history) < 2:
            return

        # Compute Elo trend (simple moving average of differences)
        elo_changes = [
            self._elo_history[i] - self._elo_history[i - 1]
            for i in range(1, len(self._elo_history))
        ]
        avg_elo_change = sum(elo_changes) / len(elo_changes)

        # Determine adjustment based on trend
        if avg_elo_change >= self.elo_rising_threshold:
            # Elo is rising - keep or slightly increase LR
            factor = self.lr_increase_factor
            trend = "rising"
        elif avg_elo_change <= self.elo_falling_threshold:
            # Elo is falling - reduce LR significantly
            factor = self.lr_decrease_factor
            trend = "falling"
        else:
            # Elo is flat - slight reduction to fine-tune
            factor = self.lr_plateau_factor
            trend = "plateau"

        self._pending_adjustment = factor

        logger.info(
            f"[EvaluationFeedbackHandler] Elo trend for {self.config_key}: "
            f"{trend} (Δ={avg_elo_change:+.1f}), "
            f"pending LR adjustment: ×{factor:.2f}"
        )

    def should_adjust_lr(self) -> bool:
        """Check if there's a pending LR adjustment."""
        return self._pending_adjustment is not None

    def apply_lr_adjustment(self, current_epoch: int = 0) -> float | None:
        """Apply the pending LR adjustment.

        Args:
            current_epoch: Current training epoch (for debouncing)

        Returns:
            New learning rate if adjusted, None otherwise.
        """
        if self._pending_adjustment is None:
            return None

        # Debounce: don't adjust more than once per epoch
        if current_epoch <= self._last_adjustment_epoch:
            return None

        factor = self._pending_adjustment
        self._pending_adjustment = None
        self._last_adjustment_epoch = current_epoch

        # Get current LR and compute new LR
        current_lr = self.optimizer.param_groups[0]["lr"]
        new_lr = current_lr * factor

        # Clamp to valid range
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))

        # Apply adjustment
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        logger.info(
            f"[EvaluationFeedbackHandler] Adjusted LR for {self.config_key}: "
            f"{current_lr:.2e} → {new_lr:.2e} (×{factor:.2f})"
        )

        return new_lr

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def get_elo_history(self) -> list[float]:
        """Get recorded Elo history."""
        return list(self._elo_history)

    def get_status(self) -> dict[str, Any]:
        """Get handler status."""
        return {
            "config_key": self.config_key,
            "subscribed": self._subscribed,
            "current_lr": self.get_current_lr(),
            "elo_history": self._elo_history,
            "pending_adjustment": self._pending_adjustment,
            "last_adjustment_epoch": self._last_adjustment_epoch,
        }


def create_evaluation_feedback_handler(
    optimizer: optim.Optimizer,
    config_key: str,
    **kwargs: Any,
) -> EvaluationFeedbackHandler:
    """Factory function to create and subscribe an EvaluationFeedbackHandler.

    Args:
        optimizer: PyTorch optimizer
        config_key: Board configuration (e.g., "hex8_2p")
        **kwargs: Additional handler configuration

    Returns:
        Configured and subscribed handler
    """
    handler = EvaluationFeedbackHandler(optimizer, config_key, **kwargs)
    handler.subscribe()
    return handler
