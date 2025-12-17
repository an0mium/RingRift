"""Training Triggers - Adapter Layer for Unified Signal Computation.

This module provides the stable API for training decisions.
Internally delegates to UnifiedSignalComputer for actual computation.

The 3 core signals are:
1. Data Freshness: New games available since last training
2. Model Staleness: Time since last training for config
3. Performance Regression: Elo/win rate below acceptable threshold

Usage:
    from app.training.training_triggers import TrainingTriggers, should_train

    triggers = TrainingTriggers(config)

    # Check if training should run
    decision = triggers.should_train("square8_2p", current_state)
    if decision.should_train:
        print(f"Training triggered by: {decision.reason}")

See: app.config.thresholds for canonical threshold constants.
See: app.training.unified_signals for the centralized signal computation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .unified_signals import (
    get_signal_computer,
    TrainingUrgency,
    TrainingSignals,
    UnifiedSignalComputer,
)

logger = logging.getLogger(__name__)

# Import canonical thresholds
try:
    from app.config.thresholds import (
        INITIAL_ELO_RATING,
        TRAINING_TRIGGER_GAMES,
        TRAINING_STALENESS_HOURS,
        MIN_WIN_RATE_PROMOTE,
        TRAINING_MIN_INTERVAL_SECONDS,
        TRAINING_BOOTSTRAP_GAMES,
    )
    DEFAULT_FRESHNESS_THRESHOLD = TRAINING_TRIGGER_GAMES
    DEFAULT_STALENESS_HOURS = TRAINING_STALENESS_HOURS
    DEFAULT_MIN_WIN_RATE = MIN_WIN_RATE_PROMOTE
    DEFAULT_MIN_INTERVAL_MINUTES = TRAINING_MIN_INTERVAL_SECONDS / 60
    DEFAULT_BOOTSTRAP_THRESHOLD = TRAINING_BOOTSTRAP_GAMES
except ImportError:
    INITIAL_ELO_RATING = 1500.0
    DEFAULT_FRESHNESS_THRESHOLD = 500
    DEFAULT_STALENESS_HOURS = 6
    DEFAULT_MIN_WIN_RATE = 0.45
    DEFAULT_MIN_INTERVAL_MINUTES = 20
    DEFAULT_BOOTSTRAP_THRESHOLD = 50

# Constants
AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TriggerConfig:
    """Configuration for training triggers.

    Note: Defaults sourced from app.config.thresholds.
    """
    # Data freshness
    freshness_threshold: int = DEFAULT_FRESHNESS_THRESHOLD
    freshness_weight: float = 1.0

    # Model staleness
    staleness_hours: float = DEFAULT_STALENESS_HOURS
    staleness_weight: float = 0.8

    # Performance regression
    min_win_rate: float = DEFAULT_MIN_WIN_RATE
    regression_weight: float = 1.5  # Higher weight for regression

    # Global constraints
    min_interval_minutes: float = DEFAULT_MIN_INTERVAL_MINUTES
    max_concurrent_training: int = 3

    # Bootstrap (new configs with no models)
    bootstrap_threshold: int = DEFAULT_BOOTSTRAP_THRESHOLD


@dataclass
class TriggerDecision:
    """Result of training trigger evaluation."""
    should_train: bool
    reason: str
    signal_scores: Dict[str, float] = field(default_factory=dict)
    config_key: str = ""
    priority: float = 0.0  # Higher = more urgent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_train": self.should_train,
            "reason": self.reason,
            "signal_scores": self.signal_scores,
            "config_key": self.config_key,
            "priority": self.priority,
        }


@dataclass
class ConfigState:
    """State for a single board/player configuration.

    Note: Uses INITIAL_ELO_RATING from app.config.thresholds as default.
    """
    config_key: str
    games_since_training: int = 0
    last_training_time: float = 0
    last_training_games: int = 0
    model_count: int = 0
    current_elo: float = INITIAL_ELO_RATING
    win_rate: float = 0.5
    win_rate_trend: float = 0.0


class TrainingTriggers:
    """Simplified training trigger system with 3 core signals.

    Delegates actual computation to UnifiedSignalComputer for consistency
    across all training decision systems.
    """

    def __init__(self, config: Optional[TriggerConfig] = None):
        self.config = config or TriggerConfig()
        self._config_states: Dict[str, ConfigState] = {}
        self._last_training_times: Dict[str, float] = {}
        # Delegate to unified signal computer
        self._signal_computer = get_signal_computer()

    def get_config_state(self, config_key: str) -> ConfigState:
        """Get or create state for a config."""
        if config_key not in self._config_states:
            self._config_states[config_key] = ConfigState(config_key=config_key)
        return self._config_states[config_key]

    def update_config_state(
        self,
        config_key: str,
        games_count: Optional[int] = None,
        elo: Optional[float] = None,
        win_rate: Optional[float] = None,
        model_count: Optional[int] = None,
    ) -> None:
        """Update state for a config.

        Updates both local state and the unified signal computer.
        """
        state = self.get_config_state(config_key)

        if games_count is not None:
            state.games_since_training = games_count - state.last_training_games

        if elo is not None:
            state.current_elo = elo

        if win_rate is not None:
            old_win_rate = state.win_rate
            state.win_rate = win_rate
            # Simple trend: difference from last update
            state.win_rate_trend = win_rate - old_win_rate

        if model_count is not None:
            state.model_count = model_count

        # Sync with unified signal computer
        self._signal_computer.update_config_state(
            config_key=config_key,
            model_count=model_count,
            current_elo=elo,
            win_rate=win_rate,
        )

    def record_training_complete(
        self,
        config_key: str,
        games_at_training: int,
        new_elo: Optional[float] = None,
    ) -> None:
        """Record that training completed for a config.

        Updates both local state and the unified signal computer.
        """
        state = self.get_config_state(config_key)
        state.last_training_time = time.time()
        state.last_training_games = games_at_training
        state.games_since_training = 0
        self._last_training_times[config_key] = time.time()

        # Sync with unified signal computer
        self._signal_computer.record_training_started(games_at_training, config_key)
        self._signal_computer.record_training_completed(new_elo, config_key)

    def should_train(self, config_key: str, state: Optional[ConfigState] = None) -> TriggerDecision:
        """Evaluate whether training should run for a config.

        Uses UnifiedSignalComputer for consistent signal computation.
        The 3 core signals are:
        1. Data Freshness: Are there enough new games?
        2. Model Staleness: Has it been too long since training?
        3. Performance Regression: Is the model underperforming?

        Returns a TriggerDecision with the result and reasoning.
        """
        if state is None:
            state = self.get_config_state(config_key)

        # Compute current games count from state
        current_games = state.last_training_games + state.games_since_training

        # Delegate to unified signal computer
        signals = self._signal_computer.compute_signals(
            current_games=current_games,
            current_elo=state.current_elo,
            config_key=config_key,
            win_rate=state.win_rate,
            model_count=state.model_count,
        )

        # Build signal scores for backward compatibility
        signal_scores: Dict[str, float] = {
            "freshness": signals.games_threshold_ratio,
            "staleness": signals.staleness_ratio,
            "regression": 1.0 if signals.win_rate_regression or signals.elo_regression_detected else 0.0,
        }

        # Add bootstrap indicator
        if signals.is_bootstrap:
            signal_scores["bootstrap"] = 1.0

        return TriggerDecision(
            should_train=signals.should_train,
            reason=signals.reason,
            signal_scores=signal_scores,
            config_key=config_key,
            priority=signals.priority,
        )

    def get_training_queue(self) -> List[TriggerDecision]:
        """Get all configs that should train, sorted by priority."""
        decisions = []

        for config_key in self._config_states:
            decision = self.should_train(config_key)
            if decision.should_train:
                decisions.append(decision)

        # Sort by priority (highest first)
        decisions.sort(key=lambda d: d.priority, reverse=True)
        return decisions

    def get_next_training_config(self) -> Optional[TriggerDecision]:
        """Get the highest priority config that should train."""
        queue = self.get_training_queue()
        return queue[0] if queue else None

    def get_urgency(self, config_key: str) -> TrainingUrgency:
        """Get current training urgency level for a config.

        Returns:
            TrainingUrgency enum value
        """
        state = self.get_config_state(config_key)
        current_games = state.last_training_games + state.games_since_training

        signals = self._signal_computer.compute_signals(
            current_games=current_games,
            current_elo=state.current_elo,
            config_key=config_key,
            win_rate=state.win_rate,
            model_count=state.model_count,
        )
        return signals.urgency

    def get_detailed_status(self, config_key: str) -> Dict[str, Any]:
        """Get detailed status for logging/debugging.

        Returns a dictionary with all signal details.
        """
        state = self.get_config_state(config_key)
        current_games = state.last_training_games + state.games_since_training

        signals = self._signal_computer.compute_signals(
            current_games=current_games,
            current_elo=state.current_elo,
            config_key=config_key,
            win_rate=state.win_rate,
            model_count=state.model_count,
        )

        return {
            "should_train": signals.should_train,
            "urgency": signals.urgency.value,
            "reason": signals.reason,
            "games_ratio": signals.games_threshold_ratio,
            "games_since_training": signals.games_since_last_training,
            "games_threshold": signals.games_threshold,
            "elo_trend": signals.elo_trend,
            "time_threshold_met": signals.time_threshold_met,
            "staleness_hours": signals.staleness_hours,
            "win_rate": signals.win_rate,
            "win_rate_regression": signals.win_rate_regression,
            "elo_regression_detected": signals.elo_regression_detected,
            "priority": signals.priority,
            "is_bootstrap": signals.is_bootstrap,
        }


# Convenience singleton
_default_triggers: Optional[TrainingTriggers] = None


def get_training_triggers(config: Optional[TriggerConfig] = None) -> TrainingTriggers:
    """Get the default training triggers instance."""
    global _default_triggers
    if _default_triggers is None:
        _default_triggers = TrainingTriggers(config)
    return _default_triggers


def should_train(config_key: str, games_since_training: int, **kwargs) -> TriggerDecision:
    """Quick check if training should run for a config.

    Args:
        config_key: Config identifier (e.g., "square8_2p")
        games_since_training: Number of new games since last training
        **kwargs: Additional state fields (elo, win_rate, model_count, etc.)

    Returns:
        TriggerDecision with result and reasoning
    """
    triggers = get_training_triggers()

    # Update state with provided info
    triggers.update_config_state(
        config_key,
        games_count=games_since_training + triggers.get_config_state(config_key).last_training_games,
        **kwargs,
    )

    return triggers.should_train(config_key)
