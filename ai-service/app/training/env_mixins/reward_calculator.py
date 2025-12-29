"""Reward calculation mixin for RingRiftEnv.

This module provides the RewardCalculatorMixin class which handles terminal
and shaped reward calculation, extracted from RingRiftEnv.step() for
improved testability and maintainability.
"""

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.models import GameState, Move


class RewardCalculatorMixin:
    """Mixin that handles reward calculation for game episodes.

    Supports two reward modes:
    - "terminal": +1 for win, -1 for loss, 0 for draw/stalemate
    - "shaped": Delegates to calculate_outcome for depth-aware rewards

    Attributes expected from host class:
        reward_on: Literal["terminal", "shaped"] - reward mode
        _move_count: int - current move count for shaped rewards
    """

    reward_on: Literal["terminal", "shaped"]
    _move_count: int

    def _calculate_episode_reward(
        self,
        state: "GameState",
        move: "Move",
        *,
        done: bool,
    ) -> float:
        """Calculate reward for a completed step.

        Parameters
        ----------
        state:
            The current game state after the move was applied.
        move:
            The move that was just applied.
        done:
            Whether the episode has terminated.

        Returns
        -------
        float
            The reward value from the perspective of move.player:
            - If not done: always 0.0
            - If done with terminal mode: +1 (win), -1 (loss), 0 (draw)
            - If done with shaped mode: depth-aware reward from calculate_outcome
        """
        if not done:
            return 0.0

        return self._calculate_terminal_reward(state, move)

    def _calculate_terminal_reward(
        self,
        state: "GameState",
        move: "Move",
    ) -> float:
        """Calculate terminal reward based on configured reward mode.

        Parameters
        ----------
        state:
            The terminal game state.
        move:
            The final move that was applied.

        Returns
        -------
        float
            Reward from the perspective of move.player.
        """
        if self.reward_on == "terminal":
            return self._calculate_terminal_reward_simple(state, move.player)
        else:
            return self._calculate_shaped_reward(state, move.player)

    def _calculate_terminal_reward_simple(
        self,
        state: "GameState",
        perspective: int,
    ) -> float:
        """Calculate simple terminal reward (+1/-1/0).

        Parameters
        ----------
        state:
            The terminal game state.
        perspective:
            The player number from whose perspective to calculate reward.

        Returns
        -------
        float
            +1.0 if perspective won, -1.0 if they lost, 0.0 for draw.
        """
        if state.winner is None:
            return 0.0
        elif state.winner == perspective:
            return 1.0
        else:
            return -1.0

    def _calculate_shaped_reward(
        self,
        state: "GameState",
        perspective: int,
    ) -> float:
        """Calculate shaped reward using calculate_outcome.

        Parameters
        ----------
        state:
            The terminal game state.
        perspective:
            The player number from whose perspective to calculate reward.

        Returns
        -------
        float
            Depth-aware reward from calculate_outcome.
        """
        # Lazy import to avoid circular dependency
        from app.training.generate_data import calculate_outcome

        return calculate_outcome(
            state,
            player_number=perspective,
            depth=self._move_count,
        )
