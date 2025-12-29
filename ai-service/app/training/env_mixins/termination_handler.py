"""Termination handling mixin for RingRiftEnv.

This module provides the TerminationHandlerMixin class which handles game
termination detection, victory reason inference, and terminal info population,
extracted from RingRiftEnv.step() for improved testability.
"""

import logging
from typing import TYPE_CHECKING, Any

from app.models import GameStatus
from app.training.tournament import infer_victory_reason

if TYPE_CHECKING:
    from app.models import BoardType, GameState


logger = logging.getLogger(__name__)


class TerminationHandlerMixin:
    """Mixin that handles game termination detection and info population.

    Handles:
    - Repetition detection (diagnostic, disabled by default)
    - Budget-based termination (max_moves)
    - Rules-based termination (game_status != ACTIVE)
    - Victory reason inference
    - Terminal info dict population

    Attributes expected from host class:
        _state: GameState | None
        _move_count: int
        _position_counts: dict[int, int]
        _repetition_threshold: int
        max_moves: int
        board_type: BoardType
        num_players: int
    """

    _state: "GameState | None"
    _move_count: int
    _position_counts: dict[int, int]
    _repetition_threshold: int
    max_moves: int
    board_type: "BoardType"
    num_players: int

    def _check_position_repetition(self) -> bool:
        """Check if the current position has been repeated beyond threshold.

        Updates position counts and returns True if repetition threshold reached.
        The S-invariant theoretically prevents repetition, so any detection
        indicates either a hash collision or a rules bug.

        Returns
        -------
        bool
            True if position repeated >= threshold times, False otherwise.
        """
        if self._repetition_threshold <= 0:
            return False

        if self._state is None:
            return False

        pos_hash = self._state.zobrist_hash
        if pos_hash is None or pos_hash == 0:
            return False

        self._position_counts[pos_hash] = self._position_counts.get(pos_hash, 0) + 1

        if self._position_counts[pos_hash] >= self._repetition_threshold:
            # This should never happen due to S-invariant. Log for investigation.
            logger.error(
                "REPETITION_DETECTED: position_hash=%d repeated %d times. "
                "board_type=%s, num_players=%d, move_count=%d. "
                "Possible causes: zobrist hash collision or S-invariant violation.",
                pos_hash,
                self._position_counts[pos_hash],
                self.board_type.value,
                self.num_players,
                self._move_count,
            )
            return True

        return False

    def _evaluate_termination_state(self) -> tuple[bool, bool, bool, bool]:
        """Evaluate all termination conditions.

        Returns
        -------
        tuple[bool, bool, bool, bool]
            (terminated_by_rules, terminated_by_budget, terminated_by_repetition, done)
        """
        terminated_by_repetition = self._check_position_repetition()
        terminated_by_rules = (
            self._state is not None
            and self._state.game_status != GameStatus.ACTIVE
        )
        terminated_by_budget = self._move_count >= self.max_moves
        done = terminated_by_rules or terminated_by_budget or terminated_by_repetition

        return terminated_by_rules, terminated_by_budget, terminated_by_repetition, done

    def _infer_canonical_victory_reason(
        self,
        terminated_by_budget_only: bool,
        terminated_by_repetition: bool = False,
    ) -> str:
        """Map engine-level termination state to a canonical result string.

        This helper keeps the mapping between Python/TS result enums
        and the canonical categories used by training and evaluation.

        Parameters
        ----------
        terminated_by_budget_only:
            True if game ended due to max_moves without rules termination.
        terminated_by_repetition:
            True if game ended due to position repetition.

        Returns
        -------
        str
            Canonical victory reason string.
        """
        # Repetition-based draw takes priority over budget cutoff
        if terminated_by_repetition:
            return "draw_by_repetition"

        if (
            terminated_by_budget_only
            and self._state is not None
            and self._state.game_status == GameStatus.ACTIVE
        ):
            return "max_moves"

        if self._state is None:
            return "unknown"

        engine_reason = infer_victory_reason(self._state)
        mapping = {
            "elimination": "ring_elimination",
            "territory": "territory_control",
            "last_player_standing": "last_player_standing",
            "structural": "structural_stalemate",
            "unknown": "unknown",
        }
        return mapping.get(engine_reason, engine_reason)

    def _log_non_termination_anomaly(
        self,
        terminated_by_budget: bool,
        terminated_by_rules: bool,
        theoretical_max: int,
    ) -> dict[str, Any]:
        """Log warnings/errors for games that hit max_moves without a winner.

        Parameters
        ----------
        terminated_by_budget:
            True if game ended due to max_moves.
        terminated_by_rules:
            True if game ended due to rules engine termination.
        theoretical_max:
            Theoretical maximum moves for this board/player config.

        Returns
        -------
        dict[str, Any]
            Additional info dict entries for anomaly tracking.
        """
        if not terminated_by_budget or terminated_by_rules:
            return {}

        if self._state is None:
            return {}

        extra_info: dict[str, Any] = {}

        if self._move_count >= theoretical_max:
            # Game exceeded the theoretical maximum number of moves.
            logger.error(
                "GAME_NON_TERMINATION_ERROR: exceeded theoretical "
                "maximum moves without a conclusion. board_type=%s, "
                "num_players=%d, move_count=%d, max_moves=%d, "
                "theoretical_max=%d, game_status=%s, winner=%s",
                self.board_type.value,
                self.num_players,
                self._move_count,
                self.max_moves,
                theoretical_max,
                self._state.game_status.value,
                self._state.winner,
            )
        else:
            # Hit configured max_moves but not theoretical maximum.
            logger.warning(
                "GAME_MAX_MOVES_CUTOFF: hit max_moves without a winner. "
                "board_type=%s, num_players=%d, move_count=%d, "
                "max_moves=%d, theoretical_max=%d, game_status=%s, "
                "winner=%s",
                self.board_type.value,
                self.num_players,
                self._move_count,
                self.max_moves,
                theoretical_max,
                self._state.game_status.value,
                self._state.winner,
            )

        extra_info["termination_anomaly"] = True
        extra_info["theoretical_max_moves"] = theoretical_max
        return extra_info

    def _populate_terminal_info(
        self,
        terminated_by_rules: bool,
        terminated_by_budget: bool,
        terminated_by_repetition: bool,
    ) -> dict[str, Any]:
        """Populate the info dictionary for terminal states.

        Parameters
        ----------
        terminated_by_rules:
            True if game ended due to rules engine termination.
        terminated_by_budget:
            True if game ended due to max_moves.
        terminated_by_repetition:
            True if game ended due to position repetition.

        Returns
        -------
        dict[str, Any]
            Info dictionary with terminal state information.
        """
        if self._state is None:
            return {}

        # Canonical victory_reason
        victory_reason = self._infer_canonical_victory_reason(
            terminated_by_budget_only=terminated_by_budget and not terminated_by_rules,
            terminated_by_repetition=terminated_by_repetition,
        )

        info: dict[str, Any] = {
            "victory_reason": victory_reason,
            # Raw engine-level category for debugging / compatibility
            "engine_victory_reason": infer_victory_reason(self._state),
        }

        # Rings eliminated are keyed by causing player id as strings
        # in GameState; expose a simpler int-keyed mapping.
        rings_eliminated: dict[int, int] = {}
        for pid_str, count in self._state.board.eliminated_rings.items():
            try:
                pid = int(pid_str)
            except (TypeError, ValueError):
                continue
            rings_eliminated[pid] = count
        info["rings_eliminated"] = rings_eliminated

        # Territory spaces per player from the Player models.
        territory_spaces: dict[int, int] = {}
        for player in self._state.players:
            territory_spaces[player.player_number] = player.territory_spaces
        info["territory_spaces"] = territory_spaces

        info["moves_played"] = self._move_count

        return info
