import logging
from typing import Optional, Tuple, List, Dict, Any
from app.models import GameState, Move, BoardType, GameStatus
from app.game_engine import GameEngine
from app.training.seed_utils import seed_all

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Theoretical Maximum Moves by Board Type and Player Count
# -----------------------------------------------------------------------------
# These represent upper bounds on game length based on board geometry.
# A game exceeding these thresholds is anomalous and should be logged/flagged.
#
# Derivation:
# - Square8: 64 cells, 18 rings/player (2p) = 36 total rings
#   Max placements ~36, max movements ~36*10, captures reduce rings
#   Conservative estimate: 150 moves for 2p, +50 per additional player
#
# - Square19: 361 cells, 36 rings/player (2p) = 72 total rings
#   Much larger board, longer games expected
#   Conservative estimate: 400 moves for 2p, +100 per additional player
#
# - Hexagonal: 331 cells (radius 11), 36 rings/player (2p) = 72 total rings
#   Similar to Square19
#   Conservative estimate: 400 moves for 2p, +100 per additional player
#
# Games reaching these limits without a winner indicate potential bugs.
# -----------------------------------------------------------------------------

THEORETICAL_MAX_MOVES: Dict[BoardType, Dict[int, int]] = {
    BoardType.SQUARE8: {
        2: 150,
        3: 200,
        4: 250,
    },
    BoardType.SQUARE19: {
        2: 400,
        3: 500,
        4: 600,
    },
    BoardType.HEXAGONAL: {
        2: 400,
        3: 500,
        4: 600,
    },
}


def get_theoretical_max_moves(board_type: BoardType, num_players: int) -> int:
    """Return theoretical maximum moves for a board type and player count.

    If the exact player count isn't defined, extrapolate from the pattern.
    """
    board_limits = THEORETICAL_MAX_MOVES.get(board_type, {})
    if num_players in board_limits:
        return board_limits[num_players]
    # Extrapolate: use 2-player base + increment per extra player
    base = board_limits.get(2, 200)
    increment = board_limits.get(3, base + 50) - base
    return base + increment * (num_players - 2)


# Canonical default evaluation configuration for heuristic
# training (GA, CMA-ES). This configuration is intended for
# multi-board, multi-start evaluation runs using the CMA-ES
# and GA heuristic scripts under ``ai-service/scripts``.
# It serves as a single source of truth for default board
# sets and evaluation kwargs.
DEFAULT_TRAINING_EVAL_CONFIG: Dict[str, Any] = {
    "boards": [
        BoardType.SQUARE8,
        BoardType.SQUARE19,
        BoardType.HEXAGONAL,
    ],
    "eval_mode": "multi-start",
    "state_pool_id": "v1",
    "games_per_eval": 16,
    "max_moves": 200,
    "eval_randomness": 0.0,
    # RNG seed is supplied by calling code; see `build_training_eval_kwargs`.
}

# 2-player training preset for heuristic optimisation (CMA-ES / GA).
#
# This preset encodes the recommended "serious" 2-player training regime:
#
# - Multi-board evaluation on Square8, Square19, and Hexagonal boards.
# - Multi-start evaluation from fixed mid/late-game state pools ("v1").
# - A small amount of evaluation-time randomness to break symmetry and
#   avoid degenerate 0.5 plateaus while remaining reproducible when a
#   seed is provided.
#
# Callers that need a simple 2-player training configuration should prefer
# this preset (via `get_two_player_training_kwargs`) over re-encoding
# boards / eval_mode / state_pool_id / eval_randomness by hand.
TWO_PLAYER_TRAINING_PRESET: Dict[str, Any] = {
    **DEFAULT_TRAINING_EVAL_CONFIG,
    "eval_randomness": 0.02,
}

# Canonical heuristic evaluation mode mapping for training/soaks.
#
# Training harnesses (CMA-ES, GA, self-play soaks) should consult this
# mapping when constructing HeuristicAI instances so that:
#
# - Square8 uses the full structural evaluator (historical behaviour).
# - Square19 and Hexagonal boards default to the lightweight evaluator
#   that omits Tier-2 structural/global features for better throughput.
TRAINING_HEURISTIC_EVAL_MODE_BY_BOARD: Dict[BoardType, str] = {
    BoardType.SQUARE8: "full",
    BoardType.SQUARE19: "light",
    BoardType.HEXAGONAL: "light",
}


def build_training_eval_kwargs(
    games_per_eval: Optional[int] = None,
    eval_randomness: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build canonical kwargs for heuristic training evaluation.

    This helper centralises defaults for multi-board, multi-start
    heuristic evaluation, returning a dict that can be splatted into
    calls such as ``evaluate_fitness_over_boards(...)`` and
    higher-level GA / CMA-ES harnesses.

    It is not a mandatory API; scripts may construct kwargs manually,
    but should keep semantics aligned with DEFAULT_TRAINING_EVAL_CONFIG
    to ensure comparable results across runs.
    """
    cfg: Dict[str, Any] = dict(DEFAULT_TRAINING_EVAL_CONFIG)
    if games_per_eval is not None:
        cfg["games_per_eval"] = games_per_eval
    if eval_randomness is not None:
        cfg["eval_randomness"] = eval_randomness
    # Always thread the seed explicitly so callers can control reproducibility.
    cfg["seed"] = seed
    return cfg


def get_two_player_training_kwargs(
    games_per_eval: int,
    seed: int,
) -> Dict[str, Any]:
    """Return canonical kwargs for 2-player heuristic training evaluation.

    The returned dict is suitable for splatting into
    :func:`evaluate_fitness_over_boards` and related helpers. It encodes the
    recommended 2-player CMA-ES / GA configuration:

    - multi-board evaluation over Square8, Square19, and Hexagonal boards;
    - ``eval_mode='multi-start'`` from the ``'v1'`` mid/late-game pools; and
    - a small, non-zero ``eval_randomness`` for symmetry breaking.

    Callers remain free to override any of the keys (for example,
    ``eval_mode`` or ``eval_randomness``) after calling this helper.
    """
    cfg: Dict[str, Any] = dict(TWO_PLAYER_TRAINING_PRESET)
    cfg["games_per_eval"] = games_per_eval
    cfg["seed"] = seed
    return cfg


class RingRiftEnv:
    """RL-style environment wrapper for RingRift.

    Provides a minimal `reset()`, `step()`, and `legal_moves()` interface
    for self-play style loops. The underlying Python GameEngine and
    GameState models are already N-player aware; the `num_players`
    parameter exposes that capability to training code (2–4 players).
    """

    def __init__(
        self,
        board_type: BoardType = BoardType.SQUARE8,
        max_moves: int = 200,
        reward_on: str = "terminal",  # "terminal" or "shaped"
        num_players: int = 2,
    ):
        self.board_type = board_type
        self.max_moves = max_moves
        self.reward_on = reward_on
        self.num_players = num_players
        self._state: Optional[GameState] = None
        self._move_count: int = 0

    def reset(self, seed: Optional[int] = None) -> GameState:
        """Create a fresh GameState for self-play.

        When `num_players` > 2, the initial state will contain that many
        AI-controlled players with per-player ring caps and victory
        thresholds aligned with the shared TypeScript initial-state
        helpers.

        If `seed` is provided, it is threaded into Python RNGs used by
        any stochastic components (future variants); the core game rules
        remain deterministic.
        """
        if seed is not None:
            # Use the central training seeding utility so that Python
            # random, NumPy, and torch (including CUDA/cuDNN flags) are
            # all initialised consistently for this environment.
            seed_all(seed)

        # Reuse a shared helper from generate_data
        # Avoid circular import by importing inside method
        from app.training.generate_data import create_initial_state

        self._state = create_initial_state(
            board_type=self.board_type,
            num_players=self.num_players,
        )
        self._move_count = 0
        return self._state

    @property
    def state(self) -> GameState:
        assert self._state is not None, "Call reset() before using env"
        return self._state

    def legal_moves(self) -> List[Move]:
        """
        Return legal moves for the current player, using the same logic
        as AIs: GameEngine.get_valid_moves.

        This includes forced elimination moves as first-class actions when
        the player has stacks but no placement/movement/capture available
        (per RR-CANON-R072/R100/R205).
        """
        return GameEngine.get_valid_moves(
            self.state, self.state.current_player
        )

    def step(
        self, move: Move
    ) -> Tuple[GameState, float, bool, Dict[str, Any]]:
        """
        Apply a move, returning (new_state, reward, done, info).

        - new_state: updated GameState from GameEngine.apply_move.
        - reward: from the perspective of the player who just moved,
          according to reward_on:
            * "terminal": +1/-1/0 only at terminal states, 0 otherwise.
            * "shaped": use calculate_outcome-style shaping at terminal.
        - done: True when game_status != ACTIVE or max_moves reached.
        - info: may include raw winner, reason, and move_count.
        """
        self._state = GameEngine.apply_move(self.state, move)
        self._move_count += 1

        done = (
            self._state.game_status != GameStatus.ACTIVE
            or self._move_count >= self.max_moves
        )

        reward = 0.0
        info: Dict[str, Any] = {"winner": self._state.winner}

        # Log warning/error for games that hit max_moves without a winner
        if done and self._move_count >= self.max_moves and self._state.winner is None:
            theoretical_max = get_theoretical_max_moves(self.board_type, self.num_players)
            if self._move_count >= theoretical_max:
                # This is anomalous - game exceeded theoretical maximum
                logger.error(
                    "GAME_NON_TERMINATION_ERROR: Game exceeded theoretical maximum moves "
                    "without reaching a definite conclusion. "
                    "board_type=%s, num_players=%d, move_count=%d, "
                    "max_moves=%d, theoretical_max=%d, game_status=%s, winner=%s",
                    self.board_type.value,
                    self.num_players,
                    self._move_count,
                    self.max_moves,
                    theoretical_max,
                    self._state.game_status.value,
                    self._state.winner,
                )
            else:
                # Hit configured max_moves but not theoretical max - still concerning
                logger.warning(
                    "GAME_MAX_MOVES_CUTOFF: Game hit max_moves limit without a winner. "
                    "board_type=%s, num_players=%d, move_count=%d, "
                    "max_moves=%d, theoretical_max=%d, game_status=%s, winner=%s",
                    self.board_type.value,
                    self.num_players,
                    self._move_count,
                    self.max_moves,
                    theoretical_max,
                    self._state.game_status.value,
                    self._state.winner,
                )
            info["termination_anomaly"] = True
            info["theoretical_max_moves"] = theoretical_max

        if done:
            if self.reward_on == "terminal":
                # Perspective: player who just moved is move.player
                perspective = move.player
                if self._state.winner is None:
                    reward = 0.0
                elif self._state.winner == perspective:
                    reward = 1.0
                else:
                    reward = -1.0
            else:
                # Reuse calculate_outcome-like shaping
                from app.training.generate_data import calculate_outcome
                reward = calculate_outcome(
                    self._state,
                    player_number=move.player,
                    depth=self._move_count
                )

        info["move_count"] = self._move_count
        return self._state, reward, done, info
