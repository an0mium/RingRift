"""Sample collection and encoding for training data export.

This module provides the SampleCollector class for extracting training samples
from games. It handles:
- State encoding with history stacking
- Move encoding for policy targets
- Value computation from game outcomes
- Heuristic feature extraction (optional)

Usage:
    from app.training.export.sample_collector import SampleCollector, SampleCollectorConfig

    config = SampleCollectorConfig(
        board_type="hex8",
        num_players=2,
        history_length=3,
    )
    collector = SampleCollector(config)

    samples = collector.collect_from_game(
        initial_state=state,
        moves=moves,
        final_state=final_state,
        metadata=game_meta,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from app.training.export.array_builder import Sample

if TYPE_CHECKING:
    from app.models.core import GameState, Move

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from app.ai.neural_net import INVALID_MOVE_INDEX, encode_move_for_board
    HAS_MOVE_ENCODING = True
except ImportError:
    HAS_MOVE_ENCODING = False
    INVALID_MOVE_INDEX = -1

    def encode_move_for_board(move: Any, board: Any) -> int:
        """Fallback that always returns invalid."""
        return -1

try:
    from app.training.fast_heuristic_features import (
        NUM_HEURISTIC_FEATURES,
        NUM_HEURISTIC_FEATURES_FULL,
        extract_full_heuristic_features,
        extract_heuristic_features,
    )
    HAS_HEURISTICS = True
except ImportError:
    HAS_HEURISTICS = False
    NUM_HEURISTIC_FEATURES = 21
    NUM_HEURISTIC_FEATURES_FULL = 49

try:
    from app.training.export_core import (
        compute_multi_player_values,
        encode_state_with_history,
        value_from_final_ranking,
        value_from_final_winner,
    )
    HAS_EXPORT_CORE = True
except ImportError:
    HAS_EXPORT_CORE = False

try:
    from app.training.encoding import get_encoder_for_board_type
    HAS_ENCODER_FACTORY = True
except ImportError:
    HAS_ENCODER_FACTORY = False


@dataclass
class SampleCollectorConfig:
    """Configuration for sample collection.

    Attributes:
        board_type: Board type string (hex8, square8, etc.)
        num_players: Number of players (2, 3, 4)
        history_length: Number of past feature frames to include
        feature_version: Feature encoding version
        encoder_version: Encoder version (default, v2, v3)
        use_board_aware_encoding: Use board-specific policy encoding
        use_rank_aware_values: Use rank-based values for multiplayer
        sample_every: Use every Nth move as a sample
        include_heuristics: Extract heuristic features
        full_heuristics: Use full 49-feature extraction
        include_intermediate: Include non-terminal moves
    """

    board_type: str
    num_players: int
    history_length: int = 3
    feature_version: int = 2
    encoder_version: str = "default"
    use_board_aware_encoding: bool = True
    use_rank_aware_values: bool = True
    sample_every: int = 1
    include_heuristics: bool = False
    full_heuristics: bool = False
    include_intermediate: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_players not in (2, 3, 4):
            raise ValueError(f"num_players must be 2, 3, or 4, got {self.num_players}")
        if self.history_length < 0:
            raise ValueError(f"history_length must be non-negative, got {self.history_length}")
        if self.sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {self.sample_every}")
        # full_heuristics implies include_heuristics
        if self.full_heuristics:
            self.include_heuristics = True


@dataclass
class GameMetadata:
    """Metadata about a game for sample collection.

    Attributes:
        game_id: Unique game identifier
        victory_type: Victory type string
        engine_mode: Engine/source type (gumbel, mcts, heuristic)
        opponent_elo: Opponent Elo rating
        opponent_type: Opponent type string
        quality_score: Game quality score
        timestamp: Game timestamp (Unix epoch)
        db_winner: Winner from database (fallback)
        move_probs: Optional move probabilities for soft targets
    """

    game_id: str = ""
    victory_type: str = ""
    engine_mode: str = ""
    opponent_elo: float = 0.0
    opponent_type: str = ""
    quality_score: float = 0.5
    timestamp: float = 0.0
    db_winner: int | None = None
    move_probs: dict[int, dict[str, float]] = field(default_factory=dict)


@dataclass
class CollectionResult:
    """Result of collecting samples from a game.

    Attributes:
        samples: List of collected samples
        success: Whether collection succeeded
        partial: Whether only partial samples were collected
        error: Error message if collection failed
        moves_processed: Number of moves processed
        samples_collected: Number of samples collected
    """

    samples: list[Sample] = field(default_factory=list)
    success: bool = True
    partial: bool = False
    error: str = ""
    moves_processed: int = 0
    samples_collected: int = 0


class SampleCollector:
    """Collect and encode training samples from games.

    This class handles the sample extraction process:
    1. Initialize encoder for the board type
    2. Iterate through game moves with state tracking
    3. Encode states with history stacking
    4. Encode moves for policy targets
    5. Compute values from game outcome
    6. Optionally extract heuristic features

    Example:
        config = SampleCollectorConfig(board_type="hex8", num_players=2)
        collector = SampleCollector(config)

        result = collector.collect_from_game(
            initial_state=initial_state,
            moves=moves,
            final_state=final_state,
            metadata=GameMetadata(quality_score=0.8),
        )

        if result.success:
            for sample in result.samples:
                builder.add_sample(sample)
    """

    def __init__(self, config: SampleCollectorConfig) -> None:
        """Initialize collector with configuration.

        Args:
            config: Collection configuration
        """
        self.config = config
        self._encoder = None
        self._num_heuristic_features = (
            NUM_HEURISTIC_FEATURES_FULL if config.full_heuristics
            else NUM_HEURISTIC_FEATURES
        )

    def _get_encoder(self) -> Any:
        """Get or create the state encoder.

        Returns:
            Encoder instance for the board type

        Raises:
            RuntimeError: If encoder factory is not available
        """
        if self._encoder is None:
            if not HAS_ENCODER_FACTORY:
                raise RuntimeError(
                    "Encoder factory not available. "
                    "Install app.training.encoding module."
                )
            self._encoder = get_encoder_for_board_type(
                self.config.board_type,
                encoder_version=self.config.encoder_version,
                feature_version=self.config.feature_version,
            )
        return self._encoder

    def collect_from_game(
        self,
        initial_state: GameState,
        moves: list[Move],
        final_state: GameState | None = None,
        metadata: GameMetadata | None = None,
        *,
        max_move_index: int | None = None,
    ) -> CollectionResult:
        """Collect training samples from a complete game.

        Args:
            initial_state: Game state at the start
            moves: List of moves played
            final_state: Final game state (computed if not provided)
            metadata: Game metadata for weighting features
            max_move_index: Stop processing after this move index

        Returns:
            CollectionResult with samples and status
        """
        if not HAS_EXPORT_CORE:
            return CollectionResult(
                success=False,
                error="export_core module not available",
            )

        metadata = metadata or GameMetadata()
        encoder = self._get_encoder()

        # Replay game and collect samples
        history_frames: list[np.ndarray] = []
        game_samples: list[tuple[Any, ...]] = []

        try:
            from app.game_engine import GameEngine
        except ImportError:
            return CollectionResult(
                success=False,
                error="GameEngine not available",
            )

        current_state = initial_state
        replay_succeeded = True
        num_players_in_game = len(initial_state.players)

        for move_index, move in enumerate(moves):
            if max_move_index is not None and move_index > max_move_index:
                break

            state_before = current_state

            # Apply move to get next state
            try:
                current_state = GameEngine.apply_move(current_state, move, trace_mode=True)
            except Exception as e:
                logger.debug(f"Replay failed at move {move_index}: {e}")
                replay_succeeded = False
                break

            # Skip if not sampling this move
            if self.config.sample_every > 1:
                if (move_index % self.config.sample_every) != 0:
                    continue

            # Encode state with history
            stacked, globals_vec = encode_state_with_history(
                encoder, state_before, history_frames, self.config.history_length
            )

            # Update history frames
            hex_encoder = getattr(encoder, "_hex_encoder", None)
            if hex_encoder is not None:
                base_features, _ = hex_encoder.encode_state(state_before)
            else:
                base_features, _ = encoder._extract_features(state_before)

            history_frames.append(base_features)
            if len(history_frames) > self.config.history_length + 1:
                history_frames.pop(0)

            # Encode move for policy target
            if self.config.use_board_aware_encoding:
                policy_idx = encode_move_for_board(move, state_before.board)
            else:
                policy_idx = encoder.encode_move(move, state_before.board)

            if policy_idx == INVALID_MOVE_INDEX:
                continue

            # Extract phase and move type
            phase_str = self._extract_phase(state_before)
            move_type_str = self._extract_move_type(move)

            # Extract heuristic features if enabled
            heuristic_vec = None
            if self.config.include_heuristics and HAS_HEURISTICS:
                heuristic_vec = self._extract_heuristics(
                    state_before, state_before.current_player
                )

            game_samples.append((
                stacked, globals_vec, policy_idx, state_before.current_player,
                move_index, phase_str, move_type_str, heuristic_vec
            ))

        # No samples collected
        if not game_samples:
            return CollectionResult(
                success=False,
                error="No samples collected from game",
                moves_processed=len(moves),
            )

        # Determine final state and winner
        effective_final_state = final_state if final_state else current_state
        effective_winner = getattr(effective_final_state, 'winner', None)

        if effective_winner is None or effective_winner == 0:
            effective_winner = metadata.db_winner

        # Skip games without valid winner
        if effective_winner is None or effective_winner == 0:
            return CollectionResult(
                success=False,
                error="Game has no valid winner",
                moves_processed=len(moves),
            )

        # Patch winner into final_state for partial games
        if not replay_succeeded and effective_winner is not None:
            effective_final_state = type(effective_final_state)(
                **{**effective_final_state.__dict__, 'winner': effective_winner}
            )

        # Compute multi-player values
        values_vec = np.asarray(
            compute_multi_player_values(
                effective_final_state, num_players=num_players_in_game
            ),
            dtype=np.float32,
        )

        # Convert game samples to Sample objects
        samples: list[Sample] = []
        total_moves = len(moves)

        for (stacked, globals_vec, idx, perspective,
             move_index, phase_str, move_type_str, heuristic_vec) in game_samples:

            # Compute scalar value from current player's perspective
            if self.config.use_rank_aware_values:
                value = value_from_final_ranking(
                    effective_final_state,
                    perspective=perspective,
                    num_players=self.config.num_players,
                )
            else:
                value = value_from_final_winner(effective_final_state, perspective=perspective)

            # Get soft policy targets if available
            soft_probs = metadata.move_probs.get(move_index)
            if soft_probs:
                policy_indices, policy_values = self._parse_soft_targets(
                    soft_probs, initial_state.board.size
                )
                if not policy_indices:
                    # Fallback to 1-hot
                    policy_indices = np.array([idx], dtype=np.int32)
                    policy_values = np.array([1.0], dtype=np.float32)
            else:
                policy_indices = np.array([idx], dtype=np.int32)
                policy_values = np.array([1.0], dtype=np.float32)

            sample = Sample(
                features=stacked,
                globals=globals_vec,
                value=float(value),
                values_mp=values_vec,
                num_players=num_players_in_game,
                policy_indices=policy_indices,
                policy_values=policy_values,
                move_number=move_index,
                total_game_moves=total_moves,
                phase=phase_str,
                victory_type=metadata.victory_type,
                engine_mode=metadata.engine_mode,
                move_type=move_type_str,
                opponent_elo=metadata.opponent_elo,
                opponent_type=metadata.opponent_type,
                quality_score=metadata.quality_score,
                timestamp=metadata.timestamp,
                heuristics=heuristic_vec,
            )
            samples.append(sample)

        return CollectionResult(
            samples=samples,
            success=True,
            partial=not replay_succeeded,
            moves_processed=len(moves),
            samples_collected=len(samples),
        )

    def _extract_phase(self, state: GameState) -> str:
        """Extract phase string from game state."""
        phase = getattr(state, 'current_phase', None)
        if phase is None:
            return "unknown"
        if hasattr(phase, 'value'):
            return str(phase.value)
        return str(phase)

    def _extract_move_type(self, move: Move) -> str:
        """Extract move type string from move."""
        move_type = getattr(move, 'type', None)
        if move_type is None:
            return "unknown"
        if hasattr(move_type, 'value'):
            return str(move_type.value)
        return str(move_type)

    def _extract_heuristics(
        self,
        state: GameState,
        player_number: int,
    ) -> np.ndarray:
        """Extract heuristic features from state.

        Args:
            state: Current game state
            player_number: Player to extract features for

        Returns:
            Heuristic feature vector
        """
        try:
            if self.config.full_heuristics:
                return extract_full_heuristic_features(
                    state,
                    player_number=player_number,
                    normalize=True,
                )
            else:
                return extract_heuristic_features(
                    state,
                    player_number=player_number,
                    eval_mode="full",
                    normalize=True,
                )
        except Exception as e:
            logger.debug(f"Heuristic extraction failed: {e}")
            return np.zeros(self._num_heuristic_features, dtype=np.float32)

    def _parse_soft_targets(
        self,
        soft_probs: dict[str, float],
        board_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse soft policy targets from move probabilities.

        Args:
            soft_probs: Dictionary mapping move keys to probabilities
            board_size: Size of the board for encoding

        Returns:
            Tuple of (indices, values) arrays for sparse policy
        """
        indices: list[int] = []
        values: list[float] = []

        for move_key, prob in soft_probs.items():
            try:
                if "->" in move_key:
                    # Format: "from_x,from_y->to_x,to_y"
                    parts = move_key.split("->")
                    to_part = parts[1]
                else:
                    # Format: "to_x,to_y"
                    to_part = move_key

                to_x, to_y = map(int, to_part.split(","))
                move_idx = to_y * board_size + to_x
                indices.append(move_idx)
                values.append(float(prob))
            except (ValueError, IndexError):
                continue

        if indices:
            return (
                np.array(indices, dtype=np.int32),
                np.array(values, dtype=np.float32),
            )
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)


def create_collector(
    board_type: str,
    num_players: int,
    **kwargs: Any,
) -> SampleCollector:
    """Create a SampleCollector with the given configuration.

    Convenience factory function.

    Args:
        board_type: Board type string
        num_players: Number of players
        **kwargs: Additional SampleCollectorConfig arguments

    Returns:
        Configured SampleCollector instance
    """
    config = SampleCollectorConfig(
        board_type=board_type,
        num_players=num_players,
        **kwargs,
    )
    return SampleCollector(config)
