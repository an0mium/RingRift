"""Array builder for accumulating and stacking training samples.

This module provides the ArrayBuilder class for managing sample accumulation
during export. It replaces the 17+ list accumulations in the monolithic
export script with a clean, focused interface.

Usage:
    from app.training.export.array_builder import ArrayBuilder, Sample

    builder = ArrayBuilder()
    builder.add_sample(Sample(
        features=stacked_features,
        globals=globals_vec,
        value=value,
        values_mp=values_vec,
        ...
    ))
    arrays = builder.build_arrays()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    """Individual training sample from game replay.

    Attributes:
        features: Stacked feature planes (shape: [C, H, W])
        globals: Global feature vector
        value: Scalar value target (current player's perspective)
        values_mp: Multi-player value vector (4 elements, per-player values)
        num_players: Number of players in the game
        policy_indices: Indices of non-zero policy values (sparse encoding)
        policy_values: Values at policy indices (sparse encoding)
        move_number: Move index within the game
        total_game_moves: Total moves in the game
        phase: Game phase string (e.g., "place_ring", "move_stack")
        victory_type: Victory type string for balanced sampling
        engine_mode: Engine/source type for source-based weighting
        move_type: Move type for chain-aware weighting
        opponent_elo: Opponent Elo for Elo-weighted training
        opponent_type: Opponent type for diversity tracking
        quality_score: Game quality score for quality-weighted training
        timestamp: Game timestamp for freshness weighting
        heuristics: Optional heuristic feature vector (21 or 49 features)
    """

    features: np.ndarray
    globals: np.ndarray
    value: float
    values_mp: np.ndarray
    num_players: int
    policy_indices: np.ndarray
    policy_values: np.ndarray
    move_number: int
    total_game_moves: int
    phase: str
    victory_type: str = ""
    engine_mode: str = ""
    move_type: str = "unknown"
    opponent_elo: float = 0.0
    opponent_type: str = ""
    quality_score: float = 0.5
    timestamp: float = 0.0
    heuristics: np.ndarray | None = None


@dataclass
class BuiltArrays:
    """Container for stacked arrays ready for NPZ output.

    Attributes:
        features: Stacked features (shape: [N, C, H, W])
        globals: Global features (shape: [N, G])
        values: Scalar values (shape: [N])
        values_mp: Multi-player values (shape: [N, 4])
        num_players: Player counts (shape: [N])
        policy_indices: Sparse policy indices (object array of int32 arrays)
        policy_values: Sparse policy values (object array of float32 arrays)
        move_numbers: Move indices (shape: [N])
        total_game_moves: Total moves (shape: [N])
        phases: Phase strings (object array)
        victory_types: Victory type strings (object array)
        engine_modes: Engine mode strings (object array)
        move_types: Move type strings (object array)
        opponent_elo: Opponent Elo values (shape: [N])
        opponent_types: Opponent type strings (object array)
        quality_scores: Quality scores (shape: [N])
        timestamps: Timestamps (shape: [N])
        heuristics: Optional heuristic features (shape: [N, H] or None)
        sample_count: Number of samples
    """

    features: np.ndarray
    globals: np.ndarray
    values: np.ndarray
    values_mp: np.ndarray
    num_players: np.ndarray
    policy_indices: np.ndarray
    policy_values: np.ndarray
    move_numbers: np.ndarray
    total_game_moves: np.ndarray
    phases: np.ndarray
    victory_types: np.ndarray
    engine_modes: np.ndarray
    move_types: np.ndarray
    opponent_elo: np.ndarray
    opponent_types: np.ndarray
    quality_scores: np.ndarray
    timestamps: np.ndarray
    heuristics: np.ndarray | None
    sample_count: int

    def to_save_kwargs(
        self,
        *,
        board_type: str,
        board_size: int,
        history_length: int,
        feature_version: int,
        policy_encoding: str = "board_aware",
        include_heuristics: bool = False,
        heuristic_mode: str = "fast",
        num_heuristic_features: int = 21,
    ) -> dict[str, Any]:
        """Convert to kwargs dict for np.savez_compressed.

        Args:
            board_type: Board type string (e.g., "hex8")
            board_size: Board size (e.g., 8 for hex8)
            history_length: Number of history frames
            feature_version: Feature encoding version
            policy_encoding: "board_aware" or "legacy_max_n"
            include_heuristics: Whether to include heuristic features
            heuristic_mode: "fast" (21) or "full" (49)
            num_heuristic_features: Number of heuristic features

        Returns:
            Dictionary ready for np.savez_compressed(**kwargs)
        """
        save_kwargs: dict[str, Any] = {
            "features": self.features,
            "globals": self.globals,
            "values": self.values,
            "policy_indices": self.policy_indices,
            "policy_values": self.policy_values,
            "values_mp": self.values_mp,
            "num_players": self.num_players,
            "move_numbers": self.move_numbers,
            "total_game_moves": self.total_game_moves,
            "phases": self.phases,
            "victory_types": self.victory_types,
            "engine_modes": self.engine_modes,
            "move_types": self.move_types,
            "opponent_elo": self.opponent_elo,
            "opponent_types": self.opponent_types,
            "quality_score": self.quality_scores,
            # Metadata scalars
            "board_type": np.asarray(board_type),
            "board_size": np.asarray(int(board_size)),
            "history_length": np.asarray(int(history_length)),
            "feature_version": np.asarray(int(feature_version)),
            "policy_encoding": np.asarray(policy_encoding),
        }

        # Add heuristics if present
        if include_heuristics and self.heuristics is not None:
            save_kwargs["heuristics"] = self.heuristics
            save_kwargs["num_heuristic_features"] = np.asarray(int(num_heuristic_features))
            save_kwargs["heuristic_mode"] = np.asarray(heuristic_mode)

        return save_kwargs


class ArrayBuilder:
    """Accumulate training samples and build final arrays.

    This class manages the 17+ lists needed to accumulate training samples
    during export, providing a clean interface for adding samples and
    building the final stacked arrays.

    Example:
        builder = ArrayBuilder()

        for game in games:
            for sample in extract_samples(game):
                builder.add_sample(sample)

        arrays = builder.build_arrays()
        np.savez_compressed(output_path, **arrays.to_save_kwargs(...))
    """

    def __init__(self, *, include_heuristics: bool = False) -> None:
        """Initialize builder.

        Args:
            include_heuristics: Whether to track heuristic features
        """
        self.include_heuristics = include_heuristics
        self._features_list: list[np.ndarray] = []
        self._globals_list: list[np.ndarray] = []
        self._values_list: list[float] = []
        self._values_mp_list: list[np.ndarray] = []
        self._num_players_list: list[int] = []
        self._policy_indices_list: list[np.ndarray] = []
        self._policy_values_list: list[np.ndarray] = []
        self._move_numbers_list: list[int] = []
        self._total_game_moves_list: list[int] = []
        self._phases_list: list[str] = []
        self._victory_types_list: list[str] = []
        self._engine_modes_list: list[str] = []
        self._move_types_list: list[str] = []
        self._opponent_elo_list: list[float] = []
        self._opponent_types_list: list[str] = []
        self._quality_score_list: list[float] = []
        self._timestamps_list: list[float] = []
        self._heuristics_list: list[np.ndarray] = []

    @property
    def sample_count(self) -> int:
        """Return current number of accumulated samples."""
        return len(self._features_list)

    def add_sample(self, sample: Sample) -> None:
        """Add a single training sample to the accumulator.

        Args:
            sample: Sample to add
        """
        self._features_list.append(sample.features)
        self._globals_list.append(sample.globals)
        self._values_list.append(sample.value)
        self._values_mp_list.append(sample.values_mp)
        self._num_players_list.append(sample.num_players)
        self._policy_indices_list.append(sample.policy_indices)
        self._policy_values_list.append(sample.policy_values)
        self._move_numbers_list.append(sample.move_number)
        self._total_game_moves_list.append(sample.total_game_moves)
        self._phases_list.append(sample.phase)
        self._victory_types_list.append(sample.victory_type)
        self._engine_modes_list.append(sample.engine_mode)
        self._move_types_list.append(sample.move_type)
        self._opponent_elo_list.append(sample.opponent_elo)
        self._opponent_types_list.append(sample.opponent_type)
        self._quality_score_list.append(sample.quality_score)
        self._timestamps_list.append(sample.timestamp)

        if self.include_heuristics:
            if sample.heuristics is not None:
                self._heuristics_list.append(sample.heuristics)
            else:
                # Placeholder for missing heuristics
                self._heuristics_list.append(np.zeros(21, dtype=np.float32))

    def add_samples(self, samples: list[Sample]) -> None:
        """Add multiple samples to the accumulator.

        Args:
            samples: List of samples to add
        """
        for sample in samples:
            self.add_sample(sample)

    def clear(self) -> None:
        """Clear all accumulated samples."""
        self._features_list.clear()
        self._globals_list.clear()
        self._values_list.clear()
        self._values_mp_list.clear()
        self._num_players_list.clear()
        self._policy_indices_list.clear()
        self._policy_values_list.clear()
        self._move_numbers_list.clear()
        self._total_game_moves_list.clear()
        self._phases_list.clear()
        self._victory_types_list.clear()
        self._engine_modes_list.clear()
        self._move_types_list.clear()
        self._opponent_elo_list.clear()
        self._opponent_types_list.clear()
        self._quality_score_list.clear()
        self._timestamps_list.clear()
        self._heuristics_list.clear()

    def build_arrays(self) -> BuiltArrays:
        """Stack accumulated samples into final arrays.

        Returns:
            BuiltArrays container with all stacked arrays

        Raises:
            ValueError: If no samples have been accumulated
        """
        if not self._features_list:
            raise ValueError("No samples accumulated - cannot build arrays")

        # Stack arrays with appropriate dtypes
        features_arr = np.stack(self._features_list, axis=0).astype(np.float32)
        globals_arr = np.stack(self._globals_list, axis=0).astype(np.float32)
        values_arr = np.array(self._values_list, dtype=np.float32)
        values_mp_arr = np.stack(self._values_mp_list, axis=0).astype(np.float32)
        num_players_arr = np.array(self._num_players_list, dtype=np.int32)

        # Policy uses object arrays for sparse encoding (variable-length)
        policy_indices_arr = np.array(self._policy_indices_list, dtype=object)
        policy_values_arr = np.array(self._policy_values_list, dtype=object)

        move_numbers_arr = np.array(self._move_numbers_list, dtype=np.int32)
        total_game_moves_arr = np.array(self._total_game_moves_list, dtype=np.int32)

        # String arrays use object dtype
        phases_arr = np.array(self._phases_list, dtype=object)
        victory_types_arr = np.array(self._victory_types_list, dtype=object)
        engine_modes_arr = np.array(self._engine_modes_list, dtype=object)
        move_types_arr = np.array(self._move_types_list, dtype=object)
        opponent_types_arr = np.array(self._opponent_types_list, dtype=object)

        opponent_elo_arr = np.array(self._opponent_elo_list, dtype=np.float32)
        quality_scores_arr = np.array(self._quality_score_list, dtype=np.float32)
        timestamps_arr = np.array(self._timestamps_list, dtype=np.float64)

        # Heuristics optional
        heuristics_arr = None
        if self.include_heuristics and self._heuristics_list:
            heuristics_arr = np.stack(self._heuristics_list, axis=0).astype(np.float32)

        return BuiltArrays(
            features=features_arr,
            globals=globals_arr,
            values=values_arr,
            values_mp=values_mp_arr,
            num_players=num_players_arr,
            policy_indices=policy_indices_arr,
            policy_values=policy_values_arr,
            move_numbers=move_numbers_arr,
            total_game_moves=total_game_moves_arr,
            phases=phases_arr,
            victory_types=victory_types_arr,
            engine_modes=engine_modes_arr,
            move_types=move_types_arr,
            opponent_elo=opponent_elo_arr,
            opponent_types=opponent_types_arr,
            quality_scores=quality_scores_arr,
            timestamps=timestamps_arr,
            heuristics=heuristics_arr,
            sample_count=len(self._features_list),
        )

    def get_max_policy_index(self) -> int:
        """Get the maximum policy index across all samples.

        Used for validation that policy indices fit within the expected
        action space for the board type.

        Returns:
            Maximum policy index found, or 0 if no samples
        """
        max_idx = 0
        for indices in self._policy_indices_list:
            if len(indices) > 0:
                max_idx = max(max_idx, int(np.max(indices)))
        return max_idx

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics about accumulated samples.

        Returns:
            Dictionary with sample statistics
        """
        if not self._features_list:
            return {
                "sample_count": 0,
                "unique_phases": set(),
                "unique_victory_types": set(),
                "unique_engine_modes": set(),
            }

        return {
            "sample_count": len(self._features_list),
            "feature_shape": self._features_list[0].shape,
            "globals_shape": self._globals_list[0].shape,
            "unique_phases": set(self._phases_list),
            "unique_victory_types": set(self._victory_types_list),
            "unique_engine_modes": set(self._engine_modes_list),
            "max_policy_index": self.get_max_policy_index(),
            "avg_quality_score": sum(self._quality_score_list) / len(self._quality_score_list),
            "has_heuristics": bool(self._heuristics_list),
        }


def merge_built_arrays(
    existing: BuiltArrays,
    new: BuiltArrays,
) -> BuiltArrays:
    """Merge two BuiltArrays by concatenating along sample axis.

    Used for appending to existing NPZ files.

    Args:
        existing: Arrays from existing NPZ
        new: Newly built arrays to append

    Returns:
        Merged BuiltArrays
    """
    return BuiltArrays(
        features=np.concatenate([existing.features, new.features], axis=0),
        globals=np.concatenate([existing.globals, new.globals], axis=0),
        values=np.concatenate([existing.values, new.values], axis=0),
        values_mp=np.concatenate([existing.values_mp, new.values_mp], axis=0),
        num_players=np.concatenate([existing.num_players, new.num_players], axis=0),
        policy_indices=np.concatenate([existing.policy_indices, new.policy_indices], axis=0),
        policy_values=np.concatenate([existing.policy_values, new.policy_values], axis=0),
        move_numbers=np.concatenate([existing.move_numbers, new.move_numbers], axis=0),
        total_game_moves=np.concatenate([existing.total_game_moves, new.total_game_moves], axis=0),
        phases=np.concatenate([existing.phases, new.phases], axis=0),
        victory_types=np.concatenate([existing.victory_types, new.victory_types], axis=0),
        engine_modes=np.concatenate([existing.engine_modes, new.engine_modes], axis=0),
        move_types=np.concatenate([existing.move_types, new.move_types], axis=0),
        opponent_elo=np.concatenate([existing.opponent_elo, new.opponent_elo], axis=0),
        opponent_types=np.concatenate([existing.opponent_types, new.opponent_types], axis=0),
        quality_scores=np.concatenate([existing.quality_scores, new.quality_scores], axis=0),
        timestamps=np.concatenate([existing.timestamps, new.timestamps], axis=0),
        heuristics=(
            np.concatenate([existing.heuristics, new.heuristics], axis=0)
            if existing.heuristics is not None and new.heuristics is not None
            else None
        ),
        sample_count=existing.sample_count + new.sample_count,
    )


def load_existing_arrays(npz_path: str) -> BuiltArrays | None:
    """Load arrays from existing NPZ file for appending.

    Args:
        npz_path: Path to existing NPZ file

    Returns:
        BuiltArrays if file exists and is valid, None otherwise
    """
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "features" not in data:
                return None

            # Load required arrays
            features = data["features"]
            globals_arr = data.get("globals", np.zeros((len(features), 0), dtype=np.float32))
            values = data.get("values", np.zeros(len(features), dtype=np.float32))

            # Handle optional arrays with sensible defaults
            sample_count = len(features)

            return BuiltArrays(
                features=features,
                globals=globals_arr,
                values=values,
                values_mp=data.get("values_mp", np.zeros((sample_count, 4), dtype=np.float32)),
                num_players=data.get("num_players", np.full(sample_count, 2, dtype=np.int32)),
                policy_indices=data.get("policy_indices", np.array([np.array([], dtype=np.int32)] * sample_count, dtype=object)),
                policy_values=data.get("policy_values", np.array([np.array([], dtype=np.float32)] * sample_count, dtype=object)),
                move_numbers=data.get("move_numbers", np.zeros(sample_count, dtype=np.int32)),
                total_game_moves=data.get("total_game_moves", np.zeros(sample_count, dtype=np.int32)),
                phases=data.get("phases", np.array(["unknown"] * sample_count, dtype=object)),
                victory_types=data.get("victory_types", np.array([""] * sample_count, dtype=object)),
                engine_modes=data.get("engine_modes", np.array([""] * sample_count, dtype=object)),
                move_types=data.get("move_types", np.array(["unknown"] * sample_count, dtype=object)),
                opponent_elo=data.get("opponent_elo", np.zeros(sample_count, dtype=np.float32)),
                opponent_types=data.get("opponent_types", np.array([""] * sample_count, dtype=object)),
                quality_scores=data.get("quality_score", np.full(sample_count, 0.5, dtype=np.float32)),
                timestamps=data.get("timestamps", np.zeros(sample_count, dtype=np.float64)),
                heuristics=data.get("heuristics", None),
                sample_count=sample_count,
            )
    except Exception as e:
        logger.warning(f"Failed to load existing NPZ {npz_path}: {e}")
        return None
