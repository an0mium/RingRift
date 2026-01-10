"""Configuration dataclasses for training data export.

This module provides type-safe configuration for the export pipeline:
- ExportConfig: Main export parameters (board type, output path, etc.)
- FilterConfig: Game filtering criteria (min/max moves, quality thresholds)
- ExportResult: Export operation result summary

Usage:
    from app.training.export.config import ExportConfig, FilterConfig, ExportResult

    config = ExportConfig(
        board_type="hex8",
        num_players=2,
        output_path=Path("data/training/hex8_2p.npz"),
    )
    filter_config = FilterConfig(
        min_game_length=10,
        require_completed=True,
    )
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExportConfig:
    """Configuration for exporting training samples from game databases.

    Attributes:
        board_type: Board type string (hex8, square8, square19, hexagonal)
        num_players: Number of players (2, 3, or 4)
        output_path: Path for output NPZ file
        history_length: Number of past feature frames to stack (default: 3)
        feature_version: Feature encoding version for global features (default: 2)
        sample_every: Use every Nth move as a training sample (default: 1)
        max_samples: Maximum samples to export (None for unlimited)
        include_intermediate: Include non-terminal moves as samples (default: True)
        augment_symmetries: Apply symmetry augmentations (default: False)
        min_quality_score: Minimum game quality score to include (None for no filter)
        use_rank_aware_values: Use rank-based values for multiplayer (default: True)
        use_board_aware_encoding: Use board-specific policy encoding (default: True)
        encoder_version: Encoder version for hex boards ('default', 'v2', 'v3')
        include_heuristics: Extract heuristic features for v5 training (default: False)
        full_heuristics: Use full 49-feature extraction (default: False)
        quality_weighted: Compute quality-weighted sample weights (default: False)
    """

    board_type: str
    num_players: int
    output_path: Path
    history_length: int = 3
    feature_version: int = 2
    sample_every: int = 1
    max_samples: int | None = None
    include_intermediate: bool = True
    augment_symmetries: bool = False
    min_quality_score: float | None = None
    use_rank_aware_values: bool = True
    use_board_aware_encoding: bool = True
    encoder_version: str = "default"
    include_heuristics: bool = False
    full_heuristics: bool = False
    quality_weighted: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

        # Validate board_type
        valid_board_types = {"hex8", "square8", "square19", "hexagonal"}
        if self.board_type not in valid_board_types:
            raise ValueError(
                f"Invalid board_type '{self.board_type}'. "
                f"Must be one of: {sorted(valid_board_types)}"
            )

        # Validate num_players
        if self.num_players not in (2, 3, 4):
            raise ValueError(
                f"Invalid num_players {self.num_players}. Must be 2, 3, or 4."
            )

        # Validate history_length
        if self.history_length < 0:
            raise ValueError(f"history_length must be non-negative, got {self.history_length}")

        # Validate sample_every
        if self.sample_every < 1:
            raise ValueError(f"sample_every must be >= 1, got {self.sample_every}")

        # full_heuristics implies include_heuristics
        if self.full_heuristics and not self.include_heuristics:
            self.include_heuristics = True

    @property
    def config_key(self) -> str:
        """Return canonical config key (e.g., 'hex8_2p')."""
        return f"{self.board_type}_{self.num_players}p"


@dataclass
class FilterConfig:
    """Configuration for filtering games during export.

    Attributes:
        min_game_length: Minimum number of moves to include a game (default: 0)
        max_game_length: Maximum number of moves to include a game (None for unlimited)
        require_completed: Only include games with normal termination (default: False)
        require_moves: Only include games with move data (default: True)
        exclude_recovery: Exclude games with recovery moves (default: False)
        max_move_index: Only process moves up to this index (None for all)
        fail_on_orphans: Fail export if orphan games found (default: True)
        include_sources: Source types to include (None = all, or set like {'selfplay', 'gauntlet'})
        exclude_sources: Source types to exclude (None = none excluded)
        parity_fixtures_dir: Directory with parity fixture JSONs for cutoffs
    """

    min_game_length: int = 0
    max_game_length: int | None = None
    require_completed: bool = False
    require_moves: bool = True
    exclude_recovery: bool = False
    max_move_index: int | None = None
    fail_on_orphans: bool = True
    include_sources: set[str] | None = None
    exclude_sources: set[str] | None = None
    parity_fixtures_dir: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.min_game_length < 0:
            raise ValueError(f"min_game_length must be non-negative, got {self.min_game_length}")

        if self.max_game_length is not None and self.max_game_length < 0:
            raise ValueError(f"max_game_length must be non-negative, got {self.max_game_length}")

        if (
            self.min_game_length
            and self.max_game_length
            and self.min_game_length > self.max_game_length
        ):
            raise ValueError(
                f"min_game_length ({self.min_game_length}) > max_game_length ({self.max_game_length})"
            )

    def to_query_filters(self, board_type: str, num_players: int) -> dict[str, Any]:
        """Convert to query filter dict for GameReplayDB.

        Args:
            board_type: Board type (e.g., 'hex8')
            num_players: Number of players

        Returns:
            Dictionary of query filters for iterate_games_with_probs()
        """
        # Import BoardType here to avoid circular imports
        from app.models import BoardType

        query_filters: dict[str, Any] = {
            "board_type": BoardType(board_type),
            "num_players": num_players,
            "require_moves": self.require_moves,
        }
        if self.min_game_length > 0:
            query_filters["min_moves"] = self.min_game_length
        if self.max_game_length is not None:
            query_filters["max_moves"] = self.max_game_length
        return query_filters

    def get_filter_description(self) -> list[str]:
        """Return human-readable description of active filters."""
        desc = []
        if self.require_moves:
            desc.append("require move data")
        if self.require_completed:
            desc.append("completed games only")
        if self.min_game_length > 0:
            desc.append(f"min {self.min_game_length} moves")
        if self.max_game_length is not None:
            desc.append(f"max {self.max_game_length} moves")
        if self.exclude_recovery:
            desc.append("excluding recovery games")
        if self.include_sources:
            desc.append(f"sources: {sorted(self.include_sources)}")
        if self.exclude_sources:
            desc.append(f"excluding: {sorted(self.exclude_sources)}")
        return desc


@dataclass
class ExportResult:
    """Result summary from an export operation.

    Attributes:
        total_games: Total games processed
        total_samples: Total samples exported
        filtered_games: Games skipped due to filters
        deduplicated_games: Games skipped due to deduplication
        partial_games: Games with partial sample extraction
        duration_seconds: Export duration in seconds
        output_path: Path to output NPZ file
        arrays: Dict of array names to shapes
        metadata: Additional metadata about the export
    """

    total_games: int
    total_samples: int
    filtered_games: int = 0
    deduplicated_games: int = 0
    partial_games: int = 0
    duration_seconds: float = 0.0
    output_path: Path | None = None
    arrays: dict[str, tuple[int, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result data."""
        if self.total_games < 0:
            raise ValueError(f"total_games must be non-negative, got {self.total_games}")
        if self.total_samples < 0:
            raise ValueError(f"total_samples must be non-negative, got {self.total_samples}")

    @property
    def success(self) -> bool:
        """Return True if export produced samples."""
        return self.total_samples > 0

    @property
    def samples_per_game(self) -> float:
        """Return average samples per game."""
        if self.total_games == 0:
            return 0.0
        return self.total_samples / self.total_games

    @property
    def samples_per_second(self) -> float:
        """Return export throughput in samples per second."""
        if self.duration_seconds <= 0:
            return 0.0
        return self.total_samples / self.duration_seconds

    def summary(self) -> str:
        """Return human-readable summary string."""
        lines = [
            f"Export Result: {'SUCCESS' if self.success else 'NO SAMPLES'}",
            f"  Games: {self.total_games:,} processed "
            f"({self.deduplicated_games:,} deduplicated, {self.filtered_games:,} filtered)",
        ]
        if self.partial_games > 0:
            lines.append(f"  Partial games: {self.partial_games:,}")
        lines.append(f"  Samples: {self.total_samples:,} ({self.samples_per_game:.1f} per game)")
        if self.duration_seconds > 0:
            lines.append(
                f"  Duration: {self.duration_seconds:.1f}s "
                f"({self.samples_per_second:.0f} samples/sec)"
            )
        if self.output_path:
            lines.append(f"  Output: {self.output_path}")
        return "\n".join(lines)


# Environment variable configuration (for compatibility with existing scripts)
DB_LOCK_MAX_RETRIES = int(os.getenv("RINGRIFT_DB_LOCK_MAX_RETRIES", "5"))
DB_LOCK_INITIAL_WAIT = float(os.getenv("RINGRIFT_DB_LOCK_INITIAL_WAIT", "0.5"))
DB_LOCK_MAX_WAIT = float(os.getenv("RINGRIFT_DB_LOCK_MAX_WAIT", "30.0"))

# Disk space safety margins
DISK_SPACE_SAFETY_MARGIN_MB = 2048  # Keep 2GB free after write
NPZ_COMPRESSION_RATIO = 0.30  # Conservative compression ratio estimate
