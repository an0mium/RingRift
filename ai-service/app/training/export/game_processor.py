"""Game processing and iteration for training data export.

This module provides the GameIterator class for iterating over games
from databases with filtering and deduplication.

Usage:
    from app.training.export.game_processor import GameIterator, GameIteratorConfig

    config = GameIteratorConfig(
        board_type="hex8",
        num_players=2,
        require_completed=True,
    )
    iterator = GameIterator(config)

    for game_data in iterator.iterate_databases(db_paths):
        # Process game_data
        pass
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from app.db import GameReplayDB
    from app.models.core import GameState, Move

logger = logging.getLogger(__name__)

# Database lock retry settings (from config)
DB_LOCK_MAX_RETRIES = 5
DB_LOCK_INITIAL_WAIT = 0.5
DB_LOCK_MAX_WAIT = 30.0


@dataclass
class GameIteratorConfig:
    """Configuration for game iteration.

    Attributes:
        board_type: Board type to filter by
        num_players: Number of players to filter by
        require_moves: Only include games with move data
        require_completed: Only include completed games
        min_moves: Minimum number of moves
        max_moves: Maximum number of moves
        min_quality: Minimum quality score threshold
        exclude_recovery: Exclude games with recovery moves
        include_sources: Source types to include (None = all)
        exclude_sources: Source types to exclude (None = none)
        fail_on_orphans: Fail if orphan games found
        parity_fixtures_dir: Directory with parity fixture JSONs
        max_games: Maximum games to process (None = unlimited)
    """

    board_type: str
    num_players: int
    require_moves: bool = True
    require_completed: bool = False
    min_moves: int | None = None
    max_moves: int | None = None
    min_quality: float | None = None
    exclude_recovery: bool = False
    include_sources: set[str] | None = None
    exclude_sources: set[str] | None = None
    fail_on_orphans: bool = True
    parity_fixtures_dir: str | None = None
    max_games: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_moves is not None and self.min_moves < 0:
            raise ValueError(f"min_moves must be non-negative, got {self.min_moves}")
        if self.max_moves is not None and self.max_moves < 0:
            raise ValueError(f"max_moves must be non-negative, got {self.max_moves}")


@dataclass
class GameData:
    """Data for a single game ready for sample collection.

    Attributes:
        game_id: Unique game identifier
        initial_state: Initial game state
        moves: List of moves
        move_probs: Move probabilities for soft targets
        victory_type: Normalized victory type
        engine_mode: Engine mode for weighting
        opponent_elo: Opponent Elo rating
        opponent_type: Opponent type
        quality_score: Game quality score
        timestamp: Game timestamp (Unix epoch)
        db_winner: Winner from database
        total_moves: Total moves in game
        max_safe_move_index: Maximum safe move index (parity cutoff)
    """

    game_id: str
    initial_state: Any  # GameState
    moves: list[Any]  # list[Move]
    move_probs: dict[int, dict[str, float]] = field(default_factory=dict)
    victory_type: str = ""
    engine_mode: str = "unknown"
    opponent_elo: float = 1500.0
    opponent_type: str = "unknown"
    quality_score: float = 1.0
    timestamp: float = 0.0
    db_winner: int | None = None
    total_moves: int = 0
    max_safe_move_index: int | None = None


@dataclass
class IterationStats:
    """Statistics from game iteration.

    Attributes:
        games_processed: Total games processed
        games_skipped: Games skipped by filters
        games_deduplicated: Games skipped as duplicates
        games_skipped_recovery: Games skipped for recovery moves
        samples_collected: Total samples collected (if tracked)
        databases_processed: Number of databases processed
        databases_skipped: Number of databases skipped
        newest_game_time: Timestamp of newest game
    """

    games_processed: int = 0
    games_skipped: int = 0
    games_deduplicated: int = 0
    games_skipped_recovery: int = 0
    samples_collected: int = 0
    databases_processed: int = 0
    databases_skipped: int = 0
    newest_game_time: str | None = None


def _is_db_locked_error(e: Exception) -> bool:
    """Check if exception is a database lock error."""
    error_str = str(e).lower()
    return (
        "database is locked" in error_str or
        ("locked" in error_str and "database" in error_str) or
        (isinstance(e, sqlite3.OperationalError) and "locked" in str(e))
    )


def _open_db_with_retry(
    db_path: str,
    max_retries: int = DB_LOCK_MAX_RETRIES,
    initial_wait: float = DB_LOCK_INITIAL_WAIT,
) -> GameReplayDB:
    """Open GameReplayDB with retry logic for database lock errors.

    Uses exponential backoff when the database is locked.

    Args:
        db_path: Path to the SQLite database
        max_retries: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds

    Returns:
        GameReplayDB instance

    Raises:
        Exception: If database cannot be opened after all retries
    """
    from app.db import GameReplayDB

    last_error: Exception | None = None
    wait_time = initial_wait

    for attempt in range(max_retries + 1):
        try:
            return GameReplayDB(db_path)
        except Exception as e:
            last_error = e
            if not _is_db_locked_error(e):
                raise

            if attempt < max_retries:
                actual_wait = min(wait_time, DB_LOCK_MAX_WAIT)
                logger.warning(
                    f"Database locked, retrying in {actual_wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries}): {db_path}"
                )
                time.sleep(actual_wait)
                wait_time *= 2

    logger.error(f"Failed to open database after {max_retries} retries: {db_path}")
    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to open database: {db_path}")


def _load_parity_cutoffs(fixtures_dir: str | None) -> dict[str, int]:
    """Load parity cutoffs from fixture directory.

    Args:
        fixtures_dir: Directory containing parity fixture JSONs

    Returns:
        Dict mapping game_id to max safe move index
    """
    if not fixtures_dir:
        return {}

    cutoffs: dict[str, int] = {}
    fixtures_path = os.path.abspath(fixtures_dir)

    if not os.path.isdir(fixtures_path):
        return cutoffs

    for name in os.listdir(fixtures_path):
        if not name.endswith(".json"):
            continue
        path = os.path.join(fixtures_path, name)
        try:
            with open(path, encoding="utf-8") as f:
                fixture = json.load(f)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
            continue

        game_id = fixture.get("game_id")
        diverged_at = fixture.get("diverged_at")
        if not isinstance(game_id, str) or not isinstance(diverged_at, int) or diverged_at <= 0:
            continue

        safe_max_move = diverged_at - 1
        prev = cutoffs.get(game_id)
        if prev is None or safe_max_move < prev:
            cutoffs[game_id] = safe_max_move

    return cutoffs


def _parse_timestamp(time_raw: Any) -> float:
    """Parse timestamp from various formats.

    Args:
        time_raw: Timestamp in various formats

    Returns:
        Unix timestamp (0.0 if parsing fails)
    """
    if time_raw is None:
        return 0.0

    try:
        if isinstance(time_raw, (int, float)):
            return float(time_raw)
        if isinstance(time_raw, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
                try:
                    dt = datetime.strptime(time_raw[:19], fmt[:19])
                    return dt.timestamp()
                except ValueError:
                    continue
    except Exception:
        pass

    return 0.0


def _categorize_source(source_raw: str) -> str:
    """Categorize game source into standard categories.

    Args:
        source_raw: Raw source string from metadata

    Returns:
        Categorized source (selfplay, gauntlet, tournament, human)
    """
    source_lower = source_raw.lower()
    if "gauntlet" in source_lower:
        return "gauntlet"
    if "tournament" in source_lower:
        return "tournament"
    if "human" in source_lower:
        return "human"
    return "selfplay"


def _normalize_victory_type(victory_raw: str) -> str:
    """Normalize victory type to standard categories.

    Args:
        victory_raw: Raw victory type string

    Returns:
        Normalized victory type
    """
    victory_lower = victory_raw.lower()
    if "territory" in victory_lower:
        return "territory"
    if "elimination" in victory_lower or "ring" in victory_lower:
        return "elimination"
    if "lps" in victory_lower or "last_player" in victory_lower:
        return "lps"
    if "stalemate" in victory_lower:
        return "stalemate"
    if "timeout" in victory_lower:
        return "timeout"
    return "other"


def _extract_engine_mode(source_raw: str) -> str:
    """Extract engine mode from source string.

    Args:
        source_raw: Raw source string

    Returns:
        Engine mode string
    """
    source_lower = source_raw.lower()
    if "gumbel" in source_lower:
        return "gumbel_mcts"
    if "mcts" in source_lower:
        return "mcts"
    if "policy" in source_lower:
        return "policy_only"
    if "descent" in source_lower:
        return "descent"
    if "heuristic" in source_lower or "gpu" in source_lower:
        return "heuristic"
    return "unknown"


def _infer_opponent_type(source_raw: str) -> str:
    """Infer opponent type from source string.

    Args:
        source_raw: Raw source string

    Returns:
        Opponent type string
    """
    source_lower = source_raw.lower()
    if "random" in source_lower:
        return "random"
    if "heuristic" in source_lower:
        return "heuristic"
    if "mcts" in source_lower:
        return "mcts"
    if "gumbel" in source_lower:
        return "gumbel_mcts"
    if "neural" in source_lower or "nn" in source_lower:
        return "neural"
    return "unknown"


class GameIterator:
    """Iterate over games from databases with filtering and deduplication.

    This class handles:
    - Database opening with retry logic
    - Game filtering by source, quality, completion
    - Deduplication across multiple databases
    - Metadata extraction and normalization
    - Parity cutoff handling

    Example:
        config = GameIteratorConfig(board_type="hex8", num_players=2)
        iterator = GameIterator(config)

        stats = IterationStats()
        for game_data in iterator.iterate_databases(db_paths, stats):
            samples = collector.collect_from_game(
                initial_state=game_data.initial_state,
                moves=game_data.moves,
                metadata=GameMetadata(
                    quality_score=game_data.quality_score,
                    ...
                ),
            )
    """

    def __init__(self, config: GameIteratorConfig) -> None:
        """Initialize iterator with configuration.

        Args:
            config: Iteration configuration
        """
        self.config = config
        self._seen_game_ids: set[str] = set()
        self._parity_cutoffs: dict[str, int] = _load_parity_cutoffs(
            config.parity_fixtures_dir
        )
        self._quality_scorer = None
        self._board_type_enum = None

    def _get_board_type_enum(self) -> Any:
        """Get BoardType enum value for the configured board type."""
        if self._board_type_enum is None:
            from app.models import BoardType
            self._board_type_enum = BoardType(self.config.board_type)
        return self._board_type_enum

    def _get_quality_scorer(self) -> Any:
        """Get quality scorer function if available."""
        if self._quality_scorer is None:
            try:
                from app.quality.unified_quality import compute_game_quality_from_params
                self._quality_scorer = compute_game_quality_from_params
            except ImportError:
                self._quality_scorer = False  # Mark as unavailable
        return self._quality_scorer if self._quality_scorer else None

    def _build_query_filters(self) -> dict[str, Any]:
        """Build query filters for database iteration."""
        board_type = self._get_board_type_enum()
        filters: dict[str, Any] = {
            "board_type": board_type,
            "num_players": self.config.num_players,
            "require_moves": self.config.require_moves,
        }
        if self.config.min_moves is not None:
            filters["min_moves"] = self.config.min_moves
        if self.config.max_moves is not None:
            filters["max_moves"] = self.config.max_moves
        return filters

    def _validate_database(self, db_path: str) -> tuple[bool, str]:
        """Validate database for export.

        Args:
            db_path: Path to database

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            from app.db.move_data_validator import (
                MIN_MOVES_REQUIRED,
                MoveDataValidationError,
                MoveDataValidator,
            )

            result = MoveDataValidator.validate_database(db_path, min_moves=MIN_MOVES_REQUIRED)

            if result.invalid_count > 0:
                if self.config.fail_on_orphans:
                    return False, (
                        f"{result.invalid_count} orphan games found "
                        "(games with insufficient move data)"
                    )
                else:
                    logger.warning(
                        f"Found {result.invalid_count} orphan games in {db_path}, "
                        "continuing due to fail_on_orphans=False"
                    )

            if not result.has_game_moves_table:
                return False, "Metadata-only database (no game_moves table)"

            return True, "Database valid"

        except ImportError:
            # Validator not available, skip validation
            return True, "Validation skipped (module not available)"
        except Exception as e:
            return False, f"Validation error: {e}"

    def _should_include_game(
        self,
        meta: dict[str, Any],
        moves: list[Any],
    ) -> tuple[bool, str]:
        """Check if game should be included based on filters.

        Args:
            meta: Game metadata
            moves: List of moves

        Returns:
            Tuple of (include, skip_reason)
        """
        # Source filtering
        if self.config.include_sources or self.config.exclude_sources:
            source_raw = str(meta.get("source", "") or "").lower()
            game_source = _categorize_source(source_raw)

            if self.config.include_sources and "all" not in self.config.include_sources:
                if game_source not in self.config.include_sources:
                    return False, "source_excluded"

            if self.config.exclude_sources and game_source in self.config.exclude_sources:
                return False, "source_excluded"

        # Completion filtering
        if self.config.require_completed:
            status = str(meta.get("game_status", ""))
            term = str(meta.get("termination_reason", ""))
            if status != "completed":
                return False, "not_completed"
            if term and not (term.startswith("status:completed") or term == "env_done_flag"):
                return False, "not_completed"

        # Quality filtering
        quality_scorer = self._get_quality_scorer()
        if quality_scorer and self.config.min_quality is not None:
            board_type = self._get_board_type_enum()
            quality = quality_scorer(
                game_id=meta.get("game_id", "unknown"),
                game_status=str(meta.get("game_status", "")),
                winner=meta.get("winner"),
                termination_reason=str(meta.get("termination_reason", "")),
                total_moves=len(moves) if moves else 0,
                board_type=board_type.value if hasattr(board_type, "value") else str(board_type),
                source=str(meta.get("source", "")),
            )
            if quality.quality_score < self.config.min_quality:
                return False, "low_quality"

        # Move count filtering
        total_moves = meta.get("total_moves")
        if total_moves is None:
            total_moves = len(moves) if moves else 0
        if total_moves <= 0 or not moves:
            return False, "no_moves"

        # Recovery filtering
        if self.config.exclude_recovery:
            has_recovery = any(
                "recovery" in str(getattr(m, "type", "")).lower()
                for m in moves
            )
            if has_recovery:
                return False, "has_recovery"

        return True, ""

    def _extract_game_data(
        self,
        meta: dict[str, Any],
        initial_state: Any,
        moves: list[Any],
        move_probs: dict[int, dict[str, float]],
    ) -> GameData:
        """Extract game data from database row.

        Args:
            meta: Game metadata
            initial_state: Initial game state
            moves: List of moves
            move_probs: Move probabilities

        Returns:
            GameData instance
        """
        game_id = meta.get("game_id", "")
        source_raw = str(meta.get("source", "") or "")

        # Victory type
        victory_raw = str(meta.get("victory_type", meta.get("termination_reason", "unknown")))
        victory_type = _normalize_victory_type(victory_raw)

        # Engine mode
        engine_mode = _extract_engine_mode(source_raw)

        # Opponent info
        opponent_elo = float(meta.get("opponent_elo", meta.get("model_elo", 1500.0)))
        opponent_type = meta.get("opponent_type") or _infer_opponent_type(source_raw)

        # Quality score
        quality_score = 1.0
        quality_scorer = self._get_quality_scorer()
        if quality_scorer:
            board_type = self._get_board_type_enum()
            quality = quality_scorer(
                game_id=game_id or "unknown",
                game_status=str(meta.get("game_status", "")),
                winner=meta.get("winner"),
                termination_reason=str(meta.get("termination_reason", "")),
                total_moves=len(moves) if moves else 0,
                board_type=board_type.value if hasattr(board_type, "value") else str(board_type),
                source=source_raw,
            )
            quality_score = quality.quality_score

        # Timestamp
        game_time_raw = meta.get("completed_at") or meta.get("created_at")
        timestamp = _parse_timestamp(game_time_raw)

        # Database winner (fallback for partial games)
        db_winner = meta.get("winner")

        # Total moves
        total_moves = meta.get("total_moves")
        if total_moves is None:
            total_moves = len(moves) if moves else 0
        total_moves = int(total_moves)

        # Parity cutoff
        max_safe_move_index = self._parity_cutoffs.get(game_id)

        return GameData(
            game_id=game_id,
            initial_state=initial_state,
            moves=moves,
            move_probs=move_probs,
            victory_type=victory_type,
            engine_mode=engine_mode,
            opponent_elo=opponent_elo,
            opponent_type=opponent_type,
            quality_score=quality_score,
            timestamp=timestamp,
            db_winner=db_winner,
            total_moves=total_moves,
            max_safe_move_index=max_safe_move_index,
        )

    def iterate_databases(
        self,
        db_paths: list[str | Path],
        stats: IterationStats | None = None,
    ) -> Iterator[GameData]:
        """Iterate over games from multiple databases.

        Args:
            db_paths: List of database paths
            stats: Optional stats object to update

        Yields:
            GameData for each valid game
        """
        if stats is None:
            stats = IterationStats()

        query_filters = self._build_query_filters()

        for db_idx, db_path in enumerate(db_paths):
            db_path = str(db_path)

            if not os.path.exists(db_path):
                logger.warning(f"Skipping missing database: {db_path}")
                stats.databases_skipped += 1
                continue

            logger.info(f"Processing [{db_idx + 1}/{len(db_paths)}]: {os.path.basename(db_path)}")

            # Validate database
            is_valid, message = self._validate_database(db_path)
            if not is_valid:
                logger.warning(f"Skipping database: {message}")
                stats.databases_skipped += 1
                continue

            # Open database with retry
            try:
                db = _open_db_with_retry(db_path)
            except Exception as e:
                if _is_db_locked_error(e):
                    logger.warning(f"Database locked after retries: {db_path}")
                else:
                    logger.error(f"Error opening database: {e}")
                stats.databases_skipped += 1
                continue

            stats.databases_processed += 1

            # Iterate games
            db_games = 0
            for meta, initial_state, moves, move_probs in db.iterate_games_with_probs(**query_filters):
                game_id = meta.get("game_id", "")

                # Deduplication
                if game_id in self._seen_game_ids:
                    stats.games_deduplicated += 1
                    continue
                self._seen_game_ids.add(game_id)

                # Filtering
                include, skip_reason = self._should_include_game(meta, moves)
                if not include:
                    stats.games_skipped += 1
                    if skip_reason == "has_recovery":
                        stats.games_skipped_recovery += 1
                    continue

                # Extract game data
                game_data = self._extract_game_data(meta, initial_state, moves, move_probs)

                # Skip games with parity cutoff <= 0
                if game_data.max_safe_move_index is not None:
                    if game_data.max_safe_move_index <= 0:
                        stats.games_skipped += 1
                        continue

                # Track newest game time
                game_time = meta.get("completed_at") or meta.get("created_at")
                if game_time:
                    game_time_str = str(game_time)
                    if stats.newest_game_time is None or game_time_str > stats.newest_game_time:
                        stats.newest_game_time = game_time_str

                stats.games_processed += 1
                db_games += 1

                yield game_data

                # Check max_games limit
                if self.config.max_games is not None:
                    if stats.games_processed >= self.config.max_games:
                        logger.info(f"Reached max_games limit ({self.config.max_games})")
                        return

            logger.info(f"  -> {db_games} games from {os.path.basename(db_path)}")

    def reset(self) -> None:
        """Reset iterator state for reuse."""
        self._seen_game_ids.clear()


def create_iterator(
    board_type: str,
    num_players: int,
    **kwargs: Any,
) -> GameIterator:
    """Create a GameIterator with the given configuration.

    Convenience factory function.

    Args:
        board_type: Board type string
        num_players: Number of players
        **kwargs: Additional GameIteratorConfig arguments

    Returns:
        Configured GameIterator instance
    """
    config = GameIteratorConfig(
        board_type=board_type,
        num_players=num_players,
        **kwargs,
    )
    return GameIterator(config)
