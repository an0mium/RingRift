"""Incremental Export Cache for Training Data.

Provides caching for expensive NPZ exports from GameReplayDB databases.
Tracks source database modification times and content hashes to determine
when re-export is necessary vs when cached output can be reused.

Usage:
    from app.training.export_cache import ExportCache

    cache = ExportCache()

    # Check if export is needed
    if cache.needs_export(db_paths, output_path, board_type, num_players):
        # Perform export
        export_replay_dataset_multi(...)
        # Update cache
        cache.record_export(db_paths, output_path, board_type, num_players)
    else:
        print(f"Using cached export: {output_path}")
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.utils.checksum_utils import compute_string_checksum

# Cache directory
from app.utils.paths import DATA_DIR, ensure_dir

EXPORT_CACHE_DIR = ensure_dir(DATA_DIR / "export_cache")


@dataclass
class ExportCacheEntry:
    """Cache entry for a single export operation."""
    output_path: str
    board_type: str
    num_players: int
    db_sources: dict[str, dict[str, Any]]  # path -> {mtime, size, game_count}
    export_timestamp: str
    samples_exported: int
    games_exported: int
    output_size: int
    output_mtime: float
    history_length: int | None = None
    feature_version: int | None = None
    policy_encoding: str | None = None
    # January 2026: Delta detection - track exported game IDs per database
    # Format: {db_path: [game_id1, game_id2, ...]} - list instead of set for JSON
    exported_game_ids: dict[str, list[str]] | None = None
    # Track max game ID per DB for faster delta queries
    max_game_ids: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExportCacheEntry:
        return cls(**d)


def _get_db_stats(db_path: str) -> dict[str, Any]:
    """Get stats for a database file (mtime, size, game count)."""
    path = Path(db_path)
    if not path.exists():
        return {"exists": False}

    stats = {
        "exists": True,
        "mtime": path.stat().st_mtime,
        "size": path.stat().st_size,
    }

    # Try to get game count from database
    try:
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM games")
        stats["game_count"] = cursor.fetchone()[0]
        conn.close()
    except (sqlite3.Error, OSError, TypeError, IndexError):
        stats["game_count"] = -1

    return stats


def _get_game_ids_from_db(
    db_path: str,
    board_type: str | None = None,
    num_players: int | None = None,
    require_completed: bool = False,
) -> tuple[list[str], int]:
    """Get all game IDs from a database, optionally filtered.

    January 2026: Added for delta detection in incremental exports.

    Args:
        db_path: Path to the SQLite database
        board_type: Optional filter for board type
        num_players: Optional filter for number of players
        require_completed: Only include completed games

    Returns:
        Tuple of (list of game IDs, max numeric game ID or 0)
    """
    path = Path(db_path)
    if not path.exists():
        return [], 0

    try:
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        # Build query with filters
        query = "SELECT game_id FROM games WHERE 1=1"
        params: list[Any] = []

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)

        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        if require_completed:
            query += " AND completed = 1"

        cursor.execute(query, params)
        game_ids = [str(row[0]) for row in cursor.fetchall()]

        # Get max numeric ID for faster delta queries
        max_id = 0
        for gid in game_ids:
            try:
                # Extract numeric part from game_id (handles prefixed IDs)
                numeric_part = int("".join(c for c in gid if c.isdigit()) or "0")
                max_id = max(max_id, numeric_part)
            except (ValueError, TypeError):
                pass

        conn.close()
        return game_ids, max_id

    except (sqlite3.Error, OSError, TypeError) as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to get game IDs from {db_path}: {e}")
        return [], 0


def _get_new_game_ids(
    db_path: str,
    exported_ids: set[str],
    board_type: str | None = None,
    num_players: int | None = None,
    require_completed: bool = False,
) -> list[str]:
    """Get game IDs that haven't been exported yet.

    January 2026: Core delta detection logic.

    Args:
        db_path: Path to the SQLite database
        exported_ids: Set of already-exported game IDs
        board_type: Optional filter for board type
        num_players: Optional filter for number of players
        require_completed: Only include completed games

    Returns:
        List of new game IDs not in exported_ids
    """
    all_ids, _ = _get_game_ids_from_db(
        db_path,
        board_type=board_type,
        num_players=num_players,
        require_completed=require_completed,
    )
    return [gid for gid in all_ids if gid not in exported_ids]


def _get_cache_key(
    board_type: str,
    num_players: int,
    output_path: str,
    history_length: int | None = None,
    feature_version: int | None = None,
    policy_encoding: str | None = None,
    # Quality filtering parameters (December 28, 2025)
    min_quality: float | None = None,
    require_completed: bool | None = None,
    encoder_version: str | None = None,
    include_heuristics: bool | None = None,
    full_heuristics: bool | None = None,
) -> str:
    """Generate a unique cache key for this export configuration.

    December 28, 2025: Added quality filtering parameters to cache key.
    Previously, changing min_quality or require_completed would not
    invalidate the cache, leading to stale training data.

    Args:
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players
        output_path: Path to output NPZ file
        history_length: Number of historical states to include
        feature_version: Feature extraction version
        policy_encoding: Policy encoding type
        min_quality: Minimum game quality threshold
        require_completed: Whether to require completed games
        encoder_version: Encoder version string
        include_heuristics: Whether to include heuristic features
        full_heuristics: Whether to use full (49) vs fast (21) heuristics
    """
    # Use a hash of the normalized output path plus feature context.
    output_norm = os.path.normpath(os.path.abspath(output_path))
    key_parts = [f"{board_type}_{num_players}p_{output_norm}"]
    if history_length is not None:
        key_parts.append(f"h{int(history_length)}")
    if feature_version is not None:
        key_parts.append(f"fv{int(feature_version)}")
    if policy_encoding:
        key_parts.append(str(policy_encoding))
    # Quality filtering parameters (December 28, 2025)
    if min_quality is not None and min_quality > 0.0:
        key_parts.append(f"mq{min_quality:.2f}")
    if require_completed is not None and require_completed:
        key_parts.append("req_complete")
    if encoder_version:
        key_parts.append(f"enc{encoder_version}")
    if include_heuristics:
        key_parts.append("heur")
    if full_heuristics:
        key_parts.append("full_heur")
    key_str = "_".join(key_parts)
    return compute_string_checksum(key_str, algorithm="md5", truncate=16)


def _get_cache_file(cache_key: str) -> Path:
    """Get the cache file path for a given key."""
    return EXPORT_CACHE_DIR / f"export_{cache_key}.json"


class ExportCache:
    """Manages incremental export caching for training data."""

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or EXPORT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_cache_entry(self, cache_key: str) -> ExportCacheEntry | None:
        """Load a cache entry by key."""
        cache_file = self.cache_dir / f"export_{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)
            return ExportCacheEntry.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, KeyError):
            return None

    def _save_cache_entry(self, cache_key: str, entry: ExportCacheEntry) -> None:
        """Save a cache entry."""
        cache_file = self.cache_dir / f"export_{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

    def needs_export(
        self,
        db_paths: list[str],
        output_path: str,
        board_type: str,
        num_players: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
        force: bool = False,
        # Quality filtering parameters (December 28, 2025)
        min_quality: float | None = None,
        require_completed: bool | None = None,
        encoder_version: str | None = None,
        include_heuristics: bool | None = None,
        full_heuristics: bool | None = None,
    ) -> bool:
        """Check if export is needed or if cached output is still valid.

        Returns True if export is needed, False if cache is valid.

        Cache is considered valid if:
        1. Output file exists
        2. Cache entry exists for this config
        3. All source DBs have same mtime and game_count as when cached
        4. Output file has same mtime as when cached
        """
        if force:
            return True

        # Check if output exists
        output = Path(output_path)
        if not output.exists():
            return True

        # Load cache entry
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
            min_quality=min_quality,
            require_completed=require_completed,
            encoder_version=encoder_version,
            include_heuristics=include_heuristics,
            full_heuristics=full_heuristics,
        )
        entry = self._load_cache_entry(cache_key)
        if entry is None:
            return True

        # Check output file hasn't been modified externally
        current_output_mtime = output.stat().st_mtime
        if abs(current_output_mtime - entry.output_mtime) > 1.0:
            return True

        # Check all source databases
        for db_path in db_paths:
            db_path_norm = os.path.normpath(os.path.abspath(db_path))
            cached_stats = entry.db_sources.get(db_path_norm)

            if cached_stats is None:
                # New database not in cache
                return True

            current_stats = _get_db_stats(db_path)

            if not current_stats.get("exists", False):
                # Source DB was removed - could skip, but safer to re-export
                continue

            # Check mtime (primary indicator of changes)
            if current_stats["mtime"] > cached_stats.get("mtime", 0):
                return True

            # Check game count (secondary check for content changes)
            cached_count = cached_stats.get("game_count", -1)
            current_count = current_stats.get("game_count", -1)
            if cached_count >= 0 and current_count >= 0 and current_count != cached_count:
                return True

        # Check if there are new DBs not in the cache
        cached_db_paths = set(entry.db_sources.keys())
        current_db_paths = {os.path.normpath(os.path.abspath(p)) for p in db_paths if Path(p).exists()}
        if current_db_paths - cached_db_paths:
            # New databases added
            return True

        # Cache is valid
        return False

    def record_export(
        self,
        db_paths: list[str],
        output_path: str,
        board_type: str,
        num_players: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
        samples_exported: int = 0,
        games_exported: int = 0,
        # Quality filtering parameters (December 28, 2025)
        min_quality: float | None = None,
        require_completed: bool | None = None,
        encoder_version: str | None = None,
        include_heuristics: bool | None = None,
        full_heuristics: bool | None = None,
        # January 2026: Delta detection - track exported game IDs
        track_game_ids: bool = True,
    ) -> None:
        """Record a completed export to the cache.

        Args:
            db_paths: List of database paths that were exported
            output_path: Path to the output NPZ file
            board_type: Board type
            num_players: Number of players
            samples_exported: Number of samples in the export
            games_exported: Number of games exported
            track_game_ids: If True, collect and store game IDs for delta detection
            Other args: Export configuration parameters
        """
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
            min_quality=min_quality,
            require_completed=require_completed,
            encoder_version=encoder_version,
            include_heuristics=include_heuristics,
            full_heuristics=full_heuristics,
        )

        # Collect source DB stats
        db_sources = {}
        for db_path in db_paths:
            if Path(db_path).exists():
                db_path_norm = os.path.normpath(os.path.abspath(db_path))
                db_sources[db_path_norm] = _get_db_stats(db_path)

        # Get output stats
        output = Path(output_path)
        output_size = output.stat().st_size if output.exists() else 0
        output_mtime = output.stat().st_mtime if output.exists() else 0

        # January 2026: Collect game IDs for delta detection
        exported_game_ids: dict[str, list[str]] | None = None
        max_game_ids: dict[str, int] | None = None

        if track_game_ids:
            exported_game_ids = {}
            max_game_ids = {}
            for db_path in db_paths:
                if Path(db_path).exists():
                    db_path_norm = os.path.normpath(os.path.abspath(db_path))
                    game_ids, max_id = _get_game_ids_from_db(
                        db_path,
                        board_type=board_type,
                        num_players=num_players,
                        require_completed=require_completed or False,
                    )
                    exported_game_ids[db_path_norm] = game_ids
                    max_game_ids[db_path_norm] = max_id

        entry = ExportCacheEntry(
            output_path=os.path.normpath(os.path.abspath(output_path)),
            board_type=board_type,
            num_players=num_players,
            db_sources=db_sources,
            export_timestamp=datetime.now().isoformat(),
            samples_exported=samples_exported,
            games_exported=games_exported,
            output_size=output_size,
            output_mtime=output_mtime,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
            exported_game_ids=exported_game_ids,
            max_game_ids=max_game_ids,
        )

        self._save_cache_entry(cache_key, entry)

    def invalidate(
        self,
        output_path: str,
        board_type: str,
        num_players: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
        # Quality filtering parameters (December 28, 2025)
        min_quality: float | None = None,
        require_completed: bool | None = None,
        encoder_version: str | None = None,
        include_heuristics: bool | None = None,
        full_heuristics: bool | None = None,
    ) -> bool:
        """Invalidate a cache entry. Returns True if entry was found and removed."""
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
            min_quality=min_quality,
            require_completed=require_completed,
            encoder_version=encoder_version,
            include_heuristics=include_heuristics,
            full_heuristics=full_heuristics,
        )
        cache_file = self.cache_dir / f"export_{cache_key}.json"

        if cache_file.exists():
            cache_file.unlink()
            return True
        return False

    def get_cache_info(
        self,
        output_path: str,
        board_type: str,
        num_players: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
        # Quality filtering parameters (December 28, 2025)
        min_quality: float | None = None,
        require_completed: bool | None = None,
        encoder_version: str | None = None,
        include_heuristics: bool | None = None,
        full_heuristics: bool | None = None,
    ) -> dict[str, Any] | None:
        """Get cache entry info for debugging/inspection."""
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
            min_quality=min_quality,
            require_completed=require_completed,
            encoder_version=encoder_version,
            include_heuristics=include_heuristics,
            full_heuristics=full_heuristics,
        )
        entry = self._load_cache_entry(cache_key)
        if entry:
            return entry.to_dict()
        return None

    def get_delta_info(
        self,
        db_paths: list[str],
        output_path: str,
        board_type: str,
        num_players: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
        require_completed: bool = False,
        # Quality filtering parameters
        min_quality: float | None = None,
        encoder_version: str | None = None,
        include_heuristics: bool | None = None,
        full_heuristics: bool | None = None,
    ) -> dict[str, Any]:
        """Get delta information for incremental export.

        January 2026: Core method for incremental NPZ export with delta detection.

        Returns:
            dict with:
                - needs_full_export: bool - True if full re-export needed
                - new_game_ids: dict[db_path, list[game_id]] - new games per DB
                - total_new_games: int - total count of new games
                - can_merge: bool - True if existing NPZ can be merged with delta
                - existing_samples: int - samples in existing NPZ (if can_merge)
                - reason: str - explanation of delta state
        """
        result: dict[str, Any] = {
            "needs_full_export": True,
            "new_game_ids": {},
            "total_new_games": 0,
            "can_merge": False,
            "existing_samples": 0,
            "reason": "Unknown",
        }

        # Check if output exists
        output = Path(output_path)
        if not output.exists():
            result["reason"] = "Output file does not exist"
            return result

        # Load cache entry
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
            min_quality=min_quality,
            require_completed=require_completed,
            encoder_version=encoder_version,
            include_heuristics=include_heuristics,
            full_heuristics=full_heuristics,
        )
        entry = self._load_cache_entry(cache_key)

        if entry is None:
            result["reason"] = "No cache entry found"
            return result

        # Check if we have exported game IDs tracking
        if entry.exported_game_ids is None:
            result["reason"] = "Cache entry missing game ID tracking (legacy cache)"
            return result

        # Check output file hasn't been modified externally
        current_output_mtime = output.stat().st_mtime
        if abs(current_output_mtime - entry.output_mtime) > 1.0:
            result["reason"] = "Output file modified externally"
            return result

        # Collect new game IDs per database
        new_game_ids: dict[str, list[str]] = {}
        total_new = 0

        for db_path in db_paths:
            if not Path(db_path).exists():
                continue

            db_path_norm = os.path.normpath(os.path.abspath(db_path))

            # Get previously exported IDs for this DB
            exported_ids_list = entry.exported_game_ids.get(db_path_norm, [])
            exported_ids = set(exported_ids_list)

            # Get new game IDs
            new_ids = _get_new_game_ids(
                db_path,
                exported_ids,
                board_type=board_type,
                num_players=num_players,
                require_completed=require_completed,
            )

            if new_ids:
                new_game_ids[db_path_norm] = new_ids
                total_new += len(new_ids)

        result["new_game_ids"] = new_game_ids
        result["total_new_games"] = total_new
        result["existing_samples"] = entry.samples_exported

        if total_new == 0:
            result["needs_full_export"] = False
            result["can_merge"] = False
            result["reason"] = "No new games - cache is up to date"
        else:
            result["needs_full_export"] = False
            result["can_merge"] = True
            result["reason"] = f"Found {total_new} new games for incremental export"

        return result

    def record_exported_games(
        self,
        db_path: str,
        game_ids: list[str],
        output_path: str,
        board_type: str,
        num_players: int,
        samples_added: int,
        history_length: int | None = None,
        feature_version: int | None = None,
        policy_encoding: str | None = None,
        # Quality filtering parameters
        min_quality: float | None = None,
        require_completed: bool | None = None,
        encoder_version: str | None = None,
        include_heuristics: bool | None = None,
        full_heuristics: bool | None = None,
    ) -> None:
        """Record incrementally exported games to the cache.

        January 2026: Update cache after incremental export without full re-export.

        Args:
            db_path: Database path that was exported
            game_ids: List of game IDs that were exported
            output_path: Path to the updated NPZ file
            board_type: Board type
            num_players: Number of players
            samples_added: Number of samples added in this increment
            Other args: Export configuration for cache key
        """
        cache_key = _get_cache_key(
            board_type,
            num_players,
            output_path,
            history_length=history_length,
            feature_version=feature_version,
            policy_encoding=policy_encoding,
            min_quality=min_quality,
            require_completed=require_completed,
            encoder_version=encoder_version,
            include_heuristics=include_heuristics,
            full_heuristics=full_heuristics,
        )

        entry = self._load_cache_entry(cache_key)
        if entry is None:
            # No existing entry - should have called record_export first
            return

        db_path_norm = os.path.normpath(os.path.abspath(db_path))

        # Update exported game IDs
        if entry.exported_game_ids is None:
            entry.exported_game_ids = {}

        existing_ids = set(entry.exported_game_ids.get(db_path_norm, []))
        existing_ids.update(game_ids)
        entry.exported_game_ids[db_path_norm] = list(existing_ids)

        # Update sample count
        entry.samples_exported += samples_added
        entry.games_exported += len(game_ids)

        # Update output file stats
        output = Path(output_path)
        if output.exists():
            entry.output_size = output.stat().st_size
            entry.output_mtime = output.stat().st_mtime

        # Update DB stats
        entry.db_sources[db_path_norm] = _get_db_stats(db_path)

        # Update timestamp
        entry.export_timestamp = datetime.now().isoformat()

        self._save_cache_entry(cache_key, entry)

    def cleanup_stale(self, max_age_days: int = 30) -> int:
        """Remove cache entries older than max_age_days. Returns count removed."""
        import time
        cutoff = time.time() - (max_age_days * 24 * 3600)
        removed = 0

        for cache_file in self.cache_dir.glob("export_*.json"):
            try:
                if cache_file.stat().st_mtime < cutoff:
                    cache_file.unlink()
                    removed += 1
            except (OSError, FileNotFoundError, PermissionError):
                pass

        return removed


def merge_npz_files(
    existing_path: str | Path,
    new_samples_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[int, int]:
    """Merge two NPZ training data files.

    January 2026: Utility for incremental NPZ export.

    Args:
        existing_path: Path to existing NPZ file with previous samples
        new_samples_path: Path to NPZ file with new samples to add
        output_path: Output path (defaults to existing_path, overwriting it)

    Returns:
        Tuple of (total_samples, new_samples_added)

    Raises:
        FileNotFoundError: If existing_path doesn't exist
        ValueError: If arrays are incompatible (different shapes, etc.)
    """
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)

    existing_path = Path(existing_path)
    new_samples_path = Path(new_samples_path)
    output_path = Path(output_path) if output_path else existing_path

    if not existing_path.exists():
        raise FileNotFoundError(f"Existing NPZ file not found: {existing_path}")
    if not new_samples_path.exists():
        raise FileNotFoundError(f"New samples NPZ file not found: {new_samples_path}")

    # Load both files
    with np.load(existing_path, allow_pickle=True) as existing:
        with np.load(new_samples_path, allow_pickle=True) as new:
            merged = {}
            new_sample_count = 0

            # Arrays that should be concatenated (have samples axis)
            concat_keys = [
                "features", "globals", "values",
                "policy_indices", "policy_values",
                "heuristics", "sample_weights", "quality_scores",
            ]

            for key in existing.files:
                existing_arr = existing[key]

                if key in new.files:
                    new_arr = new[key]

                    if key in concat_keys and len(existing_arr.shape) > 0:
                        # Concatenate along first axis
                        if key == "features":
                            # Validate shapes match (except first axis)
                            if existing_arr.shape[1:] != new_arr.shape[1:]:
                                raise ValueError(
                                    f"Feature shapes don't match: "
                                    f"{existing_arr.shape} vs {new_arr.shape}"
                                )
                            new_sample_count = len(new_arr)

                        merged[key] = np.concatenate([existing_arr, new_arr], axis=0)
                        logger.debug(
                            f"Merged {key}: {len(existing_arr)} + {len(new_arr)} = "
                            f"{len(merged[key])}"
                        )
                    else:
                        # Scalar or metadata - use existing
                        merged[key] = existing_arr
                else:
                    # Key only in existing - keep as is
                    merged[key] = existing_arr

            # Add any new keys not in existing
            for key in new.files:
                if key not in merged:
                    merged[key] = new[key]

            # Save merged result
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to temp file first, then rename (atomic)
            temp_path = output_path.with_suffix(".tmp.npz")
            np.savez_compressed(temp_path, **merged)
            temp_path.rename(output_path)

            total_samples = len(merged.get("values", []))
            logger.info(
                f"Merged NPZ: {total_samples} total samples "
                f"({new_sample_count} new added)"
            )

            return total_samples, new_sample_count


# Convenience functions for direct use
_default_cache = None

def get_export_cache() -> ExportCache:
    """Get the default export cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = ExportCache()
    return _default_cache


def needs_export(
    db_paths: list[str],
    output_path: str,
    board_type: str,
    num_players: int,
    history_length: int | None = None,
    feature_version: int | None = None,
    policy_encoding: str | None = None,
    force: bool = False,
    # Quality filtering parameters (December 28, 2025)
    min_quality: float | None = None,
    require_completed: bool | None = None,
    encoder_version: str | None = None,
    include_heuristics: bool | None = None,
    full_heuristics: bool | None = None,
) -> bool:
    """Check if export is needed using the default cache."""
    return get_export_cache().needs_export(
        db_paths,
        output_path,
        board_type,
        num_players,
        history_length=history_length,
        feature_version=feature_version,
        policy_encoding=policy_encoding,
        force=force,
        min_quality=min_quality,
        require_completed=require_completed,
        encoder_version=encoder_version,
        include_heuristics=include_heuristics,
        full_heuristics=full_heuristics,
    )


def record_export(
    db_paths: list[str],
    output_path: str,
    board_type: str,
    num_players: int,
    history_length: int | None = None,
    feature_version: int | None = None,
    policy_encoding: str | None = None,
    samples_exported: int = 0,
    games_exported: int = 0,
    # Quality filtering parameters (December 28, 2025)
    min_quality: float | None = None,
    require_completed: bool | None = None,
    encoder_version: str | None = None,
    include_heuristics: bool | None = None,
    full_heuristics: bool | None = None,
    # January 2026: Delta detection
    track_game_ids: bool = True,
) -> None:
    """Record an export using the default cache."""
    get_export_cache().record_export(
        db_paths,
        output_path,
        board_type,
        num_players,
        history_length=history_length,
        feature_version=feature_version,
        policy_encoding=policy_encoding,
        samples_exported=samples_exported,
        games_exported=games_exported,
        min_quality=min_quality,
        require_completed=require_completed,
        encoder_version=encoder_version,
        include_heuristics=include_heuristics,
        full_heuristics=full_heuristics,
        track_game_ids=track_game_ids,
    )


# January 2026: Delta detection convenience functions

def get_delta_info(
    db_paths: list[str],
    output_path: str,
    board_type: str,
    num_players: int,
    history_length: int | None = None,
    feature_version: int | None = None,
    policy_encoding: str | None = None,
    require_completed: bool = False,
    min_quality: float | None = None,
    encoder_version: str | None = None,
    include_heuristics: bool | None = None,
    full_heuristics: bool | None = None,
) -> dict[str, Any]:
    """Get delta information for incremental export using the default cache.

    Returns dict with:
        - needs_full_export: bool - True if full re-export needed
        - new_game_ids: dict[db_path, list[game_id]] - new games per DB
        - total_new_games: int - total count of new games
        - can_merge: bool - True if existing NPZ can be merged with delta
        - existing_samples: int - samples in existing NPZ (if can_merge)
        - reason: str - explanation of delta state
    """
    return get_export_cache().get_delta_info(
        db_paths,
        output_path,
        board_type,
        num_players,
        history_length=history_length,
        feature_version=feature_version,
        policy_encoding=policy_encoding,
        require_completed=require_completed,
        min_quality=min_quality,
        encoder_version=encoder_version,
        include_heuristics=include_heuristics,
        full_heuristics=full_heuristics,
    )


def record_exported_games(
    db_path: str,
    game_ids: list[str],
    output_path: str,
    board_type: str,
    num_players: int,
    samples_added: int,
    history_length: int | None = None,
    feature_version: int | None = None,
    policy_encoding: str | None = None,
    min_quality: float | None = None,
    require_completed: bool | None = None,
    encoder_version: str | None = None,
    include_heuristics: bool | None = None,
    full_heuristics: bool | None = None,
) -> None:
    """Record incrementally exported games using the default cache.

    Use this after an incremental export to update the cache with
    newly exported game IDs without requiring a full re-export.
    """
    get_export_cache().record_exported_games(
        db_path,
        game_ids,
        output_path,
        board_type,
        num_players,
        samples_added,
        history_length=history_length,
        feature_version=feature_version,
        policy_encoding=policy_encoding,
        min_quality=min_quality,
        require_completed=require_completed,
        encoder_version=encoder_version,
        include_heuristics=include_heuristics,
        full_heuristics=full_heuristics,
    )
