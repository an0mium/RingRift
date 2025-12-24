"""Unified Game Discovery - Find all game databases across cluster paths.

This module provides a centralized way to discover game databases regardless
of where they're stored. It handles all known storage patterns:

1. Central databases: data/games/selfplay.db, data/games/jsonl_aggregated.db
2. Per-config databases: data/games/{board_type}_{num_players}.db
3. Tournament databases: data/games/tournament_{board_type}_{num_players}.db
4. Canonical databases: data/selfplay/canonical_{board_type}_{num_players}.db
5. Unified selfplay: data/selfplay/unified_*/games.db
6. P2P selfplay: data/selfplay/p2p/{board_type}_{num_players}*/*/games.db
7. P2P hybrid: data/selfplay/p2p_hybrid/{board_type}_{num_players}/*/games.db
8. Harvested data: data/training/*/harvested_games.db

Usage:
    from app.utils.game_discovery import GameDiscovery

    # Find all databases
    discovery = GameDiscovery()
    all_dbs = discovery.find_all_databases()

    # Find databases for specific config
    hex8_2p_dbs = discovery.find_databases_for_config("hex8", 2)

    # Count games by config
    counts = discovery.count_games_by_config()
    print(counts)
    # {'square8_2p': 50000, 'hex8_2p': 34000, ...}

    # Get total games for a config
    total = discovery.get_total_games("hexagonal", 2)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Board type aliases - some databases use different names
BOARD_TYPE_ALIASES = {
    "hex8": ["hex8"],
    "hexagonal": ["hexagonal", "hex"],
    "square8": ["square8", "sq8"],
    "square19": ["square19", "sq19"],
}

# All board types in canonical form
ALL_BOARD_TYPES = ["square8", "square19", "hexagonal", "hex8"]

# All player counts
ALL_PLAYER_COUNTS = [2, 3, 4]


@dataclass
class DatabaseInfo:
    """Information about a discovered game database."""

    path: Path
    board_type: str | None = None
    num_players: int | None = None
    game_count: int = 0
    is_central: bool = False  # Central DBs contain multiple board types
    source_pattern: str = ""  # Which pattern matched this DB


@dataclass
class GameCounts:
    """Game counts aggregated by configuration."""

    by_config: dict[str, int] = field(default_factory=dict)
    by_board_type: dict[str, int] = field(default_factory=dict)
    by_num_players: dict[int, int] = field(default_factory=dict)
    total: int = 0
    databases_found: int = 0


class GameDiscovery:
    """Unified game database discovery across all storage patterns."""

    # Database path patterns (relative to ai-service root)
    # Order matters - more specific patterns first
    DB_PATTERNS = [
        # Central databases (contain all board types)
        ("data/games/selfplay.db", True),
        ("data/games/jsonl_aggregated.db", True),
        # Per-config databases
        ("data/games/{board_type}_{num_players}p.db", False),
        ("data/games/{board_type}_{num_players}.db", False),
        # Tournament databases
        ("data/games/tournament_{board_type}_{num_players}p.db", False),
        ("data/games/tournament_{board_type}_{num_players}.db", False),
        # Canonical selfplay databases
        ("data/selfplay/canonical_{board_type}_{num_players}p.db", False),
        ("data/selfplay/canonical_{board_type}.db", True),  # May contain multiple player counts
        # Unified selfplay (session-based)
        ("data/selfplay/unified_*/games.db", True),
        # P2P selfplay
        ("data/selfplay/p2p/{board_type}_{num_players}p*/*/games.db", False),
        ("data/selfplay/p2p/{board_type}_{num_players}*/*/games.db", False),
        # P2P hybrid selfplay
        ("data/selfplay/p2p_hybrid/{board_type}_{num_players}p/*/games.db", False),
        ("data/selfplay/p2p_hybrid/{board_type}_{num_players}/*/games.db", False),
        # Harvested training data
        ("data/training/*/harvested_games.db", True),
        # Legacy patterns
        ("data/games/hex8_*.db", False),
        ("data/games/canonical_*.db", True),
    ]

    def __init__(self, root_path: Path | str | None = None):
        """Initialize game discovery.

        Args:
            root_path: Root path to ai-service directory. If None, auto-detect.
        """
        if root_path is None:
            # Auto-detect based on common locations
            candidates = [
                Path(__file__).parent.parent.parent,  # From app/utils/
                Path.cwd(),
                Path.home() / "ringrift" / "ai-service",
                Path("/workspace/ringrift/ai-service"),
                Path("/lambda/nfs/RingRift/ai-service"),
            ]
            for candidate in candidates:
                if (candidate / "data").exists():
                    root_path = candidate
                    break
            else:
                root_path = Path.cwd()

        self.root_path = Path(root_path)
        self._cache: dict[str, list[DatabaseInfo]] = {}

    def find_all_databases(self, use_cache: bool = True) -> list[DatabaseInfo]:
        """Find all game databases using known patterns.

        Args:
            use_cache: If True, use cached results if available.

        Returns:
            List of DatabaseInfo objects for all found databases.
        """
        cache_key = "all"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        databases = []
        seen_paths: set[Path] = set()

        for pattern, is_central in self.DB_PATTERNS:
            for db_info in self._find_by_pattern(pattern, is_central):
                if db_info.path not in seen_paths:
                    seen_paths.add(db_info.path)
                    databases.append(db_info)

        self._cache[cache_key] = databases
        return databases

    def find_databases_for_config(
        self,
        board_type: str,
        num_players: int,
        include_central: bool = True,
        use_cache: bool = True,
    ) -> list[DatabaseInfo]:
        """Find databases containing games for a specific configuration.

        Args:
            board_type: Board type (square8, square19, hexagonal, hex8)
            num_players: Number of players (2, 3, 4)
            include_central: Include central databases that contain multiple configs
            use_cache: Use cached results if available

        Returns:
            List of DatabaseInfo objects matching the configuration.
        """
        cache_key = f"{board_type}_{num_players}p_central={include_central}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        all_dbs = self.find_all_databases(use_cache=use_cache)
        matching = []

        # Get aliases for this board type
        aliases = BOARD_TYPE_ALIASES.get(board_type, [board_type])

        for db_info in all_dbs:
            # Central databases need to be queried
            if db_info.is_central:
                if include_central:
                    # Check if this DB actually has games for this config
                    count = self._count_games_in_db(
                        db_info.path, board_type, num_players
                    )
                    if count > 0:
                        db_copy = DatabaseInfo(
                            path=db_info.path,
                            board_type=board_type,
                            num_players=num_players,
                            game_count=count,
                            is_central=True,
                            source_pattern=db_info.source_pattern,
                        )
                        matching.append(db_copy)
            else:
                # Check if DB matches this config
                if db_info.board_type in aliases and db_info.num_players == num_players:
                    matching.append(db_info)

        self._cache[cache_key] = matching
        return matching

    def count_games_by_config(self, use_cache: bool = True) -> GameCounts:
        """Count games for all board/player configurations.

        Returns:
            GameCounts with breakdown by config, board type, and player count.
        """
        counts = GameCounts()
        all_dbs = self.find_all_databases(use_cache=use_cache)
        counts.databases_found = len(all_dbs)

        # Query each database for game counts by config
        for db_info in all_dbs:
            if not db_info.path.exists():
                continue

            try:
                config_counts = self._get_config_counts(db_info.path)
                for config_key, count in config_counts.items():
                    if count > 0:
                        counts.by_config[config_key] = (
                            counts.by_config.get(config_key, 0) + count
                        )
                        counts.total += count

                        # Parse config key
                        parts = config_key.rsplit("_", 1)
                        if len(parts) == 2:
                            board_type = parts[0]
                            try:
                                num_players = int(parts[1].replace("p", ""))
                                counts.by_board_type[board_type] = (
                                    counts.by_board_type.get(board_type, 0) + count
                                )
                                counts.by_num_players[num_players] = (
                                    counts.by_num_players.get(num_players, 0) + count
                                )
                            except ValueError:
                                pass
            except Exception as e:
                logger.debug(f"Error querying {db_info.path}: {e}")

        return counts

    def get_total_games(
        self, board_type: str, num_players: int, use_cache: bool = True
    ) -> int:
        """Get total game count for a specific configuration.

        Args:
            board_type: Board type
            num_players: Number of players

        Returns:
            Total number of completed games for this configuration.
        """
        databases = self.find_databases_for_config(
            board_type, num_players, include_central=True, use_cache=use_cache
        )
        return sum(db.game_count for db in databases)

    def clear_cache(self):
        """Clear the discovery cache."""
        self._cache.clear()

    def _find_by_pattern(
        self, pattern: str, is_central: bool
    ) -> Iterator[DatabaseInfo]:
        """Find databases matching a pattern."""
        # Handle patterns with placeholders
        if "{board_type}" in pattern or "{num_players}" in pattern:
            for board_type in ALL_BOARD_TYPES:
                for num_players in ALL_PLAYER_COUNTS:
                    expanded = pattern.format(
                        board_type=board_type, num_players=num_players
                    )
                    yield from self._glob_pattern(
                        expanded, is_central, board_type, num_players, pattern
                    )
        else:
            yield from self._glob_pattern(pattern, is_central, None, None, pattern)

    def _glob_pattern(
        self,
        pattern: str,
        is_central: bool,
        board_type: str | None,
        num_players: int | None,
        source_pattern: str,
    ) -> Iterator[DatabaseInfo]:
        """Glob for databases matching a pattern."""
        full_pattern = self.root_path / pattern

        # Handle glob patterns
        if "*" in pattern:
            parent = full_pattern.parent
            while "*" in str(parent):
                parent = parent.parent
            if parent.exists():
                for match in self.root_path.glob(pattern):
                    if match.is_file() and match.stat().st_size > 0:
                        yield self._create_db_info(
                            match, is_central, board_type, num_players, source_pattern
                        )
        else:
            if full_pattern.exists() and full_pattern.stat().st_size > 0:
                yield self._create_db_info(
                    full_pattern, is_central, board_type, num_players, source_pattern
                )

    def _create_db_info(
        self,
        path: Path,
        is_central: bool,
        board_type: str | None,
        num_players: int | None,
        source_pattern: str,
    ) -> DatabaseInfo:
        """Create a DatabaseInfo object for a database."""
        # Try to infer board_type and num_players from path if not provided
        if board_type is None or num_players is None:
            inferred = self._infer_config_from_path(path)
            board_type = board_type or inferred[0]
            num_players = num_players or inferred[1]

        # Get game count
        game_count = 0
        if board_type and num_players and not is_central:
            game_count = self._count_games_in_db(path, board_type, num_players)
        elif is_central:
            # For central DBs, count all games
            game_count = self._count_all_games(path)

        return DatabaseInfo(
            path=path,
            board_type=board_type,
            num_players=num_players,
            game_count=game_count,
            is_central=is_central,
            source_pattern=source_pattern,
        )

    def _infer_config_from_path(self, path: Path) -> tuple[str | None, int | None]:
        """Try to infer board_type and num_players from the database path."""
        path_str = str(path).lower()

        # Check for board type
        board_type = None
        for bt in ALL_BOARD_TYPES:
            if bt in path_str:
                board_type = bt
                break

        # Check for player count
        num_players = None
        for np in ALL_PLAYER_COUNTS:
            if f"_{np}p" in path_str or f"_{np}/" in path_str or f"_{np}_" in path_str:
                num_players = np
                break

        return board_type, num_players

    def _count_games_in_db(
        self, db_path: Path, board_type: str, num_players: int
    ) -> int:
        """Count games for a specific config in a database."""
        if not db_path.exists():
            return 0

        # Get all aliases for this board type
        aliases = BOARD_TYPE_ALIASES.get(board_type, [board_type])
        placeholders = ",".join("?" * len(aliases))

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM games WHERE winner IS NOT NULL "
                f"AND board_type IN ({placeholders}) AND num_players = ?",
                (*aliases, num_players),
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.debug(f"Error counting games in {db_path}: {e}")
            return 0

    def _count_all_games(self, db_path: Path) -> int:
        """Count all completed games in a database."""
        if not db_path.exists():
            return 0

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
            cursor = conn.execute(
                "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL"
            )
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.debug(f"Error counting games in {db_path}: {e}")
            return 0

    def _get_config_counts(self, db_path: Path) -> dict[str, int]:
        """Get game counts broken down by board_type and num_players."""
        if not db_path.exists():
            return {}

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
            cursor = conn.execute(
                "SELECT board_type, num_players, COUNT(*) "
                "FROM games WHERE winner IS NOT NULL "
                "GROUP BY board_type, num_players"
            )
            results = {}
            for row in cursor.fetchall():
                board_type, num_players, count = row
                if board_type and num_players:
                    config_key = f"{board_type}_{num_players}p"
                    results[config_key] = count
            conn.close()
            return results
        except Exception as e:
            logger.debug(f"Error getting config counts from {db_path}: {e}")
            return {}


# Convenience functions for quick access
def find_all_game_databases(root_path: Path | str | None = None) -> list[DatabaseInfo]:
    """Find all game databases in the ai-service directory."""
    return GameDiscovery(root_path).find_all_databases()


def count_games_for_config(
    board_type: str, num_players: int, root_path: Path | str | None = None
) -> int:
    """Count total games for a specific board/player configuration."""
    return GameDiscovery(root_path).get_total_games(board_type, num_players)


def get_game_counts_summary(root_path: Path | str | None = None) -> dict[str, int]:
    """Get a summary of game counts by configuration."""
    return GameDiscovery(root_path).count_games_by_config().by_config


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Game Database Discovery")
    parser.add_argument("--root", type=str, help="Root path to ai-service")
    parser.add_argument(
        "--board-type", type=str, help="Filter by board type"
    )
    parser.add_argument(
        "--num-players", type=int, help="Filter by number of players"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    discovery = GameDiscovery(args.root)

    if args.board_type and args.num_players:
        dbs = discovery.find_databases_for_config(args.board_type, args.num_players)
        print(f"\nDatabases for {args.board_type} {args.num_players}p:")
        total = 0
        for db in dbs:
            print(f"  {db.path}: {db.game_count:,} games")
            total += db.game_count
        print(f"\nTotal: {total:,} games")
    else:
        counts = discovery.count_games_by_config()
        print(f"\nGame counts by configuration ({counts.databases_found} databases):")
        print("-" * 50)
        for config, count in sorted(counts.by_config.items()):
            print(f"  {config}: {count:,} games")
        print("-" * 50)
        print(f"Total: {counts.total:,} games")

        if args.verbose:
            print("\n\nAll databases found:")
            for db in discovery.find_all_databases():
                print(f"  {db.path}")
                print(f"    Pattern: {db.source_pattern}")
                print(f"    Games: {db.game_count:,}")
                print()
