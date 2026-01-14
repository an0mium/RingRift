"""Unified Gauntlet Results Database (December 2025, Updated January 2026).

Provides a single source of truth for all gauntlet evaluation results.

Jan 14, 2026: Storage consolidated into unified_elo.db (was gauntlet_results.db).
This places gauntlet_matches and model_gauntlet_summary tables alongside
existing gauntlet_runs, gauntlet_results, and elo_ratings tables.

This module provides:
- Persistent SQLite storage for all gauntlet results
- Query API by model, board type, time range
- Integration with existing gauntlet implementations
- Results aggregation and statistics

Usage:
    from app.training.gauntlet_results_db import GauntletResultsDB, get_gauntlet_db

    # Get singleton instance
    db = get_gauntlet_db()

    # Store a result
    db.store_result(
        model_id="canonical_hex8_2p",
        board_type="hex8",
        num_players=2,
        opponent="heuristic",
        wins=15,
        losses=5,
        draws=0,
    )

    # Query results
    results = db.get_results_for_model("canonical_hex8_2p")
    stats = db.get_model_stats("canonical_hex8_2p", "hex8", 2)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default database path
# Jan 14, 2026: Consolidated into unified_elo.db for single source of truth
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "unified_elo.db"


@dataclass
class GauntletMatch:
    """Single gauntlet match result."""
    match_id: str
    model_id: str
    board_type: str
    num_players: int
    opponent_type: str
    wins: int
    losses: int
    draws: int
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        total = self.total_games
        return self.wins / total if total > 0 else 0.0


@dataclass
class ModelStats:
    """Aggregated statistics for a model."""
    model_id: str
    board_type: str
    num_players: int
    total_games: int
    total_wins: int
    total_losses: int
    total_draws: int
    opponents_tested: list[str]
    win_rates_by_opponent: dict[str, float]
    first_evaluated: float
    last_evaluated: float

    @property
    def overall_win_rate(self) -> float:
        return self.total_wins / self.total_games if self.total_games > 0 else 0.0


class GauntletResultsDB:
    """Unified database for gauntlet evaluation results.

    Thread-safe SQLite database for storing and querying gauntlet results.
    Provides a consistent API for all gauntlet implementations.
    """

    def __init__(self, db_path: str | Path | None = None):
        """Initialize the database.

        Args:
            db_path: Path to SQLite database. Uses default if not specified.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    @contextmanager
    def _get_conn(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
        yield self._local.conn

    def _init_schema(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS gauntlet_matches (
                    match_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    opponent_type TEXT NOT NULL,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    draws INTEGER NOT NULL DEFAULT 0,
                    timestamp REAL NOT NULL,
                    metadata_json TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                );

                CREATE INDEX IF NOT EXISTS idx_matches_model
                    ON gauntlet_matches(model_id);
                CREATE INDEX IF NOT EXISTS idx_matches_config
                    ON gauntlet_matches(board_type, num_players);
                CREATE INDEX IF NOT EXISTS idx_matches_timestamp
                    ON gauntlet_matches(timestamp);

                -- Aggregated view for quick stats
                CREATE TABLE IF NOT EXISTS model_gauntlet_summary (
                    model_id TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    opponent_type TEXT NOT NULL,
                    total_games INTEGER NOT NULL DEFAULT 0,
                    total_wins INTEGER NOT NULL DEFAULT 0,
                    total_losses INTEGER NOT NULL DEFAULT 0,
                    total_draws INTEGER NOT NULL DEFAULT 0,
                    last_updated REAL,
                    PRIMARY KEY (model_id, board_type, num_players, opponent_type)
                );

                CREATE INDEX IF NOT EXISTS idx_summary_model
                    ON model_gauntlet_summary(model_id);
            """)
            conn.commit()

    def store_result(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
        opponent: str,
        wins: int,
        losses: int,
        draws: int = 0,
        timestamp: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a gauntlet match result.

        Args:
            model_id: Model identifier
            board_type: Board type (e.g., "hex8", "square8")
            num_players: Number of players (2, 3, or 4)
            opponent: Opponent type (e.g., "random", "heuristic")
            wins: Number of wins
            losses: Number of losses
            draws: Number of draws
            timestamp: Evaluation timestamp (uses current time if not specified)
            metadata: Optional additional metadata

        Returns:
            Match ID
        """
        import uuid
        match_id = str(uuid.uuid4())
        ts = timestamp or time.time()
        metadata_json = json.dumps(metadata) if metadata else None

        with self._get_conn() as conn:
            # Insert match
            conn.execute("""
                INSERT INTO gauntlet_matches
                (match_id, model_id, board_type, num_players, opponent_type,
                 wins, losses, draws, timestamp, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (match_id, model_id, board_type, num_players, opponent,
                  wins, losses, draws, ts, metadata_json))

            # Update summary
            conn.execute("""
                INSERT INTO model_gauntlet_summary
                (model_id, board_type, num_players, opponent_type,
                 total_games, total_wins, total_losses, total_draws, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_id, board_type, num_players, opponent_type)
                DO UPDATE SET
                    total_games = total_games + excluded.total_games,
                    total_wins = total_wins + excluded.total_wins,
                    total_losses = total_losses + excluded.total_losses,
                    total_draws = total_draws + excluded.total_draws,
                    last_updated = excluded.last_updated
            """, (model_id, board_type, num_players, opponent,
                  wins + losses + draws, wins, losses, draws, ts))

            conn.commit()

        logger.debug(f"Stored gauntlet result: {model_id} vs {opponent} = {wins}/{losses}/{draws}")
        return match_id

    def store_batch(self, results: list[dict[str, Any]]) -> list[str]:
        """Store multiple gauntlet results in a batch.

        Args:
            results: List of result dicts with keys:
                model_id, board_type, num_players, opponent,
                wins, losses, draws, timestamp (optional), metadata (optional)

        Returns:
            List of match IDs
        """
        match_ids = []
        for result in results:
            match_id = self.store_result(
                model_id=result["model_id"],
                board_type=result["board_type"],
                num_players=result["num_players"],
                opponent=result["opponent"],
                wins=result["wins"],
                losses=result["losses"],
                draws=result.get("draws", 0),
                timestamp=result.get("timestamp"),
                metadata=result.get("metadata"),
            )
            match_ids.append(match_id)
        return match_ids

    def get_results_for_model(
        self,
        model_id: str,
        board_type: str | None = None,
        num_players: int | None = None,
        since: float | None = None,
        limit: int | None = None,
    ) -> list[GauntletMatch]:
        """Get gauntlet results for a model.

        Args:
            model_id: Model identifier
            board_type: Optional filter by board type
            num_players: Optional filter by player count
            since: Optional filter by timestamp (results after this time)
            limit: Maximum number of results to return

        Returns:
            List of GauntletMatch objects
        """
        query = "SELECT * FROM gauntlet_matches WHERE model_id = ?"
        params: list[Any] = [model_id]

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        results = []
        with self._get_conn() as conn:
            for row in conn.execute(query, params):
                metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                results.append(GauntletMatch(
                    match_id=row["match_id"],
                    model_id=row["model_id"],
                    board_type=row["board_type"],
                    num_players=row["num_players"],
                    opponent_type=row["opponent_type"],
                    wins=row["wins"],
                    losses=row["losses"],
                    draws=row["draws"],
                    timestamp=row["timestamp"],
                    metadata=metadata,
                ))

        return results

    def get_model_stats(
        self,
        model_id: str,
        board_type: str,
        num_players: int,
    ) -> ModelStats | None:
        """Get aggregated statistics for a model.

        Args:
            model_id: Model identifier
            board_type: Board type
            num_players: Number of players

        Returns:
            ModelStats or None if no results found
        """
        with self._get_conn() as conn:
            # Get summary stats
            rows = list(conn.execute("""
                SELECT opponent_type, total_games, total_wins, total_losses, total_draws, last_updated
                FROM model_gauntlet_summary
                WHERE model_id = ? AND board_type = ? AND num_players = ?
            """, (model_id, board_type, num_players)))

            if not rows:
                return None

            total_games = sum(r["total_games"] for r in rows)
            total_wins = sum(r["total_wins"] for r in rows)
            total_losses = sum(r["total_losses"] for r in rows)
            total_draws = sum(r["total_draws"] for r in rows)

            opponents = [r["opponent_type"] for r in rows]
            win_rates = {
                r["opponent_type"]: r["total_wins"] / r["total_games"] if r["total_games"] > 0 else 0.0
                for r in rows
            }

            last_updated = max(r["last_updated"] for r in rows if r["last_updated"])

            # Get first evaluation time
            first_row = conn.execute("""
                SELECT MIN(timestamp) as first_ts
                FROM gauntlet_matches
                WHERE model_id = ? AND board_type = ? AND num_players = ?
            """, (model_id, board_type, num_players)).fetchone()
            first_evaluated = first_row["first_ts"] if first_row else last_updated

            return ModelStats(
                model_id=model_id,
                board_type=board_type,
                num_players=num_players,
                total_games=total_games,
                total_wins=total_wins,
                total_losses=total_losses,
                total_draws=total_draws,
                opponents_tested=opponents,
                win_rates_by_opponent=win_rates,
                first_evaluated=first_evaluated,
                last_evaluated=last_updated,
            )

    def get_all_models(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
    ) -> list[str]:
        """Get all model IDs with gauntlet results.

        Args:
            board_type: Optional filter by board type
            num_players: Optional filter by player count

        Returns:
            List of model IDs
        """
        query = "SELECT DISTINCT model_id FROM model_gauntlet_summary"
        params: list[Any] = []

        conditions = []
        if board_type:
            conditions.append("board_type = ?")
            params.append(board_type)
        if num_players:
            conditions.append("num_players = ?")
            params.append(num_players)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._get_conn() as conn:
            return [row["model_id"] for row in conn.execute(query, params)]

    def get_leaderboard(
        self,
        board_type: str,
        num_players: int,
        opponent: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get leaderboard of models by win rate.

        Args:
            board_type: Board type
            num_players: Number of players
            opponent: Optional filter by specific opponent
            limit: Maximum number of entries

        Returns:
            List of dicts with model_id, win_rate, total_games
        """
        if opponent:
            query = """
                SELECT model_id,
                       total_wins * 1.0 / total_games as win_rate,
                       total_games
                FROM model_gauntlet_summary
                WHERE board_type = ? AND num_players = ? AND opponent_type = ?
                    AND total_games > 0
                ORDER BY win_rate DESC
                LIMIT ?
            """
            params = (board_type, num_players, opponent, limit)
        else:
            query = """
                SELECT model_id,
                       SUM(total_wins) * 1.0 / SUM(total_games) as win_rate,
                       SUM(total_games) as total_games
                FROM model_gauntlet_summary
                WHERE board_type = ? AND num_players = ?
                GROUP BY model_id
                HAVING total_games > 0
                ORDER BY win_rate DESC
                LIMIT ?
            """
            params = (board_type, num_players, limit)

        with self._get_conn() as conn:
            return [
                {
                    "model_id": row["model_id"],
                    "win_rate": row["win_rate"],
                    "total_games": row["total_games"],
                }
                for row in conn.execute(query, params)
            ]

    def close(self):
        """Close database connections."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


# Singleton instance
_instance: GauntletResultsDB | None = None
_lock = threading.Lock()


def get_gauntlet_db(db_path: str | Path | None = None) -> GauntletResultsDB:
    """Get the singleton GauntletResultsDB instance.

    Args:
        db_path: Optional custom database path (only used on first call)

    Returns:
        GauntletResultsDB instance
    """
    global _instance

    with _lock:
        if _instance is None:
            _instance = GauntletResultsDB(db_path)
        return _instance


def reset_gauntlet_db() -> None:
    """Reset the singleton instance (for testing)."""
    global _instance

    with _lock:
        if _instance is not None:
            _instance.close()
            _instance = None
