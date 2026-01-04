"""Evaluation Status Tracker for OWC Model Backlog Automation.

Sprint 15 (January 2026): Tracks evaluation status for all discovered models
(local, OWC, S3, cluster) to enable automated backlog evaluation.

This module provides:
- Centralized tracking of model evaluation status
- Query APIs for finding unevaluated/stale models
- Coverage summary by configuration
- Priority scoring for evaluation queue

Usage:
    from app.training.evaluation_status import (
        get_evaluation_tracker,
        EvaluationStatusTracker,
        ModelEvaluationStatus,
    )

    tracker = get_evaluation_tracker()

    # Register a newly discovered model
    tracker.register_model(
        model_path="/path/to/model.pth",
        board_type="hex8",
        num_players=2,
        source="owc",
    )

    # Get unevaluated models for a config
    pending = tracker.get_unevaluated_models("hex8", 2, limit=10)

    # Record evaluation result
    tracker.record_evaluation_result(
        model_sha256="abc123...",
        board_type="hex8",
        num_players=2,
        harness_type="gumbel_mcts",
        elo_rating=1450.0,
        games_evaluated=50,
    )

    # Get coverage summary
    summary = tracker.get_coverage_summary()
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default database location (shares with unified_elo.db)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "unified_elo.db"

# Global singleton
_tracker_instance: EvaluationStatusTracker | None = None
_tracker_lock = threading.RLock()


class EvaluationStatus(Enum):
    """Status of model evaluation."""

    PENDING = "pending"  # Not yet queued for evaluation
    QUEUED = "queued"  # In evaluation queue
    RUNNING = "running"  # Evaluation in progress
    EVALUATED = "evaluated"  # Successfully evaluated with Elo rating
    FAILED = "failed"  # Evaluation failed (will retry)
    STALE = "stale"  # Evaluated but needs refresh (old rating)


class ModelSource(Enum):
    """Source of the model file."""

    LOCAL = "local"  # Local filesystem
    OWC = "owc"  # Open Weights Cloud drive
    S3 = "s3"  # AWS S3 bucket
    CLUSTER = "cluster"  # Other cluster nodes


@dataclass
class ModelEvaluationStatus:
    """Status record for a model's evaluation."""

    id: int
    model_sha256: str
    model_path: str
    board_type: str
    num_players: int
    harness_type: str
    status: EvaluationStatus
    elo_rating: float | None
    games_evaluated: int
    first_seen_at: float
    last_evaluated_at: float | None
    evaluation_error: str | None
    source: ModelSource
    priority: int

    @property
    def config_key(self) -> str:
        """Get the config key for this model."""
        return f"{self.board_type}_{self.num_players}p"

    @property
    def is_evaluated(self) -> bool:
        """Check if model has been evaluated."""
        return self.status == EvaluationStatus.EVALUATED

    @property
    def needs_evaluation(self) -> bool:
        """Check if model needs (re-)evaluation."""
        return self.status in (
            EvaluationStatus.PENDING,
            EvaluationStatus.STALE,
            EvaluationStatus.FAILED,
        )

    @property
    def age_days(self) -> float:
        """Days since model was first seen."""
        return (time.time() - self.first_seen_at) / 86400.0

    @property
    def staleness_days(self) -> float | None:
        """Days since last evaluation, or None if never evaluated."""
        if self.last_evaluated_at is None:
            return None
        return (time.time() - self.last_evaluated_at) / 86400.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_sha256": self.model_sha256,
            "model_path": self.model_path,
            "board_type": self.board_type,
            "num_players": self.num_players,
            "harness_type": self.harness_type,
            "status": self.status.value,
            "elo_rating": self.elo_rating,
            "games_evaluated": self.games_evaluated,
            "first_seen_at": self.first_seen_at,
            "last_evaluated_at": self.last_evaluated_at,
            "evaluation_error": self.evaluation_error,
            "source": self.source.value,
            "priority": self.priority,
            "config_key": self.config_key,
            "age_days": self.age_days,
            "staleness_days": self.staleness_days,
        }


@dataclass
class ConfigCoverage:
    """Evaluation coverage summary for a configuration."""

    board_type: str
    num_players: int
    total_models: int
    evaluated_models: int
    pending_models: int
    queued_models: int
    running_models: int
    failed_models: int
    stale_models: int
    avg_elo: float | None
    min_elo: float | None
    max_elo: float | None
    total_games: int

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"

    @property
    def coverage_pct(self) -> float:
        """Percentage of models with Elo ratings."""
        if self.total_models == 0:
            return 100.0
        return 100.0 * self.evaluated_models / self.total_models

    @property
    def is_well_covered(self) -> bool:
        """Check if config has good evaluation coverage (>80%)."""
        return self.coverage_pct >= 80.0


@dataclass
class CoverageSummary:
    """Overall evaluation coverage summary."""

    total_models: int
    evaluated_models: int
    pending_models: int
    by_config: dict[str, ConfigCoverage] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)

    @property
    def overall_coverage_pct(self) -> float:
        """Overall evaluation coverage percentage."""
        if self.total_models == 0:
            return 100.0
        return 100.0 * self.evaluated_models / self.total_models


class EvaluationStatusTracker:
    """Tracks evaluation status for all discovered models.

    Uses the unified_elo.db database to store model evaluation status,
    enabling cluster-wide coordination of the backlog evaluation process.

    Features:
    - Multi-harness support (separate Elo per harness type)
    - Priority scoring for evaluation queue
    - Staleness detection and auto-refresh
    - Source tracking (local, OWC, S3, cluster)
    """

    # Staleness threshold in days (models evaluated longer ago are marked stale)
    STALENESS_THRESHOLD_DAYS = 7.0

    # Priority score components (lower = higher priority)
    PRIORITY_CANONICAL = -50  # Boost for canonical_*.pth models
    PRIORITY_BEST = -30  # Boost for best_*.pth models
    PRIORITY_RECENT = -10  # Boost for models < 7 days old
    PRIORITY_LOCAL = -10  # Boost for local models over remote
    PRIORITY_BASE = 100  # Default priority

    def __init__(self, db_path: Path | None = None):
        """Initialize tracker with database path."""
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._ensure_schema()
        logger.info(f"EvaluationStatusTracker initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA busy_timeout=10000")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _ensure_schema(self) -> None:
        """Ensure the model_evaluation_status table exists."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='model_evaluation_status'"
        )
        if cursor.fetchone() is None:
            # Table doesn't exist - create it
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_evaluation_status (
                    id INTEGER PRIMARY KEY,
                    model_sha256 TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    board_type TEXT NOT NULL,
                    num_players INTEGER NOT NULL,
                    harness_type TEXT DEFAULT 'default',
                    status TEXT DEFAULT 'pending',
                    elo_rating REAL,
                    games_evaluated INTEGER DEFAULT 0,
                    first_seen_at REAL NOT NULL,
                    last_evaluated_at REAL,
                    evaluation_error TEXT,
                    source TEXT DEFAULT 'local',
                    priority INTEGER DEFAULT 100,
                    UNIQUE(model_sha256, board_type, num_players, harness_type)
                );

                CREATE INDEX IF NOT EXISTS idx_model_eval_status_pending
                    ON model_evaluation_status(status, priority) WHERE status = 'pending';

                CREATE INDEX IF NOT EXISTS idx_model_eval_status_config
                    ON model_evaluation_status(board_type, num_players, status);

                CREATE INDEX IF NOT EXISTS idx_model_eval_status_source
                    ON model_evaluation_status(source, status);

                CREATE INDEX IF NOT EXISTS idx_model_eval_status_stale
                    ON model_evaluation_status(last_evaluated_at) WHERE status = 'evaluated';
            """
            )
            conn.commit()
            logger.info("Created model_evaluation_status table")

    def close(self) -> None:
        """Close thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    # =========================================================================
    # Model Registration
    # =========================================================================

    @staticmethod
    def compute_model_hash(model_path: str | Path) -> str:
        """Compute SHA256 hash of model file for deduplication."""
        path = Path(model_path)
        if not path.exists():
            # For remote models, use path-based hash
            return hashlib.sha256(str(model_path).encode()).hexdigest()

        # Compute actual content hash
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _compute_priority(
        self,
        model_path: str,
        source: str,
        board_type: str,
        num_players: int,
    ) -> int:
        """Compute priority score for a model (lower = higher priority)."""
        priority = self.PRIORITY_BASE
        path_lower = model_path.lower()

        # Boost for canonical models
        if "canonical" in path_lower:
            priority += self.PRIORITY_CANONICAL
        elif "best" in path_lower:
            priority += self.PRIORITY_BEST

        # Boost for local models
        if source == ModelSource.LOCAL.value:
            priority += self.PRIORITY_LOCAL

        # Additional boost for underserved configs
        try:
            from app.config.thresholds import get_config_staleness_tier

            tier = get_config_staleness_tier(board_type, num_players)
            if tier == "ULTRA":
                priority -= 50
            elif tier == "EMERGENCY":
                priority -= 30
            elif tier == "CRITICAL":
                priority -= 20
        except ImportError:
            pass

        return max(0, min(200, priority))  # Clamp to 0-200

    def register_model(
        self,
        model_path: str,
        board_type: str,
        num_players: int,
        source: str = "local",
        harness_type: str = "default",
        model_sha256: str | None = None,
    ) -> int | None:
        """Register a model for evaluation tracking.

        Args:
            model_path: Path to the model file
            board_type: Board type (hex8, square8, etc.)
            num_players: Number of players (2, 3, 4)
            source: Model source (local, owc, s3, cluster)
            harness_type: Evaluation harness type
            model_sha256: Pre-computed SHA256 hash (computed if not provided)

        Returns:
            Row ID if inserted, None if model already exists
        """
        if model_sha256 is None:
            model_sha256 = self.compute_model_hash(model_path)

        priority = self._compute_priority(model_path, source, board_type, num_players)
        now = time.time()

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO model_evaluation_status
                (model_sha256, model_path, board_type, num_players, harness_type,
                 status, first_seen_at, source, priority)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(model_sha256, board_type, num_players, harness_type)
                DO UPDATE SET
                    model_path = CASE
                        WHEN excluded.source = 'local' AND model_evaluation_status.source != 'local'
                        THEN excluded.model_path
                        ELSE model_evaluation_status.model_path
                    END,
                    priority = MIN(model_evaluation_status.priority, excluded.priority)
                """,
                (
                    model_sha256,
                    model_path,
                    board_type,
                    num_players,
                    harness_type,
                    now,
                    source,
                    priority,
                ),
            )
            conn.commit()

            if cursor.rowcount > 0:
                logger.debug(
                    f"Registered model for evaluation: {model_path} "
                    f"({board_type}_{num_players}p, source={source})"
                )
                return cursor.lastrowid
            return None

        except sqlite3.Error as e:
            logger.error(f"Failed to register model {model_path}: {e}")
            return None

    def register_models_batch(
        self,
        models: list[dict[str, Any]],
    ) -> int:
        """Register multiple models in a single transaction.

        Args:
            models: List of dicts with keys:
                - model_path, board_type, num_players
                - Optional: source, harness_type, model_sha256

        Returns:
            Number of models registered
        """
        if not models:
            return 0

        conn = self._get_connection()
        registered = 0
        now = time.time()

        try:
            conn.execute("BEGIN IMMEDIATE")

            for model in models:
                model_path = model["model_path"]
                board_type = model["board_type"]
                num_players = model["num_players"]
                source = model.get("source", "local")
                harness_type = model.get("harness_type", "default")
                model_sha256 = model.get("model_sha256") or self.compute_model_hash(
                    model_path
                )
                priority = self._compute_priority(
                    model_path, source, board_type, num_players
                )

                cursor = conn.execute(
                    """
                    INSERT INTO model_evaluation_status
                    (model_sha256, model_path, board_type, num_players, harness_type,
                     status, first_seen_at, source, priority)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                    ON CONFLICT(model_sha256, board_type, num_players, harness_type)
                    DO NOTHING
                    """,
                    (
                        model_sha256,
                        model_path,
                        board_type,
                        num_players,
                        harness_type,
                        now,
                        source,
                        priority,
                    ),
                )
                if cursor.rowcount > 0:
                    registered += 1

            conn.execute("COMMIT")
            logger.info(f"Registered {registered} models for evaluation")
            return registered

        except sqlite3.Error as e:
            conn.execute("ROLLBACK")
            logger.error(f"Failed to register models batch: {e}")
            return 0

    # =========================================================================
    # Query APIs
    # =========================================================================

    def get_model_status(
        self,
        model_sha256: str,
        board_type: str,
        num_players: int,
        harness_type: str = "default",
    ) -> ModelEvaluationStatus | None:
        """Get evaluation status for a specific model."""
        conn = self._get_connection()
        row = conn.execute(
            """
            SELECT * FROM model_evaluation_status
            WHERE model_sha256 = ? AND board_type = ?
              AND num_players = ? AND harness_type = ?
            """,
            (model_sha256, board_type, num_players, harness_type),
        ).fetchone()

        if row:
            return self._row_to_status(row)
        return None

    def get_unevaluated_models(
        self,
        board_type: str | None = None,
        num_players: int | None = None,
        limit: int = 50,
        source: str | None = None,
    ) -> list[ModelEvaluationStatus]:
        """Get models pending evaluation, ordered by priority.

        Args:
            board_type: Filter by board type (optional)
            num_players: Filter by player count (optional)
            limit: Maximum models to return
            source: Filter by source (optional)

        Returns:
            List of ModelEvaluationStatus ordered by priority (lowest first)
        """
        conn = self._get_connection()

        query = """
            SELECT * FROM model_evaluation_status
            WHERE status IN ('pending', 'stale', 'failed')
        """
        params: list[Any] = []

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)
        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY priority ASC, first_seen_at ASC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_status(row) for row in rows]

    def get_stale_evaluations(
        self,
        max_age_days: float | None = None,
        limit: int = 50,
    ) -> list[ModelEvaluationStatus]:
        """Get models with stale evaluations that need refresh.

        Args:
            max_age_days: Age threshold in days (default: STALENESS_THRESHOLD_DAYS)
            limit: Maximum models to return

        Returns:
            List of stale ModelEvaluationStatus
        """
        if max_age_days is None:
            max_age_days = self.STALENESS_THRESHOLD_DAYS

        threshold_time = time.time() - (max_age_days * 86400.0)
        conn = self._get_connection()

        rows = conn.execute(
            """
            SELECT * FROM model_evaluation_status
            WHERE status = 'evaluated'
              AND last_evaluated_at < ?
            ORDER BY last_evaluated_at ASC
            LIMIT ?
            """,
            (threshold_time, limit),
        ).fetchall()

        return [self._row_to_status(row) for row in rows]

    def get_models_by_status(
        self,
        status: EvaluationStatus,
        board_type: str | None = None,
        num_players: int | None = None,
        limit: int = 100,
    ) -> list[ModelEvaluationStatus]:
        """Get models with a specific status."""
        conn = self._get_connection()

        query = "SELECT * FROM model_evaluation_status WHERE status = ?"
        params: list[Any] = [status.value]

        if board_type:
            query += " AND board_type = ?"
            params.append(board_type)
        if num_players:
            query += " AND num_players = ?"
            params.append(num_players)

        query += " ORDER BY priority ASC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_status(row) for row in rows]

    # =========================================================================
    # Status Updates
    # =========================================================================

    def update_status(
        self,
        model_sha256: str,
        board_type: str,
        num_players: int,
        harness_type: str,
        new_status: EvaluationStatus,
        error_message: str | None = None,
    ) -> bool:
        """Update the status of a model evaluation.

        Args:
            model_sha256: Model content hash
            board_type: Board type
            num_players: Player count
            harness_type: Harness type
            new_status: New status to set
            error_message: Error message (for FAILED status)

        Returns:
            True if updated, False if not found
        """
        conn = self._get_connection()

        cursor = conn.execute(
            """
            UPDATE model_evaluation_status
            SET status = ?, evaluation_error = ?
            WHERE model_sha256 = ? AND board_type = ?
              AND num_players = ? AND harness_type = ?
            """,
            (
                new_status.value,
                error_message,
                model_sha256,
                board_type,
                num_players,
                harness_type,
            ),
        )
        conn.commit()

        if cursor.rowcount > 0:
            logger.debug(
                f"Updated model status to {new_status.value}: "
                f"{board_type}_{num_players}p/{harness_type}"
            )
            return True
        return False

    def record_evaluation_result(
        self,
        model_sha256: str,
        board_type: str,
        num_players: int,
        harness_type: str,
        elo_rating: float,
        games_evaluated: int,
    ) -> bool:
        """Record a successful evaluation result.

        Args:
            model_sha256: Model content hash
            board_type: Board type
            num_players: Player count
            harness_type: Harness type used for evaluation
            elo_rating: Computed Elo rating
            games_evaluated: Number of games played

        Returns:
            True if recorded, False if model not found
        """
        conn = self._get_connection()
        now = time.time()

        cursor = conn.execute(
            """
            UPDATE model_evaluation_status
            SET status = 'evaluated',
                elo_rating = ?,
                games_evaluated = games_evaluated + ?,
                last_evaluated_at = ?,
                evaluation_error = NULL
            WHERE model_sha256 = ? AND board_type = ?
              AND num_players = ? AND harness_type = ?
            """,
            (
                elo_rating,
                games_evaluated,
                now,
                model_sha256,
                board_type,
                num_players,
                harness_type,
            ),
        )
        conn.commit()

        if cursor.rowcount > 0:
            logger.info(
                f"Recorded evaluation: {board_type}_{num_players}p/{harness_type} "
                f"Elo={elo_rating:.0f} ({games_evaluated} games)"
            )
            return True
        return False

    def mark_stale_evaluations(self, max_age_days: float | None = None) -> int:
        """Mark evaluations older than threshold as stale.

        Args:
            max_age_days: Age threshold in days

        Returns:
            Number of evaluations marked stale
        """
        if max_age_days is None:
            max_age_days = self.STALENESS_THRESHOLD_DAYS

        threshold_time = time.time() - (max_age_days * 86400.0)
        conn = self._get_connection()

        cursor = conn.execute(
            """
            UPDATE model_evaluation_status
            SET status = 'stale'
            WHERE status = 'evaluated'
              AND last_evaluated_at < ?
            """,
            (threshold_time,),
        )
        conn.commit()

        if cursor.rowcount > 0:
            logger.info(f"Marked {cursor.rowcount} evaluations as stale")
        return cursor.rowcount

    # =========================================================================
    # Coverage Summary
    # =========================================================================

    def get_coverage_summary(self) -> CoverageSummary:
        """Get overall evaluation coverage summary."""
        conn = self._get_connection()

        # Overall counts
        overall = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'evaluated' THEN 1 ELSE 0 END) as evaluated,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
            FROM model_evaluation_status
            """
        ).fetchone()

        # By config
        by_config_rows = conn.execute(
            """
            SELECT
                board_type,
                num_players,
                COUNT(*) as total,
                SUM(CASE WHEN status = 'evaluated' THEN 1 ELSE 0 END) as evaluated,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) as queued,
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'stale' THEN 1 ELSE 0 END) as stale,
                AVG(CASE WHEN status = 'evaluated' THEN elo_rating ELSE NULL END) as avg_elo,
                MIN(CASE WHEN status = 'evaluated' THEN elo_rating ELSE NULL END) as min_elo,
                MAX(CASE WHEN status = 'evaluated' THEN elo_rating ELSE NULL END) as max_elo,
                SUM(games_evaluated) as total_games
            FROM model_evaluation_status
            GROUP BY board_type, num_players
            """
        ).fetchall()

        by_config: dict[str, ConfigCoverage] = {}
        for row in by_config_rows:
            config_key = f"{row['board_type']}_{row['num_players']}p"
            by_config[config_key] = ConfigCoverage(
                board_type=row["board_type"],
                num_players=row["num_players"],
                total_models=row["total"],
                evaluated_models=row["evaluated"],
                pending_models=row["pending"],
                queued_models=row["queued"],
                running_models=row["running"],
                failed_models=row["failed"],
                stale_models=row["stale"],
                avg_elo=row["avg_elo"],
                min_elo=row["min_elo"],
                max_elo=row["max_elo"],
                total_games=row["total_games"] or 0,
            )

        # By source
        by_source_rows = conn.execute(
            """
            SELECT source, COUNT(*) as count
            FROM model_evaluation_status
            GROUP BY source
            """
        ).fetchall()
        by_source = {row["source"]: row["count"] for row in by_source_rows}

        return CoverageSummary(
            total_models=overall["total"] or 0,
            evaluated_models=overall["evaluated"] or 0,
            pending_models=overall["pending"] or 0,
            by_config=by_config,
            by_source=by_source,
        )

    def get_config_coverage(
        self, board_type: str, num_players: int
    ) -> ConfigCoverage | None:
        """Get coverage for a specific configuration."""
        summary = self.get_coverage_summary()
        config_key = f"{board_type}_{num_players}p"
        return summary.by_config.get(config_key)

    # =========================================================================
    # Utilities
    # =========================================================================

    def _row_to_status(self, row: sqlite3.Row) -> ModelEvaluationStatus:
        """Convert database row to ModelEvaluationStatus."""
        return ModelEvaluationStatus(
            id=row["id"],
            model_sha256=row["model_sha256"],
            model_path=row["model_path"],
            board_type=row["board_type"],
            num_players=row["num_players"],
            harness_type=row["harness_type"],
            status=EvaluationStatus(row["status"]),
            elo_rating=row["elo_rating"],
            games_evaluated=row["games_evaluated"],
            first_seen_at=row["first_seen_at"],
            last_evaluated_at=row["last_evaluated_at"],
            evaluation_error=row["evaluation_error"],
            source=ModelSource(row["source"]),
            priority=row["priority"],
        )

    def backfill_from_elo_ratings(self) -> int:
        """Backfill evaluation status from existing elo_ratings table.

        Finds models in elo_ratings that don't have evaluation status records
        and creates them as 'evaluated'.

        Returns:
            Number of records backfilled
        """
        conn = self._get_connection()

        # Check if elo_ratings table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='elo_ratings'"
        )
        if cursor.fetchone() is None:
            return 0

        now = time.time()

        # Find elo_ratings entries without corresponding evaluation_status
        cursor = conn.execute(
            """
            INSERT INTO model_evaluation_status
            (model_sha256, model_path, board_type, num_players, harness_type,
             status, elo_rating, games_played, first_seen_at, last_evaluated_at,
             source, priority)
            SELECT
                -- Use participant_id as hash placeholder (will be updated on re-eval)
                participant_id,
                COALESCE(p.model_path, participant_id),
                e.board_type,
                e.num_players,
                'default',
                'evaluated',
                e.rating,
                e.games_played,
                COALESCE(e.last_update, ?),
                COALESCE(e.last_update, ?),
                'local',
                100
            FROM elo_ratings e
            LEFT JOIN participants p ON e.participant_id = p.participant_id
            WHERE NOT EXISTS (
                SELECT 1 FROM model_evaluation_status m
                WHERE m.model_sha256 = e.participant_id
                  AND m.board_type = e.board_type
                  AND m.num_players = e.num_players
            )
            AND e.games_played > 0
            """,
            (now, now),
        )
        conn.commit()

        backfilled = cursor.rowcount
        if backfilled > 0:
            logger.info(f"Backfilled {backfilled} evaluation status records from elo_ratings")
        return backfilled


# =============================================================================
# Singleton Access
# =============================================================================


def get_evaluation_tracker(db_path: Path | None = None) -> EvaluationStatusTracker:
    """Get or create the singleton EvaluationStatusTracker instance.

    Args:
        db_path: Optional custom database path. If provided on first call,
                 will be used for the singleton.

    Returns:
        The singleton EvaluationStatusTracker instance.
    """
    global _tracker_instance

    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = EvaluationStatusTracker(db_path)
        return _tracker_instance


def reset_evaluation_tracker() -> None:
    """Reset the singleton (for testing)."""
    global _tracker_instance

    with _tracker_lock:
        if _tracker_instance is not None:
            _tracker_instance.close()
            _tracker_instance = None
