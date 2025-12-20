#!/usr/bin/env python3
"""Database migrations for the Composite ELO System.

This module handles schema migrations to support tracking (NN, Algorithm)
combinations as distinct ELO-rated participants.

Migrations:
    001: Add composite participant columns to participants table
    002: Create algorithm_baselines table
    003: Create nn_performance_summary table
    004: Add indices for composite queries

Usage:
    # Run migrations on existing database
    python -m app.training.composite_elo_migration

    # Or from code:
    from app.training.composite_elo_migration import run_migrations
    run_migrations(db_path)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.utils.paths import UNIFIED_ELO_DB

logger = logging.getLogger(__name__)

# Migration version tracking
MIGRATION_VERSION = 4


@contextmanager
def get_connection(db_path: Path):
    """Get a database connection with proper settings."""
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_current_version(conn: sqlite3.Connection) -> int:
    """Get current migration version from database."""
    try:
        cursor = conn.execute("""
            SELECT MAX(version) FROM schema_migrations
        """)
        row = cursor.fetchone()
        return row[0] if row and row[0] else 0
    except sqlite3.OperationalError:
        # Table doesn't exist
        return 0


def record_migration(conn: sqlite3.Connection, version: int, description: str) -> None:
    """Record a completed migration."""
    conn.execute("""
        INSERT INTO schema_migrations (version, description, applied_at)
        VALUES (?, ?, ?)
    """, (version, description, time.time()))


def migration_001_composite_participant_columns(conn: sqlite3.Connection) -> None:
    """Add composite participant columns to participants table."""
    logger.info("Running migration 001: Add composite participant columns")

    # Check existing columns
    cursor = conn.execute("PRAGMA table_info(participants)")
    existing_cols = {row["name"] for row in cursor.fetchall()}

    # Add nn_model_id column
    if "nn_model_id" not in existing_cols:
        conn.execute("""
            ALTER TABLE participants ADD COLUMN nn_model_id TEXT
        """)
        logger.info("Added nn_model_id column")

    # Add nn_model_path column
    if "nn_model_path" not in existing_cols:
        conn.execute("""
            ALTER TABLE participants ADD COLUMN nn_model_path TEXT
        """)
        logger.info("Added nn_model_path column")

    # Add ai_algorithm column (distinct from ai_type for backwards compat)
    if "ai_algorithm" not in existing_cols:
        conn.execute("""
            ALTER TABLE participants ADD COLUMN ai_algorithm TEXT
        """)
        logger.info("Added ai_algorithm column")

    # Add algorithm_config column (JSON)
    if "algorithm_config" not in existing_cols:
        conn.execute("""
            ALTER TABLE participants ADD COLUMN algorithm_config TEXT
        """)
        logger.info("Added algorithm_config column")

    # Add is_composite column
    if "is_composite" not in existing_cols:
        conn.execute("""
            ALTER TABLE participants ADD COLUMN is_composite INTEGER DEFAULT 0
        """)
        logger.info("Added is_composite column")

    record_migration(conn, 1, "Add composite participant columns")


def migration_002_algorithm_baselines(conn: sqlite3.Connection) -> None:
    """Create algorithm_baselines table."""
    logger.info("Running migration 002: Create algorithm_baselines table")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS algorithm_baselines (
            ai_algorithm TEXT NOT NULL,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            baseline_elo REAL DEFAULT 1500.0,
            games_played INTEGER DEFAULT 0,
            last_updated REAL,
            metadata TEXT,
            PRIMARY KEY (ai_algorithm, board_type, num_players)
        )
    """)

    # Insert default baselines
    default_baselines = [
        ("random", "square8", 2, 400.0),
        ("heuristic", "square8", 2, 1000.0),
        ("mcts", "square8", 2, 1500.0),
        ("gumbel_mcts", "square8", 2, 1500.0),
        ("descent", "square8", 2, 1500.0),
        ("policy_only", "square8", 2, 1200.0),
        ("ebmo", "square8", 2, 1500.0),
        ("gmo", "square8", 2, 1500.0),
    ]

    for algo, board, players, elo in default_baselines:
        conn.execute("""
            INSERT OR IGNORE INTO algorithm_baselines
            (ai_algorithm, board_type, num_players, baseline_elo, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """, (algo, board, players, elo, time.time()))

    record_migration(conn, 2, "Create algorithm_baselines table")


def migration_003_nn_performance_summary(conn: sqlite3.Connection) -> None:
    """Create nn_performance_summary table."""
    logger.info("Running migration 003: Create nn_performance_summary table")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS nn_performance_summary (
            nn_model_id TEXT NOT NULL,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            best_algorithm TEXT,
            best_elo REAL,
            avg_elo REAL,
            algorithms_tested INTEGER DEFAULT 0,
            last_updated REAL,
            metadata TEXT,
            PRIMARY KEY (nn_model_id, board_type, num_players)
        )
    """)

    record_migration(conn, 3, "Create nn_performance_summary table")


def migration_004_composite_indices(conn: sqlite3.Connection) -> None:
    """Add indices for composite participant queries."""
    logger.info("Running migration 004: Add composite indices")

    # Index on participants table
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_participants_nn_model
        ON participants(nn_model_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_participants_ai_algorithm
        ON participants(ai_algorithm)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_participants_nn_algo
        ON participants(nn_model_id, ai_algorithm)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_participants_composite
        ON participants(is_composite)
    """)

    # Index on elo_ratings for composite queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_elo_rating_desc
        ON elo_ratings(board_type, num_players, rating DESC)
    """)

    record_migration(conn, 4, "Add composite indices")


def create_migration_table(conn: sqlite3.Connection) -> None:
    """Create the schema_migrations table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at REAL NOT NULL
        )
    """)


def run_migrations(db_path: Path | None = None) -> dict[str, Any]:
    """Run all pending migrations.

    Args:
        db_path: Path to database (defaults to UNIFIED_ELO_DB)

    Returns:
        Dict with migration results
    """
    db_path = db_path or UNIFIED_ELO_DB

    logger.info(f"Running composite ELO migrations on {db_path}")

    migrations = [
        (1, migration_001_composite_participant_columns),
        (2, migration_002_algorithm_baselines),
        (3, migration_003_nn_performance_summary),
        (4, migration_004_composite_indices),
    ]

    results = {
        "db_path": str(db_path),
        "migrations_run": [],
        "current_version": 0,
        "target_version": MIGRATION_VERSION,
    }

    with get_connection(db_path) as conn:
        # Ensure migration table exists
        create_migration_table(conn)

        current_version = get_current_version(conn)
        results["current_version"] = current_version

        if current_version >= MIGRATION_VERSION:
            logger.info(f"Database already at version {current_version}, no migrations needed")
            return results

        # Run pending migrations
        for version, migration_func in migrations:
            if version > current_version:
                try:
                    migration_func(conn)
                    results["migrations_run"].append(version)
                    logger.info(f"Completed migration {version}")
                except Exception as e:
                    logger.error(f"Migration {version} failed: {e}")
                    raise

        # Commit all migrations
        conn.commit()

        results["current_version"] = get_current_version(conn)

    logger.info(f"Migrations complete. Now at version {results['current_version']}")
    return results


def migrate_legacy_participants(db_path: Path | None = None) -> dict[str, Any]:
    """Migrate existing legacy participant IDs to composite format.

    This updates the participants table to populate the new columns
    for existing participants based on their legacy IDs.

    Args:
        db_path: Path to database (defaults to UNIFIED_ELO_DB)

    Returns:
        Dict with migration results
    """
    from app.training.composite_participant import (
        encode_config_hash,
        get_standard_config,
        is_composite_id,
    )

    db_path = db_path or UNIFIED_ELO_DB

    logger.info("Migrating legacy participants to composite format")

    results = {
        "total_participants": 0,
        "already_composite": 0,
        "migrated": 0,
        "skipped": 0,
    }

    with get_connection(db_path) as conn:
        # Get all participants
        cursor = conn.execute("""
            SELECT participant_id, ai_type, model_path, is_composite
            FROM participants
        """)

        participants = cursor.fetchall()
        results["total_participants"] = len(participants)

        for p in participants:
            pid = p["participant_id"]

            # Skip already composite participants
            if p["is_composite"]:
                results["already_composite"] += 1
                continue

            # Skip if already in composite format
            if is_composite_id(pid):
                # Just mark as composite
                conn.execute("""
                    UPDATE participants SET is_composite = 1 WHERE participant_id = ?
                """, (pid,))
                results["already_composite"] += 1
                continue

            # Determine AI algorithm from ai_type or default to mcts
            ai_type = p["ai_type"] or "neural_net"
            if ai_type == "neural_net":
                ai_algorithm = "mcts"  # Default for NN models
            else:
                ai_algorithm = ai_type

            # Get standard config
            config = get_standard_config(ai_algorithm)
            config_hash = encode_config_hash(config, ai_algorithm)

            # Update participant metadata
            conn.execute("""
                UPDATE participants
                SET nn_model_id = ?,
                    nn_model_path = ?,
                    ai_algorithm = ?,
                    algorithm_config = ?,
                    is_composite = 0
                WHERE participant_id = ?
            """, (
                pid,  # Use ID as nn_model_id for legacy
                p["model_path"],
                ai_algorithm,
                json.dumps(config),
                pid,
            ))

            results["migrated"] += 1
            logger.debug(f"Migrated legacy participant: {pid}")

        conn.commit()

    logger.info(f"Legacy migration complete: {results['migrated']} migrated, "
                f"{results['already_composite']} already composite")
    return results


def update_nn_performance_summaries(
    db_path: Path | None = None,
    board_type: str = "square8",
    num_players: int = 2,
) -> dict[str, Any]:
    """Update NN performance summary table from current ratings.

    Aggregates ratings across all algorithm variants for each NN.

    Args:
        db_path: Path to database
        board_type: Board type to update
        num_players: Number of players

    Returns:
        Dict with update results
    """
    from app.training.composite_participant import extract_nn_id, is_composite_id

    db_path = db_path or UNIFIED_ELO_DB

    results = {
        "nn_models_updated": 0,
        "board_type": board_type,
        "num_players": num_players,
    }

    with get_connection(db_path) as conn:
        # Get all ratings for this config
        cursor = conn.execute("""
            SELECT e.participant_id, e.rating, e.games_played,
                   p.nn_model_id, p.ai_algorithm
            FROM elo_ratings e
            JOIN participants p ON e.participant_id = p.participant_id
            WHERE e.board_type = ? AND e.num_players = ? AND e.games_played > 0
        """, (board_type, num_players))

        # Aggregate by NN model
        nn_stats: dict[str, dict[str, Any]] = {}

        for row in cursor.fetchall():
            pid = row["participant_id"]

            # Extract NN ID
            nn_id = row["nn_model_id"]
            if not nn_id:
                if is_composite_id(pid):
                    nn_id = extract_nn_id(pid)
                else:
                    nn_id = pid

            if not nn_id or nn_id == "none":
                continue

            if nn_id not in nn_stats:
                nn_stats[nn_id] = {
                    "ratings": [],
                    "algorithms": set(),
                    "best_elo": 0,
                    "best_algo": None,
                }

            rating = row["rating"]
            algo = row["ai_algorithm"] or "mcts"

            nn_stats[nn_id]["ratings"].append(rating)
            nn_stats[nn_id]["algorithms"].add(algo)

            if rating > nn_stats[nn_id]["best_elo"]:
                nn_stats[nn_id]["best_elo"] = rating
                nn_stats[nn_id]["best_algo"] = algo

        # Update summary table
        for nn_id, stats in nn_stats.items():
            ratings = stats["ratings"]
            avg_elo = sum(ratings) / len(ratings) if ratings else 0

            conn.execute("""
                INSERT OR REPLACE INTO nn_performance_summary
                (nn_model_id, board_type, num_players, best_algorithm,
                 best_elo, avg_elo, algorithms_tested, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                nn_id, board_type, num_players,
                stats["best_algo"], stats["best_elo"], avg_elo,
                len(stats["algorithms"]), time.time()
            ))

            results["nn_models_updated"] += 1

        conn.commit()

    logger.info(f"Updated {results['nn_models_updated']} NN performance summaries")
    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("Running Composite ELO System Migrations")
    print("=" * 50)

    # Run schema migrations
    results = run_migrations()
    print(f"\nSchema Migrations:")
    print(f"  Current version: {results['current_version']}")
    print(f"  Migrations run: {results['migrations_run']}")

    # Ask about legacy migration
    if "--migrate-legacy" in sys.argv or input("\nMigrate legacy participants? [y/N]: ").lower() == "y":
        legacy_results = migrate_legacy_participants()
        print(f"\nLegacy Migration:")
        print(f"  Total participants: {legacy_results['total_participants']}")
        print(f"  Already composite: {legacy_results['already_composite']}")
        print(f"  Migrated: {legacy_results['migrated']}")

    # Update summaries
    if "--update-summaries" in sys.argv or input("\nUpdate NN performance summaries? [y/N]: ").lower() == "y":
        for board in ["square8", "square19"]:
            for players in [2, 3]:
                summary_results = update_nn_performance_summaries(
                    board_type=board, num_players=players
                )
                print(f"  {board}/{players}p: {summary_results['nn_models_updated']} NNs updated")

    print("\nMigration complete!")
