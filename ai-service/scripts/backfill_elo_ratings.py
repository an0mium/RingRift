#!/usr/bin/env python3
"""Backfill elo_ratings from existing match_history records.

One-time script to fix missing Elo data for configs that were saved
before Elo recording was fully implemented (Jan 11-12, 2026).

This script processes match_history entries that have elo_before/elo_after
data but weren't written to the elo_ratings table.

Usage:
    cd ai-service
    python scripts/backfill_elo_ratings.py

    # Dry run (preview without writing)
    python scripts/backfill_elo_ratings.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

# Add ai-service to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tournament.elo import EloCalculator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "unified_elo.db"


def get_missing_configs(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Find configs that have match_history but no elo_ratings."""
    cursor = conn.cursor()

    # Get configs in match_history
    cursor.execute("""
        SELECT DISTINCT board_type, num_players
        FROM match_history
        ORDER BY board_type, num_players
    """)
    match_configs = set(cursor.fetchall())

    # Get configs in elo_ratings
    cursor.execute("""
        SELECT DISTINCT board_type, num_players
        FROM elo_ratings
        ORDER BY board_type, num_players
    """)
    elo_configs = set(cursor.fetchall())

    # Also check for configs where match_history has more participants than elo_ratings
    missing = []
    for board_type, num_players in match_configs:
        cursor.execute("""
            SELECT COUNT(DISTINCT participant_id)
            FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
        """, (board_type, num_players))
        elo_count = cursor.fetchone()[0]

        # Get unique participants from match_history
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT DISTINCT winner_id as pid FROM match_history
                WHERE board_type = ? AND num_players = ? AND winner_id IS NOT NULL
                UNION
                SELECT DISTINCT json_each.value as pid FROM match_history, json_each(participant_ids)
                WHERE board_type = ? AND num_players = ? AND participant_ids IS NOT NULL
            )
        """, (board_type, num_players, board_type, num_players))
        match_count = cursor.fetchone()[0]

        if match_count > elo_count:
            missing.append((board_type, num_players))

    return missing


def backfill_config(
    conn: sqlite3.Connection,
    board_type: str,
    num_players: int,
    dry_run: bool = False,
) -> dict:
    """Process match_history records into elo_ratings for a config."""
    config_key = f"{board_type}_{num_players}p"
    cursor = conn.cursor()

    # Get all matches for this config, ordered by timestamp
    cursor.execute("""
        SELECT
            id, participant_ids, rankings, winner_id, game_id,
            tournament_id, game_length, duration_sec, timestamp,
            elo_before, elo_after
        FROM match_history
        WHERE board_type = ? AND num_players = ?
        ORDER BY timestamp ASC
    """, (board_type, num_players))

    matches = cursor.fetchall()
    logger.info(f"{config_key}: Found {len(matches)} matches to process")

    if not matches:
        return {"config": config_key, "processed": 0, "skipped": 0}

    # Track participants we've seen
    participants_updated = set()
    processed = 0
    skipped = 0

    for row in matches:
        match_id, participant_ids_json, rankings_json, winner_id, game_id, \
            tournament_id, game_length, duration_sec, timestamp, \
            elo_before_json, elo_after_json = row

        # Parse participant IDs
        if participant_ids_json:
            participant_ids = json.loads(participant_ids_json)
        elif winner_id:
            # Fallback: reconstruct from winner_id for 2p games
            participant_ids = [winner_id, "heuristic"]  # Assume vs heuristic
        else:
            skipped += 1
            continue

        # Parse rankings (0=winner, 1=loser for 2p)
        if rankings_json:
            rankings = json.loads(rankings_json)
        elif winner_id:
            # Winner is rank 0, other is rank 1
            rankings = [0 if pid == winner_id else 1 for pid in participant_ids]
        else:
            # No ranking info, skip
            skipped += 1
            continue

        # Use elo_after if available (pre-computed ratings)
        if elo_after_json:
            elo_after = json.loads(elo_after_json)

            if not dry_run:
                # Update elo_ratings directly from pre-computed values
                for pid, new_rating in elo_after.items():
                    # Count wins/losses/draws
                    pid_rank = None
                    for i, p in enumerate(participant_ids):
                        if p == pid:
                            pid_rank = rankings[i]
                            break

                    if pid_rank is None:
                        continue

                    # Check if entry exists
                    cursor.execute("""
                        SELECT rating, games_played, wins, losses, draws
                        FROM elo_ratings
                        WHERE participant_id = ? AND board_type = ? AND num_players = ?
                    """, (pid, board_type, num_players))
                    existing = cursor.fetchone()

                    if existing:
                        # Update existing entry
                        old_rating, games, wins, losses, draws = existing
                        games += 1
                        if pid_rank == 0:
                            wins += 1
                        else:
                            losses += 1

                        cursor.execute("""
                            UPDATE elo_ratings
                            SET rating = ?, games_played = ?, wins = ?, losses = ?, draws = ?
                            WHERE participant_id = ? AND board_type = ? AND num_players = ?
                        """, (new_rating, games, wins, losses, draws, pid, board_type, num_players))
                    else:
                        # Insert new entry
                        wins = 1 if pid_rank == 0 else 0
                        losses = 0 if pid_rank == 0 else 1
                        cursor.execute("""
                            INSERT INTO elo_ratings
                            (participant_id, board_type, num_players, rating, games_played, wins, losses, draws, rating_deviation)
                            VALUES (?, ?, ?, ?, 1, ?, ?, 0, 350.0)
                        """, (pid, board_type, num_players, new_rating, wins, losses))

                    participants_updated.add(pid)

            processed += 1
        else:
            # No pre-computed ratings, would need to recalculate
            # For simplicity, skip these
            skipped += 1

    if not dry_run:
        conn.commit()

    return {
        "config": config_key,
        "processed": processed,
        "skipped": skipped,
        "participants": len(participants_updated),
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill elo_ratings from match_history")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))

    # Show current state
    logger.info("=== Current State ===")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT board_type, num_players, COUNT(*)
        FROM match_history
        GROUP BY board_type, num_players
        ORDER BY board_type, num_players
    """)
    logger.info("\nMatch History:")
    for board, players, count in cursor.fetchall():
        logger.info(f"  {board}_{players}p: {count} matches")

    cursor.execute("""
        SELECT board_type, num_players, COUNT(*)
        FROM elo_ratings
        GROUP BY board_type, num_players
        ORDER BY board_type, num_players
    """)
    logger.info("\nElo Ratings:")
    for board, players, count in cursor.fetchall():
        logger.info(f"  {board}_{players}p: {count} participants")

    # Find configs needing backfill
    missing = get_missing_configs(conn)
    if not missing:
        logger.info("\nNo configs need backfilling!")
        return

    logger.info(f"\n=== Configs to Backfill ({len(missing)}) ===")
    for board, players in missing:
        logger.info(f"  {board}_{players}p")

    if args.dry_run:
        logger.info("\n[DRY RUN - no changes will be made]")

    # Process each config
    logger.info("\n=== Processing ===")
    results = []
    for board_type, num_players in missing:
        result = backfill_config(conn, board_type, num_players, dry_run=args.dry_run)
        results.append(result)
        logger.info(f"  {result['config']}: {result['processed']} processed, "
                   f"{result['skipped']} skipped, {result.get('participants', 0)} participants")

    # Show final state
    if not args.dry_run:
        logger.info("\n=== Final State ===")
        cursor.execute("""
            SELECT board_type, num_players, COUNT(*)
            FROM elo_ratings
            GROUP BY board_type, num_players
            ORDER BY board_type, num_players
        """)
        logger.info("\nElo Ratings (after backfill):")
        for board, players, count in cursor.fetchall():
            logger.info(f"  {board}_{players}p: {count} participants")

    conn.close()
    logger.info("\nBackfill complete!")


if __name__ == "__main__":
    main()
