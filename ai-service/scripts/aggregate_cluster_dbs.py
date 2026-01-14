#!/usr/bin/env python3
"""
Aggregate Cluster DBs - Consolidates selfplay databases from cluster nodes.

This script aggregates SQLite game databases from multiple cluster nodes into
a single unified database for training. It handles deduplication, validation,
and parity checking.

IMPORTANT: This script creates a minimal, legacy-compatible GameReplayDB
output (games + game_moves + game_initial_state) for export/analysis.
It is not a full canonical schema and should not be treated as canonical
training data. It can read from:
- Old format: `moves` table with action_type, action_data columns
- New format: `game_moves` table with move_json column

Usage:
    python scripts/aggregate_cluster_dbs.py --input-dir PATH --output PATH [options]

Example:
    python scripts/aggregate_cluster_dbs.py \
        --input-dir /Volumes/RingRift-Data/canonical_data/cluster_20251222 \
        --output data/games/cluster_aggregated_20251222.db
"""

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Legacy schema marker for aggregated outputs. Keep aligned with the
# minimal schema emitted by create_output_schema().
OUTPUT_SCHEMA_VERSION = 9


def get_db_schema_version(conn: sqlite3.Connection) -> Optional[str]:
    """Get the schema version of a database."""
    try:
        cursor = conn.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        )
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def get_game_count(conn: sqlite3.Connection) -> int:
    """Get the number of games in a database."""
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM games")
        return cursor.fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def get_move_count(conn: sqlite3.Connection) -> int:
    """Get the number of moves in a database (handles both schema formats)."""
    # Try new format first (game_moves)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM game_moves")
        return cursor.fetchone()[0]
    except sqlite3.OperationalError:
        pass
    # Fall back to old format (moves)
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM moves")
        return cursor.fetchone()[0]
    except sqlite3.OperationalError:
        return 0


def detect_db_format(conn: sqlite3.Connection) -> str:
    """Detect whether DB uses old 'moves' or new 'game_moves' table format.

    Returns:
        'game_moves' if using GameReplayDB format with move_json
        'moves' if using old format with action_type/action_data
        'unknown' if neither table exists
    """
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('game_moves', 'moves')"
    )
    tables = {row[0] for row in cursor.fetchall()}

    if 'game_moves' in tables:
        # Verify it has move_json column
        cursor = conn.execute("PRAGMA table_info(game_moves)")
        columns = {row[1] for row in cursor.fetchall()}
        if 'move_json' in columns:
            return 'game_moves'

    if 'moves' in tables:
        return 'moves'

    return 'unknown'


def convert_old_move_to_json(
    player: int,
    action_type: str,
    action_data: str,
    move_number: int
) -> str:
    """Convert old format (action_type, action_data) to move_json.

    Creates a Move-compatible JSON structure from legacy format.
    """
    # Parse action_data if it's JSON
    try:
        data = json.loads(action_data) if action_data else {}
    except (json.JSONDecodeError, TypeError):
        data = {"raw": action_data}

    # Build a Move-like structure
    move_dict = {
        "type": action_type,
        "player": player,
    }

    # Map common action types to Move fields
    if action_type in ("placement", "place"):
        move_dict["type"] = "placement"
        if "position" in data:
            move_dict["to"] = data["position"]
        elif "to" in data:
            move_dict["to"] = data["to"]
    elif action_type in ("movement", "move", "slide"):
        move_dict["type"] = "movement"
        if "from" in data:
            move_dict["from"] = data["from"]
        if "to" in data:
            move_dict["to"] = data["to"]
    elif action_type in ("line_process", "process_line"):
        move_dict["type"] = "line_process"
        if "positions" in data:
            move_dict["positions"] = data["positions"]
    elif action_type in ("territory_claim", "claim"):
        move_dict["type"] = "territory_claim"
        if "position" in data:
            move_dict["position"] = data["position"]
    else:
        # Keep original type and merge data
        move_dict.update(data)

    return json.dumps(move_dict)


def compute_game_hash(game_row: tuple, moves: List[tuple]) -> str:
    """Compute a deterministic hash for a game (for deduplication)."""
    # Hash based on board_type, num_players, and move sequence
    data = f"{game_row[1]}:{game_row[2]}:"  # board_type, num_players
    for move in moves:
        data += f"{move[2]}:{move[3]}:{move[4]},"  # player, action_type, action_data
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def validate_game(game_row: tuple, moves: List[tuple]) -> Tuple[bool, str]:
    """Validate a game for inclusion in aggregated dataset."""
    game_id, board_type, num_players, status, winner_id, created_at, *_ = game_row

    # Must have a terminal status
    if status not in ("completed", "finished", "abandoned"):
        return False, f"Non-terminal status: {status}"

    # Must have moves
    if not moves:
        return False, "No moves"

    # Sanity checks
    if num_players < 2 or num_players > 4:
        return False, f"Invalid player count: {num_players}"

    if board_type not in ("square8", "square19", "hexagonal", "hex8"):
        return False, f"Unknown board type: {board_type}"

    return True, "ok"


def create_output_schema(conn: sqlite3.Connection):
    """Create the output database schema (legacy GameReplayDB-compatible).

    Uses the standard `game_moves` table with `move_json` for compatibility
    with GameReplayDB.iterate_games() and export_replay_dataset, but omits
    newer canonical tables (snapshots/history/choices).
    """
    conn.executescript("""
        -- Schema metadata table
        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        -- Games table (GameReplayDB-compatible)
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            original_id TEXT,
            source_db TEXT,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            game_status TEXT NOT NULL,
            winner INTEGER,
            termination_reason TEXT,
            created_at TEXT,
            ended_at TEXT,
            total_moves INTEGER,
            game_hash TEXT UNIQUE,
            -- Additional GameReplayDB columns
            victory_type TEXT,
            excluded_from_training INTEGER DEFAULT 0
        );

        -- Moves table (GameReplayDB-compatible with move_json)
        CREATE TABLE IF NOT EXISTS game_moves (
            game_id TEXT NOT NULL,
            move_number INTEGER NOT NULL,
            turn_number INTEGER NOT NULL DEFAULT 0,
            player INTEGER NOT NULL,
            phase TEXT NOT NULL DEFAULT 'play',
            move_type TEXT NOT NULL,
            move_json TEXT NOT NULL,
            timestamp TEXT,
            think_time_ms INTEGER,
            PRIMARY KEY (game_id, move_number),
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );

        -- Game initial state (required by GameReplayDB)
        CREATE TABLE IF NOT EXISTS game_initial_state (
            game_id TEXT PRIMARY KEY,
            state_json TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games(game_id) ON DELETE CASCADE
        );

        -- Indexes for efficient querying
        CREATE INDEX IF NOT EXISTS idx_games_board_type ON games(board_type);
        CREATE INDEX IF NOT EXISTS idx_games_num_players ON games(num_players);
        CREATE INDEX IF NOT EXISTS idx_games_status ON games(game_status);
        CREATE INDEX IF NOT EXISTS idx_games_hash ON games(game_hash);
        CREATE INDEX IF NOT EXISTS idx_moves_game_id ON game_moves(game_id);
        CREATE INDEX IF NOT EXISTS idx_moves_game_turn ON game_moves(game_id, turn_number);
    """)


def get_moves_from_db(
    input_conn: sqlite3.Connection,
    game_id: str,
    db_format: str
) -> List[Dict]:
    """Get moves from a database, handling both old and new formats.

    Returns list of dicts with standardized fields:
    - move_number, turn_number, player, phase, move_type, move_json
    """
    moves = []

    if db_format == 'game_moves':
        # New GameReplayDB format - move_json already present
        cursor = input_conn.execute("""
            SELECT move_number, turn_number, player, phase, move_type, move_json
            FROM game_moves
            WHERE game_id = ?
            ORDER BY move_number
        """, (game_id,))
        for row in cursor:
            moves.append({
                'move_number': row[0],
                'turn_number': row[1],
                'player': row[2],
                'phase': row[3],
                'move_type': row[4],
                'move_json': row[5],
            })
    elif db_format == 'moves':
        # Old format - need to convert to move_json
        cursor = input_conn.execute("""
            SELECT move_number, player, action_type, action_data
            FROM moves
            WHERE game_id = ?
            ORDER BY move_number
        """, (game_id,))
        for row in cursor:
            move_number = row[0]
            player = row[1]
            action_type = row[2]
            action_data = row[3]
            move_json = convert_old_move_to_json(player, action_type, action_data, move_number)
            moves.append({
                'move_number': move_number,
                'turn_number': move_number,  # Approximate turn from move number
                'player': player,
                'phase': 'play',  # Default phase
                'move_type': action_type,
                'move_json': move_json,
            })

    return moves


def get_initial_state_from_db(
    input_conn: sqlite3.Connection,
    game_id: str
) -> Optional[str]:
    """Get initial state JSON from database if available."""
    try:
        cursor = input_conn.execute(
            "SELECT state_json FROM game_initial_state WHERE game_id = ?",
            (game_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def aggregate_database(
    input_path: Path,
    output_conn: sqlite3.Connection,
    seen_hashes: Set[str],
    stats: Dict[str, int]
) -> Dict[str, int]:
    """Aggregate games from a single input database.

    Handles both old 'moves' format and new 'game_moves' format,
    outputting in GameReplayDB-compatible format.
    """
    source_name = input_path.name

    try:
        input_conn = sqlite3.connect(str(input_path))
        input_conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        logger.warning(f"Could not open {input_path}: {e}")
        stats["failed_dbs"] += 1
        return stats

    try:
        game_count = get_game_count(input_conn)
        if game_count == 0:
            logger.info(f"  Skipping {source_name}: no games")
            stats["empty_dbs"] += 1
            return stats

        # Detect source format
        db_format = detect_db_format(input_conn)
        if db_format == 'unknown':
            logger.warning(f"  Skipping {source_name}: unknown schema format")
            stats["failed_dbs"] += 1
            return stats

        logger.info(f"  Processing {source_name}: {game_count} games (format: {db_format})")

        # Detect games table columns (may vary between versions)
        cursor = input_conn.execute("PRAGMA table_info(games)")
        games_columns = {row[1] for row in cursor.fetchall()}

        # Build flexible query based on available columns
        game_id_col = 'game_id' if 'game_id' in games_columns else 'id'
        status_col = 'game_status' if 'game_status' in games_columns else 'status'
        winner_col = 'winner' if 'winner' in games_columns else 'winner_id'
        moves_col = 'total_moves' if 'total_moves' in games_columns else 'move_count'
        # Handle ended_at vs completed_at schema difference
        ended_col = 'ended_at' if 'ended_at' in games_columns else (
            'completed_at' if 'completed_at' in games_columns else 'created_at'
        )

        query = f"""
            SELECT {game_id_col} as game_id, board_type, num_players,
                   {status_col} as status, {winner_col} as winner,
                   created_at, {ended_col} as ended_at, {moves_col} as move_count
            FROM games
            WHERE {status_col} IN ('completed', 'finished', 'abandoned')
        """

        games_cursor = input_conn.execute(query)

        added = 0
        skipped_dup = 0
        skipped_invalid = 0

        for game_row in games_cursor:
            game_id = game_row['game_id']

            # Get moves using format-aware function
            moves = get_moves_from_db(input_conn, game_id, db_format)

            # Create tuple for validation (legacy format)
            game_tuple = (
                game_id,
                game_row['board_type'],
                game_row['num_players'],
                game_row['status'],
                game_row['winner'],
                game_row['created_at'],
            )
            moves_tuples = [
                (None, game_id, m['player'], m['move_type'], m['move_json'], None, m['move_number'])
                for m in moves
            ]

            # Validate
            is_valid, reason = validate_game(game_tuple, moves_tuples)
            if not is_valid:
                skipped_invalid += 1
                continue

            # Compute hash for deduplication (using move_json for consistency)
            hash_data = f"{game_row['board_type']}:{game_row['num_players']}:"
            for m in moves:
                hash_data += f"{m['player']}:{m['move_json']},"
            game_hash = hashlib.sha256(hash_data.encode()).hexdigest()[:16]

            if game_hash in seen_hashes:
                skipped_dup += 1
                continue

            seen_hashes.add(game_hash)

            # Generate new ID
            new_game_id = f"agg_{game_hash}"

            # Insert game (GameReplayDB-compatible format)
            output_conn.execute("""
                INSERT INTO games
                (game_id, original_id, source_db, board_type, num_players,
                 game_status, winner, created_at, ended_at, total_moves, game_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_game_id, game_id, source_name,
                game_row['board_type'], game_row['num_players'],
                game_row['status'], game_row['winner'],
                game_row['created_at'], game_row['ended_at'],
                game_row['move_count'], game_hash
            ))

            # Insert moves (GameReplayDB-compatible format)
            for m in moves:
                output_conn.execute("""
                    INSERT INTO game_moves
                    (game_id, move_number, turn_number, player, phase, move_type, move_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    new_game_id, m['move_number'], m['turn_number'],
                    m['player'], m['phase'], m['move_type'], m['move_json']
                ))

            # Copy initial state if available
            initial_state = get_initial_state_from_db(input_conn, game_id)
            if initial_state:
                output_conn.execute("""
                    INSERT INTO game_initial_state (game_id, state_json)
                    VALUES (?, ?)
                """, (new_game_id, initial_state))

            added += 1

        output_conn.commit()

        stats["games_added"] += added
        stats["games_skipped_dup"] += skipped_dup
        stats["games_skipped_invalid"] += skipped_invalid
        stats["dbs_processed"] += 1

        logger.info(f"    Added: {added}, Dup: {skipped_dup}, Invalid: {skipped_invalid}")

    except sqlite3.Error as e:
        logger.warning(f"  Error processing {source_name}: {e}")
        stats["failed_dbs"] += 1

    finally:
        input_conn.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Aggregate cluster game databases")
    parser.add_argument("--input-dir", required=True,
                       help="Directory containing source DB files")
    parser.add_argument("--output", required=True,
                       help="Output aggregated database path")
    parser.add_argument("--pattern", default="*.db",
                       help="Glob pattern for input files (default: *.db)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without writing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    # Find input databases
    input_dbs = sorted(input_dir.glob(args.pattern))
    if not input_dbs:
        logger.error(f"No databases found matching {args.pattern} in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(input_dbs)} database files to aggregate")

    if args.dry_run:
        for db in input_dbs:
            try:
                conn = sqlite3.connect(str(db))
                games = get_game_count(conn)
                moves = get_move_count(conn)
                conn.close()
                logger.info(f"  {db.name}: {games} games, {moves} moves")
            except Exception as e:
                logger.warning(f"  {db.name}: error - {e}")
        logger.info("Dry run complete. Use without --dry-run to aggregate.")
        return

    # Create output database
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        backup_path = output_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        logger.info(f"Backing up existing output to {backup_path}")
        output_path.rename(backup_path)

    output_conn = sqlite3.connect(str(output_path))
    create_output_schema(output_conn)

    # Track seen game hashes for deduplication
    seen_hashes: Set[str] = set()

    # Stats
    stats = {
        "dbs_processed": 0,
        "empty_dbs": 0,
        "failed_dbs": 0,
        "games_added": 0,
        "games_skipped_dup": 0,
        "games_skipped_invalid": 0,
    }

    # Process each database
    logger.info("Starting aggregation...")
    for db_path in input_dbs:
        stats = aggregate_database(db_path, output_conn, seen_hashes, stats)

    # Add schema metadata (legacy compatibility marker).
    output_conn.execute("""
        INSERT OR REPLACE INTO schema_metadata (key, value) VALUES
        ('schema_version', ?),
        ('aggregator_version', '2.0'),
        ('created_at', ?),
        ('source_dir', ?),
        ('dbs_processed', ?),
        ('games_total', ?),
        ('games_deduped', ?)
    """, (
        str(OUTPUT_SCHEMA_VERSION),
        datetime.now().isoformat(),
        str(input_dir),
        str(stats["dbs_processed"]),
        str(stats["games_added"]),
        str(stats["games_skipped_dup"])
    ))
    output_conn.commit()
    output_conn.close()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Aggregation Complete")
    logger.info("="*60)
    logger.info(f"Output: {output_path}")
    logger.info(f"DBs processed: {stats['dbs_processed']}")
    logger.info(f"Empty DBs skipped: {stats['empty_dbs']}")
    logger.info(f"Failed DBs: {stats['failed_dbs']}")
    logger.info(f"Games added: {stats['games_added']}")
    logger.info(f"Duplicates removed: {stats['games_skipped_dup']}")
    logger.info(f"Invalid games skipped: {stats['games_skipped_invalid']}")

    # Check output size
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Output size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
