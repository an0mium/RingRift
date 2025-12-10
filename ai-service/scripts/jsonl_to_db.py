#!/usr/bin/env python
"""Convert JSONL game records to SQLite DB format.

This script converts lightweight JSONL game records (from run_hybrid_selfplay.py)
to the SQLite DB format used by GameReplayDB and the merge script.

Since JSONL records only contain metadata (no state snapshots or moves),
the resulting DB will have minimal game entries suitable for statistics
but not full replay.

Usage:
    python scripts/jsonl_to_db.py --input games.jsonl --output games.db
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path


def create_db_schema(conn: sqlite3.Connection) -> None:
    """Create the games table schema matching GameReplayDB."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            winner INTEGER,
            move_count INTEGER,
            game_status TEXT,
            victory_type TEXT,
            created_at TEXT,
            source TEXT,
            metadata_json TEXT
        )
    """)
    conn.commit()


def convert_jsonl_to_db(input_path: str, output_path: str) -> int:
    """Convert JSONL file to SQLite DB.

    Returns the number of games converted.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 0

    # Create/open the output database
    conn = sqlite3.connect(output_path)
    create_db_schema(conn)

    games_added = 0
    games_skipped = 0

    # Get basename of input file for source tracking
    source_name = Path(input_path).stem

    with open(input_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}",
                      file=sys.stderr)
                games_skipped += 1
                continue

            # Generate a unique game_id (original is just an integer index)
            original_id = record.get("game_id", line_num)
            game_id = f"{source_name}_{original_id}_{uuid.uuid4().hex[:8]}"

            # Extract fields
            board_type = record.get("board_type", "unknown")
            num_players = record.get("num_players", 2)
            winner = record.get("winner", 0)
            move_count = record.get("move_count", 0)
            game_status = record.get("status", "unknown")
            victory_type = record.get("victory_type", "unknown")
            timestamp = record.get("timestamp", "")
            game_time = record.get("game_time_seconds", 0)

            # Store metadata
            metadata = {
                "source": f"jsonl:{source_name}",
                "original_game_id": original_id,
                "game_time_seconds": game_time,
            }

            try:
                conn.execute("""
                    INSERT INTO games
                    (game_id, board_type, num_players, winner, move_count,
                     game_status, victory_type, created_at, source, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    game_id, board_type, num_players, winner, move_count,
                    game_status, victory_type, timestamp,
                    f"jsonl:{source_name}", json.dumps(metadata)
                ))
                games_added += 1
            except sqlite3.IntegrityError:
                # Duplicate game_id (shouldn't happen with UUID suffix)
                games_skipped += 1

    conn.commit()
    conn.close()

    print(f"Converted {games_added} games from {input_path}")
    if games_skipped > 0:
        print(f"  (skipped {games_skipped} invalid/duplicate records)")

    return games_added


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL game records to SQLite DB format"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output SQLite DB path"
    )

    args = parser.parse_args()

    count = convert_jsonl_to_db(args.input, args.output)
    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
