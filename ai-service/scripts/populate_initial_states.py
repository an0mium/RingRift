#!/usr/bin/env python3
"""Populate missing initial_state entries in game databases.

This script generates and inserts initial state records for games that are
missing them. Initial states are computed from board_type and num_players
since all games start from standard positions.

Usage:
    python scripts/populate_initial_states.py --db data/games/canonical_square19_2p.db
    python scripts/populate_initial_states.py --db data/games/canonical_square19_2p.db --dry-run
    python scripts/populate_initial_states.py --all  # Process all canonical DBs
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path


def get_games_missing_initial_state(conn: sqlite3.Connection) -> list[tuple[str, str, int]]:
    """Find games that don't have entries in game_initial_state.

    Returns:
        List of (game_id, board_type, num_players) tuples
    """
    cursor = conn.cursor()

    # Find games without initial_state entries
    cursor.execute("""
        SELECT g.game_id, g.board_type, g.num_players
        FROM games g
        WHERE NOT EXISTS (
            SELECT 1 FROM game_initial_state gis WHERE gis.game_id = g.game_id
        )
    """)

    return cursor.fetchall()


def create_initial_state_json(board_type: str, num_players: int) -> str:
    """Create initial state JSON for a given board config.

    Args:
        board_type: Board type string (e.g., 'square19', 'hex8')
        num_players: Number of players (2-4)

    Returns:
        JSON string of the serialized initial state (Pydantic-compatible)
    """
    from app.models import BoardType
    from app.training.initial_state import create_initial_state

    # Convert board type string to enum
    board_type_enum = BoardType(board_type.lower())

    # Create initial state
    state = create_initial_state(
        board_type=board_type_enum,
        num_players=num_players,
    )

    # Serialize to JSON using Pydantic's format (compatible with _deserialize_state)
    return state.model_dump_json(by_alias=True)


def get_initial_state_columns(conn: sqlite3.Connection) -> tuple[list[str], str, int]:
    """Detect the schema of game_initial_state table.

    Returns:
        Tuple of (column_names, insert_sql, num_json_params)
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(game_initial_state)")
    columns = [row[1] for row in cursor.fetchall()]

    # Check which columns exist
    has_state_json = "state_json" in columns
    has_initial_state_json = "initial_state_json" in columns
    has_compressed = "compressed" in columns

    # Build insert SQL based on available columns
    if has_state_json and has_initial_state_json:
        # Both columns exist
        if has_compressed:
            return (
                columns,
                "INSERT OR REPLACE INTO game_initial_state (game_id, state_json, initial_state_json, compressed) VALUES (?, ?, ?, 0)",
                2,  # 2 JSON params
            )
        else:
            return (
                columns,
                "INSERT OR REPLACE INTO game_initial_state (game_id, state_json, initial_state_json) VALUES (?, ?, ?)",
                2,  # 2 JSON params
            )
    elif has_initial_state_json:
        # Only initial_state_json exists
        if has_compressed:
            return (
                columns,
                "INSERT OR REPLACE INTO game_initial_state (game_id, initial_state_json, compressed) VALUES (?, ?, 0)",
                1,  # 1 JSON param
            )
        else:
            return (
                columns,
                "INSERT OR REPLACE INTO game_initial_state (game_id, initial_state_json) VALUES (?, ?)",
                1,  # 1 JSON param
            )
    elif has_state_json:
        # Only state_json exists
        if has_compressed:
            return (
                columns,
                "INSERT OR REPLACE INTO game_initial_state (game_id, state_json, compressed) VALUES (?, ?, 0)",
                1,  # 1 JSON param
            )
        else:
            return (
                columns,
                "INSERT OR REPLACE INTO game_initial_state (game_id, state_json) VALUES (?, ?)",
                1,  # 1 JSON param
            )
    else:
        raise ValueError("game_initial_state table has no state column")


def populate_initial_states(
    conn: sqlite3.Connection,
    games: list[tuple[str, str, int]],
    dry_run: bool = False,
) -> int:
    """Populate initial_state entries for games.

    Args:
        conn: Database connection
        games: List of (game_id, board_type, num_players) tuples
        dry_run: If True, don't actually insert

    Returns:
        Number of entries created
    """
    cursor = conn.cursor()
    created = 0

    # Detect schema
    try:
        columns, insert_sql, num_json_params = get_initial_state_columns(conn)
    except ValueError as e:
        print(f"  Schema error: {e}")
        return 0

    # Cache initial states by (board_type, num_players)
    # since all games with the same config have identical initial states
    cache: dict[tuple[str, int], str] = {}

    for game_id, board_type, num_players in games:
        try:
            cache_key = (board_type, num_players)

            if cache_key not in cache:
                cache[cache_key] = create_initial_state_json(board_type, num_players)

            initial_state_json = cache[cache_key]

            if not dry_run:
                # Build parameter tuple based on schema
                if num_json_params == 2:
                    # Both state_json and initial_state_json
                    params = (game_id, initial_state_json, initial_state_json)
                else:
                    # Only one column
                    params = (game_id, initial_state_json)

                cursor.execute(insert_sql, params)

            created += 1

        except Exception as e:
            print(f"  Error creating initial state for {game_id}: {e}")

    return created


def process_database(db_path: str, dry_run: bool = False) -> tuple[int, int]:
    """Process a single database.

    Args:
        db_path: Path to the database
        dry_run: If True, don't actually modify

    Returns:
        Tuple of (games_found, entries_created)
    """
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return 0, 0

    conn = sqlite3.connect(db_path)

    # Check if game_initial_state table exists
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='game_initial_state'
    """)
    if not cursor.fetchone():
        print(f"  Table 'game_initial_state' does not exist in {Path(db_path).name}")
        conn.close()
        return 0, 0

    games = get_games_missing_initial_state(conn)
    if not games:
        print(f"  No games missing initial_state in {Path(db_path).name}")
        conn.close()
        return 0, 0

    print(f"  Found {len(games)} games missing initial_state in {Path(db_path).name}")

    created = populate_initial_states(conn, games, dry_run)

    if not dry_run:
        conn.commit()

    conn.close()
    return len(games), created


def find_canonical_databases() -> list[str]:
    """Find all canonical game databases."""
    data_dir = Path("data/games")
    if not data_dir.exists():
        return []

    return sorted([
        str(p) for p in data_dir.glob("canonical_*.db")
        if not p.name.endswith(".db-journal")
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Populate missing initial_state entries in game databases"
    )
    parser.add_argument(
        "--db",
        help="Database path to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all canonical databases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually modify, just show what would be done",
    )

    args = parser.parse_args()

    if not args.db and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN MODE - No changes will be made ===\n")

    databases = []
    if args.all:
        databases = find_canonical_databases()
        if not databases:
            print("No canonical databases found in data/games/")
            sys.exit(1)
        print(f"Found {len(databases)} canonical databases\n")
    else:
        databases = [args.db]

    total_found = 0
    total_created = 0

    for db_path in databases:
        print(f"\nProcessing {db_path}...")
        found, created = process_database(db_path, args.dry_run)
        total_found += found
        total_created += created
        if created > 0:
            print(f"  Created {created} initial_state entries")

    print(f"\n{'=' * 50}")
    print(f"Total: {total_created} entries created for {total_found} games")
    if args.dry_run:
        print("(Dry run - no changes were made)")


if __name__ == "__main__":
    main()
