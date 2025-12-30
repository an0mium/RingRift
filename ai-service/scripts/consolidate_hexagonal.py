#!/usr/bin/env python3
"""Consolidate hexagonal game databases into canonical DBs.

December 2025: One-time script to fix empty canonical hexagonal databases.
The data exists in jsonl_converted_* and *_reimport.db files but was never
consolidated.
"""
import shutil
import sqlite3
from pathlib import Path


def get_table_columns(cursor, table_name: str) -> list[str]:
    """Get column names for a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def consolidate_to_canonical(source_dbs: list[str], canonical_db: str) -> int:
    """Merge games from source DBs into canonical DB.

    Handles schema mismatches by only copying columns that exist in both,
    and providing defaults for required columns missing in the source.

    Args:
        source_dbs: List of source database paths
        canonical_db: Path to canonical database

    Returns:
        Number of games imported
    """
    # Defaults for missing required columns
    DEFAULTS = {
        "total_turns": 0,
        "schema_version": 1,
    }

    can_conn = sqlite3.connect(canonical_db)
    can_cursor = can_conn.cursor()

    # Get canonical schema for each table (including NOT NULL info)
    can_columns = {}
    can_required = {}
    for table in ["games", "game_players", "game_moves", "game_initial_state", "game_state_snapshots"]:
        can_cursor.execute(f"PRAGMA table_info({table})")
        cols = set()
        required = set()
        for row in can_cursor.fetchall():
            col_name = row[1]
            notnull = row[3]
            cols.add(col_name)
            if notnull:
                required.add(col_name)
        can_columns[table] = cols
        can_required[table] = required

    can_cursor.execute("SELECT game_id FROM games")
    existing_ids = set(row[0] for row in can_cursor.fetchall())
    print(f"{Path(canonical_db).name}: {len(existing_ids)} existing games")

    total_imported = 0
    for source_db in source_dbs:
        if not Path(source_db).exists():
            print(f"  Skipping {source_db} - not found")
            continue

        src_conn = sqlite3.connect(source_db)
        src_cursor = src_conn.cursor()

        # Get source column names for games table
        src_game_cols = get_table_columns(src_cursor, "games")

        # Find common columns (preserve order from source)
        common_game_cols = [c for c in src_game_cols if c in can_columns["games"]]

        if not common_game_cols:
            print(f"  Skipping {source_db} - no common columns")
            src_conn.close()
            continue

        # Find missing required columns that need defaults
        missing_required = can_required["games"] - set(src_game_cols)
        cols_with_defaults = [(col, DEFAULTS.get(col)) for col in missing_required if col in DEFAULTS]

        # Get all games from source with only common columns
        src_cursor.execute(f"SELECT {','.join(common_game_cols)} FROM games")
        all_games = src_cursor.fetchall()

        imported = 0
        for game in all_games:
            game_id = game[common_game_cols.index("game_id")]
            if game_id in existing_ids:
                continue

            try:
                # Build insert with common columns + defaults for missing required
                insert_cols = list(common_game_cols)
                insert_vals = list(game)

                for col, default_val in cols_with_defaults:
                    if col not in insert_cols and default_val is not None:
                        insert_cols.append(col)
                        insert_vals.append(default_val)

                placeholders = ",".join("?" * len(insert_cols))
                can_cursor.execute(
                    f"INSERT OR IGNORE INTO games ({','.join(insert_cols)}) VALUES ({placeholders})",
                    insert_vals
                )

                # Copy related tables with schema mapping
                for table in ["game_players", "game_moves", "game_initial_state", "game_state_snapshots"]:
                    try:
                        src_table_cols = get_table_columns(src_cursor, table)
                        common_cols = [c for c in src_table_cols if c in can_columns.get(table, set())]

                        if not common_cols:
                            continue

                        src_cursor.execute(f"SELECT {','.join(common_cols)} FROM {table} WHERE game_id = ?", (game_id,))
                        rows = src_cursor.fetchall()
                        if rows:
                            ph = ",".join("?" * len(common_cols))
                            for row in rows:
                                can_cursor.execute(
                                    f"INSERT OR IGNORE INTO {table} ({','.join(common_cols)}) VALUES ({ph})",
                                    row
                                )
                    except sqlite3.OperationalError:
                        pass

                imported += 1
                existing_ids.add(game_id)
            except Exception as e:
                print(f"  Error: {e}")

        if imported > 0:
            print(f"  + {imported} games from {Path(source_db).name}")
            total_imported += imported

        src_conn.close()

    can_conn.commit()
    can_cursor.execute("SELECT COUNT(*) FROM games")
    final_count = can_cursor.fetchone()[0]
    print(f"  Final: {final_count} games")
    can_conn.close()
    return total_imported


def ensure_canonical_exists(canonical_path: str, template_db: str = None) -> bool:
    """Ensure a canonical database exists with proper schema.

    If it doesn't exist, create it from a template or with minimal schema.
    """
    if Path(canonical_path).exists():
        return True

    if template_db and Path(template_db).exists():
        # Copy template database structure
        shutil.copy(template_db, canonical_path)
        # Clear all data tables
        conn = sqlite3.connect(canonical_path)
        cur = conn.cursor()
        for table in ["games", "game_players", "game_moves", "game_initial_state", "game_state_snapshots"]:
            try:
                cur.execute(f"DELETE FROM {table}")
            except sqlite3.OperationalError:
                pass
        conn.commit()
        conn.close()
        print(f"  Created {Path(canonical_path).name} from template")
        return True

    return False


def main():
    """Run consolidation for all configs."""
    base = Path("data/games")

    # Use an existing canonical DB as template for schema
    template = None
    for candidate in ["canonical_hex8_2p.db", "canonical_hexagonal_2p.db", "canonical_square8_2p.db"]:
        if (base / candidate).exists():
            template = str(base / candidate)
            break

    # Consolidate hexagonal 2p
    print("\n=== Hexagonal 2-player ===")
    consolidate_to_canonical(
        [str(base / "jsonl_converted_hexagonal_2p.db")],
        str(base / "canonical_hexagonal_2p.db")
    )

    # Consolidate hexagonal 3p
    print("\n=== Hexagonal 3-player ===")
    consolidate_to_canonical(
        [
            str(base / "jsonl_converted_hexagonal_3p.db"),
            str(base / "hexagonal_3p_reimport.db"),
            str(base / "hexagonal_3p_imported.db"),
        ],
        str(base / "canonical_hexagonal_3p.db")
    )

    # Consolidate hexagonal 4p
    print("\n=== Hexagonal 4-player ===")
    consolidate_to_canonical(
        [
            str(base / "jsonl_converted_hexagonal_4p.db"),
            str(base / "hexagonal_4p_reimport.db"),
        ],
        str(base / "canonical_hexagonal_4p.db")
    )

    # Also consolidate hex8 3p and 4p
    print("\n=== hex8 3-player ===")
    consolidate_to_canonical(
        [str(base / "hex8_3p.db")],
        str(base / "canonical_hex8_3p.db")
    )

    print("\n=== hex8 4-player ===")
    consolidate_to_canonical(
        [str(base / "hex8_4p.db")],
        str(base / "canonical_hex8_4p.db")
    )

    # Consolidate square8 configs
    print("\n=== Square8 2-player ===")
    if ensure_canonical_exists(str(base / "canonical_square8_2p.db"), template):
        consolidate_to_canonical(
            [
                str(base / "jsonl_converted_square8_2p.db"),
                str(base / "tournament_square8_2p.db"),
                str(base / "square8_2p.db"),
            ],
            str(base / "canonical_square8_2p.db")
        )

    print("\n=== Square8 4-player ===")
    if ensure_canonical_exists(str(base / "canonical_square8_4p.db"), template):
        consolidate_to_canonical(
            [str(base / "square8_4p.db")],
            str(base / "canonical_square8_4p.db")
        )

    # Consolidate square19 configs
    print("\n=== Square19 2-player ===")
    if ensure_canonical_exists(str(base / "canonical_square19_2p.db"), template):
        consolidate_to_canonical(
            [
                str(base / "jsonl_converted_square19_2p.db"),
                str(base / "square19_2p.db"),
            ],
            str(base / "canonical_square19_2p.db")
        )

    print("\n=== Square19 4-player ===")
    if ensure_canonical_exists(str(base / "canonical_square19_4p.db"), template):
        consolidate_to_canonical(
            [str(base / "square19_4p.db")],
            str(base / "canonical_square19_4p.db")
        )

    print("\nConsolidation complete!")


if __name__ == "__main__":
    main()
