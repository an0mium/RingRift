from __future__ import annotations

import sqlite3
from pathlib import Path

import scripts.generate_canonical_selfplay as gen


def _make_minimal_db(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE games (game_id TEXT PRIMARY KEY, game_status TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE game_moves (game_id TEXT NOT NULL, move_number INTEGER NOT NULL, move_type TEXT NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()


def test_collect_db_stats_counts_swap_sides(tmp_path: Path) -> None:
    db_path = tmp_path / "games.db"
    _make_minimal_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            "INSERT INTO games (game_id, game_status) VALUES (?, ?)",
            [
                ("g1", "completed"),
                ("g2", "active"),
            ],
        )
        conn.executemany(
            "INSERT INTO game_moves (game_id, move_number, move_type) VALUES (?, ?, ?)",
            [
                ("g1", 1, "place_ring"),
                ("g1", 2, "swap_sides"),
                ("g2", 1, "place_ring"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    stats = gen.collect_db_stats(db_path)
    assert stats["games_total"] == 2
    assert stats["games_completed"] == 1
    assert stats["moves_total"] == 3
    assert stats["swap_sides"]["moves"] == 1
    assert stats["swap_sides"]["games"] == 1

