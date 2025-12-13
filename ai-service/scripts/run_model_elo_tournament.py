#!/usr/bin/env python3
"""Run comprehensive model-vs-model tournament with persistent Elo tracking.

This script addresses the gap where 182+ trained models exist but no cross-model
Elo leaderboard tracks their relative strengths.

Features:
1. Discovers all trained models (.pth files)
2. Runs round-robin or Swiss tournaments between models
3. Persists Elo ratings to SQLite database
4. Generates leaderboard reports

Usage:
    # Run tournament between all v3/v4/v5 models
    python scripts/run_model_elo_tournament.py --board square8 --players 2

    # Run quick tournament with top N models only
    python scripts/run_model_elo_tournament.py --board square8 --players 2 --top-n 10

    # View current leaderboard without running games
    python scripts/run_model_elo_tournament.py --leaderboard-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.tournament.elo import EloCalculator, EloRating
from app.models import (
    AIConfig, AIType, BoardType, GamePhase, GameStatus,
    BoardState, GameState, Player, TimeControl,
)
from app.rules.default_engine import DefaultRulesEngine


# ============================================
# Game Execution with Neural Networks
# ============================================

def play_nn_vs_nn_game(
    model_a_path: str,
    model_b_path: str,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 300,
    mcts_simulations: int = 100,
) -> Dict[str, Any]:
    """Play a single game between two neural network models.

    Returns dict with: winner (model_a, model_b, or draw), game_length, duration_sec
    """
    import time
    import uuid
    from datetime import datetime
    from app.ai.neural_net import NeuralNetAI, clear_model_cache

    start_time = time.time()

    # Create game state
    size = 8 if board_type == BoardType.SQUARE8 else (19 if board_type == BoardType.SQUARE19 else 5)
    board = BoardState(
        type=board_type,
        size=size,
        stacks={},
        markers={},
        collapsedSpaces={},
        eliminatedRings={},
    )

    players = []
    rings_per_player = 20 if num_players == 2 else (15 if num_players == 3 else 12)
    for i in range(num_players):
        players.append(Player(
            id=f"player{i+1}",
            username=f"NN_P{i+1}",
            type="ai",
            playerNumber=i + 1,
            isReady=True,
            timeRemaining=600000,
            aiDifficulty=10,
            ringsInHand=rings_per_player,
            eliminatedRings=0,
            territorySpaces=0,
        ))

    game_state = GameState(
        id=str(uuid.uuid4()),
        boardType=board_type,
        board=board,
        players=players,
        currentPhase=GamePhase.RING_PLACEMENT,
        currentPlayer=1,
        moveHistory=[],
        timeControl=TimeControl(initialTime=600, increment=5, type="standard"),
        gameStatus=GameStatus.ACTIVE,
        createdAt=datetime.now(),
        lastMoveAt=datetime.now(),
        isRated=False,
        maxPlayers=num_players,
        totalRingsInPlay=0,
        totalRingsEliminated=0,
        victoryThreshold=3,
        territoryVictoryThreshold=10 if num_players == 2 else 8,
        chainCaptureState=None,
        mustMoveFromStackKey=None,
        zobristHash=None,
    )

    # Create AI instances - alternate between model A and model B
    # Player 1 -> model_a, Player 2 -> model_b
    ai_configs = []
    model_paths = [model_a_path, model_b_path]

    for i in range(num_players):
        model_idx = i % 2  # Alternate models for multiplayer
        config = AIConfig(
            type=AIType.DESCENT,
            difficulty=10,
            nn_model_id=model_paths[model_idx],  # Pass full path
            mcts_simulations=mcts_simulations,
            think_time=5000,
            use_neural_net=True,
        )
        ai_configs.append(config)

    # Create neural net AIs
    ais = []
    try:
        for i, config in enumerate(ai_configs):
            ai = NeuralNetAI(player_number=i + 1, config=config, board_type=board_type)
            ais.append(ai)
    except Exception as e:
        clear_model_cache()
        return {
            "winner": "error",
            "game_length": 0,
            "duration_sec": time.time() - start_time,
            "error": str(e),
        }

    rules_engine = DefaultRulesEngine()
    move_count = 0

    # Play the game
    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = game_state.current_player
        current_ai = ais[current_player - 1]
        current_ai.player_number = current_player

        try:
            move = current_ai.select_move(game_state)
        except Exception as e:
            # AI error - opponent wins
            winner_idx = 1 if current_player == 1 else 0
            clear_model_cache()
            return {
                "winner": "model_b" if winner_idx == 1 else "model_a",
                "game_length": move_count,
                "duration_sec": time.time() - start_time,
                "error": f"AI error: {e}",
            }

        if not move:
            # No valid moves - opponent wins
            winner_idx = 1 if current_player == 1 else 0
            clear_model_cache()
            return {
                "winner": "model_b" if winner_idx == 1 else "model_a",
                "game_length": move_count,
                "duration_sec": time.time() - start_time,
            }

        try:
            game_state = rules_engine.apply_move(game_state, move)
        except Exception as e:
            winner_idx = 1 if current_player == 1 else 0
            clear_model_cache()
            return {
                "winner": "model_b" if winner_idx == 1 else "model_a",
                "game_length": move_count,
                "duration_sec": time.time() - start_time,
                "error": f"Move error: {e}",
            }

        move_count += 1

    # Determine winner
    duration = time.time() - start_time
    clear_model_cache()

    if game_state.game_status == GameStatus.COMPLETED and game_state.winner is not None:
        # Winner is 1-indexed player number
        winner = "model_a" if game_state.winner == 1 else "model_b"
        return {
            "winner": winner,
            "game_length": move_count,
            "duration_sec": duration,
        }
    else:
        return {
            "winner": "draw",
            "game_length": move_count,
            "duration_sec": duration,
        }


def run_model_matchup(
    conn: sqlite3.Connection,
    model_a: Dict[str, Any],
    model_b: Dict[str, Any],
    board_type: str,
    num_players: int,
    games: int,
    tournament_id: str,
) -> Dict[str, int]:
    """Run multiple games between two models and update Elo."""
    board_type_enum = BoardType.SQUARE8
    if board_type == "square19":
        board_type_enum = BoardType.SQUARE19
    elif board_type == "hex":
        board_type_enum = BoardType.HEXAGONAL

    results = {"model_a_wins": 0, "model_b_wins": 0, "draws": 0, "errors": 0}

    for game_num in range(games):
        # Alternate who plays first
        if game_num % 2 == 0:
            path_a, path_b = model_a["model_path"], model_b["model_path"]
            id_a, id_b = model_a["model_id"], model_b["model_id"]
        else:
            path_a, path_b = model_b["model_path"], model_a["model_path"]
            id_a, id_b = model_b["model_id"], model_a["model_id"]

        result = play_nn_vs_nn_game(
            model_a_path=path_a,
            model_b_path=path_b,
            board_type=board_type_enum,
            num_players=num_players,
            max_moves=300,
            mcts_simulations=50,  # Faster games
        )

        # Map back to original model_a/model_b
        winner_id = None
        if result["winner"] == "model_a":
            winner_id = id_a
        elif result["winner"] == "model_b":
            winner_id = id_b

        # Update stats based on original model_a vs model_b
        if winner_id == model_a["model_id"]:
            results["model_a_wins"] += 1
            winner = model_a["model_id"]
        elif winner_id == model_b["model_id"]:
            results["model_b_wins"] += 1
            winner = model_b["model_id"]
        elif result["winner"] == "error":
            results["errors"] += 1
            continue
        else:
            results["draws"] += 1
            winner = "draw"

        # Record match in database
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO match_history (model_a, model_b, board_type, num_players, winner, game_length, duration_sec, timestamp, tournament_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_a["model_id"], model_b["model_id"], board_type, num_players,
            winner, result["game_length"], result["duration_sec"],
            time.time(), tournament_id
        ))
        conn.commit()

        # Update Elo
        update_elo_after_match(
            conn,
            model_a["model_id"],
            model_b["model_id"],
            winner,
            board_type,
            num_players,
            tournament_id,
        )

    return results


# ============================================
# Persistent Elo Database
# ============================================

ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "elo_leaderboard.db"


def init_elo_database(db_path: Path = ELO_DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database for persistent Elo storage."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Models table - all known models
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            model_path TEXT,
            board_type TEXT,
            num_players INTEGER,
            model_version TEXT,
            created_at REAL,
            last_seen REAL
        )
    """)

    # Elo ratings table - current ratings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS elo_ratings (
            model_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            rating REAL DEFAULT 1500.0,
            games_played INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            last_update REAL,
            FOREIGN KEY (model_id) REFERENCES models(model_id)
        )
    """)

    # Match history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS match_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_a TEXT,
            model_b TEXT,
            board_type TEXT,
            num_players INTEGER,
            winner TEXT,
            game_length INTEGER,
            duration_sec REAL,
            timestamp REAL,
            tournament_id TEXT
        )
    """)

    # Rating history table (for tracking Elo over time)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rating_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT,
            rating REAL,
            games_played INTEGER,
            timestamp REAL,
            tournament_id TEXT
        )
    """)

    conn.commit()
    return conn


def discover_models(
    models_dir: Path,
    board_type: str = "square8",
    num_players: int = 2,
) -> List[Dict[str, Any]]:
    """Discover all trained models for a given board type."""
    models = []

    # Look for .pth files matching the board/player config
    pattern = f"{board_type.replace('square', 'sq')}_{num_players}p"

    for f in models_dir.glob("*.pth"):
        name = f.stem

        # Extract version info
        version = "unknown"
        if "ringrift_v5" in name:
            version = "v5"
        elif "ringrift_v4" in name:
            version = "v4"
        elif "ringrift_v3" in name:
            version = "v3"
        elif "nn_baseline" in name:
            version = "baseline"

        # Check if it matches the board/player pattern
        if pattern in name or "ringrift_v" in name:
            models.append({
                "model_id": name,
                "model_path": str(f),
                "board_type": board_type,
                "num_players": num_players,
                "version": version,
                "size_mb": f.stat().st_size / (1024 * 1024),
                "created_at": f.stat().st_mtime,
            })

    return sorted(models, key=lambda x: x["created_at"], reverse=True)


def register_models(conn: sqlite3.Connection, models: List[Dict[str, Any]]):
    """Register discovered models in the database."""
    cursor = conn.cursor()
    now = time.time()

    for m in models:
        # Insert or update model
        cursor.execute("""
            INSERT INTO models (model_id, model_path, board_type, num_players, model_version, created_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET last_seen = ?
        """, (
            m["model_id"], m["model_path"], m["board_type"], m["num_players"],
            m["version"], m["created_at"], now, now
        ))

        # Initialize Elo rating if not exists
        cursor.execute("""
            INSERT OR IGNORE INTO elo_ratings (model_id, board_type, num_players, rating, games_played, last_update)
            VALUES (?, ?, ?, 1500.0, 0, ?)
        """, (m["model_id"], m["board_type"], m["num_players"], now))

    conn.commit()


def get_leaderboard(
    conn: sqlite3.Connection,
    board_type: str = None,
    num_players: int = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get current Elo leaderboard."""
    cursor = conn.cursor()

    query = """
        SELECT e.model_id, e.board_type, e.num_players, e.rating, e.games_played,
               e.wins, e.losses, e.draws, e.last_update, m.model_version
        FROM elo_ratings e
        JOIN models m ON e.model_id = m.model_id
        WHERE 1=1
    """
    params = []

    if board_type:
        query += " AND e.board_type = ?"
        params.append(board_type)
    if num_players:
        query += " AND e.num_players = ?"
        params.append(num_players)

    query += " ORDER BY e.rating DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)

    results = []
    for row in cursor.fetchall():
        games = row[4]
        wins = row[5]
        win_rate = (wins / games * 100) if games > 0 else 0

        results.append({
            "rank": len(results) + 1,
            "model_id": row[0],
            "board_type": row[1],
            "num_players": row[2],
            "rating": round(row[3], 1),
            "games_played": games,
            "wins": wins,
            "losses": row[6],
            "draws": row[7],
            "win_rate": round(win_rate, 1),
            "version": row[9],
            "last_update": datetime.fromtimestamp(row[8]).isoformat() if row[8] else None,
        })

    return results


def update_elo_after_match(
    conn: sqlite3.Connection,
    model_a: str,
    model_b: str,
    winner: str,  # model_a, model_b, or "draw"
    board_type: str,
    num_players: int,
    tournament_id: str = None,
    k_factor: float = 32.0,
):
    """Update Elo ratings after a match."""
    cursor = conn.cursor()

    # Get current ratings
    cursor.execute("SELECT rating, games_played, wins, losses, draws FROM elo_ratings WHERE model_id = ?", (model_a,))
    row_a = cursor.fetchone()
    cursor.execute("SELECT rating, games_played, wins, losses, draws FROM elo_ratings WHERE model_id = ?", (model_b,))
    row_b = cursor.fetchone()

    if not row_a or not row_b:
        print(f"Warning: Model not found in database: {model_a if not row_a else model_b}")
        return

    rating_a, games_a, wins_a, losses_a, draws_a = row_a
    rating_b, games_b, wins_b, losses_b, draws_b = row_b

    # Calculate expected scores
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a

    # Determine actual scores
    if winner == model_a:
        score_a, score_b = 1.0, 0.0
        wins_a += 1
        losses_b += 1
    elif winner == model_b:
        score_a, score_b = 0.0, 1.0
        losses_a += 1
        wins_b += 1
    else:  # draw
        score_a, score_b = 0.5, 0.5
        draws_a += 1
        draws_b += 1

    # Update ratings
    new_rating_a = rating_a + k_factor * (score_a - expected_a)
    new_rating_b = rating_b + k_factor * (score_b - expected_b)

    now = time.time()

    # Update database
    cursor.execute("""
        UPDATE elo_ratings
        SET rating = ?, games_played = games_played + 1, wins = ?, losses = ?, draws = ?, last_update = ?
        WHERE model_id = ?
    """, (new_rating_a, wins_a, losses_a, draws_a, now, model_a))

    cursor.execute("""
        UPDATE elo_ratings
        SET rating = ?, games_played = games_played + 1, wins = ?, losses = ?, draws = ?, last_update = ?
        WHERE model_id = ?
    """, (new_rating_b, wins_b, losses_b, draws_b, now, model_b))

    # Record rating history
    cursor.execute("""
        INSERT INTO rating_history (model_id, rating, games_played, timestamp, tournament_id)
        VALUES (?, ?, ?, ?, ?)
    """, (model_a, new_rating_a, games_a + 1, now, tournament_id))

    cursor.execute("""
        INSERT INTO rating_history (model_id, rating, games_played, timestamp, tournament_id)
        VALUES (?, ?, ?, ?, ?)
    """, (model_b, new_rating_b, games_b + 1, now, tournament_id))

    conn.commit()


def print_leaderboard(leaderboard: List[Dict[str, Any]], title: str = "Elo Leaderboard"):
    """Pretty print the leaderboard."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

    if not leaderboard:
        print("  No models found in leaderboard.")
        return

    print(f"{'Rank':<6}{'Model':<50}{'Elo':>8}{'Games':>8}{'Win%':>8}{'Ver':>8}")
    print("-" * 88)

    for entry in leaderboard:
        model_short = entry["model_id"][:48] if len(entry["model_id"]) > 48 else entry["model_id"]
        print(f"{entry['rank']:<6}{model_short:<50}{entry['rating']:>8.1f}{entry['games_played']:>8}{entry['win_rate']:>7.1f}%{entry['version']:>8}")

    print(f"\nTotal models: {len(leaderboard)}")


def main():
    parser = argparse.ArgumentParser(description="Run model Elo tournament")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--games", type=int, default=10, help="Games per matchup")
    parser.add_argument("--top-n", type=int, help="Only include top N models by recency")
    parser.add_argument("--leaderboard-only", action="store_true", help="Just show leaderboard")
    parser.add_argument("--run", action="store_true", help="Actually run games (otherwise just shows plan)")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--db", type=str, help="Path to Elo database")

    args = parser.parse_args()

    db_path = Path(args.db) if args.db else ELO_DB_PATH
    conn = init_elo_database(db_path)

    # Discover models
    models_dir = AI_SERVICE_ROOT / "models"
    models = discover_models(models_dir, args.board, args.players)

    print(f"\nDiscovered {len(models)} models for {args.board} {args.players}p")

    if args.top_n:
        models = models[:args.top_n]
        print(f"Using top {args.top_n} most recent models")

    # Register models
    register_models(conn, models)

    # Show leaderboard
    leaderboard = get_leaderboard(conn, args.board, args.players)
    print_leaderboard(leaderboard, f"Current Elo Leaderboard - {args.board} {args.players}p")

    if args.leaderboard_only:
        conn.close()
        return

    if len(models) < 2:
        print("\nNeed at least 2 models to run a tournament!")
        conn.close()
        return

    # Generate matchups
    matchups = []
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            matchups.append((m1, m2))

    print(f"\n{'='*80}")
    print(f" Tournament Plan")
    print(f"{'='*80}")
    print(f"Models: {len(models)}")
    print(f"Matchups: {len(matchups)}")
    print(f"Games per matchup: {args.games}")
    print(f"Total games needed: {len(matchups) * args.games}")

    # Check if --run flag provided
    if not args.run:
        print("\nSample matchups:")
        for m1, m2 in matchups[:5]:
            print(f"  {m1['model_id'][:40]} vs {m2['model_id'][:40]}")

        if len(matchups) > 5:
            print(f"  ... and {len(matchups) - 5} more")

        print("\nAdd --run flag to execute games and update Elo ratings.")
        conn.close()
        return

    # Run the tournament
    import uuid
    tournament_id = str(uuid.uuid4())[:8]

    print(f"\n{'='*80}")
    print(f" Running Tournament {tournament_id}")
    print(f"{'='*80}")

    total_games = len(matchups) * args.games
    games_completed = 0
    start_time = time.time()

    for matchup_idx, (m1, m2) in enumerate(matchups):
        print(f"\nMatchup {matchup_idx + 1}/{len(matchups)}: {m1['model_id'][:35]} vs {m2['model_id'][:35]}")

        try:
            results = run_model_matchup(
                conn=conn,
                model_a=m1,
                model_b=m2,
                board_type=args.board,
                num_players=args.players,
                games=args.games,
                tournament_id=tournament_id,
            )

            games_completed += args.games
            elapsed = time.time() - start_time
            rate = games_completed / elapsed if elapsed > 0 else 0

            print(f"  Results: A={results['model_a_wins']} B={results['model_b_wins']} D={results['draws']} E={results['errors']}")
            print(f"  Progress: {games_completed}/{total_games} games ({rate:.1f} games/sec)")

        except Exception as e:
            print(f"  Error in matchup: {e}")
            continue

    # Show final leaderboard
    final_leaderboard = get_leaderboard(conn, args.board, args.players, limit=100)
    print_leaderboard(final_leaderboard, f"Final Elo Leaderboard - {args.board} {args.players}p (Tournament {tournament_id})")

    # Summary
    elapsed = time.time() - start_time
    print(f"\nTournament completed in {elapsed:.1f} seconds")
    print(f"Total games played: {games_completed}")

    conn.close()


if __name__ == "__main__":
    main()
