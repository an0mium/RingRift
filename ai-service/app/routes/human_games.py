"""FastAPI router for human vs AI game export endpoints.

Provides REST API for exporting human games against AI opponents
for use in training pipelines. Human games (especially wins) provide
high-quality training signal for neural network improvement.

Created: January 2026
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.db.game_replay import GameReplayDB
from app.utils.game_discovery import GameDiscovery

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/human-games", tags=["human-games"])

# Singleton discovery instance
_discovery: GameDiscovery | None = None


def get_discovery() -> GameDiscovery:
    """Get or create the game discovery instance."""
    global _discovery
    if _discovery is None:
        _discovery = GameDiscovery()
    return _discovery


# =============================================================================
# Request/Response Models
# =============================================================================


class HumanGameMetadata(BaseModel):
    """Metadata for a human vs AI game."""

    game_id: str
    board_type: str
    num_players: int
    winner: int | None = None
    human_player: int | None = None
    human_won: bool = False
    ai_difficulty: int | None = None
    ai_type: str | None = None
    total_moves: int
    created_at: str
    completed_at: str | None = None
    eligible_for_training: bool = True


class HumanGameMoveRecord(BaseModel):
    """Move record for training export."""

    move_number: int
    player: int
    move_type: str
    from_pos: dict[str, int] | None = None
    to_pos: dict[str, int] | None = None


class HumanGameForTraining(BaseModel):
    """Complete human game for training pipeline."""

    game_id: str
    board_type: str
    num_players: int
    winner: int | None = None
    human_player: int | None = None
    human_won: bool = False
    ai_difficulty: int | None = None
    moves: list[HumanGameMoveRecord]
    training_weight: float = 1.0


class HumanGamesListResponse(BaseModel):
    """Response for human games list."""

    games: list[HumanGameMetadata]
    total: int
    has_more: bool


class HumanGamesExportResponse(BaseModel):
    """Response for human games export (for training)."""

    board_type: str
    num_players: int
    total_games: int
    human_wins: int
    human_losses: int
    games: list[HumanGameForTraining]


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_training_weight(
    human_won: bool,
    ai_difficulty: int | None,
    game_length: int,
) -> float:
    """Compute training weight for a human game.

    Human wins against stronger AI = highest value.
    Longer games = more learning signal.

    Args:
        human_won: Whether human won this game
        ai_difficulty: AI difficulty level (1-10)
        game_length: Number of moves in the game

    Returns:
        Training weight (typically 0.5 to 6.0)
    """
    base_weight = 1.0

    # Human wins are 3x more valuable (show AI weaknesses)
    if human_won:
        base_weight *= 3.0

    # Difficulty scaling: harder AI = more valuable game
    if ai_difficulty is not None:
        difficulty_mult = 1.0 + (ai_difficulty / 10.0)  # 1.1 to 2.0x
        base_weight *= difficulty_mult

    # Length bonus: longer games have more training signal
    length_mult = min(2.0, 1.0 + game_length / 50)
    base_weight *= length_mult

    return round(base_weight, 3)


def _parse_position(pos_dict: dict[str, Any] | None) -> dict[str, int] | None:
    """Parse position dict from database."""
    if pos_dict is None:
        return None
    return {"x": pos_dict.get("x", 0), "y": pos_dict.get("y", 0)}


async def _get_human_games_from_db(
    db_path: str,
    human_wins_only: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Query human vs AI games from a specific database.

    Args:
        db_path: Path to the game database
        human_wins_only: If True, only return games where human won
        limit: Maximum games to return
        offset: Offset for pagination

    Returns:
        Tuple of (games list, total count)
    """

    def _query() -> tuple[list[dict[str, Any]], int]:
        db = GameReplayDB(db_path)
        try:
            # Query games with human_vs_ai source
            games = db.list_games(
                source="human_vs_ai",
                limit=limit,
                offset=offset,
            )

            # Get total count
            total = db.count_games(source="human_vs_ai")

            # If human_wins_only, filter further
            if human_wins_only:
                games = [g for g in games if g.get("metadata", {}).get("humanWon", False)]

            return games, total
        finally:
            db.close()

    return await asyncio.to_thread(_query)


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/list", response_model=HumanGamesListResponse)
async def list_human_games(
    board_type: str = Query(..., description="Board type (e.g., hex8, square8)"),
    num_players: int = Query(..., ge=2, le=4, description="Number of players"),
    human_wins_only: bool = Query(False, description="Only return human wins"),
    limit: int = Query(50, ge=1, le=500, description="Maximum games to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> HumanGamesListResponse:
    """List human vs AI games for a specific configuration.

    Returns metadata about human games without full move data.
    Use /export for full training data.
    """
    discovery = get_discovery()
    config_key = f"{board_type}_{num_players}p"

    # Find databases for this config
    databases = discovery.find_databases_for_config(board_type, num_players)

    all_games: list[HumanGameMetadata] = []
    total_count = 0

    for db_info in databases:
        try:
            games, count = await _get_human_games_from_db(
                db_info.path,
                human_wins_only=human_wins_only,
                limit=limit - len(all_games),
                offset=max(0, offset - total_count),
            )
            total_count += count

            for g in games:
                metadata = g.get("metadata", {})
                all_games.append(
                    HumanGameMetadata(
                        game_id=g.get("game_id", ""),
                        board_type=board_type,
                        num_players=num_players,
                        winner=g.get("winner"),
                        human_player=metadata.get("humanPlayer"),
                        human_won=metadata.get("humanWon", False),
                        ai_difficulty=metadata.get("aiDifficulty"),
                        ai_type=metadata.get("aiType"),
                        total_moves=g.get("total_moves", 0),
                        created_at=g.get("created_at", ""),
                        completed_at=g.get("completed_at"),
                        eligible_for_training=metadata.get("eligibleForTraining", True),
                    )
                )

                if len(all_games) >= limit:
                    break

        except Exception as e:
            logger.warning(f"Error querying {db_info.path}: {e}")
            continue

    return HumanGamesListResponse(
        games=all_games[:limit],
        total=total_count,
        has_more=total_count > offset + limit,
    )


@router.get("/export", response_model=HumanGamesExportResponse)
async def export_human_games_for_training(
    board_type: str = Query(..., description="Board type (e.g., hex8, square8)"),
    num_players: int = Query(..., ge=2, le=4, description="Number of players"),
    human_wins_only: bool = Query(True, description="Only export human wins"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum games to export"),
    since_timestamp: str | None = Query(None, description="Only games after this ISO timestamp"),
) -> HumanGamesExportResponse:
    """Export human vs AI games in training format.

    Returns full game data with moves for training pipeline consumption.
    By default, only exports human wins (higher training value).
    """
    discovery = get_discovery()

    # Find databases for this config
    databases = discovery.find_databases_for_config(board_type, num_players)

    all_games: list[HumanGameForTraining] = []
    human_wins = 0
    human_losses = 0

    for db_info in databases:
        try:

            def _export_games() -> list[dict[str, Any]]:
                db = GameReplayDB(db_info.path)
                try:
                    games = db.list_games(
                        source="human_vs_ai",
                        limit=limit - len(all_games),
                    )

                    result = []
                    for g in games:
                        metadata = g.get("metadata", {})
                        is_human_win = metadata.get("humanWon", False)

                        # Skip losses if human_wins_only
                        if human_wins_only and not is_human_win:
                            continue

                        # Get full game with moves
                        game_id = g.get("game_id")
                        if game_id:
                            full_game = db.get_game(game_id)
                            if full_game:
                                result.append(
                                    {
                                        "game": g,
                                        "moves": full_game.get("moves", []),
                                        "is_human_win": is_human_win,
                                    }
                                )

                    return result
                finally:
                    db.close()

            games_data = await asyncio.to_thread(_export_games)

            for data in games_data:
                g = data["game"]
                moves = data["moves"]
                is_win = data["is_human_win"]
                metadata = g.get("metadata", {})

                if is_win:
                    human_wins += 1
                else:
                    human_losses += 1

                # Convert moves to training format
                move_records = [
                    HumanGameMoveRecord(
                        move_number=m.get("moveNumber", i + 1),
                        player=m.get("player", 1),
                        move_type=m.get("moveType", "unknown"),
                        from_pos=_parse_position(m.get("from")),
                        to_pos=_parse_position(m.get("to")),
                    )
                    for i, m in enumerate(moves)
                ]

                training_weight = _compute_training_weight(
                    human_won=is_win,
                    ai_difficulty=metadata.get("aiDifficulty"),
                    game_length=len(moves),
                )

                all_games.append(
                    HumanGameForTraining(
                        game_id=g.get("game_id", ""),
                        board_type=board_type,
                        num_players=num_players,
                        winner=g.get("winner"),
                        human_player=metadata.get("humanPlayer"),
                        human_won=is_win,
                        ai_difficulty=metadata.get("aiDifficulty"),
                        moves=move_records,
                        training_weight=training_weight,
                    )
                )

                if len(all_games) >= limit:
                    break

        except Exception as e:
            logger.warning(f"Error exporting from {db_info.path}: {e}")
            continue

    return HumanGamesExportResponse(
        board_type=board_type,
        num_players=num_players,
        total_games=len(all_games),
        human_wins=human_wins,
        human_losses=human_losses,
        games=all_games,
    )


@router.get("/stats")
async def get_human_games_stats(
    board_type: str | None = Query(None, description="Filter by board type"),
    num_players: int | None = Query(None, ge=2, le=4, description="Filter by player count"),
) -> dict[str, Any]:
    """Get statistics about human vs AI games.

    Returns counts and summary statistics for human games
    across all configurations or filtered by board type/players.
    """
    discovery = get_discovery()

    # Find relevant databases
    if board_type and num_players:
        databases = discovery.find_databases_for_config(board_type, num_players)
    else:
        databases = discovery.find_all_databases()

    stats: dict[str, Any] = {
        "total_games": 0,
        "human_wins": 0,
        "human_losses": 0,
        "by_config": {},
    }

    for db_info in databases:
        try:

            def _get_stats() -> dict[str, Any]:
                db = GameReplayDB(db_info.path)
                try:
                    games = db.list_games(source="human_vs_ai", limit=10000)
                    wins = sum(
                        1 for g in games if g.get("metadata", {}).get("humanWon", False)
                    )
                    return {"total": len(games), "wins": wins, "losses": len(games) - wins}
                finally:
                    db.close()

            db_stats = await asyncio.to_thread(_get_stats)
            config_key = f"{db_info.board_type}_{db_info.num_players}p"

            if config_key not in stats["by_config"]:
                stats["by_config"][config_key] = {
                    "total_games": 0,
                    "human_wins": 0,
                    "human_losses": 0,
                }

            stats["total_games"] += db_stats["total"]
            stats["human_wins"] += db_stats["wins"]
            stats["human_losses"] += db_stats["losses"]
            stats["by_config"][config_key]["total_games"] += db_stats["total"]
            stats["by_config"][config_key]["human_wins"] += db_stats["wins"]
            stats["by_config"][config_key]["human_losses"] += db_stats["losses"]

        except Exception as e:
            logger.warning(f"Error getting stats from {db_info.path}: {e}")
            continue

    # Calculate win rate
    if stats["total_games"] > 0:
        stats["human_win_rate"] = round(stats["human_wins"] / stats["total_games"], 3)
    else:
        stats["human_win_rate"] = 0.0

    return stats
