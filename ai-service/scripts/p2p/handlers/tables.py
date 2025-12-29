"""Table Handlers Mixin for P2P Orchestrator.

December 2025 - Phase 8 decomposition.

Provides HTTP handlers for table-format endpoints used by Grafana Infinity.
These endpoints return flat JSON arrays suitable for table visualization.

Table Endpoints:
- GET /elo/table - Elo leaderboard
- GET /nodes/table - Node status table
- GET /holdout/table - Holdout validation metrics
- GET /mcts/table - MCTS search statistics
- GET /matchups/table - Head-to-head matchups
- GET /models/lineage/table - Model ancestry
- GET /data/quality/table - Data quality metrics
- GET /training/efficiency/table - Training efficiency
- GET /trends/table - Performance trends
- GET /abtest/table - A/B test status
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from .base import BaseP2PHandler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TableHandlersMixin(BaseP2PHandler):
    """Mixin providing table-format HTTP handlers for Grafana visualization.

    All table handlers:
    - Return JSON arrays suitable for Grafana Infinity data source
    - Include error handling with error object in array
    - Use cached data fetch methods where available
    """

    # =========================================================================
    # Elo Table
    # =========================================================================

    async def handle_elo_table(self, request: web.Request) -> web.Response:
        """GET /elo/table - Elo leaderboard in flat table format for Grafana Infinity.

        Query params:
            - source: "tournament" (default) or "trained" (actual trained NN models)
            - limit: Max entries (default 50)
            - board_type: Filter by board type
            - num_players: Filter by player count
            - nn_only: If "true", filter to NN models only (for tournament source)

        Returns a simple JSON array of model entries with rank, suitable for table display.
        """
        try:
            source = request.query.get("source", "tournament")
            limit = int(request.query.get("limit", "50"))
            board_type_filter = request.query.get("board_type")
            num_players_filter = request.query.get("num_players")
            nn_only = request.query.get("nn_only", "").lower() == "true"

            ai_root = Path(self.ringrift_path) / "ai-service"

            if source == "trained":
                return await self._handle_elo_table_trained(
                    ai_root, limit, nn_only
                )
            else:
                return await self._handle_elo_table_tournament(
                    limit, board_type_filter, num_players_filter, nn_only
                )

        except ImportError:
            return web.json_response([{"error": "Elo database module not available"}])
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    async def _handle_elo_table_trained(
        self,
        ai_root: Path,
        limit: int,
        nn_only: bool,
    ) -> web.Response:
        """Handle trained source for Elo table."""
        db_path = ai_root / "data" / "unified_elo.db"
        if not db_path.exists():
            return web.json_response([])

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
            SELECT model_id, rating, games_played, wins, losses
            FROM elo_ratings
            WHERE games_played >= 10
        """
        params: list[Any] = []

        if nn_only:
            query += " AND (model_id LIKE '%nn%' OR model_id LIKE '%NN%' OR model_id LIKE '%baseline%')"

        query += " ORDER BY rating DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        table_data = []
        for rank, row in enumerate(rows, 1):
            model_id, rating, games, wins, losses = row

            # Extract config from model name
            if "sq8" in model_id.lower() or "square8" in model_id.lower():
                config = "square8_2p"
            elif "sq19" in model_id.lower() or "square19" in model_id.lower():
                config = "square19_2p"
            elif "hex" in model_id.lower():
                config = "hexagonal_2p"
            else:
                config = "unknown"

            # Calculate win rate
            total_decided = wins + losses
            win_rate = wins / total_decided if total_decided > 0 else 0.5

            table_data.append({
                "Rank": rank,
                "Model": model_id,
                "Elo": round(rating, 1),
                "WinRate": round(win_rate * 100, 1),
                "Games": games,
                "Wins": wins,
                "Losses": losses,
                "Draws": 0,
                "Config": config,
            })

        return web.json_response(table_data)

    async def _handle_elo_table_tournament(
        self,
        limit: int,
        board_type_filter: str | None,
        num_players_filter: str | None,
        nn_only: bool,
    ) -> web.Response:
        """Handle tournament source for Elo table."""
        from scripts.run_model_elo_tournament import (
            ELO_DB_PATH,
            init_elo_database,
        )

        if not ELO_DB_PATH or not ELO_DB_PATH.exists():
            return web.json_response([])

        db = init_elo_database()
        conn = db._get_connection()
        cursor = conn.cursor()

        id_col = db.id_column

        query = f"""
            SELECT
                {id_col},
                board_type,
                num_players,
                rating,
                games_played,
                wins,
                losses,
                draws,
                last_update
            FROM elo_ratings
            WHERE games_played >= 5
        """
        params: list[Any] = []

        if board_type_filter:
            query += " AND board_type = ?"
            params.append(board_type_filter)

        if num_players_filter:
            query += " AND num_players = ?"
            params.append(int(num_players_filter))

        if nn_only:
            query += f" AND ({id_col} LIKE '%NN%' OR {id_col} LIKE '%nn%')"

        query += " ORDER BY rating DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        db.close()

        table_data = []
        for rank, row in enumerate(rows, 1):
            participant_id, board_type, num_players, rating, games, wins, losses, draws, _last_update = row

            # Extract model name from participant_id
            model_name = participant_id
            if participant_id.startswith("nn:"):
                model_name = Path(participant_id[3:]).stem

            # Calculate win rate
            total_decided = wins + losses
            win_rate = wins / total_decided if total_decided > 0 else 0.5

            # Format config
            config = f"{board_type}_{num_players}p"

            table_data.append({
                "Rank": rank,
                "Model": model_name,
                "Elo": round(rating, 1),
                "WinRate": round(win_rate * 100, 1),
                "Games": games,
                "Wins": wins,
                "Losses": losses,
                "Draws": draws,
                "Config": config,
            })

        return web.json_response(table_data)

    # =========================================================================
    # Holdout Table
    # =========================================================================

    async def handle_holdout_table(self, request: web.Request) -> web.Response:
        """GET /holdout/table - Holdout validation data in table format for Grafana Infinity.

        Returns holdout metrics as flat table rows.
        """
        try:
            metrics = await self._get_holdout_metrics_cached()

            table_data = []
            for config, data in metrics.get("configs", {}).items():
                row = {
                    "Config": config,
                    "HoldoutGames": data.get("holdout_games", 0),
                    "HoldoutPositions": data.get("holdout_positions", 0),
                    "HoldoutLoss": round(data.get("holdout_loss", 0), 4) if data.get("holdout_loss") else None,
                    "HoldoutAccuracy": round(data.get("holdout_accuracy", 0) * 100, 1) if data.get("holdout_accuracy") else None,
                    "OverfitGap": round(data.get("overfit_gap", 0), 4) if data.get("overfit_gap") else None,
                    "Status": "OK" if (data.get("overfit_gap") or 0) < 0.15 else "OVERFITTING",
                }
                table_data.append(row)

            return web.json_response(table_data)

        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # MCTS Table
    # =========================================================================

    async def handle_mcts_table(self, request: web.Request) -> web.Response:
        """GET /mcts/table - MCTS stats in table format for Grafana Infinity.

        Returns MCTS statistics as flat table rows.
        """
        try:
            stats = await self._get_mcts_stats_cached()

            table_data = []
            # Add summary row
            summary = stats.get("summary", {})
            if summary:
                table_data.append({
                    "Config": "CLUSTER AVERAGE",
                    "AvgNodes": round(summary.get("avg_nodes_per_move", 0), 0),
                    "MaxNodes": summary.get("max_nodes_per_move", 0),
                    "AvgDepth": round(summary.get("avg_search_depth", 0), 1),
                    "MaxDepth": summary.get("max_search_depth", 0),
                    "AvgTime": round(summary.get("avg_time_per_move", 0), 3) if summary.get("avg_time_per_move") else None,
                })

            # Add per-config rows
            for config, data in stats.get("configs", {}).items():
                table_data.append({
                    "Config": config,
                    "AvgNodes": round(data.get("avg_nodes", 0), 0) if data.get("avg_nodes") else None,
                    "MaxNodes": None,
                    "AvgDepth": round(data.get("avg_depth", 0), 1) if data.get("avg_depth") else None,
                    "MaxDepth": None,
                    "AvgTime": None,
                })

            return web.json_response(table_data)

        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Matchup Table
    # =========================================================================

    async def handle_matchup_table(self, request: web.Request) -> web.Response:
        """GET /matchups/table - Matchups in table format for Grafana Infinity."""
        try:
            matrix = await self._get_matchup_matrix_cached()
            table_data = []
            for matchup in matrix.get("matchups", []):
                table_data.append({
                    "ModelA": matchup["model_a"],
                    "ModelB": matchup["model_b"],
                    "AWins": matchup["a_wins"],
                    "BWins": matchup["b_wins"],
                    "Draws": matchup["draws"],
                    "Total": matchup["total"],
                    "AWinRate": round(matchup["a_win_rate"] * 100, 1),
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Model Lineage Table
    # =========================================================================

    async def handle_model_lineage_table(self, request: web.Request) -> web.Response:
        """GET /models/lineage/table - Model lineage in table format for Grafana Infinity."""
        try:
            lineage = await self._get_model_lineage_cached()
            table_data = []
            for model in lineage.get("models", []):
                table_data.append({
                    "Name": model["name"],
                    "Config": model["config"],
                    "Generation": model["generation"],
                    "SizeMB": model["size_mb"],
                    "AgeHours": model["age_hours"],
                })
            return web.json_response(sorted(table_data, key=lambda x: (-x["Generation"], x["Config"])))
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Data Quality Table
    # =========================================================================

    async def handle_data_quality_table(self, request: web.Request) -> web.Response:
        """GET /data/quality/table - Data quality in table format for Grafana Infinity."""
        try:
            quality = await self._get_data_quality_cached()
            table_data = []
            for config, metrics in quality.get("configs", {}).items():
                status = "OK"
                for issue in quality.get("issues", []):
                    if issue["config"] == config and issue["severity"] == "warning":
                        status = "WARNING"
                        break
                table_data.append({
                    "Config": config,
                    "Games": metrics["total_games"],
                    "AvgLength": metrics["avg_length"],
                    "ShortRate": metrics["short_game_rate"],
                    "StalemateRate": metrics["stalemate_rate"],
                    "OpeningDiv": metrics["opening_diversity"],
                    "Status": status,
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Training Efficiency Table
    # =========================================================================

    async def handle_training_efficiency_table(self, request: web.Request) -> web.Response:
        """GET /training/efficiency/table - Efficiency in table format for Grafana Infinity."""
        try:
            efficiency = await self._get_training_efficiency_cached()
            table_data = []
            for config, metrics in efficiency.get("configs", {}).items():
                table_data.append({
                    "Config": config,
                    "GPUHours": metrics["gpu_hours"],
                    "EloGain": metrics["elo_gain"],
                    "EloPerHour": metrics["elo_per_gpu_hour"],
                    "CostUSD": metrics["estimated_cost_usd"],
                    "CostPerElo": metrics["cost_per_elo_point"],
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # A/B Test Table
    # =========================================================================

    async def handle_abtest_table(self, request: web.Request) -> web.Response:
        """GET /abtest/table - A/B tests in table format for Grafana Infinity.

        Query params:
            status: Filter by status (optional)
        """
        try:
            status_filter = request.query.get("status")

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            if status_filter:
                cursor.execute(
                    "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                    "FROM ab_tests WHERE status = ? ORDER BY created_at DESC LIMIT 100",
                    (status_filter,)
                )
            else:
                cursor.execute(
                    "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                    "FROM ab_tests ORDER BY created_at DESC LIMIT 100"
                )

            rows = cursor.fetchall()
            conn.close()

            table_data = []
            for row in rows:
                test_id = row[0]
                stats = self._calculate_ab_test_stats(test_id)
                from datetime import datetime
                created = datetime.fromtimestamp(row[8]).strftime("%Y-%m-%d %H:%M") if row[8] else ""

                table_data.append({
                    "Test ID": test_id[:8],
                    "Name": row[1],
                    "Config": f"{row[2]}_{row[3]}p",
                    "Model A": row[4].split("/")[-1] if "/" in row[4] else row[4],
                    "Model B": row[5].split("/")[-1] if "/" in row[5] else row[5],
                    "Games": stats.get("games_played", 0),
                    "A Win%": f"{stats.get('model_a_winrate', 0):.1%}",
                    "B Win%": f"{stats.get('model_b_winrate', 0):.1%}",
                    "Confidence": f"{stats.get('confidence', 0):.1%}",
                    "Status": row[6],
                    "Winner": row[7] or "-",
                    "Created": created,
                })

            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Autoscale Recommendations Table
    # =========================================================================

    async def handle_autoscale_recommendations(self, request: web.Request) -> web.Response:
        """GET /autoscale/recommendations - Autoscaling recommendations table."""
        try:
            metrics = await self._get_autoscaling_metrics()
            table_data = []
            for rec in metrics.get("recommendations", []):
                table_data.append({
                    "Action": rec["action"].upper(),
                    "Reason": rec["reason"],
                    "SuggestedWorkers": rec["suggested_workers"],
                })
            if not table_data:
                table_data.append({
                    "Action": "NONE",
                    "Reason": "Cluster is properly sized",
                    "SuggestedWorkers": metrics.get("current_workers", 0),
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Rollback Candidates Table
    # =========================================================================

    async def handle_rollback_candidates(self, request: web.Request) -> web.Response:
        """GET /rollback/candidates - Rollback candidates in table format."""
        try:
            status = await self._check_rollback_conditions()
            table_data = []
            for candidate in status.get("candidates", []):
                table_data.append({
                    "Config": candidate["config"],
                    "Reasons": ", ".join(candidate["reasons"]),
                    "Recommended": "YES" if candidate["rollback_recommended"] else "NO",
                })
            return web.json_response(table_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])

    # =========================================================================
    # Data Quality Issues Table
    # =========================================================================

    async def handle_data_quality_issues(self, request: web.Request) -> web.Response:
        """GET /data/quality/issues - Data quality issues in table format."""
        try:
            quality = await self._get_data_quality_cached()
            return web.json_response(quality.get("issues", []))
        except Exception as e:  # noqa: BLE001
            return web.json_response([{"error": str(e)}])
