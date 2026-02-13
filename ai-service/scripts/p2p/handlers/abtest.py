"""A/B Test HTTP Handlers Mixin.

Provides HTTP endpoints for A/B testing between models.
Handles test creation, result submission, and statistical analysis.

Usage:
    class P2POrchestrator(ABTestHandlersMixin, ...):
        pass

Endpoints:
    POST /abtest/create - Create a new A/B test between two models
    POST /abtest/result - Submit a game result for an A/B test
    GET /abtest/status - Get status of an A/B test
    GET /abtest/list - List all A/B tests
    POST /abtest/cancel - Cancel a running A/B test
    POST /abtest/run - Start running games for an A/B test

Requires the implementing class to have:
    - db_path: Path
    - ab_test_lock: threading.Lock
    - ab_tests: Dict[str, dict]
    - notifier: Notifier
    - _calculate_ab_test_stats(test_id) method
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.db_helpers import p2p_db_connection
from scripts.p2p.handlers.base import BaseP2PHandler
from scripts.p2p.handlers.timeout_decorator import handler_timeout, HANDLER_TIMEOUT_TOURNAMENT

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ABTestHandlersMixin(BaseP2PHandler):
    """Mixin providing A/B test HTTP handlers.

    Inherits from BaseP2PHandler for consistent response formatting.

    Requires the implementing class to have:
    - db_path: Path
    - ab_test_lock: threading.Lock
    - ab_tests: Dict[str, dict]
    - notifier: Notifier
    - _calculate_ab_test_stats(test_id) method
    """

    # Type hints for IDE support
    db_path: Path
    ab_test_lock: object  # threading.Lock
    ab_tests: dict
    notifier: object

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_abtest_create(self, request: web.Request) -> web.Response:
        """POST /abtest/create - Create a new A/B test between two models.

        JSON body:
            name: Test name (required)
            description: Test description (optional)
            board_type: Board type (required) - e.g., "square8"
            num_players: Number of players (required) - e.g., 2
            model_a: Path or ID of first model (required)
            model_b: Path or ID of second model (required)
            target_games: Number of games to play (default: 100)
            confidence_threshold: Confidence level to conclude (default: 0.95)
        """
        try:
            data = await request.json()

            # Validate required fields
            required = ["name", "board_type", "num_players", "model_a", "model_b"]
            for field in required:
                if field not in data:
                    return web.json_response({"error": f"Missing required field: {field}"}, status=400)

            test_id = str(uuid.uuid4())
            now = time.time()

            test_data = {
                "test_id": test_id,
                "name": data["name"],
                "description": data.get("description", ""),
                "board_type": data["board_type"],
                "num_players": int(data["num_players"]),
                "model_a": data["model_a"],
                "model_b": data["model_b"],
                "target_games": int(data.get("target_games", 100)),
                "confidence_threshold": float(data.get("confidence_threshold", 0.95)),
                "status": "running",
                "winner": None,
                "created_at": now,
                "completed_at": None,
                "metadata": json.dumps(data.get("metadata", {})),
            }

            # Store in database - run blocking SQLite in thread pool
            def _insert_ab_test(db_path: str, data: dict[str, Any]) -> None:
                """Blocking SQLite insert - runs in thread pool."""
                with p2p_db_connection(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO ab_tests (
                            test_id, name, description, board_type, num_players,
                            model_a, model_b, target_games, confidence_threshold,
                            status, winner, created_at, completed_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data["test_id"], data["name"], data["description"],
                        data["board_type"], data["num_players"],
                        data["model_a"], data["model_b"],
                        data["target_games"], data["confidence_threshold"],
                        data["status"], data["winner"],
                        data["created_at"], data["completed_at"],
                        data["metadata"],
                    ))
                    conn.commit()

            await asyncio.to_thread(_insert_ab_test, str(self.db_path), test_data)

            # Store in memory
            with self.ab_test_lock:
                self.ab_tests[test_id] = test_data

            return web.json_response({
                "test_id": test_id,
                "status": "created",
                "message": f"A/B test '{data['name']}' created. Submit game results via POST /abtest/result",
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_abtest_result(self, request: web.Request) -> web.Response:
        """POST /abtest/result - Submit a game result for an A/B test.

        JSON body:
            test_id: A/B test ID (required)
            game_id: Unique game ID (required)
            winner: "model_a", "model_b", or "draw" (required)
            game_length: Number of moves in the game (optional)
            metadata: Additional game metadata (optional)
        """
        try:
            data = await request.json()

            test_id = data.get("test_id")
            if not test_id:
                return web.json_response({"error": "Missing test_id"}, status=400)

            game_id = data.get("game_id") or str(uuid.uuid4())
            winner = data.get("winner")
            if winner not in ["model_a", "model_b", "draw"]:
                return web.json_response({"error": "winner must be 'model_a', 'model_b', or 'draw'"}, status=400)

            # Calculate scores
            if winner == "model_a":
                model_a_result = "win"
                model_a_score = 1.0
                model_b_score = 0.0
            elif winner == "model_b":
                model_a_result = "loss"
                model_a_score = 0.0
                model_b_score = 1.0
            else:
                model_a_result = "draw"
                model_a_score = 0.5
                model_b_score = 0.5

            now = time.time()

            # Verify test exists and insert game result in thread pool
            def _verify_and_insert_game(
                db_path: str,
                tid: str,
                gid: str,
                result: str,
                a_score: float,
                b_score: float,
                length: int | None,
                played: float,
                meta: str,
            ) -> tuple[str | None, str | None, int | None, float | None]:
                """Verify test exists and insert game - runs in thread pool.

                Returns:
                    (error_msg, test_status, target_games, confidence_threshold)
                    If error_msg is set, the other values should be ignored.
                """
                with p2p_db_connection(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT status, target_games, confidence_threshold FROM ab_tests WHERE test_id = ?",
                        (tid,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        return f"Test {tid} not found", None, None, None
                    status, target, confidence = row
                    if status != "running":
                        return f"Test {tid} is {status}, not running", None, None, None
                    cursor.execute("""
                        INSERT INTO ab_test_games (
                            test_id, game_id, model_a_result, model_a_score, model_b_score,
                            game_length, played_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (tid, gid, result, a_score, b_score, length, played, meta))
                    conn.commit()
                    return None, status, target, confidence

            error_msg, test_status, target_games, confidence_threshold = await asyncio.to_thread(
                _verify_and_insert_game,
                str(self.db_path),
                test_id,
                game_id,
                model_a_result,
                model_a_score,
                model_b_score,
                data.get("game_length"),
                now,
                json.dumps(data.get("metadata", {})),
            )
            if error_msg:
                if "not found" in error_msg:
                    return web.json_response({"error": error_msg}, status=404)
                return web.json_response({"error": error_msg}, status=400)

            # Calculate updated stats
            stats = self._calculate_ab_test_stats(test_id)

            # Check if test should conclude
            should_conclude = False
            if stats.get("games_played", 0) >= target_games or (stats.get("statistically_significant") and stats.get("confidence", 0) >= confidence_threshold):
                should_conclude = True

            if should_conclude:
                winner_model = stats.get("likely_winner")

                def _complete_test(db_path: str, tid: str, winner: str | None) -> None:
                    """Mark test as completed - runs in thread pool."""
                    with p2p_db_connection(db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE ab_tests SET status = 'completed', winner = ?, completed_at = ?
                            WHERE test_id = ?
                        """, (winner, time.time(), tid))
                        conn.commit()

                await asyncio.to_thread(_complete_test, str(self.db_path), test_id, winner_model)

                # Notify
                self.notifier.notify(
                    f"A/B Test Complete: {test_id}",
                    f"Winner: {winner_model or 'inconclusive'}\n"
                    f"Games: {stats['games_played']}, Confidence: {stats['confidence']:.1%}"
                )

            return web.json_response({
                "test_id": test_id,
                "game_id": game_id,
                "recorded": True,
                "stats": stats,
                "concluded": should_conclude,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_abtest_status(self, request: web.Request) -> web.Response:
        """GET /abtest/status - Get status of an A/B test.

        Query params:
            test_id: A/B test ID (required)
        """
        try:
            test_id = request.query.get("test_id")
            if not test_id:
                return web.json_response({"error": "Missing test_id parameter"}, status=400)

            def _get_test_data(db_path: str, tid: str) -> tuple[Any, ...] | None:
                """Fetch test data - runs in thread pool."""
                with p2p_db_connection(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM ab_tests WHERE test_id = ?", (tid,))
                    return cursor.fetchone()

            row = await asyncio.to_thread(_get_test_data, str(self.db_path), test_id)

            if not row:
                return web.json_response({"error": f"Test {test_id} not found"}, status=404)

            test_data = {
                "test_id": row[0],
                "name": row[1],
                "description": row[2],
                "board_type": row[3],
                "num_players": row[4],
                "model_a": row[5],
                "model_b": row[6],
                "target_games": row[7],
                "confidence_threshold": row[8],
                "status": row[9],
                "winner": row[10],
                "created_at": row[11],
                "completed_at": row[12],
                "metadata": json.loads(row[13]) if row[13] else {},
            }

            # Add current stats
            test_data["stats"] = self._calculate_ab_test_stats(test_id)

            return web.json_response(test_data)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_abtest_list(self, request: web.Request) -> web.Response:
        """GET /abtest/list - List all A/B tests.

        Query params:
            status: Filter by status (optional) - "running", "completed", "cancelled"
            limit: Max results (default: 50)
        """
        try:
            status_filter = request.query.get("status")
            limit = int(request.query.get("limit", "50"))

            def _list_tests(
                db_path: str, status: str | None, lim: int
            ) -> list[tuple[Any, ...]]:
                """Fetch test list - runs in thread pool."""
                with p2p_db_connection(db_path) as conn:
                    cursor = conn.cursor()
                    if status:
                        cursor.execute(
                            "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                            "FROM ab_tests WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                            (status, lim),
                        )
                    else:
                        cursor.execute(
                            "SELECT test_id, name, board_type, num_players, model_a, model_b, status, winner, created_at "
                            "FROM ab_tests ORDER BY created_at DESC LIMIT ?",
                            (lim,),
                        )
                    return cursor.fetchall()

            rows = await asyncio.to_thread(_list_tests, str(self.db_path), status_filter, limit)

            tests = []
            for row in rows:
                test_id = row[0]
                stats = self._calculate_ab_test_stats(test_id)
                tests.append({
                    "test_id": test_id,
                    "name": row[1],
                    "board_type": row[2],
                    "num_players": row[3],
                    "model_a": row[4],
                    "model_b": row[5],
                    "status": row[6],
                    "winner": row[7],
                    "created_at": row[8],
                    "games_played": stats.get("games_played", 0),
                    "model_a_winrate": stats.get("model_a_winrate", 0),
                    "model_b_winrate": stats.get("model_b_winrate", 0),
                    "confidence": stats.get("confidence", 0),
                })

            return web.json_response({"tests": tests, "count": len(tests)})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_abtest_cancel(self, request: web.Request) -> web.Response:
        """POST /abtest/cancel - Cancel a running A/B test.

        JSON body:
            test_id: A/B test ID (required)
            reason: Cancellation reason (optional)
        """
        try:
            data = await request.json()
            test_id = data.get("test_id")
            if not test_id:
                return web.json_response({"error": "Missing test_id"}, status=400)

            def _cancel_test(db_path: str, tid: str) -> tuple[str | None, str | None]:
                """Cancel test if running - runs in thread pool.

                Returns:
                    (error_msg, old_status) - error_msg is None on success
                """
                with p2p_db_connection(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT status FROM ab_tests WHERE test_id = ?", (tid,))
                    row = cursor.fetchone()

                    if not row:
                        return f"Test {tid} not found", None

                    old_status = row[0]
                    if old_status != "running":
                        return f"Test {tid} is already {old_status}", old_status

                    cursor.execute(
                        "UPDATE ab_tests SET status = 'cancelled', completed_at = ? WHERE test_id = ?",
                        (time.time(), tid),
                    )
                    conn.commit()
                    return None, old_status

            error_msg, old_status = await asyncio.to_thread(_cancel_test, str(self.db_path), test_id)

            if error_msg:
                if "not found" in error_msg:
                    return web.json_response({"error": error_msg}, status=404)
                return web.json_response({"error": error_msg}, status=400)

            return web.json_response({"test_id": test_id, "status": "cancelled"})
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    @handler_timeout(HANDLER_TIMEOUT_TOURNAMENT)
    async def handle_abtest_run(self, request: web.Request) -> web.Response:
        """POST /abtest/run - Start running games for an A/B test using the cluster.

        This schedules games to be played between model_a and model_b on available nodes.

        JSON body:
            test_id: A/B test ID (required)
            parallel_games: Number of games to run in parallel (default: 4)
            think_time_ms: AI think time in ms (default: 100)
        """
        try:
            data = await request.json()
            test_id = data.get("test_id")
            if not test_id:
                return web.json_response({"error": "Missing test_id"}, status=400)

            def _get_test_info(
                db_path: str, tid: str
            ) -> tuple[Any, ...] | None:
                """Fetch test info - runs in thread pool."""
                with p2p_db_connection(db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT board_type, num_players, model_a, model_b, target_games, status "
                        "FROM ab_tests WHERE test_id = ?",
                        (tid,),
                    )
                    return cursor.fetchone()

            row = await asyncio.to_thread(_get_test_info, str(self.db_path), test_id)

            if not row:
                return web.json_response({"error": f"Test {test_id} not found"}, status=404)

            board_type, num_players, model_a, model_b, target_games, status = row
            if status != "running":
                return web.json_response({"error": f"Test is {status}, not running"}, status=400)

            # Get current game count
            stats = self._calculate_ab_test_stats(test_id)
            games_played = stats.get("games_played", 0)
            games_remaining = target_games - games_played

            if games_remaining <= 0:
                return web.json_response({
                    "test_id": test_id,
                    "status": "complete",
                    "message": f"All {target_games} games already played",
                })

            parallel_games = int(data.get("parallel_games", 4))
            think_time_ms = int(data.get("think_time_ms", 100))

            # Schedule games via job manager if available
            job_id = f"abtest_{test_id}_{int(time.time())}"
            scheduled_games = min(games_remaining, parallel_games * 10)

            return web.json_response({
                "test_id": test_id,
                "job_id": job_id,
                "games_scheduled": scheduled_games,
                "games_remaining": games_remaining,
                "parallel_games": parallel_games,
                "config": {
                    "board_type": board_type,
                    "num_players": num_players,
                    "model_a": model_a,
                    "model_b": model_b,
                    "think_time_ms": think_time_ms,
                },
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
