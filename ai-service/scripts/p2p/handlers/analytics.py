"""P2P Analytics HTTP handlers.

January 2026: Extracted from p2p_orchestrator.py to reduce file size (~298 LOC).

Endpoints:
- GET /holdout/metrics - Holdout validation metrics
- GET /mcts/stats - MCTS search statistics
- GET /matchups/matrix - Head-to-head matchup statistics
- GET /models/lineage - Model ancestry and generation tracking
- GET /data/quality - Data quality metrics and issue detection
- GET /training/efficiency - Training efficiency and cost metrics
- GET /autoscale/metrics - Autoscaling metrics and recommendations
- GET /resource/history - Resource utilization history for graphing
- POST /webhook/test - Test webhook notification
- GET /trends/summary - Get summary of metrics over time period
- GET /trends/history - Get historical metrics data
- GET /training/status - Training pipeline status

The handler accesses orchestrator state via `self.*` since it's designed
as a mixin that gets inherited by P2POrchestrator.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Lazy import for coordination modules
HAS_NEW_COORDINATION = True
try:
    from app.coordination.resource_optimizer import get_resource_optimizer
except ImportError:
    HAS_NEW_COORDINATION = False

    def get_resource_optimizer():
        return None


class AnalyticsHandlersMixin:
    """Mixin providing analytics HTTP handlers.

    Must be mixed into a class that provides:
    - self.analytics_cache_manager (AnalyticsCacheManager instance)
    - self.get_metrics_summary()
    - self.get_metrics_history()
    - self.notifier
    - self.node_id
    - self._get_ai_service_path()

    January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
    January 2026: Updated to use analytics_cache_manager directly (-30 LOC from orchestrator).
    """

    async def handle_holdout_metrics(self, request: web.Request) -> web.Response:
        """GET /holdout/metrics - Holdout validation metrics.

        Returns holdout set statistics and evaluation results for overfitting detection.
        Supports optional query params:
            - config: Filter by config (e.g., square8_2p)

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            config_filter = request.query.get("config")
            metrics = await self.analytics_cache_manager.get_holdout_metrics_cached()

            if config_filter:
                # Filter to specific config
                filtered = {
                    "configs": {k: v for k, v in metrics.get("configs", {}).items() if k == config_filter},
                    "evaluations": [e for e in metrics.get("evaluations", []) if e.get("config") == config_filter],
                    "summary": metrics.get("summary", {}),
                }
                return web.json_response(filtered)

            return web.json_response(metrics)

        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})

    async def handle_mcts_stats(self, request: web.Request) -> web.Response:
        """GET /mcts/stats - MCTS search statistics.

        Returns MCTS performance metrics including nodes/move, search depth, and timing.

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            stats = await self.analytics_cache_manager.get_mcts_stats_cached()
            return web.json_response(stats)

        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})

    async def handle_matchup_matrix(self, request: web.Request) -> web.Response:
        """GET /matchups/matrix - Head-to-head matchup statistics.

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            matrix = await self.analytics_cache_manager.get_matchup_matrix_cached()
            return web.json_response(matrix)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})

    async def handle_model_lineage(self, request: web.Request) -> web.Response:
        """GET /models/lineage - Model ancestry and generation tracking.

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            lineage = await self.analytics_cache_manager.get_model_lineage_cached()
            return web.json_response(lineage)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})

    async def handle_data_quality(self, request: web.Request) -> web.Response:
        """GET /data/quality - Data quality metrics and issue detection.

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            quality = await self.analytics_cache_manager.get_data_quality_cached()
            return web.json_response(quality)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})

    async def handle_training_efficiency(self, request: web.Request) -> web.Response:
        """GET /training/efficiency - Training efficiency and cost metrics.

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            efficiency = await self.analytics_cache_manager.get_training_efficiency_cached()
            return web.json_response(efficiency)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})

    async def handle_autoscale_metrics(self, request: web.Request) -> web.Response:
        """GET /autoscale/metrics - Autoscaling metrics and recommendations.

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            metrics = await self.analytics_cache_manager.get_autoscaling_metrics()
            return web.json_response(metrics)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)})

    async def handle_resource_utilization_history(self, request: web.Request) -> web.Response:
        """GET /resource/history - Resource utilization history for graphing.

        Query params:
            node_id: Specific node (optional, defaults to cluster average)
            hours: Hours of history (default: 1)

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            if not HAS_NEW_COORDINATION:
                return web.json_response([])

            node_id = request.query.get("node_id")
            hours = float(request.query.get("hours", "1"))

            optimizer = get_resource_optimizer()
            history = optimizer.get_utilization_history(node_id=node_id, hours=hours)
            return web.json_response(history)
        except (ValueError, AttributeError):
            return web.json_response([])

    async def handle_webhook_test(self, request: web.Request) -> web.Response:
        """POST /webhook/test - Test webhook notification.

        Query params:
            level: debug/info/warning/error (default: info)
            message: Custom message (default: "Test notification")

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            level = request.query.get("level", "info")
            message = request.query.get("message", "Test notification from RingRift AI orchestrator")

            has_slack = bool(self.notifier.slack_webhook)
            has_discord = bool(self.notifier.discord_webhook)

            if not has_slack and not has_discord:
                return web.json_response({
                    "success": False,
                    "message": "No webhooks configured. Set RINGRIFT_SLACK_WEBHOOK and/or RINGRIFT_DISCORD_WEBHOOK",
                })

            await self.notifier.send(
                title="Webhook Test",
                message=message,
                level=level,
                fields={
                    "Node": self.node_id,
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Level": level.upper(),
                },
                node_id=self.node_id,
            )

            return web.json_response({
                "success": True,
                "message": f"Test notification sent to {'Slack' if has_slack else ''}{' and ' if has_slack and has_discord else ''}{'Discord' if has_discord else ''}",
                "level": level,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_trends_summary(self, request: web.Request) -> web.Response:
        """GET /trends/summary - Get summary of metrics over time period.

        Query params:
            hours: Time period in hours (default: 24)

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            hours = float(request.query.get("hours", "24"))
            # Wrap blocking SQLite call to avoid blocking event loop
            summary = await asyncio.to_thread(self.get_metrics_summary, hours)
            return web.json_response(summary)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_trends_history(self, request: web.Request) -> web.Response:
        """GET /trends/history - Get historical metrics data.

        Query params:
            metric: Metric type (required) - e.g., "best_elo", "games_generated", "training_loss"
            hours: Time period in hours (default: 24)
            board: Board type filter (optional) - e.g., "square8"
            players: Number of players filter (optional) - e.g., 2
            limit: Max records to return (default: 1000)

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            metric_type = request.query.get("metric")
            if not metric_type:
                return web.json_response({"error": "Missing required parameter: metric"}, status=400)

            hours = float(request.query.get("hours", "24"))
            board_type = request.query.get("board")
            num_players = int(request.query.get("players")) if request.query.get("players") else None
            limit = int(request.query.get("limit", "1000"))

            # Wrap blocking SQLite call to avoid blocking event loop
            history = await asyncio.to_thread(
                self.get_metrics_history,
                metric_type=metric_type,
                board_type=board_type,
                num_players=num_players,
                hours=hours,
                limit=limit,
            )

            return web.json_response({
                "metric": metric_type,
                "period_hours": hours,
                "count": len(history),
                "data": history,
            })
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_training_status(self, request: web.Request) -> web.Response:
        """Get training pipeline status including NNUE, CMAES, and auto-promotion state.

        Returns daemon state for NNUE training, CMAES optimization, and model promotion.

        January 2026: Moved from p2p_orchestrator.py to AnalyticsHandlersMixin.
        """
        try:
            from datetime import datetime

            ai_root = Path(self._get_ai_service_path())

            # Load daemon state (from continuous_improvement_daemon.py)
            daemon_state_path = ai_root / "logs" / "improvement_daemon" / "state.json"
            daemon_state = {}
            daemon_running = False
            daemon_pid = None
            daemon_uptime = 0

            # Check if daemon is running
            pid_file = ai_root / "logs" / "improvement_daemon" / "daemon.pid"
            if pid_file.exists():
                try:
                    daemon_pid = int(pid_file.read_text().strip())
                    # Check if process is running
                    import os
                    os.kill(daemon_pid, 0)  # Doesn't kill, just checks
                    daemon_running = True
                except (ValueError, ProcessLookupError, PermissionError):
                    daemon_running = False

            if daemon_state_path.exists():
                try:
                    daemon_state = json.loads(daemon_state_path.read_text())
                    # Calculate uptime if daemon is running
                    if daemon_running and daemon_state.get("started_at"):
                        started = datetime.fromisoformat(daemon_state["started_at"])
                        daemon_uptime = (datetime.now() - started).total_seconds()
                except (json.JSONDecodeError, ValueError, OSError):
                    pass

            # Load runtime overrides (promoted models)
            overrides_path = ai_root / "data" / "ladder_runtime_overrides.json"
            runtime_overrides = {}
            if overrides_path.exists():
                with contextlib.suppress(json.JSONDecodeError, ValueError, OSError):
                    runtime_overrides = json.loads(overrides_path.read_text())

            # Load auto-promotion log
            promotion_log_path = (
                ai_root / "runs" / "promotion" / "model_promotion_history.json"
                if (ai_root / "runs" / "promotion" / "model_promotion_history.json").exists()
                else (ai_root / "data" / "auto_promotion_log.json")
            )
            promotion_log = []
            if promotion_log_path.exists():
                try:
                    promotion_log = json.loads(promotion_log_path.read_text())
                    if isinstance(promotion_log, list):
                        promotion_log = promotion_log[-10:]  # Last 10 entries
                except (json.JSONDecodeError, ValueError, OSError):
                    pass

            # Check NNUE model timestamps
            nnue_models = {}
            nnue_dir = ai_root / "models" / "nnue"
            if nnue_dir.exists():
                for model_file in nnue_dir.glob("*.pt"):
                    if "_prev" not in model_file.name:
                        stat = model_file.stat()
                        nnue_models[model_file.stem] = {
                            "path": str(model_file),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "size_mb": round(stat.st_size / 1024 / 1024, 2),
                        }

            # Check trained heuristic profiles
            profiles_path = ai_root / "data" / "trained_heuristic_profiles.json"
            heuristic_profiles = {}
            if profiles_path.exists():
                try:
                    profiles_data = json.loads(profiles_path.read_text())
                    heuristic_profiles = {
                        "count": len(profiles_data),
                        "profiles": list(profiles_data.keys())[:20],
                    }
                except (json.JSONDecodeError, ValueError, OSError):
                    pass

            return web.json_response({
                "success": True,
                "daemon": {
                    "running": daemon_running,
                    "pid": daemon_pid,
                    "uptime_seconds": daemon_uptime,
                    "current_cycle": daemon_state.get("total_cycles", 0),
                    "last_cycle_at": daemon_state.get("last_cycle_at", ""),
                    "total_games_generated": daemon_state.get("total_games_generated", 0),
                    "total_training_runs": daemon_state.get("total_training_runs", 0),
                    "total_tournaments": daemon_state.get("total_tournaments", 0),
                    "total_auto_promotions": daemon_state.get("total_auto_promotions", 0),
                    "last_auto_promote_time": daemon_state.get("last_auto_promote_time", 0),
                    "consecutive_failures": daemon_state.get("consecutive_failures", 0),
                },
                "nnue": {
                    "state": "idle" if not daemon_state.get("nnue_state") else "active",
                    "models": list(nnue_models.keys()),
                    "model_details": nnue_models,
                    "per_config_state": daemon_state.get("nnue_state", {}),
                    "last_gate_result": daemon_state.get("last_nnue_gate_result", None),
                },
                "cmaes": {
                    "state": "idle" if not daemon_state.get("cmaes_state") else "active",
                    "profiles": heuristic_profiles.get("profiles", []) if heuristic_profiles else [],
                    "profile_count": heuristic_profiles.get("count", 0) if heuristic_profiles else 0,
                    "per_config_state": daemon_state.get("cmaes_state", {}),
                    "generations": sum(s.get("generations", 0) for s in daemon_state.get("cmaes_state", {}).values()),
                },
                "promotion": {
                    "runtime_overrides": runtime_overrides,
                    "recent_promotions": promotion_log,
                },
                "timestamp": time.time(),
            })

        except Exception as e:  # noqa: BLE001
            return web.json_response({"success": False, "error": str(e)}, status=500)
