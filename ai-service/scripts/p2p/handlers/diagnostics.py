"""P2P Diagnostics HTTP Handlers Mixin.

January 2026: Extracted from p2p_orchestrator.py to reduce file size.

Provides HTTP endpoints for cluster diagnostics, progress tracking,
and stability monitoring.

Endpoints:
    GET /progress - Elo progress report for demonstrating iterative improvement
    GET /stability - Stability controller status
    GET /p2p/diagnostics - P2P diagnostic tracker data
    GET /swim/status - SWIM membership protocol status

Usage:
    class P2POrchestrator(DiagnosticsHandlersMixin, ...):
        pass
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import web

from scripts.p2p.handlers.base import BaseP2PHandler

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DiagnosticsHandlersMixin(BaseP2PHandler):
    """Mixin providing diagnostics HTTP handlers.

    Inherits from BaseP2PHandler for standardized response formatting.

    Requires the implementing class to have:
    - node_id: str
    - _stability_controller
    - _adaptive_timeouts
    - _effectiveness_tracker
    - _peer_state_tracker
    - _conn_failure_tracker
    - _probe_tracker
    - _swim_manager
    - _swim_started
    - _get_stability_metrics() method

    January 2026: Moved from p2p_orchestrator.py to DiagnosticsHandlersMixin.
    """

    # Type hints for IDE support
    node_id: str
    _stability_controller: Any
    _adaptive_timeouts: Any
    _effectiveness_tracker: Any
    _peer_state_tracker: Any
    _conn_failure_tracker: Any
    _probe_tracker: Any
    _swim_manager: Any
    _swim_started: bool

    async def handle_progress(self, request: web.Request) -> web.Response:
        """GET /progress - Return Elo progress report for demonstrating iterative improvement.

        January 16, 2026: Added to provide visibility into NN strength improvement.
        January 2026: Moved from p2p_orchestrator.py to DiagnosticsHandlersMixin.

        Query parameters:
            config: Optional config filter (e.g., "hex8_2p")
            days: Lookback period in days (default: 30)

        Returns JSON with:
            - configs: Per-config progress data (starting_elo, current_elo, delta, iterations)
            - overall: Summary stats (total_iterations, configs_improving, avg_elo_gain)
            - generated_at: Timestamp
        """
        config_filter = request.query.get("config")
        try:
            days = float(request.query.get("days", "30"))
        except ValueError:
            days = 30.0

        try:
            # Import progress report logic
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from scripts.elo_progress_report import get_full_report
            from dataclasses import asdict

            report = get_full_report(days=days, config_filter=config_filter)

            # Convert to JSON-serializable dict
            data = {
                "configs": {k: asdict(v) for k, v in report.configs.items()},
                "overall": asdict(report.overall),
                "generated_at": report.generated_at,
            }

            return self.json_response(data)

        except ImportError as e:
            logger.warning(f"[handle_progress] Import error: {e}")
            return self.json_response({
                "error": "progress_report_unavailable",
                "detail": str(e),
            }, status=500)
        except Exception as e:  # noqa: BLE001
            logger.error(f"[handle_progress] Error generating progress report: {e}")
            return self.json_response({
                "error": "internal_error",
                "detail": str(e),
            }, status=500)

    async def handle_stability(self, request: web.Request) -> web.Response:
        """GET /stability - Return stability controller status.

        January 2026: Added as part of P2P Self-Healing Architecture.
        January 2026: Moved from p2p_orchestrator.py to DiagnosticsHandlersMixin.

        Returns JSON with:
            - controller: StabilityController status (symptoms, actions, running state)
            - adaptive_timeouts: Per-node adaptive timeouts
            - effectiveness: Recovery action effectiveness tracking
            - metrics: Current stability metrics
        """
        response: dict[str, Any] = {
            "node_id": self.node_id,
            "timestamp": time.time(),
        }

        # Stability controller status
        if self._stability_controller:
            response["controller"] = self._stability_controller.get_status()
        else:
            response["controller"] = {"enabled": False, "reason": "not_initialized"}

        # Adaptive timeout status
        if self._adaptive_timeouts:
            response["adaptive_timeouts"] = self._adaptive_timeouts.get_status()
        else:
            response["adaptive_timeouts"] = {"enabled": False}

        # Effectiveness tracking status
        if self._effectiveness_tracker:
            response["effectiveness"] = self._effectiveness_tracker.get_status()
        else:
            response["effectiveness"] = {"enabled": False}

        # Current stability metrics
        try:
            response["metrics"] = self._get_stability_metrics()
        except Exception as e:
            response["metrics"] = {"error": str(e)}

        return self.json_response(response)

    async def handle_p2p_diagnostics(self, request: web.Request) -> web.Response:
        """GET /p2p/diagnostics - Return P2P diagnostic tracker data.

        January 2026: Phase 0 diagnostic instrumentation endpoint.
        January 2026: Moved from p2p_orchestrator.py to DiagnosticsHandlersMixin.

        Returns JSON with:
            - peer_state: Peer state transition tracking (flapping, death reasons)
            - connection_failures: Connection failure tracking by type/transport
            - probe_effectiveness: Probe success rates and false positives
        """
        response: dict[str, Any] = {
            "node_id": self.node_id,
            "timestamp": time.time(),
        }

        # Peer state tracker
        if self._peer_state_tracker:
            try:
                response["peer_state"] = self._peer_state_tracker.get_diagnostics()
            except Exception as e:
                response["peer_state"] = {"error": str(e)}
        else:
            response["peer_state"] = {"enabled": False}

        # Connection failure tracker
        if self._conn_failure_tracker:
            try:
                response["connection_failures"] = self._conn_failure_tracker.get_diagnostics()
            except Exception as e:
                response["connection_failures"] = {"error": str(e)}
        else:
            response["connection_failures"] = {"enabled": False}

        # Probe effectiveness tracker
        if self._probe_tracker:
            try:
                response["probe_effectiveness"] = self._probe_tracker.get_diagnostics()
            except Exception as e:
                response["probe_effectiveness"] = {"error": str(e)}
        else:
            response["probe_effectiveness"] = {"enabled": False}

        return self.json_response(response)

    async def handle_swim_status(self, request: web.Request) -> web.Response:
        """GET /swim/status - Return SWIM membership protocol status.

        January 2026: Moved from p2p_orchestrator.py to DiagnosticsHandlersMixin.

        SWIM provides leaderless gossip-based membership with:
        - O(1) bandwidth per node (constant message complexity)
        - <5 second failure detection (vs 60+ seconds with heartbeats)
        - Suspicion mechanism to reduce false positives
        """
        try:
            if not self._swim_manager:
                return self.json_response({
                    "status": "disabled",
                    "reason": "swim-p2p not installed or SWIM adapter not available",
                    "node_id": self.node_id,
                    "fallback": "http_heartbeats",
                })

            summary = self._swim_manager.get_membership_summary()
            alive_peers = self._swim_manager.get_alive_peers() if self._swim_started else []

            return self.json_response({
                "status": "enabled" if self._swim_started else "initialized",
                "node_id": self.node_id,
                "swim": summary,
                "alive_peers": alive_peers,
                "peer_count": len(alive_peers),
            })

        except Exception as e:  # noqa: BLE001
            logger.error(f"Error getting SWIM status: {e}")
            return self.json_response({
                "status": "error",
                "error": str(e),
                "node_id": self.node_id,
            }, status=500)
