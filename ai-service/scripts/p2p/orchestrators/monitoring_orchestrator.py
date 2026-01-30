"""Monitoring orchestrator for P2P validation and health servers.

January 2026: Created as part of Phase 4 P2POrchestrator decomposition.
Handles startup validation, health endpoints, and monitoring infrastructure.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import socket
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from aiohttp import web

from .base_orchestrator import BaseOrchestrator, HealthCheckResult

if TYPE_CHECKING:
    from scripts.p2p_orchestrator import P2POrchestrator

logger = logging.getLogger(__name__)


class MonitoringOrchestrator(BaseOrchestrator):
    """Orchestrator for monitoring, validation, and health infrastructure.

    Responsibilities:
    - Startup validation (SWIM/Raft, managers, voters, PyTorch CUDA)
    - Isolated health server for liveness/readiness probes
    - Validation result tracking
    """

    def __init__(self, p2p: "P2POrchestrator") -> None:
        """Initialize the monitoring orchestrator."""
        super().__init__(p2p)
        self._validation_result: dict[str, Any] | None = None
        self._health_server_started = False

    @property
    def name(self) -> str:
        """Return the orchestrator name."""
        return "monitoring"

    def health_check(self) -> HealthCheckResult:
        """Check health of the monitoring orchestrator."""
        details = {
            "validation_completed": self._validation_result is not None,
            "health_server_started": self._health_server_started,
        }

        if self._validation_result:
            details["validation_warnings"] = len(self._validation_result.get("warnings", []))
            details["validation_errors"] = len(self._validation_result.get("errors", []))

        healthy = (
            self._validation_result is not None
            and len(self._validation_result.get("errors", [])) == 0
        )

        return HealthCheckResult(
            healthy=healthy,
            message="Monitoring orchestrator operational" if healthy else "Validation incomplete or has errors",
            details=details,
        )

    def validate_critical_subsystems(self) -> dict:
        """Validate critical subsystems at startup.

        Returns a status dict with protocol and manager availability.
        Logs clear messages about which protocols are active.

        December 2025: Added to address silent fallback behavior
        where operators couldn't tell if SWIM/Raft was running.
        """
        p2p = self._p2p

        # Import constants with fallback
        try:
            from app.p2p.constants import (
                SWIM_ENABLED, RAFT_ENABLED, MEMBERSHIP_MODE, CONSENSUS_MODE
            )
        except ImportError:
            SWIM_ENABLED = False
            RAFT_ENABLED = False
            MEMBERSHIP_MODE = "http"
            CONSENSUS_MODE = "bully"

        status: dict[str, Any] = {
            "protocols": {
                "membership_mode": MEMBERSHIP_MODE,
                "consensus_mode": CONSENSUS_MODE,
                "swim_enabled": SWIM_ENABLED,
                "raft_enabled": RAFT_ENABLED,
            },
            "managers": {},
            "warnings": [],
            "errors": [],
        }

        # Check SWIM availability
        try:
            from app.p2p.swim_adapter import SWIM_AVAILABLE
            status["protocols"]["swim_available"] = SWIM_AVAILABLE
            if SWIM_ENABLED and not SWIM_AVAILABLE:
                msg = "SWIM_ENABLED=true but swim-p2p not installed. Install: pip install swim-p2p>=1.2.0"
                status["warnings"].append(msg)
                self._log_warning(f"[Startup Validation] {msg}")
            elif SWIM_AVAILABLE:
                self._log_info(f"[Startup Validation] SWIM protocol available (membership_mode={MEMBERSHIP_MODE})")
        except ImportError:
            status["protocols"]["swim_available"] = False
            if SWIM_ENABLED:
                status["warnings"].append("swim_adapter import failed")

        # Check Raft availability
        try:
            from app.p2p.raft_state import PYSYNCOBJ_AVAILABLE
            status["protocols"]["raft_available"] = PYSYNCOBJ_AVAILABLE
            if RAFT_ENABLED and not PYSYNCOBJ_AVAILABLE:
                msg = "RAFT_ENABLED=true but pysyncobj not installed. Install: pip install pysyncobj>=0.3.14"
                status["warnings"].append(msg)
                self._log_warning(f"[Startup Validation] {msg}")
            elif PYSYNCOBJ_AVAILABLE:
                self._log_info(f"[Startup Validation] Raft protocol available (consensus_mode={CONSENSUS_MODE})")
        except ImportError:
            status["protocols"]["raft_available"] = False
            if RAFT_ENABLED:
                status["warnings"].append("raft_state import failed")

        # Log active protocol configuration
        self._log_info(
            f"[Startup Validation] Protocol config: membership={MEMBERSHIP_MODE}, consensus={CONSENSUS_MODE}"
        )

        # Check critical managers (lazy load check - don't fail, just report)
        manager_checks = [
            ("work_queue", "app.coordination.work_queue", "get_work_queue"),
            ("health_manager", "app.coordination.unified_health_manager", "get_unified_health_manager"),
            ("sync_router", "app.coordination.sync_router", "get_sync_router"),
        ]

        for name, module_path, getter_name in manager_checks:
            try:
                module = importlib.import_module(module_path)
                getter = getattr(module, getter_name, None)
                status["managers"][name] = getter is not None
                if getter:
                    self._log_debug(f"[Startup Validation] Manager {name} available")
            except ImportError as e:
                status["managers"][name] = False
                status["warnings"].append(f"{name} import failed: {e}")
                self._log_warning(f"[Startup Validation] Manager {name} unavailable: {e}")

        # December 2025: P2P voter connectivity validation
        voter_node_ids = getattr(p2p, "voter_node_ids", [])
        voter_quorum_size = getattr(p2p, "voter_quorum_size", 0)
        port = getattr(p2p, "port", 8770)

        status["voters"] = {
            "configured": len(voter_node_ids),
            "quorum": voter_quorum_size,
            "reachable": 0,
            "unreachable": [],
        }

        if voter_node_ids:
            reachable_count = 0
            for voter_id in voter_node_ids:
                try:
                    from app.config.cluster_config import get_cluster_nodes
                    nodes = get_cluster_nodes()
                    node = nodes.get(voter_id)
                    if node:
                        voter_ip = node.best_ip
                        if voter_ip:
                            with contextlib.suppress(Exception):
                                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                sock.settimeout(2.0)
                                result = sock.connect_ex((voter_ip, port))
                                sock.close()
                                if result == 0:
                                    reachable_count += 1
                                    continue
                    status["voters"]["unreachable"].append(voter_id)
                except (socket.error, socket.timeout, OSError, TimeoutError, ConnectionRefusedError):
                    status["voters"]["unreachable"].append(voter_id)

            status["voters"]["reachable"] = reachable_count

            if reachable_count < voter_quorum_size:
                msg = (
                    f"Only {reachable_count}/{len(voter_node_ids)} voters reachable, "
                    f"need {voter_quorum_size} for quorum. Unreachable: {status['voters']['unreachable']}"
                )
                status["warnings"].append(msg)
                self._log_warning(f"[Startup Validation] {msg}")
                # Emit QUORUM_VALIDATION_FAILED event
                try:
                    from app.distributed.data_events import DataEventType, get_event_bus
                    get_event_bus().emit(
                        DataEventType.QUORUM_VALIDATION_FAILED,
                        {
                            "node_id": self.node_id,
                            "reachable_voters": reachable_count,
                            "total_voters": len(voter_node_ids),
                            "quorum_required": voter_quorum_size,
                            "unreachable": status["voters"]["unreachable"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                except (ImportError, Exception) as e:
                    self._log_debug(f"[Startup Validation] Could not emit quorum validation event: {e}")
            else:
                self._log_info(
                    f"[Startup Validation] Voter quorum OK: "
                    f"{reachable_count}/{len(voter_node_ids)} voters reachable"
                )
        else:
            self._log_info("[Startup Validation] No voters configured - quorum checks disabled")

        # Jan 9, 2026: PyTorch CUDA validation
        resource_detector = getattr(p2p, "_resource_detector", None)
        if resource_detector and hasattr(resource_detector, "validate_pytorch_cuda"):
            try:
                pytorch_status = resource_detector.validate_pytorch_cuda()
                status["pytorch"] = pytorch_status

                if pytorch_status.get("warning"):
                    status["warnings"].append(pytorch_status["warning"])
                    self._log_warning(f"[Startup Validation] {pytorch_status['warning']}")

                    try:
                        from app.distributed.data_events import DataEventType, get_event_bus
                        get_event_bus().emit(
                            DataEventType.PYTORCH_CUDA_MISMATCH,
                            {
                                "node_id": self.node_id,
                                "warning": pytorch_status["warning"],
                                "gpu_detected": pytorch_status.get("gpu_detected", False),
                                "pytorch_cuda_available": pytorch_status.get("pytorch_cuda_available", False),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except (ImportError, Exception) as e:
                        self._log_debug(f"[Startup Validation] Could not emit PyTorch CUDA event: {e}")
                elif pytorch_status.get("pytorch_cuda_available"):
                    self._log_info(
                        f"[Startup Validation] PyTorch CUDA OK: "
                        f"version={pytorch_status.get('pytorch_cuda_version')}, "
                        f"devices={pytorch_status.get('cuda_device_count')}"
                    )
                elif pytorch_status.get("error"):
                    self._log_debug(f"[Startup Validation] PyTorch not installed: {pytorch_status.get('error')}")
            except Exception as e:
                self._log_debug(f"[Startup Validation] PyTorch validation failed: {e}")

        # Summary log
        available_count = sum(1 for v in status["managers"].values() if v)
        total_count = len(status["managers"])
        if status["warnings"]:
            self._log_warning(
                f"[Startup Validation] Completed with {len(status['warnings'])} warnings. "
                f"Managers: {available_count}/{total_count} available"
            )
        else:
            self._log_info(
                f"[Startup Validation] All checks passed. "
                f"Managers: {available_count}/{total_count} available"
            )

        self._validation_result = status
        return status

    def start_isolated_health_server(self) -> None:
        """Start a lightweight health HTTP server in a separate thread.

        January 2026: This server runs in its own thread with its own event loop,
        guaranteeing that /health endpoints respond even when the main event loop
        is blocked by background tasks.

        The isolated server:
        - Listens on port + 2 (8772 for P2P on 8770)
        - Only serves /health and /ready endpoints
        - Does not access any state that requires the main event loop
        - Responds within 100ms even under heavy load
        """
        p2p = self._p2p
        health_port = getattr(p2p, "port", 8770) + 2

        def _run_health_server_in_thread() -> None:
            """Run the health server in a separate thread with its own event loop."""
            import asyncio as thread_asyncio

            async def handle_health(request: web.Request) -> web.Response:
                """Liveness probe - returns 200 if P2P process is alive."""
                uptime = time.time() - getattr(p2p, "start_time", time.time())
                role = getattr(p2p, "role", None)
                role_value = role.value if hasattr(role, 'value') else str(role)
                return web.json_response({
                    "alive": True,
                    "node_id": p2p.node_id,
                    "role": role_value,
                    "uptime_seconds": uptime,
                    "main_port": getattr(p2p, "port", 8770),
                    "isolated_health_server": True,
                    "timestamp": datetime.utcnow().isoformat(),
                })

            async def handle_ready(request: web.Request) -> web.Response:
                """Readiness probe - returns 200 if P2P has started up."""
                uptime = time.time() - getattr(p2p, "start_time", time.time())
                is_ready = uptime >= 30.0
                return web.json_response({
                    "ready": is_ready,
                    "node_id": p2p.node_id,
                    "uptime_seconds": uptime,
                    "startup_complete": is_ready,
                    "timestamp": datetime.utcnow().isoformat(),
                }, status=200 if is_ready else 503)

            async def run_server() -> None:
                """Set up and run the health server."""
                app = web.Application()
                app.router.add_get('/health', handle_health)
                app.router.add_get('/ready', handle_ready)

                runner = web.AppRunner(app)
                await runner.setup()

                try:
                    site = web.TCPSite(runner, '0.0.0.0', health_port, reuse_address=True)
                    await site.start()
                    logger.info(f"Isolated health server started on 0.0.0.0:{health_port}")

                    while True:
                        await thread_asyncio.sleep(3600)
                except OSError as e:
                    if "Address already in use" in str(e):
                        logger.warning(f"Isolated health server port {health_port} in use, skipping")
                    else:
                        logger.error(f"Isolated health server failed: {e}")
                except Exception as e:
                    logger.error(f"Isolated health server error: {e}")

            loop = thread_asyncio.new_event_loop()
            thread_asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_server())
            except Exception as e:
                logger.error(f"Isolated health server thread failed: {e}")
            finally:
                loop.close()

        health_thread = threading.Thread(
            target=_run_health_server_in_thread,
            name="isolated-health-server",
            daemon=True,
        )
        health_thread.start()
        self._health_server_started = True
        self._log_info(f"Started isolated health server thread (port {health_port})")
