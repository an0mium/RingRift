"""Maintenance Loops for P2P Orchestrator.

December 2025: Background loops for system maintenance tasks.

Loops:
- GitUpdateLoop: Periodically checks for and applies git updates with auto-restart
- CircuitBreakerDecayLoop: Decays old circuit breakers to prevent permanent exclusion
- WalCleanupLoop: Periodic SQLite WAL checkpoint to reclaim disk space
- HttpPoolMonitorLoop: Monitors HTTP connection pool and recycles stale sessions

Usage:
    from scripts.p2p.loops import GitUpdateLoop, GitUpdateConfig

    git_loop = GitUpdateLoop(
        check_for_updates=orchestrator._check_for_updates,
        perform_update=orchestrator._perform_git_update,
        restart_orchestrator=orchestrator._restart_orchestrator,
    )
    await git_loop.run_forever()
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from .base import BaseLoop

from .loop_constants import LoopIntervals, AUTO_UPDATE_ENABLED

logger = logging.getLogger(__name__)


# Backward-compat alias (Sprint 10: use LoopIntervals.GIT_UPDATE_CHECK instead)
GIT_UPDATE_CHECK_INTERVAL = LoopIntervals.GIT_UPDATE_CHECK
# AUTO_UPDATE_ENABLED is now imported from loop_constants


@dataclass
class GitUpdateConfig:
    """Configuration for git update loop.

    December 2025: Extracted from p2p_orchestrator._git_update_loop
    """

    check_interval_seconds: float = GIT_UPDATE_CHECK_INTERVAL
    error_retry_seconds: float = 60.0  # Wait before retry on error
    enabled: bool = AUTO_UPDATE_ENABLED

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.error_retry_seconds <= 0:
            raise ValueError("error_retry_seconds must be > 0")


class GitUpdateLoop(BaseLoop):
    """Background loop to periodically check for and apply git updates.

    When updates are detected:
    1. Calculates commits behind remote
    2. Performs git pull
    3. Restarts orchestrator to apply changes

    December 2025: Extracted from p2p_orchestrator._git_update_loop
    """

    def __init__(
        self,
        check_for_updates: Callable[[], tuple[bool, str | None, str | None]],
        perform_update: Callable[[], Coroutine[Any, Any, tuple[bool, str]]],
        restart_orchestrator: Callable[[], Coroutine[Any, Any, None]],
        get_commits_behind: Callable[[str, str], int] | None = None,
        config: GitUpdateConfig | None = None,
    ):
        """Initialize git update loop.

        Args:
            check_for_updates: Callback returning (has_updates, local_commit, remote_commit)
            perform_update: Async callback to perform git pull, returns (success, message)
            restart_orchestrator: Async callback to restart the orchestrator
            get_commits_behind: Optional callback to count commits behind (local, remote) -> count
            config: Update configuration
        """
        self.config = config or GitUpdateConfig()
        super().__init__(
            name="git_update",
            interval=self.config.check_interval_seconds,
        )
        self._check_for_updates = check_for_updates
        self._perform_update = perform_update
        self._restart_orchestrator = restart_orchestrator
        self._get_commits_behind = get_commits_behind

        # Statistics
        self._checks_count = 0
        self._updates_found = 0
        self._updates_applied = 0
        self._update_failures = 0

    async def _on_start(self) -> None:
        """Check if auto-update is enabled."""
        if not self.config.enabled:
            logger.info("[GitUpdate] Auto-update disabled, loop will skip checks")
        else:
            logger.info(
                f"[GitUpdate] Auto-update loop started "
                f"(interval: {self.config.check_interval_seconds}s)"
            )

    async def _run_once(self) -> None:
        """Check for updates and apply if available."""
        if not self.config.enabled:
            return

        self._checks_count += 1

        try:
            # Check for updates
            # Feb 2026: _check_for_updates may be sync or async (orchestrator's is async)
            check_result = self._check_for_updates()
            if asyncio.iscoroutine(check_result):
                check_result = await check_result
            has_updates, local_commit, remote_commit = check_result

            if has_updates and local_commit and remote_commit:
                self._updates_found += 1

                # Calculate commits behind (if callback provided)
                commits_behind = 0
                if self._get_commits_behind:
                    commits_behind = self._get_commits_behind(local_commit, remote_commit)
                    logger.info(f"[GitUpdate] Update available: {commits_behind} commits behind")

                logger.info(f"[GitUpdate] Local:  {local_commit[:8]}")
                logger.info(f"[GitUpdate] Remote: {remote_commit[:8]}")

                # Perform update
                success, message = await self._perform_update()

                if success:
                    self._updates_applied += 1
                    logger.info("[GitUpdate] Update successful, restarting...")
                    await self._restart_orchestrator()
                else:
                    self._update_failures += 1
                    logger.warning(f"[GitUpdate] Update failed: {message}")

        except Exception as e:
            logger.warning(f"[GitUpdate] Error in update check: {e}")
            await asyncio.sleep(self.config.error_retry_seconds)

    def get_update_stats(self) -> dict[str, Any]:
        """Get update statistics."""
        return {
            "enabled": self.config.enabled,
            "checks_count": self._checks_count,
            "updates_found": self._updates_found,
            "updates_applied": self._updates_applied,
            "update_failures": self._update_failures,
            "success_rate": (
                self._updates_applied / self._updates_found * 100
                if self._updates_found > 0
                else 100.0
            ),
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        stats = self.get_update_stats()

        if not self.running:
            status = "ERROR"
            message = "Git update loop not running"
        elif not self.config.enabled:
            status = "HEALTHY"
            message = "Auto-update disabled"
        elif self._update_failures > 3:
            status = "DEGRADED"
            message = f"Multiple update failures: {self._update_failures}"
        else:
            status = "HEALTHY"
            message = f"Applied {self._updates_applied} updates"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "enabled": self.config.enabled,
                "checks_count": self._checks_count,
                "updates_found": self._updates_found,
                "updates_applied": self._updates_applied,
                "update_failures": self._update_failures,
                "run_count": self.stats.total_runs,
            },
        }


# =============================================================================
# Circuit Breaker Decay Loop (Sprint 17.6)
# =============================================================================


@dataclass
class CircuitBreakerDecayConfig:
    """Configuration for circuit breaker TTL decay loop.

    January 2026 Sprint 17.6: Prevents circuits from staying OPEN indefinitely.
    January 5, 2026 (Phase 3): Reduced TTL from 6h to 1h for faster node recovery.
    """

    check_interval_seconds: float = 1800.0  # Check every 30 minutes (was 1 hour)
    ttl_seconds: float = 3600.0  # Reset circuits open > 1 hour (was 6 hours)
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")


class CircuitBreakerDecayLoop(BaseLoop):
    """Background loop to periodically decay old circuit breakers.

    This prevents circuits from being stuck OPEN indefinitely after transient
    failures. Circuits that have been OPEN longer than ttl_seconds are
    automatically reset to CLOSED.

    January 2026 Sprint 17.6: Added as part of stability improvements.
    January 5, 2026 (Phase 3): TTL reduced from 6h to 1h for faster recovery.
    January 20, 2026: Added external_alive_check for gossip-integrated recovery.

    Benefits:
    - Prevents 1h+ stuck circuits blocking health checks (was 6h)
    - Reduces manual interventions from stuck states
    - Enables graceful recovery after network partitions
    - External alive check enables immediate recovery when gossip reports node alive
    """

    def __init__(
        self,
        config: CircuitBreakerDecayConfig | None = None,
    ):
        """Initialize circuit breaker decay loop.

        Args:
            config: Optional configuration (uses defaults if not provided)
        """
        self.config = config or CircuitBreakerDecayConfig()
        super().__init__(
            name="circuit_breaker_decay",
            interval=self.config.check_interval_seconds,
        )
        self._decay_count = 0
        self._last_decay_result: dict[str, Any] = {}
        # Jan 20, 2026: External alive check for gossip-integrated recovery
        self._external_alive_check: callable | None = None

    def set_external_alive_check(self, callback: callable) -> None:
        """Set callback for external alive verification (e.g., from gossip).

        January 20, 2026: Enables immediate circuit recovery when gossip
        reports a node is alive, instead of waiting for TTL expiry.

        Args:
            callback: Callable(host: str) -> bool that returns True if the
                host is known to be alive from external source (gossip/P2P).
        """
        self._external_alive_check = callback
        logger.info("[CircuitBreakerDecay] External alive check callback configured")

    async def _run_once(self) -> None:
        """Run one decay cycle."""
        if not self.config.enabled:
            return

        try:
            # Import here to avoid circular imports
            from app.coordination.circuit_breaker_base import (
                decay_all_circuit_breakers,
            )
            from app.coordination.node_circuit_breaker import (
                get_node_circuit_registry,
            )

            # Decay operation and transport circuits (uniform TTL)
            result = decay_all_circuit_breakers(self.config.ttl_seconds)

            # Also decay node circuits
            try:
                node_registry = get_node_circuit_registry()
                node_result = node_registry.decay_all_old_circuits(self.config.ttl_seconds)
                result["node_circuits"] = node_result
            except Exception as e:
                logger.debug(f"[CircuitBreakerDecay] Node registry not available: {e}")

            # Jan 5, 2026: Also decay with transport-specific TTLs
            # This provides faster recovery for transports that typically
            # recover quickly (relay: 15min, tailscale: 30min, ssh: 1hr)
            # Jan 20, 2026: Added external_alive_check for gossip-integrated recovery
            try:
                from app.distributed.circuit_breaker import (
                    decay_transport_circuit_breakers,
                )

                transport_result = decay_transport_circuit_breakers(
                    external_alive_check=self._external_alive_check
                )
                result["transport_specific_decay"] = transport_result

                # Track external recoveries separately
                external_count = len(transport_result.get("external_recovered", []))
                if external_count > 0:
                    logger.info(
                        f"[CircuitBreakerDecay] {external_count} circuits recovered via gossip"
                    )
            except Exception as e:
                logger.debug(
                    f"[CircuitBreakerDecay] Transport-specific decay not available: {e}"
                )

            self._last_decay_result = result

            # Count total decayed (including transport-specific decay)
            total_decayed = (
                result.get("operation_registry", {}).get("total_decayed", 0)
                + len(result.get("transport_breakers", {}).get("decayed", []))
                + result.get("node_circuits", {}).get("total_decayed", 0)
                + len(result.get("transport_specific_decay", {}).get("decayed", []))
            )

            if total_decayed > 0:
                self._decay_count += total_decayed
                logger.info(
                    f"[CircuitBreakerDecay] Decayed {total_decayed} old circuits "
                    f"(total lifetime: {self._decay_count})"
                )

        except Exception as e:
            logger.warning(f"[CircuitBreakerDecay] Error in decay cycle: {e}")

    def get_decay_stats(self) -> dict[str, Any]:
        """Get decay statistics."""
        return {
            "enabled": self.config.enabled,
            "ttl_seconds": self.config.ttl_seconds,
            "check_interval_seconds": self.config.check_interval_seconds,
            "total_decayed_lifetime": self._decay_count,
            "last_result": self._last_decay_result,
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        if not self.running:
            status = "ERROR"
            message = "Circuit breaker decay loop not running"
        elif not self.config.enabled:
            status = "HEALTHY"
            message = "CB decay disabled"
        else:
            status = "HEALTHY"
            message = f"Decayed {self._decay_count} circuits lifetime"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "enabled": self.config.enabled,
                "ttl_seconds": self.config.ttl_seconds,
                "total_decayed_lifetime": self._decay_count,
                "run_count": self.stats.total_runs,
            },
        }


# =============================================================================
# WAL Cleanup Loop (March 2026 - 7-Day Autonomous Operation)
# =============================================================================


# Default interval: 6 hours (21600 seconds)
WAL_CLEANUP_INTERVAL = 21600.0


@dataclass
class WalCleanupConfig:
    """Configuration for SQLite WAL checkpoint cleanup loop.

    March 2026: WAL files accumulate 1-2GB each and are never globally
    checkpointed, leading to false disk pressure alerts after 3-4 days
    of autonomous operation. This loop runs PRAGMA wal_checkpoint(TRUNCATE)
    on all discovered .db files to reclaim disk space.
    """

    check_interval_seconds: float = WAL_CLEANUP_INTERVAL
    enabled: bool = True
    # Directories to scan for .db files (relative to ai-service root)
    scan_dirs: list[str] = field(default_factory=lambda: ["data/games", "data"])

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")


class WalCleanupLoop(BaseLoop):
    """Background loop to periodically checkpoint SQLite WAL files.

    SQLite WAL (Write-Ahead Log) files grow unbounded when many writers
    append data but no reader triggers a checkpoint. On a busy cluster
    coordinator with 12+ canonical databases, gauntlet databases, Elo
    databases, and tracking databases, WAL files can reach 1-2GB each
    after 3-4 days, consuming 10-20GB of disk and triggering false
    DISK_PRODUCTION_HALT alerts.

    This loop finds all .db files in configured directories and runs
    PRAGMA wal_checkpoint(TRUNCATE) on each, which:
    1. Writes all WAL content back to the main database file
    2. Truncates the WAL file to zero bytes
    3. Removes the shared-memory (-shm) file

    March 2026: Added for 7-day autonomous operation support.

    Benefits:
    - Prevents WAL file accumulation (saves 10-20GB over 7 days)
    - Eliminates false disk pressure alerts from WAL growth
    - Per-database error handling ensures one failure doesn't block others
    - Logs before/after WAL sizes for observability
    """

    def __init__(
        self,
        config: WalCleanupConfig | None = None,
        ai_service_root: str | None = None,
    ):
        """Initialize WAL cleanup loop.

        Args:
            config: Optional configuration (uses defaults if not provided).
            ai_service_root: Root directory of ai-service. If None, auto-detected
                from this file's location.
        """
        self.config = config or WalCleanupConfig()
        super().__init__(
            name="wal_cleanup",
            interval=self.config.check_interval_seconds,
        )

        # Resolve ai-service root
        if ai_service_root:
            self._root = Path(ai_service_root)
        else:
            # This file is at ai-service/scripts/p2p/loops/maintenance_loops.py
            self._root = Path(__file__).resolve().parent.parent.parent.parent

        # Statistics
        self._databases_checkpointed = 0
        self._total_bytes_reclaimed = 0
        self._checkpoint_errors = 0
        self._last_result: dict[str, Any] = {}

    def _find_databases(self) -> list[Path]:
        """Find all .db files in configured scan directories.

        Returns:
            List of absolute paths to .db files.
        """
        db_files: list[Path] = []
        seen: set[Path] = set()

        for scan_dir in self.config.scan_dirs:
            directory = self._root / scan_dir
            if not directory.is_dir():
                logger.debug(f"[WalCleanup] Scan directory not found: {directory}")
                continue

            for db_path in directory.glob("*.db"):
                resolved = db_path.resolve()
                if resolved not in seen and resolved.is_file():
                    seen.add(resolved)
                    db_files.append(resolved)

        return sorted(db_files)

    @staticmethod
    def _get_wal_size(db_path: Path) -> int:
        """Get the size of the WAL file for a database.

        Returns:
            Size in bytes, or 0 if no WAL file exists.
        """
        wal_path = Path(str(db_path) + "-wal")
        try:
            if wal_path.exists():
                return wal_path.stat().st_size
        except OSError:
            pass
        return 0

    def _checkpoint_database(self, db_path: Path) -> dict[str, Any]:
        """Run PRAGMA wal_checkpoint(TRUNCATE) on a single database.

        Args:
            db_path: Absolute path to the .db file.

        Returns:
            Dict with checkpoint result details.
        """
        result: dict[str, Any] = {
            "database": db_path.name,
            "path": str(db_path),
            "success": False,
            "wal_size_before": 0,
            "wal_size_after": 0,
            "bytes_reclaimed": 0,
            "error": None,
        }

        wal_size_before = self._get_wal_size(db_path)
        result["wal_size_before"] = wal_size_before

        # Skip databases with no WAL file or empty WAL
        if wal_size_before == 0:
            result["success"] = True
            result["skipped"] = True
            return result

        try:
            # Open with short timeout to avoid blocking on locked databases
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            try:
                # WAL checkpoint: TRUNCATE mode writes all WAL content to the
                # main db file and then truncates the WAL file to zero length
                checkpoint_result = conn.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()

                # checkpoint_result is (blocked, wal_pages, checkpointed_pages)
                # blocked=0 means success, blocked=1 means some pages couldn't be checkpointed
                if checkpoint_result:
                    result["checkpoint_blocked"] = checkpoint_result[0]
                    result["wal_pages"] = checkpoint_result[1]
                    result["checkpointed_pages"] = checkpoint_result[2]

                result["success"] = True
            finally:
                conn.close()

            # Measure WAL size after checkpoint
            wal_size_after = self._get_wal_size(db_path)
            result["wal_size_after"] = wal_size_after
            result["bytes_reclaimed"] = max(0, wal_size_before - wal_size_after)

        except sqlite3.OperationalError as e:
            # Database locked, busy, etc. - not fatal
            result["error"] = f"SQLite operational error: {e}"
        except Exception as e:
            result["error"] = str(e)

        return result

    async def _run_once(self) -> None:
        """Run one WAL cleanup cycle across all databases."""
        if not self.config.enabled:
            return

        try:
            # Find databases (sync I/O, but fast - just directory listing)
            db_files = await asyncio.get_event_loop().run_in_executor(
                None, self._find_databases
            )

            if not db_files:
                logger.debug("[WalCleanup] No databases found to checkpoint")
                return

            cycle_results: list[dict[str, Any]] = []
            cycle_bytes_reclaimed = 0
            cycle_checkpointed = 0
            cycle_errors = 0
            cycle_skipped = 0

            for db_path in db_files:
                # Run checkpoint in thread pool to avoid blocking event loop
                # (SQLite I/O is synchronous)
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._checkpoint_database, db_path
                )
                cycle_results.append(result)

                if result.get("skipped"):
                    cycle_skipped += 1
                elif result["success"]:
                    cycle_checkpointed += 1
                    reclaimed = result.get("bytes_reclaimed", 0)
                    cycle_bytes_reclaimed += reclaimed
                    if reclaimed > 0:
                        logger.info(
                            f"[WalCleanup] {result['database']}: "
                            f"WAL {result['wal_size_before']:,} -> "
                            f"{result['wal_size_after']:,} bytes "
                            f"(reclaimed {reclaimed:,} bytes)"
                        )
                else:
                    cycle_errors += 1
                    logger.warning(
                        f"[WalCleanup] Failed to checkpoint {result['database']}: "
                        f"{result.get('error', 'unknown error')}"
                    )

            # Update lifetime statistics
            self._databases_checkpointed += cycle_checkpointed
            self._total_bytes_reclaimed += cycle_bytes_reclaimed
            self._checkpoint_errors += cycle_errors

            self._last_result = {
                "databases_found": len(db_files),
                "checkpointed": cycle_checkpointed,
                "skipped_no_wal": cycle_skipped,
                "errors": cycle_errors,
                "bytes_reclaimed": cycle_bytes_reclaimed,
                "details": cycle_results,
            }

            if cycle_checkpointed > 0 or cycle_errors > 0:
                logger.info(
                    f"[WalCleanup] Cycle complete: {cycle_checkpointed} checkpointed, "
                    f"{cycle_skipped} skipped (no WAL), {cycle_errors} errors, "
                    f"{cycle_bytes_reclaimed:,} bytes reclaimed"
                )

        except Exception as e:
            logger.warning(f"[WalCleanup] Error in cleanup cycle: {e}")

    def get_cleanup_stats(self) -> dict[str, Any]:
        """Get WAL cleanup statistics."""
        return {
            "enabled": self.config.enabled,
            "check_interval_seconds": self.config.check_interval_seconds,
            "scan_dirs": self.config.scan_dirs,
            "total_databases_checkpointed": self._databases_checkpointed,
            "total_bytes_reclaimed": self._total_bytes_reclaimed,
            "total_bytes_reclaimed_mb": round(self._total_bytes_reclaimed / (1024 * 1024), 2),
            "total_errors": self._checkpoint_errors,
            "last_result": self._last_result,
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        if not self.running:
            status = "ERROR"
            message = "WAL cleanup loop not running"
        elif not self.config.enabled:
            status = "HEALTHY"
            message = "WAL cleanup disabled"
        elif self._checkpoint_errors > 5:
            status = "DEGRADED"
            message = f"Multiple checkpoint errors: {self._checkpoint_errors}"
        else:
            mb_reclaimed = round(self._total_bytes_reclaimed / (1024 * 1024), 1)
            status = "HEALTHY"
            message = f"Reclaimed {mb_reclaimed} MB from WAL files"

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "enabled": self.config.enabled,
                "total_databases_checkpointed": self._databases_checkpointed,
                "total_bytes_reclaimed_mb": round(
                    self._total_bytes_reclaimed / (1024 * 1024), 2
                ),
                "total_errors": self._checkpoint_errors,
                "run_count": self.stats.total_runs,
            },
        }


# =============================================================================
# HTTP Pool Monitor Loop (March 2026 - 7-Day Autonomous Operation)
# =============================================================================


# Default interval: 4 hours (14400 seconds)
HTTP_POOL_MONITOR_INTERVAL = 14400.0

# Max session age before forced recreation: 24 hours
HTTP_POOL_MAX_SESSION_AGE = 86400.0

# Connection usage threshold: 80% of pool limit triggers recreation
HTTP_POOL_USAGE_THRESHOLD = 0.80


@dataclass
class HttpPoolMonitorConfig:
    """Configuration for HTTP connection pool monitoring loop.

    March 2026: After 72+ hours of autonomous operation, the HTTP connection
    pool (aiohttp TCPConnector) can exhaust with TIME_WAIT sockets due to
    long-lived connections that are never recycled. This loop monitors pool
    health and proactively recreates the session when:
    1. Connection count exceeds 80% of the pool limit
    2. The session has been alive longer than 24 hours
    """

    check_interval_seconds: float = HTTP_POOL_MONITOR_INTERVAL
    max_session_age_seconds: float = HTTP_POOL_MAX_SESSION_AGE
    usage_threshold: float = HTTP_POOL_USAGE_THRESHOLD
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        if self.max_session_age_seconds <= 0:
            raise ValueError("max_session_age_seconds must be > 0")
        if not (0.0 < self.usage_threshold <= 1.0):
            raise ValueError("usage_threshold must be in (0.0, 1.0]")


class HttpPoolMonitorLoop(BaseLoop):
    """Background loop to monitor and recycle the HTTP connection pool.

    After 72+ hours of continuous operation, the orchestrator's shared
    aiohttp.ClientSession accumulates connections in TIME_WAIT state.
    The TCPConnector has a finite pool (default 100 total, 10 per host),
    and once exhausted, all outbound HTTP requests block or fail.

    This loop runs every 4 hours and:
    1. Inspects the TCPConnector's internal connection state
    2. Logs pool utilization metrics (connections, free slots, waiters)
    3. Triggers a graceful session recreation if:
       - Connection count exceeds 80% of the pool limit, OR
       - The session has been alive longer than 24 hours
    4. Also monitors the PeerConnectionPool (connection_pool.py) stats

    Session recreation is done via the orchestrator's recreate_http_session()
    method, which closes the old session and creates a fresh one. All callers
    that access orchestrator.http_session will transparently get the new session
    because it is a lazy property.

    March 2026: Added for 7-day autonomous operation support.

    Benefits:
    - Prevents TIME_WAIT socket exhaustion after 72+ hours
    - Proactive recycling before pool is fully exhausted
    - Zero-downtime recreation (old session closed, new one created)
    - Logs pool health metrics for observability
    """

    def __init__(
        self,
        get_http_session: Callable[[], Any] | None = None,
        get_session_created_at: Callable[[], float] | None = None,
        recreate_http_session: Callable[[], Coroutine[Any, Any, None]] | None = None,
        config: HttpPoolMonitorConfig | None = None,
    ):
        """Initialize HTTP pool monitor loop.

        Args:
            get_http_session: Callback returning the orchestrator's current
                aiohttp.ClientSession (or None if not yet created).
            get_session_created_at: Callback returning the timestamp when the
                current HTTP session was created (float, time.time()).
            recreate_http_session: Async callback to close the old session and
                create a fresh one on the orchestrator.
            config: Optional configuration (uses defaults if not provided).
        """
        self.config = config or HttpPoolMonitorConfig()
        super().__init__(
            name="http_pool_monitor",
            interval=self.config.check_interval_seconds,
        )

        self._get_http_session = get_http_session
        self._get_session_created_at = get_session_created_at
        self._recreate_http_session = recreate_http_session

        # Statistics
        self._checks_count = 0
        self._recreations_triggered = 0
        self._recreation_reason: str = ""
        self._last_pool_stats: dict[str, Any] = {}

    async def _on_start(self) -> None:
        """Log startup configuration."""
        logger.info(
            f"[HttpPoolMonitor] Started "
            f"(interval: {self.config.check_interval_seconds:.0f}s, "
            f"max_age: {self.config.max_session_age_seconds:.0f}s, "
            f"usage_threshold: {self.config.usage_threshold:.0%})"
        )

    def _get_connector_stats(self, session: Any) -> dict[str, Any]:
        """Extract connection pool statistics from an aiohttp.ClientSession.

        Inspects the TCPConnector's internal state to determine pool
        utilization. This uses internal aiohttp attributes (_conns, _limit,
        _acquired) which may change between aiohttp versions, so all
        access is wrapped in try/except.

        Args:
            session: An aiohttp.ClientSession instance.

        Returns:
            Dict with pool statistics, or empty dict on failure.
        """
        stats: dict[str, Any] = {}
        try:
            connector = getattr(session, "connector", None)
            if connector is None:
                return stats

            # Total connection limit
            limit = getattr(connector, "_limit", 0)
            stats["limit"] = limit

            # Per-host connection limit
            limit_per_host = getattr(connector, "_limit_per_host", 0)
            stats["limit_per_host"] = limit_per_host

            # Active/acquired connections
            acquired = getattr(connector, "_acquired", set())
            acquired_count = len(acquired) if acquired else 0
            stats["acquired"] = acquired_count

            # Pooled (idle) connections in _conns dict
            conns = getattr(connector, "_conns", {})
            pooled_count = 0
            hosts_with_connections = 0
            if conns:
                hosts_with_connections = len(conns)
                for key, conns_list in conns.items():
                    pooled_count += len(conns_list) if conns_list else 0
            stats["pooled"] = pooled_count
            stats["hosts_with_connections"] = hosts_with_connections

            # Total connections (acquired + pooled)
            total = acquired_count + pooled_count
            stats["total_connections"] = total

            # Usage ratio
            if limit > 0:
                stats["usage_ratio"] = total / limit
            else:
                stats["usage_ratio"] = 0.0

            # Waiters (requests waiting for a connection)
            # aiohttp uses _waiters dict of {key: list[asyncio.Future]}
            waiters = getattr(connector, "_waiters", {})
            waiter_count = 0
            if waiters:
                for key, waiter_list in waiters.items():
                    waiter_count += len(waiter_list) if waiter_list else 0
            stats["waiters"] = waiter_count

            # Whether the connector is closed
            stats["closed"] = getattr(connector, "closed", False)

        except Exception as e:
            logger.debug(f"[HttpPoolMonitor] Error reading connector stats: {e}")
            stats["error"] = str(e)

        return stats

    async def _run_once(self) -> None:
        """Run one monitoring cycle."""
        if not self.config.enabled:
            return

        self._checks_count += 1

        try:
            session = self._get_http_session() if self._get_http_session else None
            session_created_at = (
                self._get_session_created_at() if self._get_session_created_at else 0.0
            )
            now = time.time()

            # Collect connector stats
            connector_stats: dict[str, Any] = {}
            if session is not None and not getattr(session, "closed", True):
                connector_stats = self._get_connector_stats(session)

            # Collect PeerConnectionPool stats
            peer_pool_stats: dict[str, Any] = {}
            try:
                from scripts.p2p.connection_pool import get_connection_pool

                pool = get_connection_pool()
                peer_pool_stats = pool.get_pool_stats()
            except Exception:
                pass  # Connection pool not initialized

            # Compute session age
            session_age = now - session_created_at if session_created_at > 0 else 0.0

            self._last_pool_stats = {
                "session_age_seconds": round(session_age, 1),
                "session_age_hours": round(session_age / 3600, 2),
                "connector": connector_stats,
                "peer_pool": peer_pool_stats,
            }

            # Log stats
            usage_ratio = connector_stats.get("usage_ratio", 0.0)
            total_conns = connector_stats.get("total_connections", 0)
            limit = connector_stats.get("limit", 0)
            waiters = connector_stats.get("waiters", 0)
            acquired = connector_stats.get("acquired", 0)
            pooled = connector_stats.get("pooled", 0)

            logger.info(
                f"[HttpPoolMonitor] Pool stats: "
                f"connections={total_conns}/{limit} ({usage_ratio:.1%}), "
                f"acquired={acquired}, pooled={pooled}, waiters={waiters}, "
                f"session_age={session_age / 3600:.1f}h"
            )

            if peer_pool_stats:
                logger.info(
                    f"[HttpPoolMonitor] PeerPool: "
                    f"total={peer_pool_stats.get('current_total', 0)}"
                    f"/{peer_pool_stats.get('max_total', 0)}, "
                    f"peers={peer_pool_stats.get('peer_count', 0)}, "
                    f"resizes={peer_pool_stats.get('resize_count', 0)}"
                )

            # Check if recreation is needed
            needs_recreation = False
            reason = ""

            # Condition 1: Connection usage exceeds threshold
            if usage_ratio >= self.config.usage_threshold and limit > 0:
                needs_recreation = True
                reason = (
                    f"connection usage {usage_ratio:.1%} >= "
                    f"threshold {self.config.usage_threshold:.0%} "
                    f"({total_conns}/{limit})"
                )

            # Condition 2: Session age exceeds max
            if (
                not needs_recreation
                and session_age > self.config.max_session_age_seconds
                and session_created_at > 0
            ):
                needs_recreation = True
                reason = (
                    f"session age {session_age / 3600:.1f}h > "
                    f"max {self.config.max_session_age_seconds / 3600:.0f}h"
                )

            # Condition 3: Waiters indicate pool exhaustion is imminent
            if not needs_recreation and waiters > 5:
                needs_recreation = True
                reason = f"high waiter count ({waiters} requests waiting for connections)"

            if needs_recreation and self._recreate_http_session:
                logger.warning(
                    f"[HttpPoolMonitor] Triggering session recreation: {reason}"
                )
                try:
                    await self._recreate_http_session()
                    self._recreations_triggered += 1
                    self._recreation_reason = reason
                    logger.info(
                        "[HttpPoolMonitor] Session recreated successfully "
                        f"(total recreations: {self._recreations_triggered})"
                    )
                except Exception as e:
                    logger.error(
                        f"[HttpPoolMonitor] Failed to recreate session: {e}"
                    )

        except Exception as e:
            logger.warning(f"[HttpPoolMonitor] Error in monitoring cycle: {e}")

    def get_monitor_stats(self) -> dict[str, Any]:
        """Get HTTP pool monitor statistics."""
        return {
            "enabled": self.config.enabled,
            "check_interval_seconds": self.config.check_interval_seconds,
            "max_session_age_seconds": self.config.max_session_age_seconds,
            "usage_threshold": self.config.usage_threshold,
            "checks_count": self._checks_count,
            "recreations_triggered": self._recreations_triggered,
            "last_recreation_reason": self._recreation_reason,
            "last_pool_stats": self._last_pool_stats,
            **self.stats.to_dict(),
        }

    def health_check(self) -> dict[str, Any]:
        """Return health status for DaemonManager integration.

        Returns:
            HealthCheckResult-compatible dict with status, message, and details.
        """
        if not self.running:
            status = "ERROR"
            message = "HTTP pool monitor loop not running"
        elif not self.config.enabled:
            status = "HEALTHY"
            message = "HTTP pool monitor disabled"
        else:
            usage = self._last_pool_stats.get("connector", {}).get("usage_ratio", 0.0)
            if usage >= 0.95:
                status = "DEGRADED"
                message = f"Pool near exhaustion: {usage:.0%} used"
            else:
                status = "HEALTHY"
                message = (
                    f"Pool healthy ({usage:.0%} used), "
                    f"{self._recreations_triggered} recreations"
                )

        return {
            "status": status,
            "message": message,
            "details": {
                "is_running": self.running,
                "enabled": self.config.enabled,
                "checks_count": self._checks_count,
                "recreations_triggered": self._recreations_triggered,
                "last_recreation_reason": self._recreation_reason,
                "run_count": self.stats.total_runs,
            },
        }


__all__ = [
    "GitUpdateConfig",
    "GitUpdateLoop",
    "CircuitBreakerDecayConfig",
    "CircuitBreakerDecayLoop",
    "WalCleanupConfig",
    "WalCleanupLoop",
    "HttpPoolMonitorConfig",
    "HttpPoolMonitorLoop",
]
