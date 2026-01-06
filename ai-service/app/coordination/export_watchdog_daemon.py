"""Export Watchdog Daemon - Kill hung export scripts.

This daemon monitors export_replay_dataset.py processes and kills them
if they hang for too long, preventing pipeline deadlocks.

Jan 6, 2026: Created for Phase 3 automation - prevents export deadlocks
from blocking the training pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)


@dataclass
class ExportWatchdogConfig:
    """Configuration for ExportWatchdogDaemon."""

    # Maximum runtime before killing an export process (seconds)
    max_export_runtime: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_EXPORT_MAX_RUNTIME", "3600")  # 60 min
        )
    )

    # How often to check for stuck exports (seconds)
    check_interval: float = field(
        default_factory=lambda: float(
            os.environ.get("RINGRIFT_EXPORT_CHECK_INTERVAL", "60")
        )
    )

    # Minimum cycles before considering a process stuck
    min_checks_before_kill: int = field(
        default_factory=lambda: int(
            os.environ.get("RINGRIFT_EXPORT_MIN_CHECKS", "5")
        )
    )

    # Enable dry run (log but don't kill)
    dry_run: bool = field(
        default_factory=lambda: os.environ.get(
            "RINGRIFT_EXPORT_WATCHDOG_DRY_RUN", ""
        ).lower()
        in ("true", "1", "yes")
    )


@dataclass
class ExportProcessInfo:
    """Information about a running export process."""

    pid: int
    config_key: str
    start_time: float
    checks_count: int = 0
    last_check_time: float = field(default_factory=time.time)

    @property
    def runtime_seconds(self) -> float:
        """Get runtime in seconds."""
        return time.time() - self.start_time

    @property
    def runtime_minutes(self) -> float:
        """Get runtime in minutes."""
        return self.runtime_seconds / 60


class ExportWatchdogDaemon(HandlerBase):
    """Daemon that kills export scripts that hang for too long.

    Monitors export_replay_dataset.py processes and terminates them
    if they exceed the maximum runtime threshold.

    Events emitted:
    - EXPORT_TIMEOUT: When an export process is killed
    - EXPORT_WATCHDOG_STARTED: When watchdog monitoring starts
    - EXPORT_WATCHDOG_HEALTHY: When all exports are running normally
    """

    _event_source = "ExportWatchdogDaemon"
    _config_class = ExportWatchdogConfig

    def __init__(self, config: ExportWatchdogConfig | None = None):
        """Initialize the export watchdog daemon.

        Args:
            config: Optional configuration override
        """
        self._watchdog_config = config or ExportWatchdogConfig()
        super().__init__(
            name="export_watchdog",
            cycle_interval=self._watchdog_config.check_interval,
        )

        # Track export processes we're monitoring
        self._tracked_exports: dict[int, ExportProcessInfo] = {}

        # Stats
        self._exports_killed = 0
        self._exports_completed_normally = 0
        self._false_positives = 0  # Processes that finished before we killed them

    def _get_event_subscriptions(self) -> dict:
        """Get event subscriptions.

        Currently no subscriptions - this daemon proactively monitors processes.
        """
        return {}

    async def _run_cycle(self) -> None:
        """Main watchdog cycle - check for stuck exports."""
        try:
            # Find export processes
            export_processes = await self._find_export_processes()

            # Update tracking
            current_pids = set(export_processes.keys())
            tracked_pids = set(self._tracked_exports.keys())

            # Add new processes
            for pid, info in export_processes.items():
                if pid not in tracked_pids:
                    self._tracked_exports[pid] = info
                    logger.info(
                        f"[ExportWatchdog] Now tracking export PID {pid} "
                        f"for config {info.config_key}"
                    )

            # Remove completed processes
            for pid in tracked_pids - current_pids:
                info = self._tracked_exports.pop(pid, None)
                if info:
                    self._exports_completed_normally += 1
                    logger.debug(
                        f"[ExportWatchdog] Export PID {pid} completed normally "
                        f"after {info.runtime_minutes:.1f}min"
                    )

            # Check for stuck exports
            for pid, info in list(self._tracked_exports.items()):
                info.checks_count += 1
                info.last_check_time = time.time()

                if (
                    info.runtime_seconds > self._watchdog_config.max_export_runtime
                    and info.checks_count >= self._watchdog_config.min_checks_before_kill
                ):
                    await self._kill_export(pid, info)

        except Exception as e:
            logger.error(f"[ExportWatchdog] Error in cycle: {e}")
            self._record_error(f"Cycle error: {e}", e)

    async def _find_export_processes(self) -> dict[int, ExportProcessInfo]:
        """Find running export_replay_dataset.py processes."""
        processes: dict[int, ExportProcessInfo] = {}

        try:
            # Use pgrep to find export processes
            result = await asyncio.to_thread(
                subprocess.run,
                ["pgrep", "-af", "export_replay_dataset"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue

                    parts = line.split(maxsplit=1)
                    if len(parts) < 2:
                        continue

                    try:
                        pid = int(parts[0])
                        cmdline = parts[1]

                        # Parse config from command line
                        config_key = self._parse_config_from_cmdline(cmdline)

                        # Get process start time
                        start_time = await self._get_process_start_time(pid)

                        processes[pid] = ExportProcessInfo(
                            pid=pid,
                            config_key=config_key,
                            start_time=start_time,
                        )
                    except (ValueError, IndexError):
                        continue

        except subprocess.TimeoutExpired:
            logger.warning("[ExportWatchdog] pgrep timed out")
        except Exception as e:
            logger.error(f"[ExportWatchdog] Error finding exports: {e}")

        return processes

    def _parse_config_from_cmdline(self, cmdline: str) -> str:
        """Parse board_type and num_players from command line."""
        board_type = "unknown"
        num_players = 2

        # Look for --board-type
        if "--board-type" in cmdline:
            parts = cmdline.split("--board-type")
            if len(parts) > 1:
                value = parts[1].split()[0] if parts[1].split() else "unknown"
                board_type = value.strip()

        # Look for --num-players
        if "--num-players" in cmdline:
            parts = cmdline.split("--num-players")
            if len(parts) > 1:
                try:
                    value = parts[1].split()[0]
                    num_players = int(value)
                except (ValueError, IndexError):
                    pass

        return f"{board_type}_{num_players}p"

    async def _get_process_start_time(self, pid: int) -> float:
        """Get the start time of a process."""
        try:
            # Use ps to get process start time
            result = await asyncio.to_thread(
                subprocess.run,
                ["ps", "-p", str(pid), "-o", "lstart="],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                # Parse the date (format: "Mon Jan  6 16:58:00 2026")
                import datetime

                try:
                    dt = datetime.datetime.strptime(
                        result.stdout.strip(), "%a %b %d %H:%M:%S %Y"
                    )
                    return dt.timestamp()
                except ValueError:
                    pass

        except Exception:
            pass

        # Fallback: use current time (process just started)
        return time.time()

    async def _kill_export(self, pid: int, info: ExportProcessInfo) -> None:
        """Kill a stuck export process."""
        if self._watchdog_config.dry_run:
            logger.warning(
                f"[ExportWatchdog] DRY RUN: Would kill export PID {pid} "
                f"({info.config_key}) after {info.runtime_minutes:.1f}min"
            )
            return

        logger.warning(
            f"[ExportWatchdog] Killing stuck export PID {pid} "
            f"({info.config_key}) after {info.runtime_minutes:.1f}min"
        )

        try:
            # First try SIGTERM
            os.kill(pid, 15)
            await asyncio.sleep(5)

            # Check if still running
            try:
                os.kill(pid, 0)  # Check if process exists
                # Still running, use SIGKILL
                os.kill(pid, 9)
                logger.warning(f"[ExportWatchdog] Sent SIGKILL to PID {pid}")
            except ProcessLookupError:
                pass  # Process already terminated

            self._exports_killed += 1
            self._tracked_exports.pop(pid, None)

            # Emit event
            await self._safe_emit_event_async(
                "EXPORT_TIMEOUT",
                {
                    "pid": pid,
                    "config_key": info.config_key,
                    "runtime_minutes": info.runtime_minutes,
                    "reason": "exceeded_max_runtime",
                },
            )

        except ProcessLookupError:
            # Process already gone
            self._false_positives += 1
            self._tracked_exports.pop(pid, None)
            logger.info(
                f"[ExportWatchdog] Export PID {pid} already terminated"
            )
        except PermissionError:
            logger.error(
                f"[ExportWatchdog] Permission denied killing PID {pid}"
            )
        except Exception as e:
            logger.error(f"[ExportWatchdog] Error killing PID {pid}: {e}")

    def health_check(self) -> HealthCheckResult:
        """Return health check status."""
        base_health = super().health_check()

        # Add watchdog-specific details
        details = base_health.details.copy() if base_health.details else {}
        details.update(
            {
                "tracked_exports": len(self._tracked_exports),
                "exports_killed": self._exports_killed,
                "exports_completed": self._exports_completed_normally,
                "false_positives": self._false_positives,
                "max_runtime_min": self._watchdog_config.max_export_runtime / 60,
                "dry_run": self._watchdog_config.dry_run,
            }
        )

        # Check for any long-running exports
        long_running = [
            (pid, info)
            for pid, info in self._tracked_exports.items()
            if info.runtime_minutes > 15
        ]
        if long_running:
            details["long_running_exports"] = [
                {"pid": pid, "config": info.config_key, "runtime_min": info.runtime_minutes}
                for pid, info in long_running
            ]

        return HealthCheckResult(
            status=base_health.status,
            message=base_health.message,
            details=details,
        )


def get_export_watchdog_daemon() -> ExportWatchdogDaemon:
    """Get the singleton ExportWatchdogDaemon instance."""
    return ExportWatchdogDaemon.get_instance()
