"""Elo Progress Daemon - Periodically snapshots best model Elo for each config.

This daemon runs hourly (default) to capture the Elo rating of the best model
for each board configuration, enabling tracking of improvement over time.

Usage:
    # As standalone daemon
    python -m app.coordination.elo_progress_daemon

    # Via DaemonManager
    from app.coordination.daemon_manager import get_daemon_manager
    from app.coordination.daemon_types import DaemonType
    dm = get_daemon_manager()
    await dm.start(DaemonType.ELO_PROGRESS)

December 31, 2025: Created for training loop effectiveness tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.coordination.handler_base import HandlerBase, HealthCheckResult

logger = logging.getLogger(__name__)

# Default snapshot interval: 1 hour
DEFAULT_SNAPSHOT_INTERVAL = 3600.0

# Minimum interval between snapshots (prevent spam)
MIN_SNAPSHOT_INTERVAL = 300.0  # 5 minutes


@dataclass
class EloProgressDaemonConfig:
    """Configuration for EloProgressDaemon."""
    snapshot_interval: float = DEFAULT_SNAPSHOT_INTERVAL
    enabled: bool = True


class EloProgressDaemon(HandlerBase):
    """Daemon that periodically snapshots best model Elo for each config."""

    _instance: EloProgressDaemon | None = None

    def __init__(self, config: EloProgressDaemonConfig | None = None):
        super().__init__(
            name="elo_progress_daemon",
            cycle_interval=config.snapshot_interval if config else DEFAULT_SNAPSHOT_INTERVAL,
        )
        self.config = config or EloProgressDaemonConfig()
        self._last_snapshot_time: float = 0.0
        self._snapshot_count: int = 0
        self._last_error: str | None = None

    @classmethod
    def get_instance(cls) -> EloProgressDaemon:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    async def _run_cycle(self) -> None:
        """Take Elo snapshots for all configs."""
        if not self.config.enabled:
            logger.debug("[EloProgress] Daemon disabled, skipping snapshot")
            return

        # Rate limit snapshots
        now = time.time()
        if now - self._last_snapshot_time < MIN_SNAPSHOT_INTERVAL:
            logger.debug("[EloProgress] Skipping snapshot - too soon since last")
            return

        try:
            from app.coordination.elo_progress_tracker import snapshot_all_configs

            logger.info("[EloProgress] Taking Elo snapshots for all configs...")
            results = await snapshot_all_configs()

            # Count successful snapshots
            success_count = sum(1 for v in results.values() if v is not None)
            total_count = len(results)

            self._last_snapshot_time = now
            self._snapshot_count += 1
            self._last_error = None

            logger.info(
                f"[EloProgress] Snapshot complete: {success_count}/{total_count} configs recorded "
                f"(total snapshots: {self._snapshot_count})"
            )

            # Log any configs that had models
            for config_key, snapshot in sorted(results.items()):
                if snapshot:
                    logger.debug(
                        f"[EloProgress]   {config_key}: {snapshot.best_elo:.1f} Elo "
                        f"({snapshot.games_played} games)"
                    )

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"[EloProgress] Snapshot failed: {e}")

    def _get_event_subscriptions(self) -> dict[str, Any]:
        """Subscribe to evaluation events for immediate snapshots."""
        return {
            "EVALUATION_COMPLETED": self._on_evaluation_completed,
            "MODEL_PROMOTED": self._on_model_promoted,
        }

    async def _on_evaluation_completed(self, event: dict[str, Any]) -> None:
        """Take a snapshot when a model evaluation completes."""
        # Only snapshot if enough time has passed
        if time.time() - self._last_snapshot_time < MIN_SNAPSHOT_INTERVAL:
            return

        config_key = event.get("config_key") or event.get("payload", {}).get("config_key")
        if not config_key:
            return

        logger.info(f"[EloProgress] Evaluation completed for {config_key}, triggering snapshot")
        await self._run_cycle()

    async def _on_model_promoted(self, event: dict[str, Any]) -> None:
        """Take a snapshot when a model is promoted."""
        # Only snapshot if enough time has passed
        if time.time() - self._last_snapshot_time < MIN_SNAPSHOT_INTERVAL:
            return

        config_key = event.get("config_key") or event.get("payload", {}).get("config_key")
        if not config_key:
            return

        logger.info(f"[EloProgress] Model promoted for {config_key}, triggering snapshot")
        await self._run_cycle()

    def health_check(self) -> HealthCheckResult:
        """Return health status."""
        is_healthy = self._last_error is None

        return HealthCheckResult(
            healthy=is_healthy,
            status="healthy" if is_healthy else "degraded",
            message=self._last_error or f"Snapshots: {self._snapshot_count}",
            details={
                "snapshot_count": self._snapshot_count,
                "last_snapshot_time": self._last_snapshot_time,
                "last_error": self._last_error,
                "enabled": self.config.enabled,
                "interval": self.config.snapshot_interval,
            },
        )


def get_elo_progress_daemon() -> EloProgressDaemon:
    """Get singleton EloProgressDaemon."""
    return EloProgressDaemon.get_instance()


async def create_elo_progress_daemon() -> EloProgressDaemon:
    """Factory function for DaemonManager."""
    return get_elo_progress_daemon()


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Elo Progress Daemon")
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_SNAPSHOT_INTERVAL,
        help=f"Snapshot interval in seconds (default: {DEFAULT_SNAPSHOT_INTERVAL})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Take one snapshot and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    async def main():
        config = EloProgressDaemonConfig(snapshot_interval=args.interval)
        daemon = EloProgressDaemon(config)

        if args.once:
            await daemon._run_cycle()
        else:
            await daemon.start()
            try:
                # Run forever
                while daemon._running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                await daemon.stop()

    asyncio.run(main())
