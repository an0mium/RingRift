"""Standalone daemon factory functions (legacy compatibility layer).

This module historically provided small, test-friendly factory functions for
spawning individual background daemons.

In Dec 2025 the daemon lifecycle stack was refactored and most production
startup moved to [`app.coordination.daemon_runners`](ai-service/app/coordination/daemon_runners.py:1)
(and the declarative registry used by [`DaemonManager`](ai-service/app/coordination/daemon_manager.py:97)).

Some tests (and potentially external tooling) still import
`app.coordination.daemon_factory_implementations`. To keep the test suite
stable (and avoid forcing callers to update immediately), we retain this module
as a thin shim.

The factories here intentionally:
- Import their target classes lazily (inside the function) to ease mocking.
- Start the daemon/controller.
- Return quickly when the daemon indicates it is not running.

These factories are *not* the primary production startup path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


async def _wait_while_running(obj: Any, *, poll_interval: float = 0.1) -> None:
    """Best-effort wait loop for legacy daemons.

    Many daemons expose a `_running` flag (or similar). Tests typically patch this
    to `False` so factories return immediately.
    """

    while bool(getattr(obj, "_running", False)):
        await asyncio.sleep(poll_interval)


async def create_auto_sync() -> None:
    """Create and start [`AutoSyncDaemon`](ai-service/app/coordination/auto_sync_daemon.py:1)."""

    from app.coordination.auto_sync_daemon import AutoSyncDaemon

    daemon = AutoSyncDaemon()
    await daemon.start()
    await _wait_while_running(daemon)


async def create_model_distribution() -> None:
    """Create and start a model distribution daemon."""

    from app.coordination.unified_distribution_daemon import DataType, UnifiedDistributionDaemon

    daemon = UnifiedDistributionDaemon(data_type=DataType.MODEL)
    await daemon.start()
    await _wait_while_running(daemon)


async def create_npz_distribution() -> None:
    """Create and start an NPZ distribution daemon."""

    from app.coordination.unified_distribution_daemon import DataType, UnifiedDistributionDaemon

    daemon = UnifiedDistributionDaemon(data_type=DataType.NPZ)
    await daemon.start()
    await _wait_while_running(daemon)


async def create_evaluation() -> None:
    """Create and start [`EvaluationDaemon`](ai-service/app/coordination/evaluation_daemon.py:1)."""

    from app.coordination.evaluation_daemon import EvaluationDaemon

    daemon = EvaluationDaemon()
    await daemon.start()
    await _wait_while_running(daemon)


async def create_auto_promotion() -> None:
    """Create and run [`PromotionController`](ai-service/app/training/promotion_controller.py:1)."""

    from app.training.promotion_controller import PromotionController

    controller = PromotionController()
    await controller.run()


async def create_quality_monitor() -> None:
    """Create and start [`QualityMonitorDaemon`](ai-service/app/coordination/quality_monitor_daemon.py:1)."""

    from app.coordination.quality_monitor_daemon import QualityMonitorDaemon

    daemon = QualityMonitorDaemon()
    await daemon.start()
    await _wait_while_running(daemon)


async def create_feedback_loop() -> None:
    """Create and start [`FeedbackLoopController`](ai-service/app/coordination/feedback_loop_controller.py:1)."""

    from app.coordination.feedback_loop_controller import FeedbackLoopController

    controller = FeedbackLoopController()
    await controller.start()
    await _wait_while_running(controller)


async def create_data_pipeline() -> None:
    """Create and start [`DataPipelineOrchestrator`](ai-service/app/coordination/data_pipeline_orchestrator.py:1)."""

    from app.coordination.data_pipeline_orchestrator import DataPipelineOrchestrator

    orchestrator = DataPipelineOrchestrator()
    await orchestrator.start()
    await _wait_while_running(orchestrator)


FACTORY_REGISTRY: dict[str, Callable[[], Awaitable[None]]] = {
    "AUTO_SYNC": create_auto_sync,
    "MODEL_DISTRIBUTION": create_model_distribution,
    "NPZ_DISTRIBUTION": create_npz_distribution,
    "EVALUATION": create_evaluation,
    "AUTO_PROMOTION": create_auto_promotion,
    "QUALITY_MONITOR": create_quality_monitor,
    "FEEDBACK_LOOP": create_feedback_loop,
    "DATA_PIPELINE": create_data_pipeline,
}


def get_factory(name: str) -> Callable[[], Awaitable[None]] | None:
    """Look up a factory by registry key.

    The lookup is case-sensitive by design.
    """

    return FACTORY_REGISTRY.get(name)


__all__ = [
    "create_auto_sync",
    "create_model_distribution",
    "create_npz_distribution",
    "create_evaluation",
    "create_auto_promotion",
    "create_quality_monitor",
    "create_feedback_loop",
    "create_data_pipeline",
    "FACTORY_REGISTRY",
    "get_factory",
]
