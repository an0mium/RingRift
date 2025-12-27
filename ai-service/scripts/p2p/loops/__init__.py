"""P2P Orchestrator Background Loops.

Created as part of Phase 4 of the P2P orchestrator decomposition (December 2025).

This package contains background loop implementations extracted from p2p_orchestrator.py
for better modularity and testability.

Loop Categories:
- base.py: BaseLoop abstract class with error handling and backoff
- data_loops.py: JSONL conversion, model sync (TODO)
- network_loops.py: IP updates, Tailscale recovery (TODO)
- coordination_loops.py: Elo sync, work queue, auto-scaling (TODO)
- job_loops.py: Job reaper, idle detection (TODO)

Usage:
    from scripts.p2p.loops import BaseLoop, BackoffConfig, LoopManager, LoopStats

    class MyLoop(BaseLoop):
        async def _run_once(self) -> None:
            # Your loop logic here
            pass

    # Single loop usage
    loop = MyLoop(name="my_loop", interval=60.0)
    await loop.run_forever()

    # Multiple loops with manager
    manager = LoopManager()
    manager.register(MyLoop(name="loop1", interval=30.0))
    manager.register(MyLoop(name="loop2", interval=60.0))
    await manager.start_all()
    # ... later ...
    await manager.stop_all()
"""

from .base import (
    BackoffConfig,
    BaseLoop,
    LoopManager,
    LoopStats,
)

__all__ = [
    "BackoffConfig",
    "BaseLoop",
    "LoopManager",
    "LoopStats",
]
