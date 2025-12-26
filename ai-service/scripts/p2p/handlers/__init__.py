"""P2P Handler Mixins.

This package contains mixin classes for HTTP handlers extracted from p2p_orchestrator.py.
These mixins are designed to be inherited by P2POrchestrator.

Usage:
    from scripts.p2p.handlers import (
        ElectionHandlersMixin,
        GauntletHandlersMixin,
        RelayHandlersMixin,
        WorkQueueHandlersMixin,
    )

    class P2POrchestrator(
        WorkQueueHandlersMixin,
        ElectionHandlersMixin,
        RelayHandlersMixin,
        GauntletHandlersMixin,
    ):
        pass
"""

from .admin import AdminHandlersMixin
from .election import ElectionHandlersMixin
from .elo_sync import EloSyncHandlersMixin
from .gauntlet import GauntletHandlersMixin
from .gossip import GossipHandlersMixin
from .relay import RelayHandlersMixin
from .work_queue import WorkQueueHandlersMixin

__all__ = [
    "AdminHandlersMixin",
    "ElectionHandlersMixin",
    "EloSyncHandlersMixin",
    "GauntletHandlersMixin",
    "GossipHandlersMixin",
    "RelayHandlersMixin",
    "WorkQueueHandlersMixin",
]
