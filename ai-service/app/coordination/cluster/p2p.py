"""P2P backend coordination (December 2025).

Re-exports from p2p_backend.py for unified access.

Usage:
    from app.coordination.cluster.p2p import P2PBackend
"""

from app.coordination.p2p_backend import (
    P2PBackend,
    P2PConfig,
    PeerInfo,
    LeaderElection,
)

__all__ = [
    "P2PBackend",
    "P2PConfig",
    "PeerInfo",
    "LeaderElection",
]
