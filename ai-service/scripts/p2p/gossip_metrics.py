"""Gossip metrics mixin - DEPRECATED.

This module is deprecated and will be removed in Q2 2026.
The GossipMetricsMixin has been merged into GossipProtocolMixin.

For new code, use GossipProtocolMixin directly:

    from scripts.p2p.gossip_protocol import GossipProtocolMixin

    class MyClass(GossipProtocolMixin):
        pass

The standalone calculate_compression_ratio function is still available here
for backward compatibility but will also be removed in Q2 2026.

December 28, 2025: Merged into gossip_protocol.py for consolidation.
"""

from __future__ import annotations

import warnings

# Issue deprecation warning on import
warnings.warn(
    "scripts.p2p.gossip_metrics is deprecated. "
    "Use GossipProtocolMixin from scripts.p2p.gossip_protocol directly. "
    "This module will be removed in Q2 2026.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export GossipProtocolMixin as GossipMetricsMixin for backward compatibility
from scripts.p2p.gossip_protocol import GossipProtocolMixin as GossipMetricsMixin


def calculate_compression_ratio(original: int, compressed: int) -> float:
    """Calculate compression ratio.

    DEPRECATED: This function will be removed in Q2 2026.
    Use GossipProtocolMixin._record_gossip_compression() instead.

    Args:
        original: Original size in bytes
        compressed: Compressed size in bytes

    Returns:
        Ratio of bytes saved (0.0 to 1.0). Negative if expansion occurred.
    """
    if original <= 0:
        return 0.0
    return 1.0 - (compressed / original)


__all__ = ["GossipMetricsMixin", "calculate_compression_ratio"]
