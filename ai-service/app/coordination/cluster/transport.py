"""Cluster transport layer (December 2025).

.. deprecated:: December 2025
    Import directly from app.coordination.cluster_transport instead.

Re-exports from cluster_transport.py for unified access.

Usage:
    # DEPRECATED:
    from app.coordination.cluster.transport import ClusterTransport

    # RECOMMENDED:
    from app.coordination.cluster_transport import ClusterTransport
"""

import warnings

warnings.warn(
    "app.coordination.cluster.transport is deprecated. "
    "Import directly from app.coordination.cluster_transport instead.",
    DeprecationWarning,
    stacklevel=2,
)

from app.coordination.cluster_transport import (
    ClusterTransport,
    NodeConfig,
    PermanentTransportError,
    RetryableTransportError,
    TransportConfig,
    TransportError,
    TransportResult,
    get_cluster_transport,
)

__all__ = [
    "ClusterTransport",
    "NodeConfig",
    "PermanentTransportError",
    "RetryableTransportError",
    "TransportConfig",
    "TransportError",
    "TransportResult",
    "get_cluster_transport",
]
