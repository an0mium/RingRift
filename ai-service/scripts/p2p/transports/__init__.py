"""
Transport implementations for the transport cascade.

Dec 30, 2025: Comprehensive transport layer with tiered failover.
Jan 19, 2026: Added P2PD UDP hole punching transport for CGNAT bypass.
"""

from .http_transports import (
    DirectHTTPTransport,
    TailscaleHTTPTransport,
    CloudflareHTTPTransport,
)
from .ssh_transport import SSHTunnelTransport
from .relay_transport import P2PRelayTransport
from .notification_transports import (
    SlackWebhookTransport,
    DiscordWebhookTransport,
    TelegramTransport,
    EmailTransport,
    SMSTransport,
    PagerDutyTransport,
)

# P2PD transport - optional, graceful fallback if not installed
try:
    from .p2pd_transport import P2PDUDPTransport

    _P2PD_AVAILABLE = True
except ImportError:
    P2PDUDPTransport = None  # type: ignore
    _P2PD_AVAILABLE = False

__all__ = [
    # Tier 1-2: Fast/Reliable
    "DirectHTTPTransport",
    "TailscaleHTTPTransport",
    "P2PDUDPTransport",  # Tier 1.5: UDP hole punching (CGNAT bypass)
    # Tier 3: Tunneled
    "CloudflareHTTPTransport",
    "SSHTunnelTransport",
    # Tier 4: Relay
    "P2PRelayTransport",
    # Tier 5: External
    "SlackWebhookTransport",
    "DiscordWebhookTransport",
    "TelegramTransport",
    "EmailTransport",
    # Tier 6: Manual
    "SMSTransport",
    "PagerDutyTransport",
]
