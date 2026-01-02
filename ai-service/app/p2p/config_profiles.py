"""P2P Configuration Profiles (Phase 3.1 - January 2026).

Semantic P2P configuration profiles that replace 60+ individual parameters
with easy-to-understand presets. This simplifies cluster configuration and
reduces misconfiguration risks.

Usage:
    from app.p2p.config_profiles import (
        P2PProfile,
        get_profile,
        apply_profile,
        PROFILES,
    )

    # Get profile from environment (RINGRIFT_P2P_PROFILE)
    profile = get_profile()

    # Apply profile to set environment variables
    apply_profile("aggressive")

    # Or use specific profile
    profile = PROFILES["production"]
    print(f"Heartbeat: {profile.heartbeat_interval}s")

Environment Variables:
    RINGRIFT_P2P_PROFILE: Profile name (production, aggressive, relaxed, development)

January 2026: Created as part of long-term stability improvements.
Expected impact: Simpler configuration, fewer misconfigurations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ProfileName",
    "P2PProfile",
    "PROFILES",
    "get_profile",
    "get_profile_name",
    "apply_profile",
    "apply_profile_to_env",
]


class ProfileName(Enum):
    """Available P2P profile names."""

    PRODUCTION = "production"
    AGGRESSIVE = "aggressive"
    RELAXED = "relaxed"
    DEVELOPMENT = "development"


@dataclass(frozen=True)
class P2PProfile:
    """Semantic P2P configuration profile.

    Groups related P2P settings into coherent presets that match
    common deployment scenarios.

    Attributes:
        name: Profile identifier
        description: Human-readable description of when to use this profile

        # Heartbeat and Timeout Settings
        heartbeat_interval: Seconds between heartbeat messages
        peer_timeout: Seconds without heartbeat before marking peer dead
        peer_timeout_nat_blocked: Extended timeout for NAT-blocked peers
        election_timeout: Seconds to wait for election responses

        # Gossip Protocol Settings
        gossip_interval: Seconds between gossip rounds
        gossip_fanout: Number of peers to gossip to each round

        # Leader Settings
        leader_lease_duration: How long a leader lease is valid
        leader_renew_interval: How often leader renews its lease

        # Recovery Settings
        partition_heal_interval: Seconds between partition healing attempts
        peer_retry_interval: Seconds between retrying dead peers

        # Job Management
        job_check_interval: Seconds between job status checks
        job_timeout_multiplier: Multiplier for job timeouts
    """

    name: str
    description: str

    # Heartbeat and Timeout
    heartbeat_interval: float
    peer_timeout: float
    peer_timeout_nat_blocked: float
    election_timeout: float

    # Gossip Protocol
    gossip_interval: float
    gossip_fanout: int

    # Leader Settings
    leader_lease_duration: float
    leader_renew_interval: float

    # Recovery Settings
    partition_heal_interval: float
    peer_retry_interval: float

    # Job Management
    job_check_interval: float
    job_timeout_multiplier: float = 1.0


# =============================================================================
# Profile Definitions
# =============================================================================

PROFILES: dict[str, P2PProfile] = {
    "production": P2PProfile(
        name="production",
        description=(
            "Production profile for stable clusters. Balanced between "
            "responsiveness and resource efficiency. Recommended for "
            "24/7 autonomous operation."
        ),
        # Heartbeat: 15s interval, 90s timeout
        heartbeat_interval=15.0,
        peer_timeout=90.0,
        peer_timeout_nat_blocked=120.0,
        election_timeout=15.0,
        # Gossip: moderate frequency and fanout
        gossip_interval=15.0,
        gossip_fanout=5,
        # Leader: 5-minute lease
        leader_lease_duration=300.0,
        leader_renew_interval=30.0,
        # Recovery: 30s heal interval
        partition_heal_interval=30.0,
        peer_retry_interval=300.0,
        # Jobs: 15s check interval
        job_check_interval=15.0,
        job_timeout_multiplier=1.0,
    ),
    "aggressive": P2PProfile(
        name="aggressive",
        description=(
            "Aggressive profile for fast failure detection. Uses more "
            "network bandwidth but detects issues faster. Suitable for "
            "low-latency networks with reliable connectivity."
        ),
        # Heartbeat: 10s interval, 60s timeout
        heartbeat_interval=10.0,
        peer_timeout=60.0,
        peer_timeout_nat_blocked=90.0,
        election_timeout=10.0,
        # Gossip: high frequency and fanout
        gossip_interval=10.0,
        gossip_fanout=7,
        # Leader: 3-minute lease
        leader_lease_duration=180.0,
        leader_renew_interval=15.0,
        # Recovery: 15s heal interval
        partition_heal_interval=15.0,
        peer_retry_interval=120.0,
        # Jobs: 10s check interval
        job_check_interval=10.0,
        job_timeout_multiplier=0.8,
    ),
    "relaxed": P2PProfile(
        name="relaxed",
        description=(
            "Relaxed profile for resource-constrained or high-latency networks. "
            "Uses less bandwidth and tolerates temporary disconnections. "
            "Suitable for heterogeneous clusters with varying connectivity."
        ),
        # Heartbeat: 30s interval, 120s timeout
        heartbeat_interval=30.0,
        peer_timeout=120.0,
        peer_timeout_nat_blocked=180.0,
        election_timeout=30.0,
        # Gossip: low frequency and fanout
        gossip_interval=30.0,
        gossip_fanout=3,
        # Leader: 10-minute lease
        leader_lease_duration=600.0,
        leader_renew_interval=60.0,
        # Recovery: 60s heal interval
        partition_heal_interval=60.0,
        peer_retry_interval=600.0,
        # Jobs: 30s check interval
        job_check_interval=30.0,
        job_timeout_multiplier=1.5,
    ),
    "development": P2PProfile(
        name="development",
        description=(
            "Development profile for local testing. Very fast failure detection "
            "for quick debugging. NOT suitable for production use."
        ),
        # Heartbeat: 5s interval, 30s timeout
        heartbeat_interval=5.0,
        peer_timeout=30.0,
        peer_timeout_nat_blocked=45.0,
        election_timeout=5.0,
        # Gossip: very high frequency
        gossip_interval=5.0,
        gossip_fanout=3,
        # Leader: 1-minute lease
        leader_lease_duration=60.0,
        leader_renew_interval=10.0,
        # Recovery: 10s heal interval
        partition_heal_interval=10.0,
        peer_retry_interval=30.0,
        # Jobs: 5s check interval
        job_check_interval=5.0,
        job_timeout_multiplier=0.5,
    ),
}


# =============================================================================
# Profile Access
# =============================================================================

def get_profile_name() -> str:
    """Get the active profile name from environment."""
    return os.environ.get("RINGRIFT_P2P_PROFILE", "production")


def get_profile(name: str | None = None) -> P2PProfile:
    """Get P2P profile by name or from environment.

    Args:
        name: Profile name. If None, reads from RINGRIFT_P2P_PROFILE env var.

    Returns:
        P2PProfile instance. Defaults to 'production' if name not found.
    """
    if name is None:
        name = get_profile_name()

    return PROFILES.get(name, PROFILES["production"])


def apply_profile_to_env(profile: P2PProfile) -> None:
    """Apply profile settings to environment variables.

    This sets the individual RINGRIFT_P2P_* environment variables based on
    the profile settings. Call this early in startup to configure the P2P
    orchestrator.

    Args:
        profile: Profile to apply
    """
    env_mappings = {
        "RINGRIFT_P2P_HEARTBEAT_INTERVAL": profile.heartbeat_interval,
        "RINGRIFT_P2P_PEER_TIMEOUT": profile.peer_timeout,
        "RINGRIFT_P2P_PEER_TIMEOUT_NAT_BLOCKED": profile.peer_timeout_nat_blocked,
        "RINGRIFT_P2P_ELECTION_TIMEOUT": profile.election_timeout,
        "RINGRIFT_P2P_GOSSIP_INTERVAL": profile.gossip_interval,
        "RINGRIFT_P2P_GOSSIP_FANOUT": profile.gossip_fanout,
        "RINGRIFT_P2P_LEADER_LEASE_DURATION": profile.leader_lease_duration,
        "RINGRIFT_P2P_LEADER_RENEW_INTERVAL": profile.leader_renew_interval,
        "RINGRIFT_P2P_PARTITION_HEAL_INTERVAL": profile.partition_heal_interval,
        "RINGRIFT_P2P_PEER_RETRY_INTERVAL": profile.peer_retry_interval,
        "RINGRIFT_P2P_JOB_CHECK_INTERVAL": profile.job_check_interval,
    }

    for key, value in env_mappings.items():
        os.environ[key] = str(value)


def apply_profile(name: str | None = None) -> P2PProfile:
    """Get profile and apply it to environment variables.

    Convenience function that combines get_profile() and apply_profile_to_env().

    Args:
        name: Profile name. If None, reads from RINGRIFT_P2P_PROFILE env var.

    Returns:
        The applied P2PProfile instance.
    """
    profile = get_profile(name)
    apply_profile_to_env(profile)
    return profile


# =============================================================================
# Profile Utilities
# =============================================================================

def list_profiles() -> list[str]:
    """List available profile names."""
    return list(PROFILES.keys())


def describe_profiles() -> str:
    """Get human-readable description of all profiles."""
    lines = ["Available P2P Profiles:", ""]
    for name, profile in PROFILES.items():
        lines.append(f"  {name}:")
        lines.append(f"    {profile.description}")
        lines.append(f"    Heartbeat: {profile.heartbeat_interval}s, Timeout: {profile.peer_timeout}s")
        lines.append("")
    return "\n".join(lines)
