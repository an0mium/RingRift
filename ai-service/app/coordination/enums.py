"""Canonical enum definitions for coordination modules.

This module provides the centralized source of truth for enums that were
previously duplicated across multiple modules. Each enum is given a unique,
descriptive name to avoid naming collisions.

December 2025: Created to resolve enum naming collisions identified in
strategic code quality assessment.

Usage:
    from app.coordination.enums import (
        ScaleAction,
        CatalogDataType,
        DistributionDataType,
        SystemRecoveryAction,
    )

Migration Notes:
    - ScaleAction: Consolidated from resource_optimizer.py and capacity_planner.py
    - CatalogDataType: Renamed from DataType in data_catalog.py
    - DistributionDataType: Renamed from DataType in unified_distribution_daemon.py
    - SystemRecoveryAction: Renamed from RecoveryAction in recovery_engine.py
"""

from __future__ import annotations

from enum import Enum, auto


class ScaleAction(str, Enum):
    """Scaling actions for cluster capacity management.

    Consolidated from resource_optimizer.py and capacity_planner.py.
    The 'str' base enables direct use as string values.
    """

    NONE = "none"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REBALANCE = "rebalance"


class CatalogDataType(Enum):
    """Types of data tracked in the data catalog.

    Renamed from DataType in data_catalog.py to avoid collision with
    DistributionDataType.
    """

    GAMES = "games"  # SQLite game databases (.db)
    MODELS = "models"  # PyTorch model checkpoints (.pth)
    NPZ = "npz"  # NumPy training data (.npz)
    CHECKPOINT = "checkpoint"  # Training checkpoints
    CONFIG = "config"  # Configuration files
    LOG = "log"  # Log files
    UNKNOWN = "unknown"

    @classmethod
    def from_path(cls, path: str) -> CatalogDataType:
        """Infer data type from file path."""
        path_lower = path.lower()
        if path_lower.endswith(".db"):
            return cls.GAMES
        elif path_lower.endswith(".pth") or path_lower.endswith(".pt"):
            return cls.MODELS
        elif path_lower.endswith(".npz"):
            return cls.NPZ
        elif "checkpoint" in path_lower or path_lower.endswith(".ckpt"):
            return cls.CHECKPOINT
        elif path_lower.endswith((".yaml", ".yml", ".json", ".toml")):
            return cls.CONFIG
        elif path_lower.endswith(".log"):
            return cls.LOG
        return cls.UNKNOWN


class DistributionDataType(Enum):
    """Types of data that can be distributed across cluster.

    Renamed from DataType in unified_distribution_daemon.py to avoid collision
    with CatalogDataType.
    """

    MODEL = auto()
    NPZ = auto()
    TORRENT = auto()


class SystemRecoveryAction(Enum):
    """Recovery action types for system-level recovery.

    Renamed from RecoveryAction in recovery_engine.py.
    Ordered by escalation level (least to most disruptive).

    See also:
    - JobRecoveryAction in unified_health_manager.py (job-level)
    - NodeRecoveryAction in node_recovery_daemon.py (node-level)
    """

    RESTART_P2P = auto()  # 10s, soft restart of P2P process
    RESTART_TAILSCALE = auto()  # 30s, network reset
    REBOOT_INSTANCE = auto()  # 2min, provider reboot API
    RECREATE_INSTANCE = auto()  # 5min, destroy and recreate

    @property
    def timeout_seconds(self) -> int:
        """Get timeout for this action."""
        timeouts = {
            SystemRecoveryAction.RESTART_P2P: 30,
            SystemRecoveryAction.RESTART_TAILSCALE: 60,
            SystemRecoveryAction.REBOOT_INSTANCE: 180,
            SystemRecoveryAction.RECREATE_INSTANCE: 600,
        }
        return timeouts.get(self, 60)

    @property
    def description(self) -> str:
        """Human-readable description of the action."""
        descriptions = {
            SystemRecoveryAction.RESTART_P2P: "Restart P2P orchestrator process",
            SystemRecoveryAction.RESTART_TAILSCALE: "Restart Tailscale VPN",
            SystemRecoveryAction.REBOOT_INSTANCE: "Reboot cloud instance",
            SystemRecoveryAction.RECREATE_INSTANCE: "Destroy and recreate instance",
        }
        return descriptions.get(self, "Unknown action")


# Backward-compatible aliases (deprecated, remove Q2 2026)
# These allow gradual migration without breaking existing code
DataType = CatalogDataType  # Use CatalogDataType or DistributionDataType instead
RecoveryAction = SystemRecoveryAction  # Use SystemRecoveryAction instead


__all__ = [
    # Canonical enum names
    "ScaleAction",
    "CatalogDataType",
    "DistributionDataType",
    "SystemRecoveryAction",
    # Deprecated aliases (for backward compatibility)
    "DataType",
    "RecoveryAction",
]
