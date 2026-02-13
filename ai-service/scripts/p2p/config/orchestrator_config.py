"""Orchestrator Configuration Dataclasses.

February 2026: Extracted from p2p_orchestrator.py __init__ to consolidate
hardcoded configuration values into typed, documented dataclasses.

These dataclasses centralize default values that were previously scattered
throughout the __init__ method, making them easier to find, tune, and test.

Usage:
    from scripts.p2p.config.orchestrator_config import (
        OrchestratorConfig,
        SyncConfig,
        TrainingConfig,
        ManifestConfig,
        PartitionConfig,
        SafeguardConfig,
    )

    config = OrchestratorConfig()
    sync = SyncConfig()
    training = TrainingConfig()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from scripts.p2p.constants import (
    DEFAULT_PORT,
    TRAINING_SYNC_INTERVAL,
)


@dataclass
class OrchestratorConfig:
    """General P2P orchestrator configuration.

    Controls the core orchestrator behavior: networking, auth, storage,
    and cluster coordination settings.
    """

    # Network
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT

    # Storage
    storage_type: str = "auto"  # "disk", "ramdrive", or "auto"
    sync_to_disk_interval: int = 300  # Sync ramdrive to disk every N seconds

    # Selfplay stats history (leader-only, in-memory)
    selfplay_stats_history_max_samples: int = 288  # ~24h at 5-min cadence

    # CMA-ES concurrency
    max_concurrent_cmaes_evals: int = 2

    # Status endpoint caching
    status_cache_ttl: float = 5.0  # seconds

    # Startup
    startup_grace_period: float = 10.0  # seconds before health checks apply

    @classmethod
    def from_env(cls) -> OrchestratorConfig:
        """Create config with environment variable overrides."""
        config = cls()
        raw = (os.environ.get("RINGRIFT_P2P_MAX_CONCURRENT_CMAES_EVALS", "") or "").strip()
        if raw:
            try:
                config.max_concurrent_cmaes_evals = max(1, int(raw))
            except (ValueError, AttributeError):
                pass
        return config


@dataclass
class SyncConfig:
    """Configuration for data synchronization.

    Controls sync intervals, timeouts, and data freshness settings
    used by the orchestrator for cluster data synchronization.
    """

    # Auto-sync interval (when data is missing)
    auto_sync_interval: float = 600.0  # 10 minutes

    # Training node priority sync
    training_sync_interval: float = TRAINING_SYNC_INTERVAL

    # Manifest collection
    manifest_collection_interval: float = 300.0  # 5 minutes


@dataclass
class TrainingConfig:
    """Configuration for training pipeline coordination.

    Controls training check intervals and improvement cycle settings
    used by the leader for orchestrating the training pipeline.
    """

    # Training readiness check interval
    training_check_interval: float = 300.0  # 5 minutes

    # Improvement cycle check interval
    improvement_cycle_check_interval: float = 600.0  # 10 minutes


@dataclass
class PartitionConfig:
    """Configuration for network partition handling.

    Controls read-only mode behavior when in a minority partition,
    preventing data divergence during split-brain scenarios.
    """

    # Partition check interval
    check_interval: float = 30.0  # seconds

    # Partition read-only mode defaults
    readonly_mode: bool = False
    readonly_since: float = 0.0
    last_check: float = 0.0


@dataclass
class SafeguardConfig:
    """Configuration for process spawning safeguards.

    Rate-limiting and coordinator integration settings to prevent
    runaway process spawning.
    """

    # These are loaded from constants at runtime
    agent_mode: bool = False
    coordinator_url: str = ""
    coordinator_available: bool = False
    last_coordinator_check: float = 0.0
