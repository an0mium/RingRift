"""State Checker Base Class (December 2025).

Defines the interface and data structures for checking cloud provider instance states.
Each provider (Vast.ai, Lambda, RunPod, etc.) implements a subclass.

Key concepts:
- ProviderInstanceState: Unified enum for instance states across all providers
- InstanceInfo: Dataclass containing instance information from provider API
- StateChecker: Abstract base class for provider-specific implementations
- STATE_TO_YAML_STATUS: Maps provider states to distributed_hosts.yaml status values
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ProviderInstanceState(Enum):
    """Unified instance state across all cloud providers.

    Each provider maps their specific state strings to these canonical states.
    """

    RUNNING = "running"
    STARTING = "starting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


# Map provider instance states to distributed_hosts.yaml status values
STATE_TO_YAML_STATUS: dict[ProviderInstanceState, str] = {
    ProviderInstanceState.RUNNING: "ready",
    ProviderInstanceState.STARTING: "setup",
    ProviderInstanceState.STOPPING: "offline",
    ProviderInstanceState.STOPPED: "offline",
    ProviderInstanceState.TERMINATED: "retired",
    ProviderInstanceState.UNKNOWN: "offline",
}


@dataclass
class InstanceInfo:
    """Information about a cloud provider instance.

    Returned by StateChecker.get_instance_states() for each instance.
    """

    # Required fields
    instance_id: str
    state: ProviderInstanceState
    provider: str  # "vast", "lambda", "runpod", "vultr", "hetzner"

    # Optional fields for correlation with distributed_hosts.yaml
    node_name: Optional[str] = None  # Name in distributed_hosts.yaml
    tailscale_ip: Optional[str] = None
    public_ip: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_port: int = 22

    # Provider-specific metadata
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    gpu_vram_gb: float = 0.0
    hostname: Optional[str] = None

    # Timestamps
    created_at: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    # Raw provider response (for debugging)
    raw_data: dict = field(default_factory=dict)

    @property
    def yaml_status(self) -> str:
        """Get the corresponding distributed_hosts.yaml status."""
        return STATE_TO_YAML_STATUS.get(self.state, "offline")

    def __str__(self) -> str:
        name = self.node_name or self.instance_id
        return f"{name} ({self.provider}): {self.state.value}"


class StateChecker(ABC):
    """Abstract base class for provider-specific state checkers.

    Each cloud provider implements a subclass that:
    1. Queries the provider's API for current instance states
    2. Maps provider-specific states to ProviderInstanceState enum
    3. Correlates instances with node names in distributed_hosts.yaml
    """

    def __init__(self, provider_name: str):
        """Initialize state checker.

        Args:
            provider_name: Name of the provider (e.g., "vast", "lambda")
        """
        self.provider_name = provider_name
        self._enabled = True
        self._last_check: Optional[datetime] = None
        self._last_error: Optional[str] = None

    @property
    def is_enabled(self) -> bool:
        """Check if this provider checker is enabled."""
        return self._enabled

    def disable(self, reason: str) -> None:
        """Disable this checker (e.g., due to missing API key)."""
        self._enabled = False
        logger.warning(f"{self.provider_name} checker disabled: {reason}")

    def enable(self) -> None:
        """Re-enable this checker."""
        self._enabled = True
        logger.info(f"{self.provider_name} checker enabled")

    @abstractmethod
    async def get_instance_states(self) -> list[InstanceInfo]:
        """Query provider API and return current instance states.

        Returns:
            List of InstanceInfo objects for all known instances.
            Returns empty list if API is unavailable or no instances exist.

        Raises:
            Should not raise - catch and log errors internally, return empty list.
        """
        ...

    @abstractmethod
    async def check_api_availability(self) -> bool:
        """Check if the provider API is available and credentials are valid.

        Returns:
            True if API is accessible, False otherwise.
        """
        ...

    @abstractmethod
    def correlate_with_config(
        self,
        instances: list[InstanceInfo],
        config_hosts: dict[str, dict],
    ) -> list[InstanceInfo]:
        """Match instances to node names in distributed_hosts.yaml.

        Args:
            instances: List of instances from get_instance_states()
            config_hosts: The 'hosts' section from distributed_hosts.yaml

        Returns:
            Updated instances with node_name field populated where matches found.
        """
        ...

    def get_status(self) -> dict:
        """Get current status of this checker."""
        return {
            "provider": self.provider_name,
            "enabled": self._enabled,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "last_error": self._last_error,
        }
