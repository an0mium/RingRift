"""P2P Orchestrator Initialization Manager.

January 30, 2026 - Priority 2.1 Decomposition

This manager consolidates initialization logic that was previously inlined
in P2POrchestrator.__init__. It handles:

1. Bootstrap configuration (seeds, relay mode)
2. Storage type detection (disk vs ramdrive)
3. Advertise host resolution (Tailscale, public IP)
4. Security configuration (auth tokens, git safe directory)

Usage:
    from scripts.p2p.managers.initialization_manager import (
        InitializationManager,
        InitializationConfig,
        create_initialization_manager,
    )

    manager = create_initialization_manager(config)
    bootstrap_result = manager.resolve_bootstrap_config(cli_peers, relay_peers)
    storage_result = manager.resolve_storage_config()
    advertise_result = manager.resolve_advertise_host()
"""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Constants (imported from p2p_orchestrator constants)
# =============================================================================

# Bootstrap seeds for initial peer discovery
BOOTSTRAP_SEEDS: list[str] = [
    "100.107.168.125:8770",  # mac-studio (primary coordinator)
    "100.94.201.92:8770",    # vultr-a100-20gb (stable GPU node)
    "100.110.28.41:8770",    # nebius-backbone-1 (backbone)
    "100.94.174.19:8770",    # hetzner-cpu1 (relay-capable)
    "100.67.131.72:8770",    # hetzner-cpu2 (relay-capable)
]

# Initial cluster epoch for split-brain resolution
INITIAL_CLUSTER_EPOCH = 1


@dataclass
class InitializationConfig:
    """Configuration for initialization manager."""

    # Bootstrap
    default_seeds: list[str] = field(default_factory=lambda: list(BOOTSTRAP_SEEDS))
    shuffle_seeds: bool = True

    # Storage
    auto_detect_storage: bool = True
    ramdrive_path: str = "/dev/shm/ringrift/data"
    sync_to_disk_interval: int = 300

    # Advertise host
    tailscale_wait_timeout: float = 90.0
    tailscale_check_interval: float = 1.0
    prefer_public_ip: bool = False

    # Auth
    auth_token_env: str = "RINGRIFT_P2P_AUTH_TOKEN"
    auth_token_file_env: str = "RINGRIFT_P2P_AUTH_TOKEN_FILE"


@dataclass
class BootstrapResult:
    """Result of bootstrap configuration resolution."""

    known_peers: list[str]
    bootstrap_seeds: list[str]
    relay_peers: set[str]
    force_relay_mode: bool
    cluster_epoch: int


@dataclass
class StorageResult:
    """Result of storage configuration resolution."""

    storage_type: str  # "disk" or "ramdrive"
    ramdrive_path: str
    sync_to_disk_interval: int
    total_ram_gb: float
    free_disk_gb: float
    disk_usage_percent: float


@dataclass
class AdvertiseResult:
    """Result of advertise host resolution."""

    advertise_host: str
    advertise_port: int
    source: str  # "yaml", "tailscale", "public_ip", "local_ip"
    is_tailscale: bool


@dataclass
class AuthResult:
    """Result of auth configuration resolution."""

    auth_token: str
    require_auth: bool
    source: str  # "env", "file", "arg", "none"


class InitializationManager:
    """Manages P2P orchestrator initialization logic."""

    def __init__(
        self,
        config: InitializationConfig | None = None,
        node_id: str = "",
        ringrift_path: str | None = None,
    ):
        self.config = config or InitializationConfig()
        self.node_id = node_id
        self.ringrift_path = ringrift_path or self._detect_ringrift_path()

    def _detect_ringrift_path(self) -> str:
        """Detect the RingRift AI service path."""
        # Check environment variable
        env_path = os.environ.get("RINGRIFT_AI_SERVICE_PATH", "")
        if env_path and Path(env_path).exists():
            return env_path

        # Try common locations
        candidates = [
            Path.home() / "ringrift" / "ai-service",
            Path.home() / "Development" / "RingRift" / "ai-service",
            Path("/root/ringrift/ai-service"),
            Path.cwd(),
        ]
        for candidate in candidates:
            if (candidate / "scripts" / "p2p_orchestrator.py").exists():
                return str(candidate)

        return str(Path.cwd())

    def resolve_bootstrap_config(
        self,
        cli_peers: list[str] | None = None,
        relay_peers: list[str] | None = None,
    ) -> BootstrapResult:
        """Resolve bootstrap configuration (seeds, relay mode).

        Priority: CLI peers first, then hardcoded seeds (shuffled).

        Args:
            cli_peers: Peers specified via CLI --known-peers
            relay_peers: Peers that should receive relay heartbeats

        Returns:
            BootstrapResult with resolved configuration
        """
        cli_peers = cli_peers or []

        # Merge CLI peers with hardcoded seeds
        merged_seeds = list(cli_peers)
        for seed in self.config.default_seeds:
            if seed not in merged_seeds:
                merged_seeds.append(seed)

        # Shuffle only the hardcoded portion to distribute load
        if self.config.shuffle_seeds and len(merged_seeds) > len(cli_peers):
            hardcoded_portion = merged_seeds[len(cli_peers):]
            random.shuffle(hardcoded_portion)
            merged_seeds = merged_seeds[:len(cli_peers)] + hardcoded_portion

        # Load force relay mode from config
        force_relay_mode = self._load_force_relay_mode()

        logger.info(
            f"Bootstrap: {len(cli_peers)} CLI + {len(self.config.default_seeds)} "
            f"hardcoded = {len(merged_seeds)} total peers"
        )

        return BootstrapResult(
            known_peers=merged_seeds,
            bootstrap_seeds=list(self.config.default_seeds),
            relay_peers=set(relay_peers or []),
            force_relay_mode=force_relay_mode,
            cluster_epoch=INITIAL_CLUSTER_EPOCH,
        )

    def _load_force_relay_mode(self) -> bool:
        """Load force_relay_mode from distributed_hosts.yaml."""
        try:
            import yaml
            config_path = Path(self.ringrift_path) / "config" / "distributed_hosts.yaml"
            if not config_path.exists():
                return False

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            hosts = config.get("hosts", {})
            node_config = hosts.get(self.node_id, {})

            return bool(
                node_config.get("force_relay_mode", False) or
                node_config.get("nat_blocked", False)
            )
        except Exception as e:
            logger.debug(f"Failed to load force_relay_mode: {e}")
            return False

    def resolve_storage_config(self, storage_type: str = "auto") -> StorageResult:
        """Resolve storage configuration (disk vs ramdrive).

        Args:
            storage_type: "disk", "ramdrive", or "auto" (detected)

        Returns:
            StorageResult with resolved configuration
        """
        try:
            from scripts.p2p.ramdrive import (
                get_system_resources,
                should_use_ramdrive,
                log_storage_recommendation,
            )

            resources = get_system_resources()

            if storage_type == "auto" and self.config.auto_detect_storage:
                if should_use_ramdrive():
                    resolved_type = "ramdrive"
                    logger.info(
                        f"Auto-detected storage: RAMDRIVE "
                        f"(RAM: {resources.total_ram_gb:.0f}GB, "
                        f"Disk: {resources.free_disk_gb:.0f}GB free / "
                        f"{resources.disk_usage_percent:.0f}% used)"
                    )
                else:
                    resolved_type = "disk"
                    logger.info(
                        f"Auto-detected storage: DISK "
                        f"(RAM: {resources.total_ram_gb:.0f}GB, "
                        f"Disk: {resources.free_disk_gb:.0f}GB free / "
                        f"{resources.disk_usage_percent:.0f}% used)"
                    )
                log_storage_recommendation()
            else:
                resolved_type = storage_type

            return StorageResult(
                storage_type=resolved_type,
                ramdrive_path=self.config.ramdrive_path,
                sync_to_disk_interval=self.config.sync_to_disk_interval,
                total_ram_gb=resources.total_ram_gb,
                free_disk_gb=resources.free_disk_gb,
                disk_usage_percent=resources.disk_usage_percent,
            )
        except ImportError:
            logger.warning("Ramdrive module not available, defaulting to disk storage")
            return StorageResult(
                storage_type="disk",
                ramdrive_path=self.config.ramdrive_path,
                sync_to_disk_interval=self.config.sync_to_disk_interval,
                total_ram_gb=0.0,
                free_disk_gb=0.0,
                disk_usage_percent=0.0,
            )

    def resolve_advertise_host(
        self,
        advertise_host_arg: str | None = None,
        advertise_port_arg: int | None = None,
        default_port: int = 8770,
    ) -> AdvertiseResult:
        """Resolve advertise host and port.

        Priority:
        1. Explicit argument
        2. Environment variable (RINGRIFT_ADVERTISE_HOST)
        3. YAML config tailscale_ip
        4. Tailscale CLI detection (with retry)
        5. Local IP fallback

        Args:
            advertise_host_arg: Explicit host argument
            advertise_port_arg: Explicit port argument
            default_port: Default P2P port

        Returns:
            AdvertiseResult with resolved configuration
        """
        prefer_public = (
            self.config.prefer_public_ip or
            os.environ.get("RINGRIFT_PREFER_PUBLIC_IP", "").lower() in ("1", "true", "yes")
        )

        # Check explicit argument
        advertise_host = (advertise_host_arg or "").strip()
        source = "arg" if advertise_host else "none"

        # Check environment variable
        if not advertise_host:
            env_host = os.environ.get("RINGRIFT_ADVERTISE_HOST", "").strip()
            if env_host:
                advertise_host = env_host
                source = "env"

        # Try YAML config
        if not advertise_host and not prefer_public:
            yaml_ip = self._get_yaml_tailscale_ip()
            if yaml_ip:
                advertise_host = yaml_ip
                source = "yaml"
                logger.info(f"Using YAML config tailscale_ip: {yaml_ip}")

        # Try Tailscale CLI
        if not advertise_host and not prefer_public:
            ts_ip = self._wait_for_tailscale_ip(
                timeout_seconds=self.config.tailscale_wait_timeout,
                interval_seconds=self.config.tailscale_check_interval,
            )
            if ts_ip:
                advertise_host = ts_ip
                source = "tailscale"
            else:
                logger.warning("Tailscale unavailable, falling back to local IP")

        # Fallback to local IP
        if not advertise_host:
            advertise_host = self._get_local_ip()
            source = "public_ip" if prefer_public else "local_ip"

        # Resolve port
        advertise_port = advertise_port_arg or self._infer_advertise_port(default_port)

        return AdvertiseResult(
            advertise_host=advertise_host,
            advertise_port=advertise_port,
            source=source,
            is_tailscale=(source in ("yaml", "tailscale")),
        )

    def _get_yaml_tailscale_ip(self) -> str | None:
        """Get tailscale_ip from distributed_hosts.yaml."""
        try:
            import yaml
            config_path = Path(self.ringrift_path) / "config" / "distributed_hosts.yaml"
            if not config_path.exists():
                return None

            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            hosts = config.get("hosts", {})
            node_config = hosts.get(self.node_id, {})
            return node_config.get("tailscale_ip")
        except Exception as e:
            logger.debug(f"Failed to get YAML tailscale_ip: {e}")
            return None

    def _wait_for_tailscale_ip(
        self,
        timeout_seconds: float = 90.0,
        interval_seconds: float = 1.0,
    ) -> str | None:
        """Wait for Tailscale to report an IP address."""
        import subprocess

        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                result = subprocess.run(
                    ["tailscale", "ip", "-4"],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if result.returncode == 0:
                    ip = result.stdout.strip().split("\n")[0]
                    if ip and ip.startswith("100."):
                        return ip
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass
            time.sleep(interval_seconds)

        return None

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        import socket
        try:
            # Connect to a public DNS to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _infer_advertise_port(self, default_port: int = 8770) -> int:
        """Infer advertise port from environment or Vast.ai mapping."""
        # Check environment variable
        env_port = os.environ.get("RINGRIFT_ADVERTISE_PORT", "")
        if env_port:
            try:
                return int(env_port)
            except ValueError:
                pass

        # Check Vast.ai port mapping
        vast_port = os.environ.get("PUBLIC_8770", "")
        if vast_port:
            try:
                return int(vast_port)
            except ValueError:
                pass

        return default_port

    def resolve_auth_config(
        self,
        auth_token_arg: str | None = None,
        require_auth: bool = False,
    ) -> AuthResult:
        """Resolve authentication configuration.

        Priority:
        1. Explicit argument
        2. Environment variable (RINGRIFT_P2P_AUTH_TOKEN)
        3. Token file (RINGRIFT_P2P_AUTH_TOKEN_FILE)

        Args:
            auth_token_arg: Explicit token argument
            require_auth: Whether to require authentication

        Returns:
            AuthResult with resolved configuration

        Raises:
            ValueError: If require_auth is True but no token is configured
        """
        token = ""
        source = "none"

        # Check explicit argument
        if auth_token_arg:
            token = auth_token_arg.strip()
            source = "arg"

        # Check environment variable
        if not token:
            env_token = os.environ.get(self.config.auth_token_env, "").strip()
            if env_token:
                token = env_token
                source = "env"

        # Check token file
        if not token:
            token_file = os.environ.get(self.config.auth_token_file_env, "").strip()
            if token_file:
                try:
                    token = Path(token_file).read_text().strip()
                    source = "file"
                except Exception as e:
                    logger.info(
                        f"Auth: failed to read {self.config.auth_token_file_env}={token_file}: {e}"
                    )

        # Validate require_auth
        if require_auth and not token:
            raise ValueError(
                f"--require-auth set but {self.config.auth_token_env}/"
                f"{self.config.auth_token_file_env}/--auth-token is empty"
            )

        return AuthResult(
            auth_token=token,
            require_auth=require_auth,
            source=source,
        )


def create_initialization_manager(
    config: InitializationConfig | None = None,
    node_id: str = "",
    ringrift_path: str | None = None,
) -> InitializationManager:
    """Factory function to create an InitializationManager.

    Args:
        config: Optional configuration
        node_id: Node identifier
        ringrift_path: Path to RingRift AI service

    Returns:
        Configured InitializationManager instance
    """
    return InitializationManager(
        config=config,
        node_id=node_id,
        ringrift_path=ringrift_path,
    )
