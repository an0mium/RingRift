"""Cluster-Wide Model Discovery for RingRift.

Extends the local ModelDiscovery to find models across all cluster nodes.
Uses ClusterManifest for tracking and SSH for remote discovery.

Features:
- Discover models on any cluster node
- Sync models from remote nodes to local
- Track model locations in ClusterManifest
- Prioritize training nodes for model discovery

Usage:
    from app.models.cluster_discovery import ClusterModelDiscovery

    discovery = ClusterModelDiscovery()

    # Find all hex8 2-player models across cluster
    models = discovery.discover_cluster_models(board_type="hex8", num_players=2)

    # Sync a model to local node
    local_path = discovery.sync_model_to_local(models[0])

    # Ensure a model exists locally for selfplay
    local_path = discovery.ensure_model_available(
        board_type="hex8", num_players=2
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.distributed.cluster_manifest import (
    ClusterManifest,
    DataType,
    ModelLocation,
    get_cluster_manifest,
)
from app.models.discovery import ModelInfo, get_model_info, discover_models

logger = logging.getLogger(__name__)

__all__ = [
    # Data classes
    "RemoteModelInfo",
    # Main class
    "ClusterModelDiscovery",
    # Singleton accessors
    "get_cluster_model_discovery",
    "reset_cluster_model_discovery",
]


@dataclass
class RemoteModelInfo:
    """Model information with remote location details."""
    model_info: ModelInfo
    node_id: str
    remote_path: str
    is_local: bool = False
    sync_priority: int = 0  # Higher = sync first
    last_seen: float = 0.0


class ClusterModelDiscovery:
    """Discovers and syncs models across all cluster nodes.

    Integrates with:
    - ClusterManifest for model location tracking
    - SSH for remote model discovery
    - Local ModelDiscovery for local models
    """

    def __init__(
        self,
        config_path: Path | None = None,
        manifest: ClusterManifest | None = None,
    ):
        """Initialize cluster model discovery.

        Args:
            config_path: Path to distributed_hosts.yaml
            manifest: ClusterManifest instance (uses singleton if None)
        """
        self.node_id = socket.gethostname()
        self._manifest = manifest or get_cluster_manifest()

        # Load host configuration
        self._hosts_config: dict[str, Any] = {}
        self._ssh_configs: dict[str, dict[str, str]] = {}
        self._load_config(config_path)

        # Cache for remote discoveries
        self._discovery_cache: dict[str, list[RemoteModelInfo]] = {}
        self._cache_ttl = 300  # 5 minute cache

        # Local paths
        self._ai_service_root = Path(__file__).resolve().parent.parent.parent
        self._models_dir = self._ai_service_root / "models"

        logger.info(f"ClusterModelDiscovery initialized: {len(self._hosts_config)} hosts")

    def _load_config(self, config_path: Path | None = None) -> None:
        """Load host configuration for SSH access."""
        if config_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            config_path = base_dir / "config" / "distributed_hosts.yaml"

        if not config_path.exists():
            logger.warning(f"No config found at {config_path}")
            return

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            self._hosts_config = config.get("hosts", {})

            # Extract SSH configurations
            for host_name, host_config in self._hosts_config.items():
                ssh_host = (
                    host_config.get("tailscale_ip") or
                    host_config.get("ssh_host")
                )
                if ssh_host:
                    self._ssh_configs[host_name] = {
                        "host": ssh_host,
                        "user": host_config.get("ssh_user", "ubuntu"),
                        "key": host_config.get("ssh_key", "~/.ssh/id_cluster"),
                        "path": host_config.get("ringrift_path", "~/ringrift/ai-service"),
                    }

        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def discover_cluster_models(
        self,
        board_type: str,
        num_players: int,
        include_local: bool = True,
        include_remote: bool = True,
        max_remote_nodes: int = 5,
        timeout: float = 30.0,
    ) -> list[RemoteModelInfo]:
        """Discover models across the cluster.

        Args:
            board_type: Board type to search for
            num_players: Number of players
            include_local: Include local models
            include_remote: Include remote models via SSH
            max_remote_nodes: Max number of remote nodes to query
            timeout: SSH timeout in seconds

        Returns:
            List of RemoteModelInfo sorted by priority
        """
        cache_key = f"{board_type}_{num_players}p"

        # Check cache
        if cache_key in self._discovery_cache:
            cache_time = self._discovery_cache.get(f"{cache_key}_time", 0)
            if time.time() - cache_time < self._cache_ttl:
                return self._discovery_cache[cache_key]

        results: list[RemoteModelInfo] = []

        # 1. Check local models first
        if include_local:
            local_models = self._discover_local_models(board_type, num_players)
            results.extend(local_models)

        # 2. Check ClusterManifest for known locations
        manifest_models = self._discover_from_manifest(board_type, num_players)
        for rm in manifest_models:
            if not any(r.remote_path == rm.remote_path for r in results):
                results.append(rm)

        # 3. Query remote nodes if needed
        if include_remote and len(results) < max_remote_nodes:
            remote_models = self._discover_remote_models(
                board_type, num_players,
                max_nodes=max_remote_nodes,
                timeout=timeout,
            )
            for rm in remote_models:
                if not any(r.remote_path == rm.remote_path for r in results):
                    results.append(rm)

        # Sort by priority (local first, then by sync_priority)
        results.sort(
            key=lambda m: (not m.is_local, -m.sync_priority),
        )

        # Update cache
        self._discovery_cache[cache_key] = results
        self._discovery_cache[f"{cache_key}_time"] = time.time()

        return results

    def _discover_local_models(
        self,
        board_type: str,
        num_players: int,
    ) -> list[RemoteModelInfo]:
        """Discover models on the local node."""
        results: list[RemoteModelInfo] = []

        # Use existing discover_models function
        local_models = discover_models(
            board_type=board_type,
            num_players=num_players,
        )

        for model in local_models:
            results.append(RemoteModelInfo(
                model_info=model,
                node_id=self.node_id,
                remote_path=model.path,
                is_local=True,
                sync_priority=100,  # Local is highest priority
                last_seen=time.time(),
            ))

            # Register in manifest
            self._manifest.register_model(
                model_path=model.path,
                node_id=self.node_id,
                board_type=board_type,
                num_players=num_players,
                model_version=model.architecture_version,
                file_size=model.size_bytes,
            )

        return results

    def _discover_from_manifest(
        self,
        board_type: str,
        num_players: int,
    ) -> list[RemoteModelInfo]:
        """Discover models from ClusterManifest."""
        results: list[RemoteModelInfo] = []

        locations = self._manifest.find_models_for_config(board_type, num_players)

        for loc in locations:
            # Skip if this is the local node
            if loc.node_id == self.node_id:
                continue

            # Create ModelInfo from location
            model = ModelInfo(
                path=loc.model_path,
                name=Path(loc.model_path).stem,
                model_type="nn",
                board_type=board_type,
                num_players=num_players,
                architecture_version=loc.model_version,
                size_bytes=loc.file_size,
                source="manifest",
            )

            results.append(RemoteModelInfo(
                model_info=model,
                node_id=loc.node_id,
                remote_path=loc.model_path,
                is_local=False,
                sync_priority=self._compute_sync_priority(loc.node_id),
                last_seen=loc.last_seen,
            ))

        return results

    def _discover_remote_models(
        self,
        board_type: str,
        num_players: int,
        max_nodes: int = 5,
        timeout: float = 30.0,
    ) -> list[RemoteModelInfo]:
        """Discover models on remote nodes via SSH."""
        results: list[RemoteModelInfo] = []

        # Prioritize training nodes
        priority_hosts = self._get_priority_hosts(max_nodes)

        for host_name in priority_hosts:
            if host_name == self.node_id:
                continue

            ssh_config = self._ssh_configs.get(host_name)
            if not ssh_config:
                continue

            try:
                remote_models = self._query_remote_node(
                    host_name,
                    ssh_config,
                    board_type,
                    num_players,
                    timeout,
                )
                results.extend(remote_models)

            except Exception as e:
                logger.debug(f"Failed to query {host_name}: {e}")

        return results

    def _get_priority_hosts(self, max_hosts: int) -> list[str]:
        """Get prioritized list of hosts to query."""
        hosts_with_priority: list[tuple[str, int]] = []

        for host_name, host_config in self._hosts_config.items():
            role = host_config.get("role", "selfplay")
            priority = 0

            # Training nodes first
            if "training" in role:
                priority += 50

            # Prefer non-ephemeral
            if not host_config.get("ephemeral", False):
                priority += 20

            # Prefer nodes with known connectivity
            if host_name in self._ssh_configs:
                priority += 10

            hosts_with_priority.append((host_name, priority))

        # Sort by priority descending
        hosts_with_priority.sort(key=lambda x: -x[1])

        return [h[0] for h in hosts_with_priority[:max_hosts]]

    def _query_remote_node(
        self,
        host_name: str,
        ssh_config: dict[str, str],
        board_type: str,
        num_players: int,
        timeout: float,
    ) -> list[RemoteModelInfo]:
        """Query a remote node for models via SSH."""
        results: list[RemoteModelInfo] = []

        # Build SSH command to discover models
        remote_path = ssh_config["path"]
        python_script = f'''
import sys
sys.path.insert(0, "{remote_path}")
try:
    from app.models.discovery import discover_models
    import json
    models = discover_models(board_type="{board_type}", num_players={num_players})
    for m in models:
        print(json.dumps(m.to_dict()))
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
'''

        ssh_key = os.path.expanduser(ssh_config["key"])
        ssh_cmd = [
            "ssh",
            "-i", ssh_key,
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            f"{ssh_config['user']}@{ssh_config['host']}",
            f"cd {remote_path} && python3 -c '{python_script}'",
        ]

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        model = ModelInfo.from_dict(data)

                        rm = RemoteModelInfo(
                            model_info=model,
                            node_id=host_name,
                            remote_path=model.path,
                            is_local=False,
                            sync_priority=self._compute_sync_priority(host_name),
                            last_seen=time.time(),
                        )
                        results.append(rm)

                        # Register in manifest
                        self._manifest.register_model(
                            model_path=model.path,
                            node_id=host_name,
                            board_type=board_type,
                            num_players=num_players,
                            model_version=model.architecture_version,
                            file_size=model.size_bytes,
                        )

                    except json.JSONDecodeError:
                        continue

        except subprocess.TimeoutExpired:
            logger.debug(f"SSH timeout querying {host_name}")
        except Exception as e:
            logger.debug(f"SSH error querying {host_name}: {e}")

        return results

    def _compute_sync_priority(self, node_id: str) -> int:
        """Compute sync priority for a node."""
        priority = 50  # Base

        host_config = self._hosts_config.get(node_id, {})
        role = host_config.get("role", "selfplay")

        if "training" in role:
            priority += 30

        # Prefer non-ephemeral
        if not host_config.get("ephemeral", False):
            priority += 10

        return priority

    def sync_model_to_local(
        self,
        remote_model: RemoteModelInfo,
        local_dir: Path | None = None,
        timeout: float = 120.0,
    ) -> Path | None:
        """Sync a remote model to the local node.

        Args:
            remote_model: Remote model to sync
            local_dir: Local directory (defaults to models/)
            timeout: Timeout for rsync

        Returns:
            Local path to synced model, or None on failure
        """
        if remote_model.is_local:
            return Path(remote_model.remote_path)

        if local_dir is None:
            local_dir = self._models_dir

        local_dir.mkdir(parents=True, exist_ok=True)

        # Get SSH config
        ssh_config = self._ssh_configs.get(remote_model.node_id)
        if not ssh_config:
            logger.error(f"No SSH config for {remote_model.node_id}")
            return None

        # Build local path
        model_name = Path(remote_model.remote_path).name
        local_path = local_dir / model_name

        # Rsync command
        ssh_key = os.path.expanduser(ssh_config["key"])
        remote_full_path = f"{ssh_config['user']}@{ssh_config['host']}:{remote_model.remote_path}"

        rsync_cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e", f"ssh -i {ssh_key} -o StrictHostKeyChecking=no",
            remote_full_path,
            str(local_path),
        ]

        try:
            logger.info(f"Syncing model from {remote_model.node_id}: {model_name}")

            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0 and local_path.exists():
                logger.info(f"Model synced: {local_path}")

                # Also sync sidecar if exists
                sidecar_remote = remote_full_path + ".json"
                sidecar_local = str(local_path) + ".json"
                subprocess.run(
                    [*rsync_cmd[:-2], sidecar_remote, sidecar_local],
                    capture_output=True,
                    timeout=30,
                )

                # Register local copy
                self._manifest.register_model(
                    model_path=str(local_path),
                    node_id=self.node_id,
                    board_type=remote_model.model_info.board_type,
                    num_players=remote_model.model_info.num_players,
                    model_version=remote_model.model_info.architecture_version,
                    file_size=local_path.stat().st_size,
                )

                return local_path
            else:
                logger.error(f"Rsync failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Rsync timeout syncing from {remote_model.node_id}")
            return None
        except Exception as e:
            logger.error(f"Rsync error: {e}")
            return None

    def ensure_model_available(
        self,
        board_type: str,
        num_players: int,
        prefer_canonical: bool = True,
    ) -> Path | None:
        """Ensure a model is available locally, syncing if needed.

        Args:
            board_type: Board type
            num_players: Number of players
            prefer_canonical: Prefer canonical models

        Returns:
            Local path to model, or None if not found
        """
        # Check local first
        local_models = self._discover_local_models(board_type, num_players)

        if prefer_canonical:
            # Look for canonical model
            for m in local_models:
                if "canonical" in m.model_info.name.lower():
                    return Path(m.model_info.path)

        if local_models:
            return Path(local_models[0].model_info.path)

        # Try to sync from cluster
        logger.info(f"No local model for {board_type}_{num_players}p, searching cluster...")

        remote_models = self.discover_cluster_models(
            board_type=board_type,
            num_players=num_players,
            include_local=False,
            include_remote=True,
        )

        if not remote_models:
            logger.warning(f"No models found for {board_type}_{num_players}p in cluster")
            return None

        # Prefer canonical
        if prefer_canonical:
            for rm in remote_models:
                if "canonical" in rm.model_info.name.lower():
                    return self.sync_model_to_local(rm)

        # Sync best available
        return self.sync_model_to_local(remote_models[0])

    def get_status(self) -> dict[str, Any]:
        """Get discovery status."""
        return {
            "node_id": self.node_id,
            "hosts_configured": len(self._hosts_config),
            "ssh_configs": len(self._ssh_configs),
            "cached_discoveries": len(self._discovery_cache),
        }


# Module-level singleton
_cluster_model_discovery: ClusterModelDiscovery | None = None


def get_cluster_model_discovery() -> ClusterModelDiscovery:
    """Get the singleton ClusterModelDiscovery instance."""
    global _cluster_model_discovery
    if _cluster_model_discovery is None:
        _cluster_model_discovery = ClusterModelDiscovery()
    return _cluster_model_discovery


def reset_cluster_model_discovery() -> None:
    """Reset the singleton (for testing)."""
    global _cluster_model_discovery
    _cluster_model_discovery = None
