"""Vast.ai State Checker (December 2025).

Queries Vast.ai API to get current instance states and maps them to
ProviderInstanceState enum for distributed_hosts.yaml synchronization.

Uses `vastai show instances --raw` CLI command for API access.

Vast.ai State Mappings:
- "running" -> RUNNING (ready)
- "loading" -> STARTING (setup)
- "exited" -> STOPPED (offline)
- "destroying" -> STOPPING (offline)
- Not in list -> TERMINATED (retired)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional

from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
    StateChecker,
)

logger = logging.getLogger(__name__)

# Vast.ai state mappings
VAST_STATE_MAP: dict[str, ProviderInstanceState] = {
    "running": ProviderInstanceState.RUNNING,
    "loading": ProviderInstanceState.STARTING,
    "exited": ProviderInstanceState.STOPPED,
    "destroying": ProviderInstanceState.STOPPING,
    "created": ProviderInstanceState.STARTING,
    "starting": ProviderInstanceState.STARTING,
}


class VastChecker(StateChecker):
    """State checker for Vast.ai instances.

    Uses vastai CLI to query instance states. Requires VAST_API_KEY environment
    variable or ~/.vastai_api_key file.

    Instance correlation with distributed_hosts.yaml:
    - Matches by instance ID (vast-{id} pattern)
    - Falls back to SSH port matching
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Vast.ai checker.

        Args:
            api_key: Vast.ai API key (uses env/file if None)
        """
        super().__init__("vast")
        self._api_key = api_key or os.environ.get("VAST_API_KEY")

        # Check for API key file if not in env
        # Try multiple common locations
        if not self._api_key:
            key_files = [
                os.path.expanduser("~/.config/vastai/vast_api_key"),  # Default CLI location
                os.path.expanduser("~/.vastai_api_key"),  # Legacy location
            ]
            for key_file in key_files:
                if os.path.exists(key_file):
                    try:
                        with open(key_file) as f:
                            self._api_key = f.read().strip()
                        if self._api_key:
                            break
                    except OSError:
                        pass

        if not self._api_key:
            self.disable("No VAST_API_KEY found")

    async def check_api_availability(self) -> bool:
        """Check if vastai CLI and API key are available."""
        if not self._api_key:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "vastai", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.wait()
            return proc.returncode == 0
        except FileNotFoundError:
            logger.warning("vastai CLI not found")
            return False
        except Exception as e:
            logger.warning(f"Failed to check vastai CLI: {e}")
            return False

    async def get_instance_states(self) -> list[InstanceInfo]:
        """Query Vast.ai for all instance states.

        Returns:
            List of InstanceInfo for all Vast.ai instances.
        """
        if not self.is_enabled:
            return []

        try:
            # Run vastai show instances --raw
            env = os.environ.copy()
            if self._api_key:
                env["VAST_API_KEY"] = self._api_key

            proc = await asyncio.create_subprocess_exec(
                "vastai", "show", "instances", "--raw",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                self._last_error = error_msg
                logger.error(f"vastai show instances failed: {error_msg}")
                return []

            self._last_check = datetime.now()
            self._last_error = None

            # Parse JSON response
            data = json.loads(stdout.decode())
            instances = []

            for instance in data:
                instance_id = str(instance.get("id", ""))
                actual_status = instance.get("actual_status", "unknown")
                state = VAST_STATE_MAP.get(actual_status, ProviderInstanceState.UNKNOWN)

                # Extract instance details
                info = InstanceInfo(
                    instance_id=instance_id,
                    state=state,
                    provider="vast",
                    ssh_host=instance.get("ssh_host"),
                    ssh_port=instance.get("ssh_port", 22),
                    public_ip=instance.get("public_ipaddr"),
                    gpu_type=instance.get("gpu_name"),
                    gpu_count=instance.get("num_gpus", 1),
                    gpu_vram_gb=instance.get("gpu_ram", 0) / 1024,  # MB to GB
                    raw_data=instance,
                )

                instances.append(info)

            logger.debug(f"Found {len(instances)} Vast.ai instances")
            return instances

        except json.JSONDecodeError as e:
            self._last_error = f"JSON parse error: {e}"
            logger.error(f"Failed to parse vastai output: {e}")
            return []
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Error querying Vast.ai instances: {e}")
            return []

    def correlate_with_config(
        self,
        instances: list[InstanceInfo],
        config_hosts: dict[str, dict],
    ) -> list[InstanceInfo]:
        """Match Vast.ai instances to node names in config.

        Matching strategies:
        1. Node name contains instance ID (vast-{id})
        2. SSH host/port matches
        """
        # Build lookup maps from config
        id_to_node: dict[str, str] = {}
        ssh_to_node: dict[tuple[str, int], str] = {}

        for node_name, host_config in config_hosts.items():
            # Match vast-{id} pattern
            if node_name.startswith("vast-"):
                parts = node_name.split("-")
                if len(parts) >= 2 and parts[1].isdigit():
                    id_to_node[parts[1]] = node_name

            # Build SSH lookup
            ssh_host = host_config.get("ssh_host")
            ssh_port = host_config.get("ssh_port", 22)
            if ssh_host:
                ssh_to_node[(ssh_host, ssh_port)] = node_name

        # Correlate instances
        for instance in instances:
            # Try ID match first
            if instance.instance_id in id_to_node:
                instance.node_name = id_to_node[instance.instance_id]
                continue

            # Try SSH match
            ssh_key = (instance.ssh_host, instance.ssh_port)
            if ssh_key in ssh_to_node:
                instance.node_name = ssh_to_node[ssh_key]

        return instances

    async def get_terminated_instances(
        self,
        config_hosts: dict[str, dict],
    ) -> list[str]:
        """Find nodes in config that are no longer in Vast.ai.

        Args:
            config_hosts: The 'hosts' section from distributed_hosts.yaml

        Returns:
            List of node names that appear terminated (not in API response).
        """
        # Get current instances
        instances = await self.get_instance_states()
        active_ids = {inst.instance_id for inst in instances}

        # Find vast nodes in config that aren't in API response
        terminated = []
        for node_name, host_config in config_hosts.items():
            if not node_name.startswith("vast-"):
                continue

            # Extract instance ID from node name
            parts = node_name.split("-")
            if len(parts) >= 2 and parts[1].isdigit():
                instance_id = parts[1]
                if instance_id not in active_ids:
                    # Instance not in API response - likely terminated
                    current_status = host_config.get("status", "")
                    if current_status != "retired":
                        terminated.append(node_name)

        return terminated
