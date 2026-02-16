"""Lambda Labs State Checker (December 2025).

Queries Lambda Labs REST API to get current instance states and maps them to
ProviderInstanceState enum for distributed_hosts.yaml synchronization.

Lambda Labs API Documentation: https://cloud.lambdalabs.com/api/v1

Lambda Labs State Mappings:
- "active" -> RUNNING (ready)
- "booting" -> STARTING (setup)
- "unhealthy" -> UNKNOWN (offline)
- "terminated" -> TERMINATED (retired)
- Not in list -> TERMINATED (retired)
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Optional

from app.coordination.node_availability.state_checker import (
    InstanceInfo,
    ProviderInstanceState,
    StateChecker,
)

logger = logging.getLogger(__name__)

# Lambda Labs state mappings
LAMBDA_STATE_MAP: dict[str, ProviderInstanceState] = {
    "active": ProviderInstanceState.RUNNING,
    "booting": ProviderInstanceState.STARTING,
    "unhealthy": ProviderInstanceState.UNKNOWN,
    "terminated": ProviderInstanceState.TERMINATED,
}

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"


class LambdaChecker(StateChecker):
    """State checker for Lambda Labs instances.

    Uses Lambda Labs REST API to query instance states.
    Requires LAMBDA_API_KEY environment variable.

    Instance correlation with distributed_hosts.yaml:
    - Matches by hostname pattern (lambda-*)
    - Falls back to IP address matching
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Lambda Labs checker.

        Args:
            api_key: Lambda Labs API key (uses env if None)
        """
        super().__init__("lambda")
        self._api_key = api_key or os.environ.get("LAMBDA_API_KEY")
        self._http_session = None

        # Check for API key file if not in env
        if not self._api_key:
            key_files = [
                os.path.expanduser("~/.lambda_api_key"),
                os.path.expanduser("~/.config/lambda/api_key"),
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
            self.disable("No LAMBDA_API_KEY found")

    async def _get_session(self) -> Any:
        """Get or create aiohttp session."""
        if self._http_session is None:
            try:
                import aiohttp
                self._http_session = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    }
                )
            except ImportError:
                logger.warning("aiohttp not available, Lambda checker disabled")
                self.disable("aiohttp not installed")
                return None
        return self._http_session

    async def close(self) -> None:
        """Close HTTP session."""
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    async def check_api_availability(self) -> bool:
        """Check if Lambda Labs API is accessible."""
        if not self._api_key:
            return False

        session = await self._get_session()
        if not session:
            return False

        try:
            import aiohttp
            async with session.get(f"{LAMBDA_API_BASE}/instances") as resp:
                return resp.status == 200
        except asyncio.TimeoutError:
            logger.warning("Lambda API check timed out")
            return False
        except aiohttp.ClientError as e:
            logger.warning(f"Lambda API network error: {e}")
            return False
        except (OSError, RuntimeError) as e:
            logger.warning(f"Lambda API check failed: {e}")
            return False

    async def get_instance_states(self) -> list[InstanceInfo]:
        """Query Lambda Labs for all instance states.

        Returns:
            List of InstanceInfo for all Lambda Labs instances.
        """
        if not self.is_enabled:
            return []

        session = await self._get_session()
        if not session:
            return []

        try:
            async with session.get(f"{LAMBDA_API_BASE}/instances") as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    self._last_error = f"API error {resp.status}: {error_text}"
                    logger.error(f"Lambda API error: {self._last_error}")
                    return []

                data = await resp.json()
                self._last_check = datetime.now()
                self._last_error = None

                instances = []
                for instance in data.get("data", []):
                    instance_id = instance.get("id", "")
                    status = instance.get("status", "unknown")
                    state = LAMBDA_STATE_MAP.get(status, ProviderInstanceState.UNKNOWN)

                    # Extract instance details
                    info = InstanceInfo(
                        instance_id=instance_id,
                        state=state,
                        provider="lambda",
                        hostname=instance.get("name"),
                        public_ip=instance.get("ip"),
                        ssh_host=instance.get("ip"),
                        ssh_port=22,
                        gpu_type=instance.get("instance_type", {}).get("name"),
                        gpu_count=instance.get("instance_type", {}).get("specs", {}).get("gpus", 1),
                        raw_data=instance,
                    )

                    instances.append(info)

                logger.debug(f"Found {len(instances)} Lambda Labs instances")
                return instances

        except asyncio.TimeoutError:
            self._last_error = "Request timed out"
            logger.error("Lambda Labs API request timed out")
            return []
        except aiohttp.ClientError as e:
            self._last_error = f"Network error: {e}"
            logger.error(f"Lambda Labs API network error: {e}")
            return []
        except (KeyError, ValueError, TypeError) as e:
            self._last_error = f"Data parse error: {e}"
            logger.error(f"Error parsing Lambda Labs response: {e}")
            return []
        except (OSError, RuntimeError) as e:
            self._last_error = str(e)
            logger.error(f"Error querying Lambda Labs instances: {e}")
            return []

    def correlate_with_config(
        self,
        instances: list[InstanceInfo],
        config_hosts: dict[str, dict],
    ) -> list[InstanceInfo]:
        """Match Lambda Labs instances to node names in config.

        Matching strategies:
        1. Node name contains instance hostname
        2. IP address matches ssh_host
        """
        # Build lookup maps from config
        hostname_to_node: dict[str, str] = {}
        ip_to_node: dict[str, str] = {}

        for node_name, host_config in config_hosts.items():
            if not node_name.startswith("lambda-"):
                continue

            # Map node name directly
            hostname_to_node[node_name] = node_name

            # Build IP lookup
            ssh_host = host_config.get("ssh_host")
            if ssh_host:
                ip_to_node[ssh_host] = node_name

            # Also check tailscale_ip
            tailscale_ip = host_config.get("tailscale_ip")
            if tailscale_ip:
                ip_to_node[tailscale_ip] = node_name

        # Correlate instances
        for instance in instances:
            # Try hostname match
            hostname = instance.hostname or ""
            if hostname in hostname_to_node:
                instance.node_name = hostname_to_node[hostname]
                continue

            # Try IP match
            if instance.public_ip and instance.public_ip in ip_to_node:
                instance.node_name = ip_to_node[instance.public_ip]
                continue

            # Try matching lambda-gh200-* pattern by instance name
            for node_name in hostname_to_node.keys():
                # Check if instance hostname contains node name or vice versa
                if hostname and (hostname in node_name or node_name in hostname):
                    instance.node_name = node_name
                    break

        return instances

    async def get_terminated_instances(
        self,
        config_hosts: dict[str, dict],
    ) -> list[str]:
        """Find nodes in config that are no longer in Lambda Labs.

        Args:
            config_hosts: The 'hosts' section from distributed_hosts.yaml

        Returns:
            List of node names that appear terminated (not in API response).
        """
        # Get current instances
        instances = await self.get_instance_states()
        active_ips = {inst.public_ip for inst in instances if inst.public_ip}
        active_hostnames = {inst.hostname for inst in instances if inst.hostname}

        # Find lambda nodes in config that aren't in API response
        terminated = []
        for node_name, host_config in config_hosts.items():
            if not node_name.startswith("lambda-"):
                continue

            # Check if node is active
            ssh_host = host_config.get("ssh_host")
            if ssh_host and ssh_host in active_ips:
                continue

            # Check hostname match
            if node_name in active_hostnames:
                continue

            # Node not found in active instances
            current_status = host_config.get("status", "")
            if current_status not in ("retired", "offline", "archived"):
                terminated.append(node_name)

        return terminated
