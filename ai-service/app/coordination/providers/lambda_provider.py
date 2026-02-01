"""Lambda Labs cloud provider integration.

Implements the CloudProvider interface for Lambda Labs GPU cloud.
Supports GH200, H100, A100, and other GPU types.

API Documentation: https://cloud.lambdalabs.com/api/v1/docs

Created: Dec 28, 2025
December 30, 2025: Added circuit breaker protection for API resilience.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp

from .base import (
    CloudProvider,
    GPUType,
    Instance,
    InstanceStatus,
    ProviderType,
)
from app.distributed.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)

logger = logging.getLogger(__name__)

# Circuit breaker for Lambda Labs API calls (December 30, 2025)
# Lambda handles training-only workloads on 6 GH200 nodes, so we use
# tolerant thresholds to avoid blocking training during transient issues
_lambda_circuit_breaker: CircuitBreaker | None = None


def get_lambda_circuit_breaker() -> CircuitBreaker:
    """Get the Lambda Labs API circuit breaker singleton."""
    global _lambda_circuit_breaker
    if _lambda_circuit_breaker is None:
        _lambda_circuit_breaker = CircuitBreaker(
            failure_threshold=4,  # Open after 4 consecutive failures
            recovery_timeout=60.0,  # Wait 60s before testing recovery
            half_open_max_calls=1,  # Single test call in half-open
            success_threshold=1,
            operation_type="lambda_api",
            max_backoff=300.0,  # Cap at 5 minutes
        )
    return _lambda_circuit_breaker


def reset_lambda_circuit_breaker() -> None:
    """Reset the circuit breaker (for testing)."""
    global _lambda_circuit_breaker
    if _lambda_circuit_breaker is not None:
        _lambda_circuit_breaker.reset_all()
    _lambda_circuit_breaker = None


@dataclass
class LambdaConfig:
    """Configuration for Lambda Labs provider."""
    api_key: str | None = None
    api_base: str = "https://cloud.lambdalabs.com/api/v1"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "LambdaConfig":
        """Load configuration from environment variables."""
        return cls(
            api_key=os.environ.get("LAMBDA_API_KEY"),
        )


# GPU type mappings for Lambda Labs
LAMBDA_INSTANCE_TYPES = {
    # GH200 instances
    GPUType.GH200_96GB: "gpu_1x_gh200",
    # H100 instances
    GPUType.H100_80GB: "gpu_1x_h100_sxm5",
    # A100 instances
    GPUType.A100_80GB: "gpu_1x_a100_sxm4",
    GPUType.A100_40GB: "gpu_1x_a100",
    # A10 instances
    GPUType.A10: "gpu_1x_a10",
    # Consumer GPUs
    GPUType.RTX_4090: "gpu_1x_rtx4090",
    GPUType.RTX_3090: "gpu_1x_rtx3090",
}

# Reverse mapping for parsing
LAMBDA_GPU_TYPES = {v: k for k, v in LAMBDA_INSTANCE_TYPES.items()}

# Cost per hour (approximate, check Lambda pricing for current rates)
LAMBDA_COSTS = {
    GPUType.GH200_96GB: 2.49,
    GPUType.H100_80GB: 2.49,
    GPUType.A100_80GB: 1.29,
    GPUType.A100_40GB: 1.10,
    GPUType.A10: 0.60,
    GPUType.RTX_4090: 0.80,
    GPUType.RTX_3090: 0.50,
}


class LambdaProvider(CloudProvider):
    """Lambda Labs cloud provider implementation.

    Requires LAMBDA_API_KEY environment variable to be set.

    Example:
        provider = LambdaProvider()
        if provider.is_configured():
            instances = await provider.list_instances()
    """

    def __init__(self, config: LambdaConfig | None = None):
        self.config = config or LambdaConfig.from_env()
        self._session: aiohttp.ClientSession | None = None

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LAMBDA

    @property
    def name(self) -> str:
        return "Lambda Labs"

    def is_configured(self) -> bool:
        """Check if API key is set."""
        return bool(self.config.api_key)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated API request with circuit breaker protection.

        December 30, 2025: Added circuit breaker to prevent cascading failures
        when Lambda Labs API is experiencing issues.

        Raises:
            ValueError: If API key not configured
            CircuitOpenError: If circuit is open due to recent failures
        """
        if not self.config.api_key:
            raise ValueError("Lambda API key not configured")

        breaker = get_lambda_circuit_breaker()
        target = "lambda_api"

        # Check circuit state before attempting call
        if not breaker.can_execute(target):
            state = breaker.get_status(target)
            logger.warning(
                f"Lambda Labs circuit breaker is {state.state.value}, "
                f"skipping API call (failures={state.failure_count})"
            )
            raise CircuitOpenError(f"Lambda Labs API circuit is {state.state.value}")

        url = f"{self.config.api_base}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        session = await self._get_session()

        try:
            async with session.request(
                method,
                url,
                headers=headers,
                json=json,
            ) as resp:
                data = await resp.json()

                if resp.status >= 400:
                    error_msg = data.get("error", {}).get("message", str(data))
                    # Record failure for API errors
                    breaker.record_failure(target)
                    raise Exception(f"Lambda API error ({resp.status}): {error_msg}")

                # Success - record it
                breaker.record_success(target)
                return data

        except asyncio.TimeoutError as e:
            logger.warning(f"Lambda API timeout: {e}")
            breaker.record_failure(target)
            raise
        except (ConnectionError, OSError) as e:
            logger.warning(f"Lambda API connection error: {e}")
            breaker.record_failure(target)
            raise
        except aiohttp.ClientError as e:
            breaker.record_failure(target)
            raise Exception(f"Lambda API request failed: {e}")

    def _parse_instance_status(self, status: str) -> InstanceStatus:
        """Parse Lambda instance status to enum."""
        status_map = {
            "active": InstanceStatus.RUNNING,
            "booting": InstanceStatus.STARTING,
            "unhealthy": InstanceStatus.ERROR,
            "terminating": InstanceStatus.STOPPING,
            "terminated": InstanceStatus.TERMINATED,
        }
        return status_map.get(status, InstanceStatus.UNKNOWN)

    def _parse_gpu_type(self, instance_type: str) -> GPUType:
        """Parse Lambda instance type to GPU type."""
        return LAMBDA_GPU_TYPES.get(instance_type, GPUType.UNKNOWN)

    async def list_instances(self) -> list[Instance]:
        """List all Lambda instances."""
        if not self.is_configured():
            logger.warning("Lambda provider not configured")
            return []

        try:
            data = await self._api_request("GET", "/instances")
            instances_data = data.get("data", [])

            instances = []
            for inst in instances_data:
                gpu_type = self._parse_gpu_type(inst.get("instance_type", {}).get("name", ""))

                instance = Instance(
                    id=inst["id"],
                    provider=ProviderType.LAMBDA,
                    name=inst.get("name", inst["id"]),
                    status=self._parse_instance_status(inst.get("status", "unknown")),
                    gpu_type=gpu_type,
                    gpu_count=1,  # Lambda instances are typically single-GPU
                    gpu_memory_gb=self._get_gpu_memory(gpu_type),
                    ip_address=inst.get("ip"),
                    ssh_port=22,
                    ssh_user="ubuntu",
                    region=inst.get("region", {}).get("name", ""),
                    cost_per_hour=self.get_cost_per_hour(gpu_type),
                    raw_data=inst,
                )
                instances.append(instance)

            return instances

        except Exception as e:
            logger.error(f"Failed to list Lambda instances: {e}")
            return []

    async def get_instance(self, instance_id: str) -> Instance | None:
        """Get a specific instance by ID."""
        if not self.is_configured():
            return None

        try:
            data = await self._api_request("GET", f"/instances/{instance_id}")
            inst = data.get("data")

            if not inst:
                return None

            gpu_type = self._parse_gpu_type(inst.get("instance_type", {}).get("name", ""))

            return Instance(
                id=inst["id"],
                provider=ProviderType.LAMBDA,
                name=inst.get("name", inst["id"]),
                status=self._parse_instance_status(inst.get("status", "unknown")),
                gpu_type=gpu_type,
                gpu_count=1,
                gpu_memory_gb=self._get_gpu_memory(gpu_type),
                ip_address=inst.get("ip"),
                ssh_port=22,
                ssh_user="ubuntu",
                region=inst.get("region", {}).get("name", ""),
                cost_per_hour=self.get_cost_per_hour(gpu_type),
                raw_data=inst,
            )

        except Exception as e:
            logger.error(f"Failed to get Lambda instance {instance_id}: {e}")
            return None

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        """Get current status of an instance."""
        instance = await self.get_instance(instance_id)
        return instance.status if instance else InstanceStatus.UNKNOWN

    async def scale_up(
        self,
        gpu_type: GPUType,
        count: int = 1,
        region: str | None = None,
        name_prefix: str = "ringrift",
    ) -> list[Instance]:
        """Create new Lambda instances."""
        if not self.is_configured():
            raise ValueError("Lambda provider not configured")

        instance_type = LAMBDA_INSTANCE_TYPES.get(gpu_type)
        if not instance_type:
            raise ValueError(f"GPU type {gpu_type} not supported on Lambda")

        # Get available regions for this instance type
        if not region:
            region = await self._get_best_region(instance_type)
            if not region:
                raise ValueError(f"No available regions for {instance_type}")

        try:
            data = await self._api_request(
                "POST",
                "/instance-operations/launch",
                json={
                    "instance_type_name": instance_type,
                    "region_name": region,
                    "quantity": count,
                    "name": f"{name_prefix}-{gpu_type.value}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "file_system_names": [],
                },
            )

            instance_ids = data.get("data", {}).get("instance_ids", [])
            logger.info(f"Launched {len(instance_ids)} Lambda instance(s): {instance_ids}")

            # Wait for instances to be ready
            instances = []
            for inst_id in instance_ids:
                instance = await self._wait_for_instance(inst_id, timeout=300)
                if instance:
                    instances.append(instance)

            return instances

        except Exception as e:
            logger.error(f"Failed to launch Lambda instances: {e}")
            raise

    async def scale_down(self, instance_ids: list[str]) -> dict[str, bool]:
        """Terminate Lambda instances."""
        if not self.is_configured():
            return {inst_id: False for inst_id in instance_ids}

        results = {}

        try:
            data = await self._api_request(
                "POST",
                "/instance-operations/terminate",
                json={"instance_ids": instance_ids},
            )

            terminated_ids = data.get("data", {}).get("terminated_instances", [])

            for inst_id in instance_ids:
                # Lambda returns terminated instance objects
                success = any(
                    t.get("id") == inst_id for t in terminated_ids
                ) if isinstance(terminated_ids, list) else inst_id in terminated_ids
                results[inst_id] = success

            logger.info(f"Terminated Lambda instances: {results}")

        except Exception as e:
            logger.error(f"Failed to terminate Lambda instances: {e}")
            for inst_id in instance_ids:
                results[inst_id] = False

        return results

    async def reboot_instance(self, instance_id: str) -> bool:
        """Reboot a Lambda instance."""
        if not self.is_configured():
            return False

        try:
            await self._api_request(
                "POST",
                "/instance-operations/restart",
                json={"instance_ids": [instance_id]},
            )
            logger.info(f"Rebooted Lambda instance: {instance_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to reboot Lambda instance {instance_id}: {e}")
            return False

    def get_cost_per_hour(self, gpu_type: GPUType) -> float:
        """Get hourly cost for a GPU type."""
        return LAMBDA_COSTS.get(gpu_type, 0.0)

    def _get_gpu_memory(self, gpu_type: GPUType) -> float:
        """Get GPU memory in GB."""
        memory_map = {
            GPUType.GH200_96GB: 96.0,
            GPUType.H100_80GB: 80.0,
            GPUType.A100_80GB: 80.0,
            GPUType.A100_40GB: 40.0,
            GPUType.A10: 24.0,
            GPUType.RTX_4090: 24.0,
            GPUType.RTX_3090: 24.0,
        }
        return memory_map.get(gpu_type, 0.0)

    async def _get_best_region(self, instance_type: str) -> str | None:
        """Get the best available region for an instance type."""
        try:
            data = await self._api_request("GET", "/instance-types")
            instance_types = data.get("data", {})

            for type_name, type_info in instance_types.items():
                if type_name == instance_type:
                    regions = type_info.get("regions_with_capacity_available", [])
                    if regions:
                        # Prefer US regions
                        for region in regions:
                            region_name = region.get("name", "")
                            if "us-" in region_name:
                                return region_name
                        # Fall back to first available
                        return regions[0].get("name")

        except Exception as e:
            logger.error(f"Failed to get Lambda regions: {e}")

        return None

    async def _wait_for_instance(
        self,
        instance_id: str,
        timeout: int = 300,
    ) -> Instance | None:
        """Wait for an instance to be ready."""
        start_time = asyncio.get_running_loop().time()

        while asyncio.get_running_loop().time() - start_time < timeout:
            instance = await self.get_instance(instance_id)
            if instance and instance.status == InstanceStatus.RUNNING:
                return instance

            await asyncio.sleep(10)

        logger.warning(f"Timeout waiting for Lambda instance {instance_id}")
        return await self.get_instance(instance_id)

    async def get_instance_types(self) -> dict[str, dict]:
        """Get available instance types and their availability."""
        if not self.is_configured():
            return {}

        try:
            data = await self._api_request("GET", "/instance-types")
            return data.get("data", {})
        except Exception as e:
            logger.error(f"Failed to get Lambda instance types: {e}")
            return {}

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def health_check(self) -> "HealthCheckResult":
        """Check provider health for CoordinatorProtocol compliance.

        December 30, 2025: Added for daemon health monitoring integration.
        """
        from app.coordination.protocols import CoordinatorStatus, HealthCheckResult

        configured = self.is_configured()

        # Get circuit breaker status
        breaker = get_lambda_circuit_breaker()
        circuit_status = breaker.get_status("lambda_api")
        circuit_state = circuit_status.state

        if not configured:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.ERROR,
                message="LambdaProvider: API key not configured",
                details={
                    "configured": False,
                    "circuit_state": circuit_state.value,
                },
            )

        # Check if circuit breaker is open
        if circuit_state == CircuitState.OPEN:
            return HealthCheckResult(
                healthy=False,
                status=CoordinatorStatus.DEGRADED,
                message=f"LambdaProvider: API circuit open (failures={circuit_status.failure_count})",
                details={
                    "configured": True,
                    "circuit_state": circuit_state.value,
                    "circuit_failures": circuit_status.failure_count,
                    "circuit_opened_at": circuit_status.opened_at,
                },
            )

        return HealthCheckResult(
            healthy=True,
            status=CoordinatorStatus.RUNNING,
            message="LambdaProvider: API configured and operational",
            details={
                "configured": True,
                "circuit_state": circuit_state.value,
                "circuit_failures": circuit_status.failure_count,
            },
        )


# Singleton instance
_lambda_provider: LambdaProvider | None = None


def get_lambda_provider() -> LambdaProvider:
    """Get the singleton Lambda provider instance."""
    global _lambda_provider
    if _lambda_provider is None:
        _lambda_provider = LambdaProvider()
    return _lambda_provider


def reset_lambda_provider() -> None:
    """Reset the singleton (for testing)."""
    global _lambda_provider
    _lambda_provider = None
