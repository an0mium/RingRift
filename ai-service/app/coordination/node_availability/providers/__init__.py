"""Provider-specific state checkers for NodeAvailabilityDaemon.

Each provider implements a StateChecker subclass that:
1. Queries the provider API for instance states
2. Maps provider-specific states to ProviderInstanceState enum
3. Correlates instances with node names in distributed_hosts.yaml
"""

from app.coordination.node_availability.providers.vast_checker import VastChecker
from app.coordination.node_availability.providers.lambda_checker import LambdaChecker
from app.coordination.node_availability.providers.runpod_checker import RunPodChecker

__all__ = [
    "VastChecker",
    "LambdaChecker",
    "RunPodChecker",
]
