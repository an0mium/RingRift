# ADR-010: Provider Abstraction Layer

## Status

Proposed (December 2025)

## Context

Cloud provider logic (Lambda, Vast, RunPod, Nebius) was scattered across 12+ files with hardcoded checks like `if "vast" in node_name`.

## Decision

Create a `ProviderRegistry` with:

1. **ProviderConfig**: Dataclass with idle thresholds, shutdown methods
2. **Node name matching**: Regex patterns per provider
3. **Centralized defaults**: Single file for all provider configs

## Implementation

```python
from app.coordination.providers import ProviderRegistry

config = ProviderRegistry.get_for_node("vast-12345")
print(config.idle_threshold_seconds)  # 900
```

## Consequences

- Adding provider: Single file change
- No scattered `if provider ==` checks
- Testable configuration
- Runtime provider detection

## Migration

1. Create providers/registry.py
2. Update daemon_types.py to use registry
3. Update idle shutdown daemons
4. Remove hardcoded provider checks
