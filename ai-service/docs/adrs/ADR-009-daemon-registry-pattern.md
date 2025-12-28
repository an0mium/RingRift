# ADR-009: Daemon Registry Pattern

## Status

Accepted (December 2025)

## Context

The DaemonManager had 66 factory methods creating daemons imperatively. Adding new daemons required modifying multiple files.

## Decision

We adopted a declarative registry pattern:

1. **DaemonSpec dataclass**: Frozen configuration per daemon type
2. **DAEMON_REGISTRY**: Dict[DaemonType, DaemonSpec] as single source of truth
3. **daemon_runners.py**: Separated runner functions from manager

## Implementation

```python
DAEMON_REGISTRY = {
    DaemonType.AUTO_SYNC: DaemonSpec(
        runner_name="create_auto_sync",
        depends_on=(DaemonType.EVENT_ROUTER,),
        category="sync",
        auto_restart=True,
    ),
}
```

## Consequences

- Adding daemon: 6-8 lines vs 20-25 LOC
- Dependency graph validation at startup
- Runners testable in isolation
- Categories enable grouped operations

## Alternatives Considered

- Decorator-based registration (rejected: less explicit)
- YAML configuration (rejected: no type checking)
