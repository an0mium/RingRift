# ADR-008: Event Router Architecture

## Status

Accepted (December 2025)

## Context

The coordination layer needed a reliable way for 66+ daemons to communicate without tight coupling. Direct method calls created circular dependencies and made testing difficult.

## Decision

We implemented a unified event-driven architecture using `UnifiedEventRouter`:

1. **Single Event Bus**: All events flow through one router
2. **Content-Based Deduplication**: SHA256 hash prevents duplicate processing
3. **Dead Letter Queue**: Failed events are captured for retry
4. **Type Normalization**: Both string and enum event types work

## Architecture

```
Emitter -> UnifiedEventRouter -> [DataEventType normalization]
                              -> [Deduplication check]
                              -> [Async handler dispatch]
                              -> [DLQ on failure]
```

## Consequences

- Loose coupling between coordinators
- Easy to add new event types
- Testable in isolation
- Slight latency overhead (~1ms per event)

## Alternatives Considered

- Direct method calls (rejected: circular deps)
- Message queue (rejected: infrastructure complexity)
- Observer pattern (rejected: no deduplication)
