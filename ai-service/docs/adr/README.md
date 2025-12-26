# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the RingRift AI Service.

## Index

| ADR                                               | Title                                                      | Status   |
| ------------------------------------------------- | ---------------------------------------------------------- | -------- |
| [ADR-001](ADR-001-event-driven-architecture.md)   | Event-Driven Architecture for Training Pipeline            | Accepted |
| [ADR-002](ADR-002-daemon-lifecycle-management.md) | Daemon Lifecycle Management                                | Accepted |
| [ADR-003](ADR-003-pfsp-opponent-selection.md)     | PFSP (Prioritized Fictitious Self-Play) Opponent Selection | Accepted |
| [ADR-004](ADR-004-quality-gate-feedback-loop.md)  | Quality Gate and Feedback Loop Architecture                | Accepted |

## ADR Format

Each ADR follows this structure:

```markdown
# ADR-NNN: Title

**Status**: Proposed | Accepted | Deprecated | Superseded
**Date**: Month Year
**Author**: Author Name

## Context

Why was this decision needed?

## Decision

What was decided?

## Consequences

What are the positive and negative effects?

## Implementation Notes

Any implementation details worth noting.

## Related ADRs

Links to related decisions.
```

## Creating New ADRs

1. Copy the template above
2. Use sequential numbering (ADR-005, ADR-006, ...)
3. Start with Status: Proposed
4. Update this README index
5. Get team review before changing to Accepted
