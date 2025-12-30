# Events Module Unit Tests

This directory contains comprehensive unit tests for the `app/events` module.

## Test Coverage

**100% code coverage** achieved for the events module.

### Tested Modules

- `app/events/__init__.py` - Package exports
- `app/events/types.py` - Event type definitions and utilities

### Test Files

1. **test_types.py** - Core event type system tests
   - RingRiftEventType enum validation
   - EventCategory functionality
   - Event category mapping
   - Utility functions (get_events_by_category, is_cross_process_event)
   - Backwards compatibility (StageEvent alias)
   - Event coverage and categorization
   - Event type semantics
   - Cross-process event patterns

2. **test_event_usage.py** - Event usage pattern tests
   - Event type imports and exports
   - Event filtering and routing
   - Event subscription patterns
   - Event value access and iteration
   - String representation
   - Type safety

3. **test_event_integration.py** - Integration scenario tests
   - Training pipeline event sequences
   - Evaluation pipeline workflows
   - Promotion pipeline flows
   - Data sync pipeline events
   - Quality feedback loops
   - Regression detection workflows
   - Cluster coordination events
   - Work queue lifecycles
   - Optimization events (CMA-ES, NAS, PBT)
   - Stage completion events
   - Cross-process communication patterns

## Test Statistics

- **Total Tests**: 109
- **Pass Rate**: 100%
- **Code Coverage**: 100%

## Test Categories

### 1. Event Type Validation (22 tests)

- Enum structure and values
- Event existence across categories
- Event naming conventions
- Value format validation

### 2. Event Categorization (15 tests)

- Category mapping
- Event retrieval by category
- Category completeness
- No duplicate categorization

### 3. Event Utilities (12 tests)

- get_events_by_category()
- is_cross_process_event()
- EventCategory.from_event()
- Cross-process event set validation

### 4. Backwards Compatibility (4 tests)

- StageEvent alias
- DataEventType alias
- Value mapping

### 5. Event Usage Patterns (18 tests)

- Dictionary keys and sets
- Filtering and routing
- Subscription patterns
- Iteration and counting

### 6. Integration Scenarios (38 tests)

- Training lifecycle
- Evaluation workflows
- Promotion flows
- Data sync pipelines
- Quality feedback
- Regression detection
- Cluster coordination
- Work queues
- Optimization workflows

## Running Tests

### Run all events tests

```bash
python -m pytest tests/unit/events/ -v
```

### Run with coverage

```bash
python -m pytest tests/unit/events/ --cov=app/events --cov-report=term-missing
```

### Run specific test file

```bash
python -m pytest tests/unit/events/test_types.py -v
python -m pytest tests/unit/events/test_event_usage.py -v
python -m pytest tests/unit/events/test_event_integration.py -v
```

### Run specific test class

```bash
python -m pytest tests/unit/events/test_types.py::TestRingRiftEventType -v
```

### Run specific test

```bash
python -m pytest tests/unit/events/test_types.py::TestRingRiftEventType::test_event_type_values_are_strings -v
```

## Key Test Highlights

### Comprehensive Event Coverage

- All 173 unified event types are validated (`RingRiftEventType`)
- All 15 event categories are tested
- Cross-process event set verified (39 events)

### Pipeline Integration

- Complete training pipeline: threshold → started → progress → completed/failed
- Evaluation workflow: started → progress → completed → Elo updates → curriculum
- Promotion flow: candidate → started → promoted/rejected/failed
- Data sync: started → completed/failed with freshness monitoring

### Error Recovery

- Training rollback sequences
- Regression detection and recovery
- Error handling and timeout scenarios
- Graceful degradation patterns

### Cluster Coordination

- Node lifecycle: online → unhealthy → recovered → offline
- Leader election: elected → lost → stepdown
- P2P health monitoring
- Work queue management

### Quality Assurance

- Event naming consistency
- Lifecycle completeness (started/completed pairs)
- Category organization
- Type safety and immutability

## Design Principles Tested

1. **Single Source of Truth**: All events defined in RingRiftEventType
2. **Category Organization**: Events grouped by functional area
3. **Cross-Process Awareness**: Clear distinction between local and distributed events
4. **Backwards Compatibility**: Aliases maintained for legacy code
5. **Type Safety**: Enum-based for compile-time checking
6. **Naming Conventions**: Consistent snake_case values, descriptive names
7. **Lifecycle Completeness**: Paired started/completed/failed events

## Future Considerations

These tests validate the static event type definitions. For dynamic event behavior (creation, serialization, dispatching, handlers), see:

- `tests/unit/coordination/test_event_emitters.py` - Event emission
- `tests/unit/coordination/test_event_router.py` - Event routing
- `tests/unit/coordination/test_cross_process_events.py` - Cross-process propagation

## Maintenance Notes

When adding new events to `app/events/types.py`:

1. Add the event to the appropriate category in `_EVENT_CATEGORIES` dict
2. Add to `CROSS_PROCESS_EVENT_TYPES` if it should be distributed
3. Run tests to ensure:
   - No naming conflicts
   - Proper categorization
   - Lifecycle completeness (if applicable)
   - Value format conventions (snake_case, descriptive)

The test suite will automatically validate new events against all established patterns.
