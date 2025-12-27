# Deprecated Module Usage Tracking

**Generated**: December 26, 2025
**Purpose**: Track imports from deprecated modules for future migration

---

## Overview

This document lists all files that import from deprecated coordination modules. These imports will need to be updated before the deprecated wrappers are removed in Q2 2026.

**Total files with deprecated imports**: 49

---

## Deprecated Modules

### 1. `health_check_orchestrator.py`

**Replacement**: `cluster/health.py` → `UnifiedHealthManager`

### 2. `event_emitters.py`

**Replacement**: `event_router.py`

### 3. `cross_process_events.py`

**Replacement**: `event_router.py`

---

## Files Using Deprecated Modules (49 files)

### Core Training & Coordination

1. `app/training/train.py`
2. `app/training/promotion_controller.py`
3. `app/training/curriculum_feedback.py`
4. `app/training/validation_scheduling.py`
5. `app/training/task_lifecycle_integration.py`
6. `app/training/event_integration.py`
7. `app/training/adaptive_controller.py`

### Coordination (Main Directory)

8. `app/coordination/__init__.py`
9. `app/coordination/data_pipeline_orchestrator.py`
10. `app/coordination/unified_node_health_daemon.py`
11. `app/coordination/feedback_loop_controller.py`
12. `app/coordination/training_coordinator.py`
13. `app/coordination/resource_monitoring_coordinator.py`
14. `app/coordination/selfplay_orchestrator.py`
15. `app/coordination/task_coordinator.py`
16. `app/coordination/evaluation_daemon.py`
17. `app/coordination/utilization_optimizer.py`
18. `app/coordination/recovery_orchestrator.py`
19. `app/coordination/task_decorators.py`
20. `app/coordination/metrics_analysis_orchestrator.py`
21. `app/coordination/handler_resilience.py`
22. `app/coordination/cache_coordination_orchestrator.py`
23. `app/coordination/async_training_bridge.py`
24. `app/coordination/pipeline_actions.py`
25. `app/coordination/task_lifecycle_coordinator.py`
26. `app/coordination/optimization_coordinator.py`
27. `app/coordination/README.md`

### Distributed Systems

28. `app/distributed/sync_coordinator.py`
29. `app/distributed/data_events.py`
30. `app/distributed/sync_orchestrator.py`
31. `app/distributed/unified_data_sync.py`

### Quality & Monitoring

32. `app/quality/unified_quality.py`

### Scripts

33. `scripts/demo_auto_export.py`
34. `scripts/export_replay_dataset.py`
35. `scripts/auto_promote.py`
36. `scripts/cluster_health_cli.py`

### Tests

37. `tests/unit/coordination/conftest.py`

### Documentation

38. `docs/MIGRATION_GUIDE.md`
39. `docs/planning/INTEGRATION_ASSESSMENT_DEC2025.md`
40. `app/core/EVENT_GUIDE.md`

### Deprecated Module Wrappers (Self-referential)

41. `app/coordination/_deprecated_cross_process_events.py`
42. `app/coordination/_deprecated_event_emitters.py`
43. `app/coordination/_deprecated_health_check_orchestrator.py`

### Consolidation Documentation

44. `app/coordination/CONSOLIDATION_PROGRESS.md`
45. `app/coordination/MODULE_CONSOLIDATION_STATUS.md`

### Still Using Deprecated Modules

46. `app/coordination/cross_process_events.py` (original, not wrapper)
47. `app/coordination/event_emitters.py` (original, not wrapper)
48. `app/coordination/event_router.py` (may import originals)
49. `app/coordination/health_check_orchestrator.py` (original, not wrapper)

---

## Migration Priority

### High Priority (Core Training Loop)

🔴 These files are in the critical training path:

- `app/training/train.py`
- `app/training/promotion_controller.py`
- `app/coordination/training_coordinator.py`
- `app/coordination/data_pipeline_orchestrator.py`

### Medium Priority (Coordination Infrastructure)

🟡 These files support coordination:

- All `app/coordination/*_orchestrator.py` files
- All `app/coordination/*_coordinator.py` files
- `app/distributed/sync_coordinator.py`

### Low Priority (Scripts & Tests)

🟢 These can be updated later:

- Scripts in `scripts/`
- Test files
- Documentation

---

## Migration Checklist

### Before Removing Deprecated Modules

- [ ] Update all 49 files to use new import paths
- [ ] Run full test suite: `pytest tests/`
- [ ] Update documentation
- [ ] Update examples in docstrings
- [ ] Add linting rule to prevent deprecated imports
- [ ] Verify CI/CD passes

### Import Pattern Changes

#### Health Monitoring

```python
# OLD
from app.coordination.health_check_orchestrator import (
    HealthCheckOrchestrator,
    get_health_orchestrator,
)

# NEW
from app.coordination.cluster.health import (
    UnifiedHealthManager,
    get_health_manager,
)
```

#### Event Emitters

```python
# OLD
from app.coordination.event_emitters import (
    emit_training_completed,
    emit_model_promoted,
)

# NEW
from app.coordination.event_router import (
    emit_training_completed,
    emit_model_promoted,
)

# OR use unified publish:
from app.coordination.event_router import publish, DataEventType
await publish(DataEventType.TRAINING_COMPLETED, payload={...})
```

#### Cross-Process Events

```python
# OLD
from app.coordination.cross_process_events import (
    publish_event,
    poll_events,
    subscribe_process,
)

# NEW
from app.coordination.event_router import (
    publish,      # replaces publish_event
    get_router,   # for subscriptions
)

# Subscriptions:
router = get_router()
router.subscribe(DataEventType.MODEL_PROMOTED, my_handler)
```

---

## Automation Opportunities

### Search & Replace Patterns

```bash
# Find all health_check_orchestrator imports
git grep -l "from app.coordination.health_check_orchestrator import"

# Find all event_emitters imports
git grep -l "from app.coordination.event_emitters import"

# Find all cross_process_events imports
git grep -l "from app.coordination.cross_process_events import"
```

### Suggested Script

```python
#!/usr/bin/env python3
"""Update deprecated imports to new paths."""

import re
from pathlib import Path

REPLACEMENTS = {
    # Health
    r'from app\.coordination\.health_check_orchestrator import':
        'from app.coordination.cluster.health import',
    r'HealthCheckOrchestrator': 'UnifiedHealthManager',
    r'get_health_orchestrator': 'get_health_manager',

    # Events
    r'from app\.coordination\.event_emitters import':
        'from app.coordination.event_router import',
    r'from app\.coordination\.cross_process_events import':
        'from app.coordination.event_router import',
    r'publish_event': 'publish',
}

def update_file(path: Path):
    content = path.read_text()
    updated = content
    for old, new in REPLACEMENTS.items():
        updated = re.sub(old, new, updated)

    if updated != content:
        path.write_text(updated)
        print(f"Updated: {path}")

# Usage:
# for f in Path('app').rglob('*.py'):
#     update_file(f)
```

---

## Timeline

- **Now - Q1 2026**: Deprecation warnings guide migration
- **Q1 2026**: Update imports in phases (high → medium → low priority)
- **Q2 2026**: Remove deprecated wrappers (breaking change)

---

## Notes

- Wrappers emit `DeprecationWarning` at import time
- All functionality preserved - only import paths change
- Original modules remain for backward compatibility until Q2 2026
- Test suite should catch any breakage from updated imports

---

## Contact

For questions about migration:

- See `MODULE_CONSOLIDATION_STATUS.md` for detailed migration guides
- See `CONSOLIDATION_PROGRESS.md` for overall progress tracking
- See `CONSOLIDATION_SUMMARY.md` for executive summary
