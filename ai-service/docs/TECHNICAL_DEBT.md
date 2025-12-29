# Technical Debt Registry

Last updated: Dec 28, 2025

## Summary

| Category             | Items     | Priority | Deadline                        |
| -------------------- | --------- | -------- | ------------------------------- |
| Deprecation Warnings | 2         | Medium   | Q2 2026                         |
| Data Generation      | 5 configs | High     | Ongoing                         |
| Third-Party Updates  | 1         | Low      | When cryptography 48.0 releases |

---

## 1. Deprecation Warnings (Q2 2026)

### 1.1 event_emitters Migration

**Status**: Deprecation warning active
**Deadline**: Q2 2026
**Scope**: 41 files, ~70 import statements

**Current Location**: `app/coordination/event_emitters.py`

**Recommended Replacement**: Use `app.coordination.event_router` directly:

```python
# Current (deprecated)
from app.coordination.event_emitters import emit_training_complete
await emit_training_complete(config_key="hex8_2p", model_path="...")

# New (recommended)
from app.coordination.event_router import get_event_bus, DataEvent, DataEventType

bus = get_event_bus()
await bus.publish(DataEvent(
    event_type=DataEventType.TRAINING_COMPLETED,
    payload={"config_key": "hex8_2p", "model_path": "..."},
    source="my_component",
))
```

**Files Affected** (top consumers):

- `app/coordination/pipeline_actions.py` (7 imports)
- `app/training/curriculum_feedback.py` (3 imports)
- `app/coordination/task_decorators.py` (5 imports)
- `app/coordination/training_coordinator.py` (multiple)

**Migration Strategy**:

1. Create centralized emit helper functions in event_router.py
2. Update imports file-by-file
3. Archive event_emitters.py after all callers migrated

### 1.2 singleton_mixin Location

**Status**: FIXED (Dec 28, 2025)
**Details**: 2 files updated from `app.core.singleton_mixin` to `app.coordination.singleton_mixin`

---

## 2. Data Generation Gaps

### Underserved Configurations

Games needed for robust model training:

| Config      | Current Games | Target | Priority |
| ----------- | ------------- | ------ | -------- |
| square19_4p | 0             | 1,000+ | CRITICAL |
| hex8_4p     | 46            | 1,000+ | HIGH     |
| square19_2p | 78            | 1,000+ | HIGH     |
| hex8_3p     | 149           | 1,000+ | MEDIUM   |
| square19_3p | 150           | 1,000+ | MEDIUM   |

**Action**: Run `scripts/trigger_priority_selfplay.py` when cluster is active.

---

## 3. Third-Party Dependencies

### 3.1 Paramiko TripleDES Deprecation

**Warning**:

```
CryptographyDeprecationWarning: TripleDES has been moved to
cryptography.hazmat.decrepit.ciphers.algorithms.TripleDES
```

**Impact**: Will break when cryptography 48.0 releases
**Fix**: Update paramiko when new version with fix is available
**Priority**: Low (no immediate deadline)

---

## 4. Code Consolidation Opportunities (Low Priority)

### 4.1 Pipeline Mixin Hierarchy

Four mixin files could potentially be merged:

- `pipeline_stage_mixin.py` (419 LOC)
- `pipeline_metrics_mixin.py` (450 LOC)
- `pipeline_trigger_mixin.py` (489 LOC)
- `pipeline_event_handler_mixin.py` (1,273 LOC)

**Estimated Savings**: ~450 LOC
**Risk**: Low (internal implementation)

### 4.2 Sync Mixin Hierarchy

Five sync mixin files share common patterns:

- `sync_mixin_base.py`
- `sync_push_mixin.py` (~400 LOC)
- `sync_pull_mixin.py` (~300 LOC)
- `sync_event_mixin.py` (658 LOC)
- `sync_ephemeral_mixin.py` (888 LOC)

**Estimated Savings**: ~330 LOC
**Risk**: Medium (syncs are performance-critical)

---

## 5. Documentation Gaps (Future Work)

| Document                       | Effort  | Priority |
| ------------------------------ | ------- | -------- |
| P2P Orchestrator Architecture  | 8 hours | Medium   |
| Complete Pipeline Flow Runbook | 4 hours | Medium   |
| Resource Optimizer Runbook     | 3 hours | Low      |

---

## Verification Commands

```bash
# Check deprecation warnings
python -c "import app.coordination" 2>&1 | grep -i deprecat

# Check singleton imports
grep -r "from app.core.singleton_mixin" app/

# Check event_emitters usage
grep -r "from app.coordination.event_emitters" app/ | wc -l

# Check database game counts
python scripts/trigger_priority_selfplay.py --check-only
```
