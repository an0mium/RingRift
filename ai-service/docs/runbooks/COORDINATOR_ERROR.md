# Runbook: Coordinator Error

## Alert: CoordinatorError

**Severity:** Critical
**Component:** coordinators
**Team:** infrastructure

## Description

A coordinator (RecoveryManager, BandwidthManager, or SyncCoordinator) has entered an error state and remained there for more than 2 minutes.

## Impact

- Coordinator functionality is unavailable
- Dependent services may be affected:
  - **RecoveryManager:** Automatic node/job recovery disabled
  - **BandwidthManager:** Transfer bandwidth allocation unavailable
  - **SyncCoordinator:** Data synchronization paused

## Diagnosis

### 1. Check Coordinator Status

```bash
# Via admin API
curl -H "X-Admin-Key: $ADMIN_KEY" \
  http://localhost:8001/admin/health/coordinators

# Or via Python
python -c "
from app.coordination.coordinator_base import get_coordinator_registry
registry = get_coordinator_registry()
print(registry.get_health_summary())
"
```

### 2. Check Logs

```bash
# Find coordinator errors
grep -E "(RecoveryManager|BandwidthManager|SyncCoordinator).*ERROR" \
  /var/log/ringrift/ai-service.log | tail -50

# Check for stack traces
grep -A 20 "Traceback" /var/log/ringrift/ai-service.log | tail -100
```

### 3. Check System Resources

```bash
# Memory
free -h

# Disk (coordinators use SQLite)
df -h /tmp/ringrift_coordinator/
df -h data/coordination/

# Check if database is corrupted
sqlite3 /tmp/ringrift_coordinator/tasks.db "PRAGMA integrity_check;"
```

## Resolution

### Option 1: Restart the Service

If the error appears transient (network timeout, temporary resource exhaustion):

```bash
# Graceful restart
systemctl restart ringrift-ai-service

# Or if using Docker
docker-compose restart ai-service
```

### Option 2: Clear Coordinator State

If SQLite database appears corrupted:

```bash
# Stop service first
systemctl stop ringrift-ai-service

# Backup and remove state
mv /tmp/ringrift_coordinator/tasks.db /tmp/ringrift_coordinator/tasks.db.bak
mv data/coordination/resource_state.db data/coordination/resource_state.db.bak

# Restart
systemctl start ringrift-ai-service
```

### Option 3: Manual Coordinator Reset

If specific coordinator is stuck:

```python
from app.coordination.coordinator_base import get_coordinator_registry

registry = get_coordinator_registry()
coord = registry.get("SyncCoordinator")  # or other coordinator name

# Try to restart
await coord.stop()
await coord.start()
```

## Prevention

1. **Monitor disk space** - Coordinators use SQLite and need disk space
2. **Set resource limits** - Ensure adequate memory/CPU for coordinator tasks
3. **Regular health checks** - Use `/admin/health/coordinators` in monitoring

## Escalation

If the above steps don't resolve the issue:

1. Check for related infrastructure issues (network, storage)
2. Review recent deployments for breaking changes
3. Escalate to on-call infrastructure team

## Related Alerts

- CoordinatorStopped
- CoordinatorHighErrorRate
