# Runbook: Cluster Health Critical

## Alert: ClusterHealthCritical

**Severity:** Critical
**Component:** sync
**Team:** infrastructure

## Description

The sync cluster health score has dropped below 50%. This indicates multiple hosts are unavailable or significantly stale.

## Impact

- Training data may be significantly delayed or lost
- Self-play games from affected hosts are not being synchronized
- Model training may be using incomplete or outdated data

## Diagnosis

### 1. Check Cluster Health Details

```bash
# Via admin API
curl -H "X-Admin-Key: $ADMIN_KEY" \
  http://localhost:8001/admin/health/full | jq '.coordinators'

# Check sync-specific metrics
curl http://localhost:8001/metrics | grep -E "ringrift_sync_"
```

### 2. Identify Affected Hosts

```bash
# Check Prometheus metrics
curl -s http://prometheus:9090/api/v1/query?query=ringrift_sync_hosts_critical

# Via Python
python -c "
from app.coordination.sync_coordinator import SyncCoordinator
sync = SyncCoordinator.get_instance()
stats = sync.get_stats()
print('Healthy:', stats.get('hosts_healthy', 0))
print('Stale:', stats.get('hosts_stale', 0))
print('Critical:', stats.get('hosts_critical', 0))
"
```

### 3. Check Host Connectivity

```bash
# Test SSH connectivity to Vast.ai hosts
for host in $(cat /etc/ringrift/hosts.txt); do
  echo -n "$host: "
  ssh -o ConnectTimeout=5 $host "echo OK" 2>/dev/null || echo "FAILED"
done

# Check P2P orchestrator status
curl http://localhost:5001/health
```

### 4. Review Recent Changes

- Check for Vast.ai instance terminations
- Review cloud provider status pages
- Check for network outages

## Resolution

### Option 1: Recover Stale Hosts

```bash
# Trigger manual sync for specific host
python -c "
from app.coordination.sync_coordinator import SyncCoordinator
sync = SyncCoordinator.get_instance()
sync.trigger_host_sync('host-id-here')
"
```

### Option 2: Mark Hosts for Recovery

```bash
# Via P2P admin API
curl -X POST http://localhost:5001/admin/recover-host \
  -H "X-Admin-Key: $ADMIN_KEY" \
  -d '{"host_id": "host-id-here"}'
```

### Option 3: Scale Up Healthy Hosts

If hosts are permanently unavailable, increase capacity on healthy hosts:

```bash
# Increase selfplay rate on healthy hosts
curl -X POST http://localhost:8001/admin/rate-adjust \
  -H "X-Admin-Key: $ADMIN_KEY" \
  -d '{"action": "scale_up", "reason": "host_recovery"}'
```

### Option 4: Retire Unrecoverable Hosts

If a host cannot be recovered:

```bash
# Retire host from cluster
curl -X POST http://localhost:5001/admin/retire-host \
  -H "X-Admin-Key: $ADMIN_KEY" \
  -d '{"host_id": "host-id-here", "reason": "unrecoverable"}'
```

## Prevention

1. **Monitor host health** - Set up alerts before hosts become critical
2. **Redundancy** - Maintain N+1 capacity for selfplay hosts
3. **Regular sync checks** - Run periodic sync validation
4. **Auto-recovery** - Ensure RecoveryManager is running

## Escalation

If cluster health remains critical after 15 minutes:

1. Page on-call infrastructure team
2. Consider pausing training to prevent data quality issues
3. Review Vast.ai/cloud provider status

## Related Alerts

- ClusterHealthDegraded (warning at 80%)
- CriticalSyncHosts
- HighUnsyncedGames
