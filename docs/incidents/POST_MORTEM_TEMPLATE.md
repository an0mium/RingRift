# Post-Mortem Template

Use this template for all P1 and P2 incidents. Post-mortems should be completed within 48 hours of incident resolution.

---

## Post-Mortem: [Incident Title]

### Metadata

| Field | Value |
|-------|-------|
| **Date** | YYYY-MM-DD |
| **Duration** | HH:MM (start to resolution) |
| **Severity** | P1 Critical / P2 High / P3 Medium |
| **Author** | [Name] |
| **Reviewers** | [Names] |
| **Status** | Draft / In Review / Complete |

### Summary

*A brief 2-3 sentence summary of what happened, when, and the impact.*

Example:
> On November 25, 2024 from 14:32-15:17 UTC, the RingRift database became unresponsive due to connection pool exhaustion. This resulted in a 45-minute period where users could not log in, create games, or make moves. Approximately 150 active users were affected.

---

## Impact

### User Impact

*Describe what users experienced during the incident.*

| Metric | Value |
|--------|-------|
| Duration | XX minutes |
| Users Affected | ~XXX users |
| Error Rate Peak | XX% |
| Requests Failed | ~XXX |
| Games Interrupted | XX |

### Business Impact

*Describe any business consequences (revenue, reputation, SLA breaches, etc.)*

---

## Timeline

*Detailed timeline of the incident. All times in UTC.*

| Time (UTC) | Event |
|------------|-------|
| HH:MM | First alert fired (AlertName) |
| HH:MM | On-call acknowledged alert |
| HH:MM | Initial investigation started |
| HH:MM | Root cause identified |
| HH:MM | Mitigation started |
| HH:MM | Service restored |
| HH:MM | All-clear declared |
| HH:MM | Monitoring confirmed stable |

### Detection

- **Alert that fired:** [Alert name]
- **Time to detection:** XX minutes
- **Detected by:** Monitoring / User report / Engineer

---

## Root Cause Analysis

### What Happened

*Detailed technical explanation of what went wrong.*

Example:
> The database connection pool was configured with a maximum of 20 connections. A gradual increase in traffic over the week, combined with a slow query introduced in release v1.2.3, caused connections to be held longer than usual. By 14:32, all pool connections were exhausted, and new requests began failing.

### Contributing Factors

*List all factors that contributed to the incident.*

1. [Factor 1 - e.g., Slow query introduced in recent release]
2. [Factor 2 - e.g., Connection pool sized for lower traffic]
3. [Factor 3 - e.g., No alerting on connection pool usage]

### Why Was This Not Caught Earlier?

*Explain gaps in prevention, detection, or response.*

- [Gap 1 - e.g., No load testing for new query]
- [Gap 2 - e.g., Missing metric for pool utilization]
- [Gap 3 - e.g., Staging environment doesn't reflect production load]

---

## Resolution

### How Was It Fixed?

*Describe the immediate fix that resolved the incident.*

Example:
> The immediate fix was to:
> 1. Kill long-running database queries
> 2. Restart the application to reset connection pool
> 3. Manually verify service recovery

### Rollback or Fix?

- [ ] Rolled back to previous version
- [ ] Applied hotfix
- [ ] Configuration change
- [ ] Infrastructure change
- [ ] Other: [describe]

---

## Lessons Learned

### What Went Well

*Highlight things that worked well during incident response.*

- [e.g., Alert fired promptly when issue started]
- [e.g., Runbook was accurate and helpful]
- [e.g., Team collaboration was effective]
- [e.g., Communication was timely]

### What Went Poorly

*Identify areas that need improvement.*

- [e.g., Took too long to identify root cause]
- [e.g., Runbook was missing steps for this scenario]
- [e.g., Had to escalate due to missing access]
- [e.g., Status page wasn't updated promptly]

### Where We Got Lucky

*Identify areas where luck prevented worse outcomes.*

- [e.g., Low traffic time reduced user impact]
- [e.g., Engineer with relevant knowledge was available]
- [e.g., Similar incident recently so investigation was faster]

---

## Action Items

*Concrete, trackable actions to prevent recurrence.*

| # | Priority | Action | Owner | Due Date | Status |
|---|----------|--------|-------|----------|--------|
| 1 | High | [Action item] | @name | YYYY-MM-DD | Open |
| 2 | High | [Action item] | @name | YYYY-MM-DD | Open |
| 3 | Medium | [Action item] | @name | YYYY-MM-DD | Open |
| 4 | Low | [Action item] | @name | YYYY-MM-DD | Open |

### Action Item Categories

- **Prevention:** Actions to prevent this type of incident
- **Detection:** Actions to detect issues earlier
- **Response:** Actions to improve incident response
- **Process:** Actions to improve processes/procedures

---

## Supporting Information

### Related Alerts

| Alert | Time | Link |
|-------|------|------|
| [AlertName] | HH:MM | [Link to alert] |

### Related Logs

```
[Include relevant log snippets]
```

### Related Metrics

*Include screenshots or links to relevant dashboards/graphs.*

### Related PRs/Commits

| Type | Link | Description |
|------|------|-------------|
| Cause | [PR link] | PR that introduced the issue |
| Fix | [PR link] | PR that fixed the issue |

### Related Incidents

| Date | Incident | Similarity |
|------|----------|------------|
| YYYY-MM-DD | [Title] | [How related] |

---

## Appendix

### Commands Used

*Document useful commands run during the incident for future reference.*

```bash
# Example: Commands that helped diagnose
docker compose logs --tail 500 app | grep ERROR

# Example: Commands that helped mitigate
docker exec ringrift-postgres-1 psql -U ringrift -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '30 seconds';"
```

### Runbook Feedback

*Did the runbook help? What was missing?*

- [ ] Runbook was useful
- [ ] Runbook was missing information (specify below)
- [ ] Runbook needs update
- [ ] No runbook existed

**Suggested runbook changes:**
- [Change 1]
- [Change 2]

---

## Sign-Off

| Role | Name | Date |
|------|------|------|
| Author | [Name] | YYYY-MM-DD |
| Team Lead Review | [Name] | YYYY-MM-DD |
| Post-Mortem Review Meeting | [Attendees] | YYYY-MM-DD |

---

## Post-Mortem Checklist

Before closing the post-mortem:

- [ ] Summary is clear and accurate
- [ ] Timeline is complete
- [ ] Root cause is identified
- [ ] All contributing factors listed
- [ ] Action items are specific and assigned
- [ ] Action items have due dates
- [ ] Runbook feedback captured
- [ ] Post-mortem reviewed with team
- [ ] Action items tracked in issue tracker
- [ ] Post-mortem stored in incident history

---

## Quick Reference: Post-Mortem Principles

### Blameless Culture

- Focus on **systems and processes**, not individuals
- Ask "how did the system allow this?" not "who caused this?"
- Assume good intentions from all involved
- Encourage transparency and learning

### Good Action Items

**Good:** "Add connection pool utilization metric to dashboard"  
**Bad:** "Be more careful about database connections"

**Good:** "Update runbook with steps for connection pool exhaustion"  
**Bad:** "Improve documentation"

**Good:** "Add pre-deploy load test for queries touching users table"  
**Bad:** "More testing"

### When is Post-Mortem Required?

| Severity | Post-Mortem Required | Timeline |
|----------|---------------------|----------|
| P1 Critical | Always | Within 48 hours |
| P2 High | Always | Within 72 hours |
| P3 Medium | If learnings valuable | Within 1 week |
| P4 Low | Rarely | Optional |

---

*Template version: 1.0*
*Last updated: 2025-11-25*