# SLO Gate CI/CD Runbook

> **Created:** 2025-12-23  
> **Status:** Active  
> **Purpose:** Operational guide for SLO gate validation in CI/CD pipelines

---

## Overview

The SLO Gate system provides automated Service Level Objective validation as a quality gate for deployments. It analyzes load test results against defined thresholds and produces pass/fail decisions for staging promotion and production readiness.

### Key Components

| Component          | Location                                                                                   | Purpose                           |
| ------------------ | ------------------------------------------------------------------------------------------ | --------------------------------- |
| SLO Gate Script    | [`scripts/slo-gate-check.ts`](../../scripts/slo-gate-check.ts)                             | TypeScript CLI for SLO validation |
| SLO Gate Workflow  | [`.github/workflows/slo-gate.yml`](../../.github/workflows/slo-gate.yml)                   | GitHub Actions workflow           |
| SLO Definitions    | [`tests/load/configs/slo-definitions.json`](../../tests/load/configs/slo-definitions.json) | Consolidated SLO targets          |
| Threshold Config   | [`tests/load/config/thresholds.json`](../../tests/load/config/thresholds.json)             | k6 thresholds per environment     |
| Verify SLOs Script | [`tests/load/scripts/verify-slos.js`](../../tests/load/scripts/verify-slos.js)             | JavaScript SLO verification       |

---

## Quick Start

### Run SLO Gate Locally

```bash
# With existing load test results
npm run slo:gate -- --results-file tests/load/results/baseline.json

# Against staging thresholds
npm run slo:gate:staging -- --results-file tests/load/results/baseline.json

# Against production thresholds (fails on breach)
npm run slo:gate:production -- --results-file tests/load/results/baseline.json --fail-on-breach
```

### Run Full SLO Verification Pipeline

```bash
# Run load test + SLO verification (staging)
npm run slo:check

# Run against production thresholds
npm run slo:check:production
```

### Verify Specific Results

```bash
# Using the JavaScript verifier
npm run slo:verify tests/load/results/baseline.json console --env staging

# Using the TypeScript gate script (with JSON output)
npx ts-node scripts/slo-gate-check.ts \
  --results-file tests/load/results/baseline.json \
  --env production \
  --format json \
  --output-file slo-report.json
```

---

## SLO Gate Workflow

### Trigger Methods

1. **Manual Dispatch** - On-demand validation via GitHub UI
2. **Workflow Call** - Reusable workflow from deployment pipelines
3. **Scheduled** - Nightly production readiness assessment (3:00 AM UTC)

### Gate Types

| Gate Type              | Purpose                            | When to Use                  |
| ---------------------- | ---------------------------------- | ---------------------------- |
| `staging-promotion`    | Validate staging before production | After staging deployment     |
| `production-readiness` | Full production SLO check          | Before production deployment |
| `smoke-test`           | Quick validation                   | Local testing, quick checks  |

### Gate Statuses

| Status        | Meaning                             | CI Behavior              |
| ------------- | ----------------------------------- | ------------------------ |
| `APPROVED`    | All critical + high SLOs passed     | Continues pipeline       |
| `CONDITIONAL` | Critical passed, some high breached | Requires manual approval |
| `BLOCKED`     | One or more critical SLOs breached  | Fails pipeline           |

---

## SLO Thresholds

### Critical SLOs (Zero Tolerance)

These must pass for gate approval:

| SLO                  | Staging Target | Production Target |
| -------------------- | -------------- | ----------------- |
| Service Availability | ≥99.9%         | ≥99.9%            |
| Error Rate           | ≤1.0%          | ≤0.5%             |
| True Error Rate      | ≤0.5%          | ≤0.2%             |
| Contract Failures    | 0              | 0                 |
| Lifecycle Mismatches | 0              | 0                 |

### High Priority SLOs

Important for user experience:

| SLO                          | Staging Target | Production Target |
| ---------------------------- | -------------- | ----------------- |
| API Latency (p95)            | <800ms         | <500ms            |
| Move Latency E2E (p95)       | <300ms         | <200ms            |
| AI Response Time (p95)       | <1500ms        | <1000ms           |
| WebSocket Connection Success | ≥99.0%         | ≥99.5%            |
| Move Stall Rate              | ≤0.5%          | ≤0.2%             |
| Concurrent Games             | ≥20            | ≥100              |
| Concurrent Players           | ≥60            | ≥300              |

### Medium Priority SLOs

Performance optimization targets:

| SLO                    | Target  |
| ---------------------- | ------- |
| API Latency (p99)      | <2000ms |
| AI Response Time (p99) | <2000ms |
| AI Fallback Rate       | ≤1.0%   |

---

## Workflow Configuration

### Triggering from Deployment Pipeline

```yaml
# In your deployment workflow
jobs:
  deploy-staging:
    # ... deployment steps ...

  slo-validation:
    needs: deploy-staging
    uses: ./.github/workflows/slo-gate.yml
    with:
      environment: staging
      gate_type: staging-promotion
      target_url: https://staging.ringrift.ai
```

### Manual Trigger Parameters

| Parameter            | Options                                             | Default           | Description               |
| -------------------- | --------------------------------------------------- | ----------------- | ------------------------- |
| `environment`        | staging, production                                 | staging           | SLO threshold set         |
| `gate_type`          | staging-promotion, production-readiness, smoke-test | staging-promotion | Validation type           |
| `target_url`         | URL string                                          | (local)           | Target for load tests     |
| `load_test_scenario` | baseline, target-scale, websocket-gameplay, skip    | baseline          | Which load test to run    |
| `results_file`       | Path                                                | (auto)            | Use existing results      |
| `fail_on_breach`     | true, false                                         | true              | Exit with error on breach |

---

## Interpreting Results

### Console Output

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       RingRift SLO Gate Check Report                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Gate Type:     staging-promotion   Environment: staging                      ║
║  Status:        ✅ APPROVED                                                    ║
║  SLOs Passed:   13/15                                                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  [CRITICAL]                                                                   ║
║    ✅ Service Availability              99.95% /      99.9%                   ║
║    ✅ Error Rate                          0.1% /       1.0%                   ║
║    ✅ Contract Failures                      0 /          0                   ║
...
```

### JSON Report Structure

```json
{
  "timestamp": "2025-12-23T23:00:00.000Z",
  "environment": "staging",
  "gateType": "staging-promotion",
  "resultsFile": "tests/load/results/baseline.json",
  "overall": {
    "passed": true,
    "passedCount": 13,
    "totalCount": 15,
    "criticalBreaches": 0,
    "highBreaches": 1,
    "mediumBreaches": 1
  },
  "decision": {
    "gateStatus": "APPROVED",
    "reason": "All critical and high-priority SLOs passed",
    "recommendations": ["1 medium-priority SLO(s) should be reviewed"]
  },
  "slos": {
    "availability": {
      "name": "Service Availability",
      "target": 99.9,
      "actual": 99.95,
      "unit": "percent",
      "passed": true,
      "priority": "critical",
      "margin": 0.05
    }
    // ... other SLOs
  }
}
```

---

## Troubleshooting

### Common Issues

#### "No results file found"

**Cause:** Load test didn't produce output or path is incorrect.

**Fix:**

```bash
# Check for results files
ls -la tests/load/results/

# Run load test with explicit output
k6 run --out json=results.json tests/load/scenarios/websocket-gameplay.js
```

#### "true_errors_total not reported"

**Cause:** Load test scenario doesn't emit error classification counters.

**Fix:** This is informational. The gate will use `http_req_failed` as fallback. Update load test to emit true error counters if needed.

#### Gate Status "BLOCKED" unexpectedly

**Cause:** Critical SLO breach, often contract failures or error rate.

**Fix:**

```bash
# Get detailed report
npm run slo:gate -- --results-file results.json --format json --output-file report.json

# Check specific breaches
jq '.slos | to_entries[] | select(.value.passed == false)' report.json
```

### Validating Load Test Output

```bash
# Check k6 output format
head -5 tests/load/results/baseline.json

# Count metrics
grep -c '"type":"Point"' tests/load/results/baseline.json

# Check for summary block
grep -c '"metrics":\s*{' tests/load/results/baseline.json
```

---

## Integration with Existing Tools

### Using with verify-slos.js

The new `slo-gate-check.ts` complements the existing `verify-slos.js`:

```bash
# Run both for comprehensive output
node tests/load/scripts/verify-slos.js results.json console --env staging
npx ts-node scripts/slo-gate-check.ts --results-file results.json --env staging
```

### Generating Dashboards

```bash
# Generate HTML dashboard from report
npm run slo:dashboard tests/load/results/baseline_slo_report.json
```

---

## CI/CD Integration Examples

### GitHub Actions (Direct)

```yaml
- name: Run SLO Gate
  run: |
    npm run slo:gate:production -- \
      --results-file ${{ steps.load-test.outputs.file }} \
      --output-file slo-report.json
```

### Using Workflow Outputs

```yaml
jobs:
  slo-gate:
    uses: ./.github/workflows/slo-gate.yml
    with:
      environment: production

  deploy-prod:
    needs: slo-gate
    if: needs.slo-gate.outputs.gate_status == 'APPROVED'
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying to production..."
```

---

## Maintenance

### Updating SLO Thresholds

1. Update [`tests/load/configs/slo-definitions.json`](../../tests/load/configs/slo-definitions.json)
2. Update [`tests/load/config/thresholds.json`](../../tests/load/config/thresholds.json)
3. Update threshold values in [`scripts/slo-gate-check.ts`](../../scripts/slo-gate-check.ts) `getSLODefinitions()`
4. Update this runbook's threshold tables

### Adding New SLOs

1. Define in `slo-definitions.json` and `thresholds.json`
2. Add metric extraction in `extractMetrics()` function
3. Add evaluation in `evaluateSLO()` function
4. Document in this runbook

---

## Related Documentation

- [`docs/production/PRODUCTION_VALIDATION_GATE.md`](../production/PRODUCTION_VALIDATION_GATE.md) - Full validation checklist
- [`docs/planning/SLO_THRESHOLD_ALIGNMENT_AUDIT.md`](../planning/SLO_THRESHOLD_ALIGNMENT_AUDIT.md) - SLO threshold analysis
- [`docs/operations/SLO_VERIFICATION.md`](SLO_VERIFICATION.md) - Verification procedures
- [`tests/load/README.md`](../../tests/load/README.md) - Load test documentation
- [`.github/workflows/README.md`](../../.github/workflows/README.md) - CI/CD workflow docs

---

## Revision History

| Version | Date       | Changes          |
| ------- | ---------- | ---------------- |
| 1.0     | 2025-12-23 | Initial creation |
