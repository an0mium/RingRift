# Orchestrator Rollout Runbook

> **Audience:** On-call engineers and backend maintainers  
> **Scope:** Managing orchestrator posture, circuit breaker, and parity diagnostics for the shared turn orchestrator

> **Post-Phase 4 note:** `ORCHESTRATOR_ADAPTER_ENABLED` is hardcoded to `true` in `EnvSchema`; rollout-percentage and shadow-mode flags were removed. Treat this runbook as guidance for circuit‑breaker and parity diagnostics flows; percentage/shadow controls are historical only.

---

## 1. Quick Status Checklist

When investigating an orchestrator-related page or alert, check:

- `ringrift_orchestrator_circuit_breaker_state` (0 = closed, 1 = open)
- `ringrift_orchestrator_error_rate` (0.0–1.0 fraction)
- `ringrift_orchestrator_sessions_total{engine,selection_reason}`
- `ringrift_rules_parity_mismatches_total{suite="runtime_python_mode",mismatch_type=...}` (only when running python‑authoritative diagnostics)
- Admin API snapshot:
  - `GET /api/admin/orchestrator/status` (requires admin auth)

**Target SLO-style thresholds (steady state):**

- Orchestrator error rate: `ringrift_orchestrator_error_rate < 0.02` (2%) over a 5–10 minute window.
- Parity mismatches: near‑zero during python‑authoritative diagnostics (see `RULES_PARITY.md`).
- Circuit breaker closed: `ringrift_orchestrator_circuit_breaker_state == 0` during normal operation.

Use `/metrics` plus the admin API to decide whether to:

- Investigate parity posture or circuit-breaker state (rollout percentage is fixed at 100%).
- Trip or reset the circuit breaker.

---

## 2. Reading the Admin Status Endpoint

**Endpoint:** `GET /api/admin/orchestrator/status`  
**Auth:** Bearer token, user with `role = "admin"`

Response shape (summary):

````json
{
  "success": true,
	  "data": {
	    "config": {
	      "adapterEnabled": true,
	      "allowlistUsers": ["staff-1"],
	      "denylistUsers": [],
	      "circuitBreaker": {
	        "enabled": true,
	        "errorThresholdPercent": 5,
	        "errorWindowSeconds": 300
	      }
	    },
	    "circuitBreaker": {
	      "isOpen": false,
	      "errorCount": 3,
	      "requestCount": 200,
	      "windowStart": "2025-11-28T12:34:56.000Z",
	      "errorRatePercent": 1.5
	    }
	  }
	}
	```

Key questions:

- **Is the circuit breaker open?** (`circuitBreaker.isOpen === true`)
- **Is the error rate elevated?** (`errorRatePercent > 2–5%`)
- **Are runtime parity mismatches non-trivial?** (only applicable during python‑authoritative diagnostics)

If any of these are true, pause python-authoritative diagnostics and follow the incident steps below.

---

## 3. Normal Operations

### 3.1 Rollout Percentage (Removed)

`ORCHESTRATOR_ROLLOUT_PERCENTAGE` no longer affects routing; the adapter is hardcoded ON. Do not set this flag in current environments. Historical phase-by-phase rollout steps are archived in `docs/archive/ORCHESTRATOR_ROLLOUT_RUNBOOK_PHASES.md`.

### 3.2 Parity Diagnostics (Post‑Phase 4)

Legacy shadow mode (`ORCHESTRATOR_SHADOW_MODE_ENABLED` / `RINGRIFT_RULES_MODE=shadow`) was removed in Phase 4.
To collect runtime TS↔Python parity metrics, run explicit diagnostic jobs or staging deployments with:

```bash
RINGRIFT_RULES_MODE=python
```

In this posture the backend validates moves via Python, applies them through TS, and records parity mismatches. Keep production at `RINGRIFT_RULES_MODE=ts` unless debugging a confirmed rules regression.

---

## 4. Incident Scenarios & Actions

### 4.1 Alert: OrchestratorCircuitBreakerOpen

**Signal:**
Prometheus alert `OrchestratorCircuitBreakerOpen` fired.
Metric: `ringrift_orchestrator_circuit_breaker_state == 1`.

**Impact:**
Circuit breaker state is open, indicating elevated orchestrator errors. Routing is unchanged (orchestrator remains the canonical path), but this is a high‑severity diagnostic signal.

**Actions:**

1. Confirm via admin API:
   - `GET /api/admin/orchestrator/status` → `circuitBreaker.isOpen === true`.
2. Inspect:
   - `ringrift_orchestrator_error_rate`
   - Application logs around orchestrator adapter (`GameEngine.processMoveViaAdapter`) for repeated errors.
3. Short-term:
   - Keep production on `RINGRIFT_RULES_MODE=ts` and avoid python‑authoritative diagnostics unless explicitly debugging parity.
   - Focus on isolating the failing move types / phases via logs and regression tests.
4. Remediation:
   - Identify and fix underlying orchestrator issue (e.g. specific move types, board types, or phases).
   - Once fixed and deployed, manually reset the breaker via an admin task (if added) or by restarting the API pods.
5. Post-fix:
   - Monitor `ringrift_orchestrator_error_rate` returning to near zero.
   - Confirm breaker state closes:
     - `ringrift_orchestrator_circuit_breaker_state == 0`.

### 4.2 Alert: OrchestratorErrorRateWarning

**Signal:**
`ringrift_orchestrator_error_rate > 0.02` for > 2m.

**Actions:**

1. Keep production on `RINGRIFT_RULES_MODE=ts`; pause python‑authoritative diagnostics.
2. Check logs for adapter or orchestrator-level errors.
3. If error rate approaches threshold configured in `ORCHESTRATOR_ERROR_THRESHOLD_PERCENT`:
   - Treat as a release regression and prepare a deployment rollback to a known‑good build.
   - Use circuit‑breaker state and parity diagnostics to confirm recovery.

### 4.3 Alert: OrchestratorShadowMismatches (Deprecated)

**Signal:**
Deprecated; alert removed and metric no longer emitted.

**Actions:**

1. No action for the legacy alert.
2. If you see parity mismatches while running `RINGRIFT_RULES_MODE=python`, follow `RULES_PARITY.md`.

---

## 5. Emergency Rollback Procedures

### 5.1 Deployment Rollback (Preferred)

There is no runtime kill switch or rollout‑percentage lever. If orchestrator
behaviour is clearly wrong (incorrect winners, crashes), roll back to the last
known‑good build using the standard deployment rollback runbooks.

### 5.2 Mitigation Checklist

Use when error rates are elevated but not catastrophic:

- Keep production on `RINGRIFT_RULES_MODE=ts`.
- Pause any python‑authoritative diagnostics.
- Capture logs and parity artifacts for post‑incident analysis.

**Historical note:** `ORCHESTRATOR_ROLLOUT_PERCENTAGE` and shadow-only posture were removed in Phase 4; there is no supported “0% rollout” or `shadow` mode toggle.
Allow/deny lists and circuit-breaker state are **diagnostic signals only**; they do not change routing.
For runtime parity diagnostics, use `RINGRIFT_RULES_MODE=python` in explicit staging/diagnostic runs (see §3.2).

---

## 6. Verification After Changes

After any deployment or configuration change:

1. Confirm configuration:
   - `GET /api/admin/orchestrator/status`
   - Environment variables in deployment (`ORCHESTRATOR_*`).
2. Confirm metrics:
  - `/metrics`:
     - `ringrift_orchestrator_rollout_percentage` (fixed at `100`)
     - `ringrift_orchestrator_circuit_breaker_state`
     - `ringrift_orchestrator_error_rate`
3. Run a small set of smoke tests:
   - Create a few games with AI/human players.
   - Exercise capture, lines, territory, and victory conditions.
   - Verify no spike in 5xx or orchestrator errors.
   - Optionally run an orchestrator soak to re-check core invariants under the TS engine:
     - `npm run soak:orchestrator:smoke` (single short backend game on square8, fails on invariant violation).
     - `npm run soak:orchestrator:short` (deterministic short backend soak on square8 with multiple games, `--failOnViolation=true`; this is the concrete CI implementation of `SLO-CI-ORCH-SHORT-SOAK`).
     - For deeper offline runs, see `npm run soak:orchestrator:nightly` and [`docs/testing/STRICT_INVARIANT_SOAKS.md`](../testing/STRICT_INVARIANT_SOAKS.md) §2.3–2.4 for details on orchestrator soak profiles and related SLOs.
     - For extended **vector‑seeded** soaks (chain capture, deep chains, forced elimination, territory/line endgame, hex edge cases, and near‑victory territory), use:
       ```bash
       npx ts-node scripts/run-orchestrator-soak.ts \
         --profile=extended-vectors-short \
         --gamesPerVector=1 \
         --maxTurns=200 \
         --vectorBundle=tests/fixtures/contract-vectors/v2/chain_capture_long_tail.vectors.json \
         --vectorBundle=tests/fixtures/contract-vectors/v2/chain_capture_extended.vectors.json \
         --vectorBundle=tests/fixtures/contract-vectors/v2/forced_elimination.vectors.json \
         --vectorBundle=tests/fixtures/contract-vectors/v2/territory_line_endgame.vectors.json \
         --vectorBundle=tests/fixtures/contract-vectors/v2/hex_edge_cases.vectors.json \
         --vectorBundle=tests/fixtures/contract-vectors/v2/near_victory_territory.vectors.json \
         --outputPath=results/orchestrator_soak_extended_vectors.json
       ```
   - Optionally run a **backend HTTP load smoke** to exercise `/api` and WebSocket paths under orchestrator‑ON:
     ```bash
     TS_NODE_PROJECT=tsconfig.server.json npm run load:orchestrator:smoke
     ```
     This script:
     - Registers a small number of throwaway users via `/api/auth/register`.
     - Creates short games via `/api/games` and fetches game lists/details.
    - Samples `/metrics` for orchestrator metrics.
       Use it as a quick check that backend HTTP + orchestrator wiring behave sensibly at low concurrency before production deploys or parity diagnostics.
   - Optionally run a **metrics & observability smoke** to confirm `/metrics` is exposed and key orchestrator gauges are present:
     ```bash
     npm run test:e2e -- tests/e2e/metrics.e2e.spec.ts
     ```
     This Playwright spec:
     - Waits for `/ready` via `E2E_API_BASE_URL` (or `http://localhost:3000`).
     - Scrapes `/metrics` and asserts Prometheus output.
     - Verifies the presence of orchestrator metrics such as:
       - `ringrift_orchestrator_error_rate`
       - `ringrift_orchestrator_rollout_percentage`
       - `ringrift_orchestrator_circuit_breaker_state`
         Use it as a fast guardrail that observability wiring is intact before and after orchestrator-related deploys.

---

## 7. References

- **Design & Feature Flags (archived):**
  `docs/archive/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md`

- **Shared Engine & Adapters:**
  - `src/shared/engine/orchestration/turnOrchestrator.ts`
  - `src/server/game/turn/TurnEngineAdapter.ts`
  - `src/client/sandbox/SandboxOrchestratorAdapter.ts`

- **Rollout Services:**
  - `src/server/services/OrchestratorRolloutService.ts`

- **Metrics:**
  - `src/server/services/MetricsService.ts`
  - `monitoring/prometheus/alerts.yml`

---

## 8. Historical rollout playbook (archived)

The phase-by-phase rollout and rollback commands are archived in `docs/archive/ORCHESTRATOR_ROLLOUT_RUNBOOK_PHASES.md`. Use Sections 1-7 for current operations.
````
