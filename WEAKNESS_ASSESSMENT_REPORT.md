# RingRift Weakness Assessment & Hardest Problems Report

**Last Updated:** 2025-12-01 (Post-P18.5 Remediation)
**Status:** Active

This document tracks the project's **single weakest aspect** and **single hardest outstanding problem** over time. It serves as a high-level risk register to focus architectural and remediation efforts.

> **Post-P18.5 Note (2025-12-01):** The "hardest problem" from the initial PASS18 assessment (Deep Multi-Engine Parity) has been **substantially resolved** through P18.1-5 remediation work:
>
> - 43 extended contract vectors with 0 mismatches
> - swap_sides (pie rule) parity verified across all layers
> - Orchestrator at Phase 4 (100% rollout, staging complete, production preview ready)
> - See Section 4 for detailed remediation summary.

---

## 1. Current Assessment (Pass 19B)

> **Progress Note (Pass 19B):** Near-victory fixture API created enabling E2E game completion tests. 6 tests enabled (3 E2E, 3 scenario). V2 test bug fixed. All skip rationales documented. Focus: E2E infrastructure for complex multiplayer scenarios.

### 1.1 Weakest Aspect: E2E Test Coverage for Game Completion Scenarios

**Score: 3.5/5.0** (Lowest Coverage Area, refined from Frontend UX)

Progress made:

- ✅ **Near-Victory Fixture API:** `near_victory_elimination` scenario enables E2E game completion tests.
- ✅ **E2E Tests Enabled:** Victory modal (return to lobby, rematch), rating updates after rated game.
- ✅ **Scenario Tests Enabled:** M2 (Disconnection Ladder), C2 (Capture Chain Endgame).
- ✅ **V2 Test Bug Fixed:** Stale `gameState` reference issue resolved.
- ✅ **Chat Test Enabled:** Multiplayer chat feature test now runs.

Remaining gaps:

1. **Multiplayer Coordination:** Timeout notifications, concurrent player flows.
2. **WebSocket Interception:** Network partition simulation for reconnection tests.
3. **Time Acceleration:** Decision timeout tests require real-time waiting.
4. **Complex Fixtures:** Territory victory, multi-phase turns.

**Why it is the weakest:**
Single-player game completion is now testable via fixtures, but multiplayer scenarios require infrastructure that doesn't exist. The backend and rules engine are robust (scores ≥ 4.5).

### 1.2 Hardest Outstanding Problem: Production E2E Infrastructure

**Difficulty: 4/5** (Increased from 3/5 due to refined scope)

Testing multiplayer coordination scenarios requires infrastructure beyond current capabilities:

1. **Multi-browser synchronization** – Deterministic turn-taking with WebSocket event verification.
2. **Network simulation** – Disconnection/reconnection testing via Playwright interception.
3. **Time acceleration** – Server-side mocking for timeout scenarios.

**Why it is hard:**

- Cross-cutting concerns across frontend, backend, and test infrastructure.
- Non-determinism in WebSocket timing and network conditions.
- Maintenance burden for complex test infrastructure.

**Resolved this pass:**

- ✅ Near-victory fixture API for game completion tests.
- ✅ E2E test helpers (`createFixtureGame`, `createNearVictoryGame`).
- ✅ V2 test bug fix (stale gameState reference).
- ✅ All skip rationales documented with specific blockers.

---

## 2. Remediation Plan (Pass 19B)

The detailed backlog is in [`docs/PASS19B_ASSESSMENT_REPORT.md`](docs/PASS19B_ASSESSMENT_REPORT.md).

### P0 (Critical) – E2E Test Infrastructure

- ~~**P19B.1-1:** Near-victory fixture API.~~ ✅ Done
- **P19B.1-2:** Multi-context WebSocket coordination helper.
- **P19B.1-3:** Network partition simulation.
- **P19B.1-4:** Time acceleration mode for timeout tests.

### P1 (Important) – Additional Fixtures

- **P19B.2-1:** Near-victory territory fixture.
- **P19B.2-2:** Chain capture fixture with 4+ targets.
- **P19B.2-3:** Multi-phase turn fixture.

### P2 (Nice to Have) – Test Polish

- **P19B.3-1:** Continue `any` cast reduction.
- **P19B.3-2:** Visual regression tests.

---

## 3. P18.1-5 Completed Remediations (PASS18 Hardest Problem Resolution)

The initial PASS18 assessment identified **Deep Multi-Engine Parity** as the hardest outstanding problem. This has been substantially resolved through the following remediation work:

### 3.1 P18.1-\*: Capture/Territory Host Parity ✅

- Unified capture chain handling between backend `GameEngine` and `ClientSandboxEngine`
- Aligned territory processing order and Q23 self-elimination prerequisites
- Advanced-phase semantics verified across hosts via orchestrator adapters
- **Docs:** [`P18.1-1_CAPTURE_TERRITORY_HOST_MAP.md`](docs/P18.1-1_CAPTURE_TERRITORY_HOST_MAP.md)

### 3.2 P18.2-\*: RNG Seed Handling Alignment ✅

- RNG seeding aligned between TS and Python for deterministic game replay
- AI move selection now reproducible given same seed
- **Docs:** [`P18.2-1_AI_RNG_PATHS.md`](docs/P18.2-1_AI_RNG_PATHS.md)

### 3.3 P18.3-\*: Decision Lifecycle and Timeout Semantics ✅

- Timeout behavior for pending decisions specified
- Decision expiry handling aligned across hosts
- **Docs:** [`P18.3-1_DECISION_LIFECYCLE_SPEC.md`](docs/P18.3-1_DECISION_LIFECYCLE_SPEC.md)

### 3.4 P18.4-\*: Orchestrator Rollout (Phase 4 Complete) ✅

- All environments: `ORCHESTRATOR_ADAPTER_ENABLED=true`, `ORCHESTRATOR_ROLLOUT_PERCENTAGE=100`
- Zero invariant violations across all board types in soak tests
- Production preview (P18.4-4) success criteria defined
- **Docs:** [`ORCHESTRATOR_ROLLOUT_PLAN.md`](docs/ORCHESTRATOR_ROLLOUT_PLAN.md), [`P18.4-3_ORCHESTRATOR_STAGING_REPORT.md`](docs/P18.4-3_ORCHESTRATOR_STAGING_REPORT.md)

### 3.5 P18.5-\*: Extended Contract Vectors and swap_sides Parity ✅

- **43 contract vectors** across 4 families: chain capture, forced elimination, territory/line, hex edge cases
- **0 mismatches** between TS and Python
- swap_sides (pie rule) verified across all layers (TS backend, TS sandbox, Python)
- **Design clarification (P18.5-3):** Mid-phase vectors are for single-step parity testing, not game seeding
- **Docs:** [`P18.5-3_ORCHESTRATOR_EXTENDED_VECTOR_SOAK_REPORT.md`](docs/P18.5-3_ORCHESTRATOR_EXTENDED_VECTOR_SOAK_REPORT.md), [`P18.5-4_SWAP_SIDES_PARITY_REPORT.md`](docs/P18.5-4_SWAP_SIDES_PARITY_REPORT.md)

### 3.6 P18.18: Skipped Test Triage ✅

- Obsolete test suites removed (4 files)
- RulesMatrix.Comprehensive partially re-enabled (7 passing, 3 skipped)
- OrchestratorSInvariant.regression re-enabled and passing
- **Docs:** [`P18.18_SKIPPED_TEST_TRIAGE.md`](docs/P18.18_SKIPPED_TEST_TRIAGE.md)

---

## 4. Historical Assessments

### Pass 19B (2025-11-30, Current)

- **Weakest Area:** E2E Test Coverage for Game Completion Scenarios (3.5/5.0).
- **Hardest Problem:** Production E2E Infrastructure (4/5).
- **Note:** Near-victory fixture API created. 6 tests enabled. V2 test bug fixed. Focus: E2E infrastructure for complex multiplayer scenarios.

### Pass 19A (2025-11-30, superseded by 19B)

- **Weakest Area:** Frontend UX Polish & Feature Completeness (3.5/5.0).
- **Hardest Problem:** Remaining `any` Casts & Incremental Refinement.
- **Note:** Test failures resolved. GameHistoryPanel integrated. Legacy paths deprecated.

### Pass 18-3 (2025-11-30, superseded by 19A)

- **Weakest Area:** Frontend UX Polish (3.3/5.0).
- **Hardest Problem:** Test Suite Cleanup & Legacy Deprecation.
- **Note:** Superseded by 19A after test fixes and GameHistoryPanel integration.

### Pass 18C (2025-11-30, superseded by 18-3)

- **Weakest Area:** Frontend UX Polish (3.2/5.0).
- **Hardest Problem:** Test Suite Cleanup.
- **Note:** Accessibility improved significantly.

### Pass 18B (2025-11-30, superseded by 18C)

- **Weakest Area:** Frontend UX & Accessibility (2.5/5.0).
- **Hardest Problem:** Legacy Code Deprecation & Test Suite Cleanup.
- **Note:** Orchestrator at 100% rollout. Host integration stabilized. Focus shifted to UX.

### Pass 18A (2025-11-30, superseded by 18B)

- **Weakest Area:** TS Rules/Host Integration & Parity (stabilizing).
- **Hardest Problem:** Orchestrator-first rollout execution.
- **Note:** Test suite stabilization in progress.

### Pass 18 (2025-11-30, superseded by 18A)

- **Weakest Area:** TS Rules/Host Integration & Parity (initial assessment).
- **Hardest Problem:** Orchestrator-first rollout & deep multi-engine parity.
- **Note:** PASS18A confirmed findings and added test stabilization progress.

### Pass 17 (2025-11-30)

- **Weakest Area:** Deep rules parity & invariants for territory / chain-capture / endgames.
- **Hardest Problem:** Operationalising orchestrator-first rollout with SLO gates.
- **Resolution:** P18.5-\* extended vectors now provide 43 cases with 0 mismatches covering these scenarios.

### Pass 16 (2025-11-28)

- **Weakest Area:** Frontend host architecture & UX ergonomics.
- **Hardest Problem:** Orchestrator-first production rollout & legacy decommissioning.
- **Resolution:** P18.4-\* achieved Phase 4 (100% rollout). Legacy paths properly quarantined as DIAGNOSTICS-ONLY.

### Pass 7 (2025-11-27)

- **Weakest Area:** DevOps/CI Enforcement (3.4/5.0).
- **Hardest Problem:** Orchestrator Production Rollout.
- **Resolution:** Orchestrator rollout infrastructure complete. CI gates via `orchestrator-parity` job.
