# RingRift Test Layering Strategy

> **Purpose:** Define a clear test layering strategy to minimize redundancy, improve iteration speed, and ensure each layer has a specific purpose.

## Test Layer Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 4: E2E Tests (Playwright)                                              │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Scope: Full browser-based user journeys                                      │
│ Location: tests/e2e/                                                         │
│ Run: Before release, after major changes                                     │
│ Speed: Slow (~minutes)                                                       │
│ Count: 2 test files                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Integration Tests                                                   │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Scope: Service interactions, AI service, WebSocket flows                     │
│ Location: tests/integration/, tests/unit/*Integration*.test.ts               │
│ Run: CI pipeline, major feature changes                                      │
│ Speed: Medium (~30-60 seconds)                                               │
│ Count: ~10 test files                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 2: Contract/Scenario Tests                                             │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Scope: Cross-language parity, canonical engine behavior                      │
│ Location: tests/contracts/, tests/scenarios/, fixtures/contract-vectors/     │
│ Run: Every commit, after rules changes                                       │
│ Speed: Fast-Medium (~10-30 seconds)                                          │
│ Count: ~20 test files + vector bundles                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 1: Unit Tests                                                          │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Scope: Individual module behavior, shared engine functions                   │
│ Location: tests/unit/*.shared.test.ts, tests/unit/*Engine*.test.ts           │
│ Run: Every commit, during development                                        │
│ Speed: Fast (~5-15 seconds)                                                  │
│ Count: ~60 test files                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Unit Tests (Fast, Per-Module)

### Purpose

Test individual functions, modules, and components in isolation.

### Characteristics

- **Speed:** < 100ms per test
- **Dependencies:** Mocked or minimal
- **Determinism:** 100% deterministic
- **Granularity:** Single function or module

### Key Test Categories

#### Shared Engine Core (`*.shared.test.ts`)

| File                                      | Purpose                       | Keep/Review |
| ----------------------------------------- | ----------------------------- | ----------- |
| `movement.shared.test.ts`                 | Movement validation logic     | ✅ Keep     |
| `captureLogic.shared.test.ts`             | Capture chain logic           | ✅ Keep     |
| `lineDetection.shared.test.ts`            | Line detection algorithms     | ✅ Keep     |
| `territoryBorders.shared.test.ts`         | Territory border calculations | ✅ Keep     |
| `territoryDecisionHelpers.shared.test.ts` | Territory decision helpers    | ✅ Keep     |
| `lineDecisionHelpers.shared.test.ts`      | Line decision helpers         | ✅ Keep     |
| `victory.shared.test.ts`                  | Victory condition logic       | ✅ Keep     |
| `heuristicParity.shared.test.ts`          | Heuristic evaluation parity   | ✅ Keep     |

#### Server Components

| File                         | Purpose               | Keep/Review |
| ---------------------------- | --------------------- | ----------- |
| `auth.routes.test.ts`        | Authentication routes | ✅ Keep     |
| `rateLimiter.test.ts`        | Rate limiting         | ✅ Keep     |
| `securityHeaders.test.ts`    | Security middleware   | ✅ Keep     |
| `logger.test.ts`             | Logging utilities     | ✅ Keep     |
| `validation.schemas.test.ts` | Input validation      | ✅ Keep     |
| `MetricsService.test.ts`     | Prometheus metrics    | ✅ Keep     |

#### Infrastructure

| File                          | Purpose                  | Keep/Review |
| ----------------------------- | ------------------------ | ----------- |
| `envFlags.test.ts`            | Environment flag parsing | ✅ Keep     |
| `notation.test.ts`            | Move notation parsing    | ✅ Keep     |
| `NoRandomInCoreRules.test.ts` | Determinism guard        | ✅ Keep     |
| `RNGDeterminism.test.ts`      | RNG consistency          | ✅ Keep     |

---

## Layer 2: Contract/Scenario Tests (Canonical Engine)

### Purpose

Validate that all engine implementations (TS backend, TS sandbox, Python) produce identical outputs for the same inputs.

### Characteristics

- **Speed:** < 500ms per vector
- **Dependencies:** Canonical engine, test fixtures
- **Determinism:** 100% deterministic
- **Granularity:** State transitions, move validation

### Contract Test Vectors (`tests/fixtures/contract-vectors/v2/`)

| File                          | Category             | Vector Count |
| ----------------------------- | -------------------- | ------------ |
| `placement.vectors.json`      | Placement moves      | 10+          |
| `movement.vectors.json`       | Movement/capture     | 15+          |
| `capture.vectors.json`        | Capture chains       | 10+          |
| `line_detection.vectors.json` | Line detection       | 10+          |
| `territory.vectors.json`      | Territory processing | 10+          |

### Contract Runner

- **File:** `tests/contracts/contractVectorRunner.test.ts`
- **Role:** Single source of truth for cross-language parity
- **Python counterpart:** `ai-service/tests/contracts/test_contract_vectors.py`

### Scenario Tests (`tests/scenarios/`)

| Category     | Files                                                                   | Purpose                         |
| ------------ | ----------------------------------------------------------------------- | ------------------------------- |
| Rules Matrix | `RulesMatrix.*.test.ts`                                                 | Comprehensive rules validation  |
| FAQ Tests    | `FAQ_Q*.test.ts`                                                        | User-facing rule clarifications |
| Edge Cases   | `ComplexChainCaptures.test.ts`, `ForcedEliminationAndStalemate.test.ts` | Complex scenarios               |

---

## Layer 3: Parity/Integration Tests

### Purpose

Verify that hosts (backend, sandbox, Python) behave identically and that external service integrations work correctly.

### Parity Tests (Review for Consolidation)

#### Backend vs Sandbox Parity

| File                                     | Purpose               | Recommendation                      |
| ---------------------------------------- | --------------------- | ----------------------------------- |
| `Backend_vs_Sandbox.traceParity.test.ts` | Trace-level parity    | ⚠️ Consolidate with contract tests  |
| `Backend_vs_Sandbox.seed*.test.ts`       | Seed-specific parity  | ⚠️ Move to historical failures only |
| `*Parity*.test.ts` (10+ files)           | Various parity checks | ⚠️ Promote to contract vectors      |

#### Python vs TS Parity

| File                               | Purpose               | Recommendation                      |
| ---------------------------------- | --------------------- | ----------------------------------- |
| `Python_vs_TS.traceParity.test.ts` | Cross-language parity | ✅ Keep, supplement with contracts  |
| `ai-service/tests/parity/*.py`     | Python parity tests   | ✅ Keep, use contracts as authority |

### Integration Tests (`tests/integration/`)

| File                                         | Purpose                     | Keep/Review                 |
| -------------------------------------------- | --------------------------- | --------------------------- |
| `AIGameCreation.test.ts`                     | AI game creation flow       | ✅ Keep                     |
| `AIResilience.test.ts`                       | AI service failure handling | ✅ Keep                     |
| `FullGameFlow.test.ts`                       | Complete game lifecycle     | ✅ Keep                     |
| `GameReconnection.test.ts`                   | WebSocket reconnection      | ✅ Keep                     |
| `LobbyRealtime.test.ts`                      | Lobby real-time updates     | ✅ Keep                     |
| `PythonRulesClient.live.integration.test.ts` | Live Python service         | ⚠️ Requires running service |

---

## Layer 4: E2E Tests (Playwright)

### Purpose

Validate complete user journeys in a real browser environment.

### Test Files (`tests/e2e/`)

| File                    | Purpose             |
| ----------------------- | ------------------- |
| `auth.e2e.spec.ts`      | Authentication flow |
| `game-flow.e2e.spec.ts` | Game play flow      |

### When to Run

- Before releases
- After UI changes
- After major backend changes

---

## Test Categories by Host

### Shared Engine Tests (Authoritative)

```
tests/unit/*.shared.test.ts          # Core logic tests
tests/contracts/                      # Contract vectors
tests/fixtures/contract-vectors/      # Test vector data
```

### Backend Tests

```
tests/unit/GameEngine.*.test.ts       # Backend game engine
tests/unit/RuleEngine.*.test.ts       # Legacy rule engine
tests/unit/BoardManager.*.test.ts     # Board management
tests/integration/                    # Service integration
```

### Sandbox Tests

```
tests/unit/ClientSandboxEngine.*.test.ts   # Client sandbox
tests/unit/sandboxTerritory*.test.ts       # Territory logic
```

### Python Tests

```
ai-service/tests/contracts/           # Contract validation
ai-service/tests/parity/              # TS parity tests
ai-service/tests/rules/               # Python rules tests
ai-service/tests/invariants/          # Invariant tests
```

---

## Redundancy Analysis & Consolidation Candidates

### High Priority for Consolidation

#### Parity Tests → Contract Vectors

Many parity tests can be converted to contract vectors:

| Parity Test                       | Contract Vector            |
| --------------------------------- | -------------------------- |
| `MovementCaptureParity.*.test.ts` | → `movement.vectors.json`  |
| `PlacementParity.*.test.ts`       | → `placement.vectors.json` |
| `VictoryParity.*.test.ts`         | → Add victory vectors      |
| `TerritoryParity.*.test.ts`       | → `territory.vectors.json` |

#### Seed-Specific Tests → Historical Failures

Keep seed-specific tests only for documented historical failures:

| File                                  | Status    | Action                       |
| ------------------------------------- | --------- | ---------------------------- |
| `Seed14Move35LineParity.test.ts`      | ✅ Keep   | Documents seed-14 resolution |
| `Backend_vs_Sandbox.seed5.*.test.ts`  | ⚠️ Review | Consolidate or document      |
| `Backend_vs_Sandbox.seed17.*.test.ts` | ⚠️ Review | Consolidate or document      |
| `Backend_vs_Sandbox.seed18.*.test.ts` | ⚠️ Review | Consolidate or document      |

### Medium Priority

#### Engine-Specific Tests

Some tests duplicate behavior across engines:

| GameEngine Test                             | Sandbox Test                                         | Recommendation                    |
| ------------------------------------------- | ---------------------------------------------------- | --------------------------------- |
| `GameEngine.lines.scenarios.test.ts`        | `ClientSandboxEngine.lines.test.ts`                  | Promote shared cases to contracts |
| `GameEngine.territoryDisconnection.test.ts` | `ClientSandboxEngine.territoryDisconnection.test.ts` | Same                              |
| `GameEngine.victory.*.test.ts`              | `ClientSandboxEngine.victory.test.ts`                | Same                              |

---

## CI Pipeline Configuration

### Fast Feedback (< 2 min)

```bash
# Layer 1: Unit tests
npm run test:unit -- --testPathPattern="\.shared\.test\."

# Layer 2: Contract tests
npm run test -- --testPathPattern="contracts"
```

### Standard CI (< 5 min)

```bash
# Layers 1-2 + key scenarios
npm run test -- --testPathIgnorePatterns="e2e|integration|\.debug\.|\.seed\d+\."
```

### Full CI (< 15 min)

```bash
# All layers except E2E
npm run test

# Python contracts
cd ai-service && pytest tests/contracts/
```

### Release Gate

```bash
# All tests including E2E
npm run test
npm run test:e2e

# Python full suite
cd ai-service && pytest
```

---

## Migration Path: Adapters & Legacy Elimination

With the orchestrator adapters (`TurnEngineAdapter.ts`, `SandboxOrchestratorAdapter.ts`) now in place:

### Phase A: Enable Adapters (Current)

1. Wire feature flag `ORCHESTRATOR_ADAPTER_ENABLED`
2. Run all tests with flag disabled (baseline)
3. Run all tests with flag enabled (parity check)
4. Enable in staging/canary

### Phase B: Trim Redundant Parity Tests

Once adapters are stable:

1. Remove seed-specific parity tests (keep only historical failure docs)
2. Consolidate engine-specific tests to contract vectors
3. Remove legacy trace parity tests

### Phase C: Remove Legacy Code

After full adapter migration:

1. Remove legacy sandbox turn engines
2. Update tests to use adapters directly
3. Simplify test suite by ~30%

---

## Related Documents

- [`tests/README.md`](./README.md) - How to run tests
- [`tests/TEST_SUITE_PARITY_PLAN.md`](./TEST_SUITE_PARITY_PLAN.md) - Detailed test classification
- [`RULES_SCENARIO_MATRIX.md`](../RULES_SCENARIO_MATRIX.md) - Rules-to-test mapping
- [`docs/drafts/LEGACY_CODE_ELIMINATION_PLAN.md`](../docs/drafts/LEGACY_CODE_ELIMINATION_PLAN.md) - Legacy removal plan
