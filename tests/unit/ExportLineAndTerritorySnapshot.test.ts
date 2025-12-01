/**
 * @deprecated Phase 4 legacy path test
 *
 * Historical exporter for line+territory ComparableSnapshots used by
 * early Python parity tests. The canonical coverage for these scenarios
 * now lives in:
 *
 *   - Contract vectors under tests/fixtures/contract-vectors/v2
 *     (see scripts/generate-extended-contract-vectors.ts and
 *      scripts/generate-orchestrator-contract-vectors.ts)
 *   - Orchestrator-backed multi-phase suites:
 *       • tests/scenarios/Orchestrator.Backend.multiPhase.test.ts
 *       • tests/scenarios/Orchestrator.Sandbox.multiPhase.test.ts
 *   - RulesMatrix/FAQ scenario tests (e.g. LineAndTerritory.test.ts)
 *
 * This file is kept as a stub to preserve historical context; it no
 * longer exercises legacy GameEngine helpers or writes new fixtures.
 */

test.skip('legacy line+territory snapshot exporter (superseded by v2 contract vectors)', () => {
  // Intentionally empty: snapshots now flow through orchestrator-backed
  // contract vectors rather than legacy processLineFormations /
  // processDisconnectedRegions helpers.
});
