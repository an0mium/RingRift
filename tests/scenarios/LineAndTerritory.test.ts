/**
 * @deprecated Phase 4 legacy path test (archived stub)
 *
 * Historical backend scenario test for combined line + territory flows
 * using legacy `GameEngine.processLineFormations()` plus move-driven
 * territory processing. This path has been superseded by:
 *
 *   - Orchestrator-backed multi-phase suites:
 *       • tests/scenarios/Orchestrator.Backend.multiPhase.test.ts
 *       • tests/scenarios/Orchestrator.Sandbox.multiPhase.test.ts
 *   - Shared-engine line/territory helpers:
 *       • tests/unit/lineDecisionHelpers.shared.test.ts
 *       • tests/unit/territoryDecisionHelpers.shared.test.ts
 *       • tests/unit/GameEngine.lines.scenarios.test.ts
 *       • tests/scenarios/RulesMatrix.Territory.GameEngine.test.ts
 *   - v2 contract vectors covering line+territory endgames:
 *       • tests/fixtures/contract-vectors/v2/territory_line_endgame.vectors.json
 *
 * This file is retained only as a documentation pointer and does not call
 * any legacy GameEngine helpers. It can be removed once Wave 5.4 cleanup
 * is complete.
 */

describe.skip('Scenario: Line and Territory Interactions (FAQ 7, 20, 22, 23; backend)', () => {
  it('is covered by orchestrator multi-phase suites and v2 contract vectors', () => {
    // No-op stub: see docs/LEGACY_CODE_ELIMINATION_PLAN.md and the
    // orchestrator/contract-vector suites referenced above.
  });
});
