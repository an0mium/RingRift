/**
 * @deprecated Phase 4 legacy path test (archived stub)
 *
 * Historical integration test for legacy `processLineFormations()` +
 * PlayerInteractionManager + WebSocketInteractionHandler line reward and
 * ring-elimination PlayerChoice flows.
 *
 * The active, orchestrator-backed coverage for WebSocket decision plumbing
 * now lives in:
 *   - tests/integration/GameSession.aiOrchestrator.integration.test.ts
 *   - tests/unit/WebSocketServer.humanDecisionById.integration.test.ts
 *
 * Line and elimination decision surfaces are also covered by:
 *   - tests/unit/GameEngine.lines.scenarios.test.ts
 *   - tests/unit/Backend_vs_Sandbox.CaptureAndTerritoryParity.test.ts
 *   - v2 contract vectors under tests/fixtures/contract-vectors/v2
 *
 * This file is retained only as a documentation pointer and does not call
 * any legacy GameEngine helpers. It can be removed once Wave 5.4 cleanup
 * is complete.
 */

describe.skip('GameEngine + WebSocketInteractionHandler line reward / ring elimination integration (LEGACY)', () => {
  it('is covered by orchestrator-backed GameSession/WebSocket tests and v2 contract vectors', () => {
    // No-op stub: see docs/LEGACY_CODE_ELIMINATION_PLAN.md and the
    // orchestrator/WebSocket suites referenced above.
  });
});
