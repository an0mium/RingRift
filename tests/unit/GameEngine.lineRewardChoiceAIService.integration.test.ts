/**
 * @deprecated Phase 4 legacy path test (archived stub)
 *
 * Historical integration test for legacy `processLineFormations()` +
 * AIInteractionHandler + AIServiceClient line_reward_option flows.
 *
 * The active, orchestrator-backed coverage for AI line reward decisions now
 * lives at the GameSession/WebSocket level:
 *   - tests/integration/GameSession.aiOrchestrator.integration.test.ts
 *   - tests/integration/GameSession.aiDeterminism.test.ts
 *
 * AI service wiring and metrics for line reward decisions are also exercised
 * via:
 *   - tests/unit/AIEngine.fallback.test.ts
 *   - ai-service/tests/test_swap_rule.py (pie-rule/seat-swap meta flows)
 *
 * This file is retained only as a documentation pointer and does not call
 * any legacy GameEngine helpers. It can be removed entirely once Wave 5.4
 * cleanup is finished.
 */

describe.skip('GameEngine + AIInteractionHandler + AIServiceClient line_reward_option integration (LEGACY)', () => {
  it('is covered by orchestrator-backed GameSession AI tests and v2 contract vectors', () => {
    // No-op stub: see docs/LEGACY_CODE_ELIMINATION_PLAN.md and the
    // orchestrator-backed suites referenced above.
  });
});
