import client from 'prom-client';
import { MetricsService, getMetricsService } from '../../src/server/services/MetricsService';

/**
 * Tests for orchestrator shadow metrics bridge.
 *
 * NOTE: These tests are currently skipped because the shadow comparator
 * metrics feature (refreshOrchestratorShadowMetrics) was never implemented
 * in MetricsService. When the feature is implemented, remove the skip.
 *
 * Related: The FSM is now the canonical orchestrator (RR-CANON-R070).
 * Shadow comparison between legacy and FSM orchestration may be reconsidered
 * as part of future FSM migration work.
 */
describe('MetricsService orchestrator shadow metrics bridge', () => {
  beforeEach(() => {
    MetricsService.resetInstance();
    client.register.clear();
  });

  // TODO: Implement refreshOrchestratorShadowMetrics in MetricsService
  // or remove this test if shadow comparison metrics are no longer needed
  it.skip('exposes shadow comparator gauges in /metrics output', async () => {
    const metrics = getMetricsService();

    // Force a refresh so gauges are registered and populated at least once
    metrics.refreshOrchestratorShadowMetrics();

    const output = await metrics.getMetrics();

    expect(output).toContain('ringrift_orchestrator_shadow_comparisons_current');
    expect(output).toContain('ringrift_orchestrator_shadow_mismatches_current');
    expect(output).toContain('ringrift_orchestrator_shadow_mismatch_rate');
    expect(output).toContain('ringrift_orchestrator_shadow_orchestrator_errors_current');
    expect(output).toContain('ringrift_orchestrator_shadow_orchestrator_error_rate');
    expect(output).toContain('ringrift_orchestrator_shadow_avg_legacy_latency_ms');
    expect(output).toContain('ringrift_orchestrator_shadow_avg_orchestrator_latency_ms');
  });
});
