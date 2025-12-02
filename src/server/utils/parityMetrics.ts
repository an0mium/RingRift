import { getMetricsService } from '../services/MetricsService';

/**
 * Lightweight helper for recording TS↔Python parity check outcomes from
 * Node-only scripts and CLIs. This module must not be imported from any
 * browser bundles.
 */
export function recordParityBatchResult(passed: boolean): void {
  try {
    const metrics = getMetricsService();
    metrics.recordParityCheck(passed);
  } catch {
    // Metrics must never break parity harnesses; ignore failures.
  }
}
