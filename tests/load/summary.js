import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

/**
 * Shared handleSummary helper for RingRift k6 load tests.
 *
 * Goals for Wave 3.1:
 * - Emit a compact JSON summary per scenario under results/load/ that can be
 *   compared against SLO tables in STRATEGIC_ROADMAP.md and PROJECT_GOALS.md.
 * - Capture:
 *   - Scenario name and environment label
 *   - p95/p99 latencies for key HTTP / WebSocket / move metrics
 *   - Error and classification counters (contract vs capacity vs ID lifecycle)
 *
 * This helper is intentionally minimal. It does not try to mirror the full k6
 * JSON export format; instead it focuses on the small set of metrics that the
 * SLO docs and ALERTING_THRESHOLDS.md reference directly.
 */

/**
 * Extract p95/p99 percentiles for a Trend-like metric from the k6 summary.
 *
 * @param {string} metricName
 * @param {any} data - k6 summary object passed to handleSummary
 * @returns {{ p95: number, p99: number } | null}
 */
function extractPercentiles(metricName, data) {
  const metric = data && data.metrics && data.metrics[metricName];
  if (!metric || !metric.values) {
    return null;
  }

  const values = metric.values;
  const p95 = values['p(95)'];
  const p99 = values['p(99)'];

  if (typeof p95 !== 'number' && typeof p99 !== 'number') {
    return null;
  }

  return {
    p95: typeof p95 === 'number' ? p95 : null,
    p99: typeof p99 === 'number' ? p99 : null,
  };
}

/**
 * Extract simple count/rate information for a Counter/Rate metric.
 *
 * @param {string} metricName
 * @param {any} data
 * @returns {{ count?: number, rate?: number } | null}
 */
function extractCounterSummary(metricName, data) {
  const metric = data && data.metrics && data.metrics[metricName];
  if (!metric || !metric.values) {
    return null;
  }

  const values = metric.values;
  const hasCount = typeof values.count === 'number';
  const hasRate = typeof values.rate === 'number';

  if (!hasCount && !hasRate) {
    return null;
  }

  return {
    count: hasCount ? values.count : undefined,
    rate: hasRate ? values.rate : undefined,
  };
}

/**
 * Extract a simple rate value for Rate metrics such as success_rate.
 *
 * @param {string} metricName
 * @param {any} data
 * @returns {number | null}
 */
function extractRate(metricName, data) {
  const metric = data && data.metrics && data.metrics[metricName];
  if (!metric || !metric.values || typeof metric.values.rate !== 'number') {
    return null;
  }
  return metric.values.rate;
}

/**
 * Build the compact JSON summary object shared across all scenarios.
 *
 * @param {string} scenarioName
 * @param {string} environmentLabel
 * @param {any} data
 */
function buildSummaryObject(scenarioName, environmentLabel, data) {
  return {
    scenario: scenarioName,
    environment: environmentLabel,
    // Mirror the THRESHOLD_ENV used by thresholds.json; this is usually an
    // environment label rather than a full deployment identifier.
    thresholdsEnv: environmentLabel,

    http: {
      // Aggregate HTTP timings across all requests in the scenario.
      http_req_duration: extractPercentiles('http_req_duration', data),
      // Scenario-specific trends, present where applicable:
      game_creation_latency_ms: extractPercentiles('game_creation_latency_ms', data),
      // For move scenarios:
      move_submission_latency_ms: extractPercentiles('move_submission_latency_ms', data),
      turn_processing_latency_ms: extractPercentiles('turn_processing_latency_ms', data),
    },

    websocket: {
      websocket_message_latency_ms: extractPercentiles('websocket_message_latency_ms', data),
      websocket_connection_duration_ms: extractPercentiles(
        'websocket_connection_duration_ms',
        data
      ),
      websocket_connection_success_rate: extractRate('websocket_connection_success_rate', data),
      websocket_handshake_success_rate: extractRate('websocket_handshake_success_rate', data),
    },

    ai: {
      // Kept for future extension; current scenarios primarily validate AI via
      // Prometheus metrics rather than direct k6 metrics.
      ai_move_latency_ms: extractPercentiles('ai_move_latency_ms', data),
    },

    classifications: {
      // Shared classification counters across all scenarios:
      contract_failures_total: extractCounterSummary('contract_failures_total', data),
      id_lifecycle_mismatches_total: extractCounterSummary(
        'id_lifecycle_mismatches_total',
        data
      ),
      capacity_failures_total: extractCounterSummary('capacity_failures_total', data),

      // Scenario-specific error counters, present where defined:
      game_state_errors: extractCounterSummary('game_state_errors', data),
      move_processing_errors: extractCounterSummary('move_processing_errors', data),
      stalled_moves_total: extractCounterSummary('stalled_moves_total', data),
      websocket_connection_errors: extractCounterSummary('websocket_connection_errors', data),
      websocket_protocol_errors: extractCounterSummary('websocket_protocol_errors', data),
    },
  };
}

/**
 * Factory for per-file handleSummary implementations.
 *
 * Usage in a scenario file (e.g. game-creation.js):
 *
 *   import { makeHandleSummary } from '../summary.js';
 *   export const handleSummary = makeHandleSummary('game-creation');
 *
 * This will:
 *   - Write a JSON file under results/load/<scenario>.<env>.summary.json
 *   - Keep the standard text summary on stdout unless K6_DISABLE_STDOUT_SUMMARY=1
 *
 * Environment variables:
 *   - THRESHOLD_ENV: environment label used for thresholds (default: "staging")
 *   - K6_SUMMARY_DIR: override output directory (default: "results/load")
 *   - K6_DISABLE_STDOUT_SUMMARY=1 to turn off the human-readable text summary
 *
 * @param {string} scenarioName
 * @returns {(data: any) => Record<string, string>}
 */
export function makeHandleSummary(scenarioName) {
  const envLabel = (__ENV.THRESHOLD_ENV || 'staging').toString();
  const baseDir = (__ENV.K6_SUMMARY_DIR || 'results/load').toString();

  return function handleSummary(data) {
    const summary = buildSummaryObject(scenarioName, envLabel, data);
    const filePath = `${baseDir}/${scenarioName}.${envLabel}.summary.json`;

    const outputs = {
      [filePath]: JSON.stringify(summary, null, 2),
    };

    // Preserve the default k6 behaviour of printing a human-readable summary to
    // stdout unless explicitly disabled via env var. This keeps local/dev usage
    // friendly while enabling machine-readable JSON for Wave 7 SLO validation.
    if (__ENV.K6_DISABLE_STDOUT_SUMMARY !== '1') {
      outputs.stdout = textSummary(data, { indent: ' ', enableColors: true });
    }

    return outputs;
  };
}