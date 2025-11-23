import client from 'prom-client';
import { logger } from './logger';

/**
 * Prometheus counters for TS <-> Python rules-engine parity.
 *
 * These are incremented by the RulesBackendFacade whenever it observes
 * a discrepancy between the authoritative backend engine and the Python
 * rules service running in shadow mode (or vice versa when Python is
 * authoritative).
 */
export const rulesParityMetrics = {
  validMismatch: new client.Counter({
    name: 'rules_parity_valid_mismatch_total',
    help: 'TS vs Python rules: validation verdict mismatch count',
  }),
  hashMismatch: new client.Counter({
    name: 'rules_parity_hash_mismatch_total',
    help: 'TS vs Python rules: post-move state hash mismatch count',
  }),
  sMismatch: new client.Counter({
    name: 'rules_parity_S_mismatch_total',
    help: 'TS vs Python rules: S-invariant mismatch count',
  }),
  gameStatusMismatch: new client.Counter({
    name: 'rules_parity_gameStatus_mismatch_total',
    help: 'TS vs Python rules: gameStatus mismatch count',
  }),
};

/**
 * Core application-wide Prometheus metrics for AI, move processing, and WebSockets.
 * These share the default Node.js registry and are exported individually so
 * callers can import only what they need without re-configuring collectors.
 */
export const aiMoveLatencyHistogram = new client.Histogram({
  name: 'ai_move_latency_ms',
  help: 'Latency of AI move selection calls in milliseconds',
  labelNames: ['aiType', 'difficulty'] as const,
  buckets: [25, 50, 100, 200, 400, 800, 1600, 3200, 6400],
});

export const aiFallbackCounter = new client.Counter({
  name: 'ai_fallback_total',
  help: 'Total number of AI fallbacks by reason',
  labelNames: ['reason'] as const,
});

export const gameMoveLatencyHistogram = new client.Histogram({
  name: 'game_move_latency_ms',
  help: 'Latency of game move processing in milliseconds',
  labelNames: ['boardType', 'phase'] as const,
  buckets: [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560],
});

export const webSocketConnectionsGauge = new client.Gauge({
  name: 'websocket_connections_current',
  help: 'Current number of active WebSocket connections',
});

/**
 * Structured logging helper for rules parity discrepancies.
 *
 * All parity-related logs are emitted with a common message key so that
 * they can be discovered and aggregated easily in log pipelines.
 */
export function logRulesMismatch(
  kind: 'valid' | 'hash' | 'S' | 'gameStatus' | 'backend_fallback' | 'shadow_error',
  details: Record<string, unknown>
): void {
  logger.warn('rules_parity_mismatch', {
    kind,
    ...details,
  });
}
