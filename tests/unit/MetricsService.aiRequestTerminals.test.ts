import client from 'prom-client';
import { MetricsService } from '../../src/server/services/MetricsService';

describe('MetricsService AI request / AI turn terminal metrics', () => {
  beforeEach(() => {
    MetricsService.resetInstance();
    client.register.clear();
  });

  it('records AI turn request terminals with kind/code/ai_error_type labels', async () => {
    const metrics = MetricsService.getInstance();

    metrics.recordAITurnRequestTerminal('timed_out');
    metrics.recordAITurnRequestTerminal('failed', 'AI_SERVICE_TIMEOUT', 'timeout');
    metrics.recordAITurnRequestTerminal('canceled', 'AI_SERVICE_CANCELED', 'session_canceled');

    const output = await metrics.getMetrics();

    expect(output).toContain('ringrift_ai_turn_request_terminal_total');
    expect(output).toContain(
      'ringrift_ai_turn_request_terminal_total{kind="timed_out",code="none",ai_error_type="none"} 1'
    );
    expect(output).toContain(
      'ringrift_ai_turn_request_terminal_total{kind="failed",code="AI_SERVICE_TIMEOUT",ai_error_type="timeout"} 1'
    );
    expect(output).toContain(
      'ringrift_ai_turn_request_terminal_total{kind="canceled",code="AI_SERVICE_CANCELED",ai_error_type="session_canceled"} 1'
    );
  });

  it('records AI request latency and timeout counters for timed-out vs failed outcomes', async () => {
    const metrics = MetricsService.getInstance();

    metrics.recordAIRequestLatencyMs(1200, 'timeout');
    metrics.recordAIRequestLatencyMs(800, 'error');
    metrics.recordAIRequestTimeout();

    const output = await metrics.getMetrics();

    expect(output).toContain('ringrift_ai_request_latency_ms_bucket');
    expect(output).toContain('ringrift_ai_request_latency_ms_count');
    expect(output).toContain('ringrift_ai_request_timeout_total');
  });
});
