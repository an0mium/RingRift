import {
  parseThresholdExpression,
  computeScenarioSloStatus,
  makeHandleSummary,
} from '../load/summary.js';

describe('parseThresholdExpression', () => {
  it('parses percentile threshold expressions', () => {
    const result = parseThresholdExpression('p(95)<800');
    expect(result).toEqual({
      statistic: 'p(95)',
      comparison: '<',
      limit: 800,
    });

    const result99 = parseThresholdExpression('p(99)>=123.45');
    expect(result99).toEqual({
      statistic: 'p(99)',
      comparison: '>=',
      limit: 123.45,
    });
  });

  it('parses rate/count/max threshold expressions', () => {
    const rate = parseThresholdExpression('rate<0.01');
    expect(rate).toEqual({
      statistic: 'rate',
      comparison: '<',
      limit: 0.01,
    });

    const count = parseThresholdExpression('count<=5');
    expect(count).toEqual({
      statistic: 'count',
      comparison: '<=',
      limit: 5,
    });

    const max = parseThresholdExpression('max>10');
    expect(max).toEqual({
      statistic: 'max',
      comparison: '>',
      limit: 10,
    });
  });

  it('returns null fields for invalid expressions', () => {
    const empty = parseThresholdExpression('');
    expect(empty).toEqual({ statistic: null, comparison: null, limit: null });

    const garbage = parseThresholdExpression('this-is-not-a-threshold');
    expect(garbage).toEqual({ statistic: null, comparison: null, limit: null });

    // @ts-expect-error - runtime guard accepts non-string
    const nonString = parseThresholdExpression(123);
    expect(nonString).toEqual({ statistic: null, comparison: null, limit: null });
  });
});

describe('computeScenarioSloStatus', () => {
  it('computes per-threshold status and overallPass from k6-style summary data', () => {
    const data = {
      metrics: {
        'http_req_duration{name:create-game}': {
          values: {
            'p(95)': 700,
            'p(99)': 1200,
          },
          thresholds: {
            'p(95)<800': { ok: true, actual: 700 },
            'p(99)<1500': { ok: true, actual: 1200 },
          },
        },
        http_req_failed: {
          values: {
            rate: 0.02,
          },
          thresholds: {
            'rate<0.01': { ok: false, actual: 0.02 },
          },
        },
      },
    };

    const status = computeScenarioSloStatus('game-creation', 'staging', data as any);

    // We expect one entry per configured threshold expression
    expect(status.thresholds).toHaveLength(3);

    const byMetric = (metric: string) => status.thresholds.filter((t) => t.metric === metric);

    const createGameThresholds = byMetric('http_req_duration{name:create-game}');
    expect(createGameThresholds).toHaveLength(2);
    expect(createGameThresholds[0].passed).toBe(true);
    expect(createGameThresholds[1].passed).toBe(true);

    const errorRateThresholds = byMetric('http_req_failed');
    expect(errorRateThresholds).toHaveLength(1);
    const errorRate = errorRateThresholds[0];

    expect(errorRate.statistic).toBe('rate');
    expect(errorRate.comparison).toBe('<');
    expect(errorRate.limit).toBe(0.01);
    expect(errorRate.value).toBeCloseTo(0.02);
    expect(errorRate.passed).toBe(false);

    // overallPass should be false because at least one threshold failed
    expect(status.overallPass).toBe(false);
  });

  it('returns overallPass=false when there are no thresholds', () => {
    const data = {
      metrics: {
        some_metric_without_thresholds: {
          values: {
            count: 10,
          },
        },
      },
    };

    const status = computeScenarioSloStatus('noop', 'staging', data as any);
    expect(status.thresholds).toHaveLength(0);
    expect(status.overallPass).toBe(false);
  });
});

describe('makeHandleSummary', () => {
  const ORIGINAL_ENV = { ...process.env };

  afterEach(() => {
    // Restore environment between tests
    process.env = { ...ORIGINAL_ENV };
    // Ensure any global __ENV set in a test does not leak
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (global as any).__ENV = undefined;
  });

  it('produces a structured summary with scenario, environment, thresholds and overallPass', () => {
    process.env.THRESHOLD_ENV = 'staging';
    process.env.RINGRIFT_ENV = 'staging-stack';
    process.env.K6_SUMMARY_DIR = 'results/load-test';

    const handleSummary = makeHandleSummary('game-creation');

    const data = {
      metrics: {
        // Single passing threshold for a basic sanity check
        'http_req_duration{name:create-game}': {
          values: {
            'p(95)': 100,
          },
          thresholds: {
            'p(95)<800': { ok: true, actual: 100 },
          },
        },
      },
    };

    const outputs = handleSummary(data as any);

    const expectedPath = 'results/load-test/game-creation.staging.summary.json';
    expect(Object.keys(outputs)).toContain(expectedPath);

    const summaryRaw = outputs[expectedPath];
    expect(typeof summaryRaw).toBe('string');

    const summary = JSON.parse(summaryRaw as string);

    // Identity and environment fields
    expect(summary.scenario).toBe('game-creation');
    expect(summary.environment).toBe('staging-stack');
    expect(summary.thresholdsEnv).toBe('staging');

    // SLO block
    expect(summary.slo).toBeDefined();
    expect(summary.slo.scenario).toBe('game-creation');
    expect(summary.slo.environment).toBe('staging');
    expect(Array.isArray(summary.thresholds)).toBe(true);
    expect(typeof summary.overallPass).toBe('boolean');

    // Raw block retains the compact metric structure for backwards compatibility
    expect(summary.raw).toBeDefined();
    expect(summary.raw.scenario).toBe('game-creation');
    expect(summary.raw.environment).toBe('staging-stack');
    expect(summary.raw.thresholdsEnv).toBe('staging');

    // runTimestamp is ISO-like; we just assert that it exists and is a string.
    expect(typeof summary.runTimestamp).toBe('string');
  });

  it('falls back to THRESHOLD_ENV when RINGRIFT_ENV is not set', () => {
    process.env.THRESHOLD_ENV = 'production';
    delete process.env.RINGRIFT_ENV;
    delete process.env.K6_SUMMARY_DIR;

    const handleSummary = makeHandleSummary('player-moves');

    const data = {
      metrics: {},
    };

    const outputs = handleSummary(data as any);
    const expectedPath = 'results/load/player-moves.production.summary.json';
    expect(Object.keys(outputs)).toContain(expectedPath);

    const summary = JSON.parse(outputs[expectedPath] as string);

    expect(summary.scenario).toBe('player-moves');
    // Without RINGRIFT_ENV or explicit ENVIRONMENT, environment falls back to thresholds env
    expect(summary.environment).toBe('production');
    expect(summary.thresholdsEnv).toBe('production');
  });
});
