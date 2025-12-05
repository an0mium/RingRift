/**
 * Production Preview Go/No-Go Harness Tests
 *
 * Focus on orchestration and CLI parsing only:
 * - Topology + deployment config validation wiring
 * - Auth + game session shell harness integration
 * - AI readiness integration (via injectable helper)
 * - JSON report generation under results/ops
 * - CLI arg parsing behaviour
 */

import fs from 'fs';
import { execFile } from 'child_process';

import * as Harness from '../../scripts/run-prod-preview-go-no-go';
import { parseArgs } from '../../scripts/run-prod-preview-go-no-go';
import type { DeploymentConfigValidationResult } from '../../scripts/validate-deployment-config';
import { validateDeploymentConfigProgrammatically } from '../../scripts/validate-deployment-config';

// Jest module mocks ----------------------------------------------------------------

jest.mock('../../scripts/validate-deployment-config', () => ({
  validateDeploymentConfigProgrammatically: jest.fn(),
}));

jest.mock('fs', () => {
  const actual = jest.requireActual('fs') as typeof import('fs');
  return {
    ...actual,
    mkdirSync: jest.fn(),
    writeFileSync: jest.fn(),
  };
});

jest.mock('child_process', () => {
  const actual = jest.requireActual('child_process') as typeof import('child_process');
  return {
    ...actual,
    execFile: jest.fn(),
  };
});

describe('runProdPreviewGoNoGo (orchestration)', () => {
  const mockValidate = validateDeploymentConfigProgrammatically as jest.MockedFunction<
    typeof validateDeploymentConfigProgrammatically
  >;
  const mkdirMock = fs.mkdirSync as jest.MockedFunction<typeof fs.mkdirSync>;
  const writeFileMock = fs.writeFileSync as jest.MockedFunction<typeof fs.writeFileSync>;
  const execFileMock = execFile as jest.MockedFunction<typeof execFile>;

  let originalTopology: string | undefined;

  beforeEach(() => {
    jest.clearAllMocks();
    originalTopology = process.env.RINGRIFT_APP_TOPOLOGY;
    delete process.env.RINGRIFT_APP_TOPOLOGY;
  });

  afterEach(() => {
    if (originalTopology === undefined) {
      delete process.env.RINGRIFT_APP_TOPOLOGY;
    } else {
      process.env.RINGRIFT_APP_TOPOLOGY = originalTopology;
    }
  });

  it('runs all checks and writes a passing report when everything succeeds', async () => {
    const validationResult: DeploymentConfigValidationResult = {
      ok: true,
      errors: [],
      warnings: [],
      results: [],
    };
    mockValidate.mockReturnValue(validationResult);

    process.env.RINGRIFT_APP_TOPOLOGY = 'single';

    // First execFile: auth smoke via bash; second: game-session-load-smoke via npx.
    let callIndex = 0;
    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callIndex += 1;

      if (callIndex === 1) {
        callback(null, 'auth ok', '');
      } else {
        callback(null, 'game session ok', '');
      }

      // child_process.execFile returns a ChildProcess; we do not use it in tests.
      return {} as any;
    });

    const aiHealthCheck = jest.fn(async (_opts: any) => ({
      ok: true,
      details: {
        status: 'healthy',
        source: 'test-helper',
      },
    }));

    const { report } = await Harness.runProdPreviewGoNoGo({
      env: 'staging',
      operator: 'alice',
      baseUrl: 'http://example.com',
      // Testing-only injectable hook so we do not touch HealthCheckService directly.
      aiHealthCheck: aiHealthCheck as any,
    } as any);

    // Overall result
    expect(report.environment).toBe('staging');
    expect(report.operator).toBe('alice');
    expect(report.overallPass).toBe(true);
    expect(report.checks).toHaveLength(4);

    const names = report.checks.map((c) => c.name).sort();
    expect(names).toEqual([
      'ai_service_readiness',
      'auth_smoke_test',
      'game_session_smoke',
      'topology_and_config',
    ]);

    // Topology summary
    expect(report.topologySummary.appTopology).toBe('single');
    expect(report.topologySummary.expectedTopology).toBe('single');
    expect(report.topologySummary.configOk).toBe(true);

    for (const check of report.checks) {
      expect(check.status).toBe('pass');
    }

    // Filesystem writes
    expect(mkdirMock).toHaveBeenCalledTimes(1);
    expect(mkdirMock).toHaveBeenCalledWith(expect.stringContaining('results/ops'), {
      recursive: true,
    });

    expect(writeFileMock).toHaveBeenCalledTimes(1);
    const [outputPathArg, jsonContent] = writeFileMock.mock.calls[0];
    expect(String(outputPathArg)).toContain('results/ops/prod_preview_go_no_go.staging.');
    const parsed = JSON.parse(String(jsonContent));
    expect(parsed.drillType).toBe('prod_preview_go_no_go');
    expect(parsed.environment).toBe('staging');
    expect(parsed.overallPass).toBe(true);
    expect(Array.isArray(parsed.checks)).toBe(true);

    // AI readiness helper invoked once with options.
    expect(aiHealthCheck).toHaveBeenCalledTimes(1);
  });

  it('marks topology_and_config as fail and overallPass=false when config validation fails', async () => {
    const validationResult: DeploymentConfigValidationResult = {
      ok: false,
      errors: ['missing secret: RINGRIFT_DB_PASSWORD'],
      warnings: [],
      results: [],
    };
    mockValidate.mockReturnValue(validationResult);

    process.env.RINGRIFT_APP_TOPOLOGY = 'single';

    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callback(null, 'ok', '');
      return {} as any;
    });

    const aiHealthCheck = jest.fn(async (_opts: any) => ({
      ok: true,
      details: { status: 'healthy' },
    }));

    const { report } = await Harness.runProdPreviewGoNoGo({
      env: 'production',
      expectedTopology: 'single',
      aiHealthCheck: aiHealthCheck as any,
    } as any);

    expect(report.overallPass).toBe(false);

    const topoCheck = report.checks.find((c) => c.name === 'topology_and_config');
    expect(topoCheck).toBeDefined();
    expect(topoCheck?.status).toBe('fail');

    const details = topoCheck?.details as {
      appTopology: string;
      expectedTopology: string;
      topologyMatches: boolean;
      configErrors: string[];
    };
    expect(details.appTopology).toBe('single');
    expect(details.expectedTopology).toBe('single');
    expect(details.topologyMatches).toBe(true);
    expect(details.configErrors.some((e) => e.includes('missing secret'))).toBe(true);
  });

  it('propagates failure when auth smoke exits non-zero', async () => {
    const validationResult: DeploymentConfigValidationResult = {
      ok: true,
      errors: [],
      warnings: [],
      results: [],
    };
    mockValidate.mockReturnValue(validationResult);

    process.env.RINGRIFT_APP_TOPOLOGY = 'single';

    let callIndex = 0;
    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callIndex += 1;

      if (callIndex === 1) {
        const err = new Error('auth failed') as any;
        err.code = 1;
        callback(err, 'auth stdout', 'auth stderr');
      } else {
        callback(null, 'game ok', '');
      }

      return {} as any;
    });

    const aiHealthCheck = jest.fn(async (_opts: any) => ({
      ok: true,
      details: { status: 'healthy' },
    }));

    const { report } = await Harness.runProdPreviewGoNoGo({
      env: 'staging',
      aiHealthCheck: aiHealthCheck as any,
    } as any);

    expect(report.overallPass).toBe(false);

    const authCheck = report.checks.find((c) => c.name === 'auth_smoke_test');
    expect(authCheck).toBeDefined();
    expect(authCheck?.status).toBe('fail');

    const topoCheck = report.checks.find((c) => c.name === 'topology_and_config');
    const gameCheck = report.checks.find((c) => c.name === 'game_session_smoke');
    const aiCheck = report.checks.find((c) => c.name === 'ai_service_readiness');

    expect(topoCheck?.status).toBe('pass');
    expect(gameCheck?.status).toBe('pass');
    expect(aiCheck?.status).toBe('pass');
  });

  it('propagates failure when game session smoke exits non-zero', async () => {
    const validationResult: DeploymentConfigValidationResult = {
      ok: true,
      errors: [],
      warnings: [],
      results: [],
    };
    mockValidate.mockReturnValue(validationResult);

    process.env.RINGRIFT_APP_TOPOLOGY = 'single';

    let callIndex = 0;
    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callIndex += 1;

      if (callIndex === 1) {
        // Auth succeeds
        callback(null, 'auth ok', '');
      } else {
        const err = new Error('game session failed') as any;
        err.code = 1;
        callback(err, 'game stdout', 'game stderr');
      }

      return {} as any;
    });

    const aiHealthCheck = jest.fn(async (_opts: any) => ({
      ok: true,
      details: { status: 'healthy' },
    }));

    const { report } = await Harness.runProdPreviewGoNoGo({
      env: 'staging',
      aiHealthCheck: aiHealthCheck as any,
    } as any);

    expect(report.overallPass).toBe(false);

    const gameCheck = report.checks.find((c) => c.name === 'game_session_smoke');
    expect(gameCheck).toBeDefined();
    expect(gameCheck?.status).toBe('fail');

    const authCheck = report.checks.find((c) => c.name === 'auth_smoke_test');
    const topoCheck = report.checks.find((c) => c.name === 'topology_and_config');
    const aiCheck = report.checks.find((c) => c.name === 'ai_service_readiness');

    expect(authCheck?.status).toBe('pass');
    expect(topoCheck?.status).toBe('pass');
    expect(aiCheck?.status).toBe('pass');
  });

  it('marks ai_service_readiness as fail and overallPass=false when AI health helper reports failure', async () => {
    const validationResult: DeploymentConfigValidationResult = {
      ok: true,
      errors: [],
      warnings: [],
      results: [],
    };
    mockValidate.mockReturnValue(validationResult);

    process.env.RINGRIFT_APP_TOPOLOGY = 'single';

    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callback(null, 'ok', '');
      return {} as any;
    });

    const aiHealthCheck = jest.fn(async (_opts: any) => ({
      ok: false,
      details: {
        status: 'unhealthy',
        reason: 'simulated failure',
      },
    }));

    const { report } = await Harness.runProdPreviewGoNoGo({
      env: 'staging',
      aiHealthCheck: aiHealthCheck as any,
    } as any);

    expect(report.overallPass).toBe(false);

    const aiCheck = report.checks.find((c) => c.name === 'ai_service_readiness');
    expect(aiCheck).toBeDefined();
    expect(aiCheck?.status).toBe('fail');

    const details = aiCheck?.details as { status?: string; reason?: string };
    expect(details.status).toBe('unhealthy');
    expect(details.reason).toBe('simulated failure');
  });
});

describe('parseArgs (CLI parsing)', () => {
  it('parses required --env and optional flags', () => {
    const parsed = parseArgs([
      'node',
      'script',
      '--env',
      'staging',
      '--operator',
      'alice',
      '--output',
      'results/ops/custom.json',
      '--expectedTopology',
      'multi-sticky',
      '--baseUrl',
      'http://example.com',
    ]);

    expect(parsed.env).toBe('staging');
    expect(parsed.operator).toBe('alice');
    expect(parsed.output).toBe('results/ops/custom.json');
    expect(parsed.expectedTopology).toBe('multi-sticky');
    expect(parsed.baseUrl).toBe('http://example.com');
  });

  it('supports kebab-case aliases for expectedTopology and baseUrl', () => {
    const parsed = parseArgs([
      'node',
      'script',
      '--env=production',
      '--expected-topology=single',
      '--base-url=https://prod.example.com',
    ]);

    expect(parsed.env).toBe('production');
    expect(parsed.expectedTopology).toBe('single');
    expect(parsed.baseUrl).toBe('https://prod.example.com');
  });

  it('throws a helpful error when --env is missing', () => {
    expect(() => parseArgs(['node', 'script'])).toThrow('Missing required --env <env> argument');
  });
});
