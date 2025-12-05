/**
 * Secrets Rotation Drill Harness Tests
 *
 * Focus on orchestration and CLI parsing only:
 * - Programmatic validation passthrough
 * - Shell auth smoke test execution
 * - HTTP health check wiring
 * - JSON report generation under results/ops
 * - CLI arg parsing behaviour
 */

import fs from 'fs';
import { execFile } from 'child_process';

import type { DeploymentConfigValidationResult } from '../../scripts/validate-deployment-config';
import * as DrillHarness from '../../scripts/run-secrets-rotation-drill';
import { parseArgs } from '../../scripts/run-secrets-rotation-drill';
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

describe('runSecretsRotationDrill (orchestration)', () => {
  const mockValidate = validateDeploymentConfigProgrammatically as jest.MockedFunction<
    typeof validateDeploymentConfigProgrammatically
  >;
  const mkdirMock = fs.mkdirSync as jest.MockedFunction<typeof fs.mkdirSync>;
  const writeFileMock = fs.writeFileSync as jest.MockedFunction<typeof fs.writeFileSync>;
  const execFileMock = execFile as jest.MockedFunction<typeof execFile>;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('runs all checks and writes a passing report when everything succeeds', async () => {
    const validationResult: DeploymentConfigValidationResult = {
      ok: true,
      errors: [],
      warnings: [],
      results: [],
    };
    mockValidate.mockReturnValue(validationResult);

    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callback(null, 'auth smoke ok', '');
      // child_process.execFile returns a ChildProcess; we do not use it in tests
      return {} as any;
    });

    const httpHealthCheck = jest.fn(async (_url: string) => ({ statusCode: 200 }));

    const options: any = {
      env: 'staging',
      operator: 'alice',
      httpHealthCheck,
    };

    const { report } = await DrillHarness.runSecretsRotationDrill(options);

    // Overall result
    expect(report.environment).toBe('staging');
    expect(report.operator).toBe('alice');
    expect(report.overallPass).toBe(true);
    expect(report.checks).toHaveLength(3);

    const names = report.checks.map((c) => c.name).sort();
    expect(names).toEqual(['auth_smoke_test', 'deployment_config_validation', 'http_health_check']);

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
    expect(String(outputPathArg)).toContain('results/ops/secrets_rotation.staging.');
    const parsed = JSON.parse(String(jsonContent));
    expect(parsed.drillType).toBe('secrets_rotation');
    expect(parsed.environment).toBe('staging');
    expect(parsed.overallPass).toBe(true);
    expect(Array.isArray(parsed.checks)).toBe(true);

    // HTTP health check invoked with /health on default base URL
    expect(httpHealthCheck).toHaveBeenCalledTimes(1);
    const [healthUrl] = httpHealthCheck.mock.calls[0];
    expect(healthUrl).toBe('http://localhost:3000/health');
  });

  it('marks overallPass=false when deployment config validation fails and includes errors', async () => {
    const validationResult: DeploymentConfigValidationResult = {
      ok: false,
      errors: ['missing secret: RINGRIFT_DB_PASSWORD'],
      warnings: [],
      results: [],
    };
    mockValidate.mockReturnValue(validationResult);

    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callback(null, 'auth ok', '');
      return {} as any;
    });

    jest.spyOn(DrillHarness, 'performHttpHealthCheck').mockResolvedValue({ statusCode: 200 });

    const { report } = await DrillHarness.runSecretsRotationDrill({
      env: 'production',
      operator: 'ops-user',
      outputPath: 'results/ops/custom-report.json',
    });

    expect(report.overallPass).toBe(false);
    expect(report.checks).toHaveLength(3);

    const configCheck = report.checks.find((c) => c.name === 'deployment_config_validation');
    expect(configCheck).toBeDefined();
    expect(configCheck?.status).toBe('fail');

    const details = configCheck?.details as { errors: string[]; warnings: string[] };
    expect(details.errors.some((e) => e.includes('missing secret'))).toBe(true);

    // Explicit output path respected
    expect(writeFileMock).toHaveBeenCalledTimes(1);
    const [outputPathArg] = writeFileMock.mock.calls[0];
    expect(String(outputPathArg)).toBe('results/ops/custom-report.json');
  });

  it('propagates failure when auth smoke test exits non-zero', async () => {
    const validationResult: DeploymentConfigValidationResult = {
      ok: true,
      errors: [],
      warnings: [],
      results: [],
    };
    mockValidate.mockReturnValue(validationResult);

    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      const err = new Error('auth failed') as any;
      err.code = 1;
      callback(err, 'auth stdout', 'auth stderr');
      return {} as any;
    });

    jest.spyOn(DrillHarness, 'performHttpHealthCheck').mockResolvedValue({ statusCode: 200 });

    const { report } = await DrillHarness.runSecretsRotationDrill({
      env: 'staging',
    });

    expect(report.overallPass).toBe(false);

    const authCheck = report.checks.find((c) => c.name === 'auth_smoke_test');
    expect(authCheck).toBeDefined();
    expect(authCheck?.status).toBe('fail');

    const details = authCheck?.details as {
      exitCode: number | null;
      stdoutSnippet: string;
      stderrSnippet: string;
    };
    expect(details.exitCode).toBe(1);
    expect(details.stdoutSnippet).toContain('auth stdout');
    expect(details.stderrSnippet).toContain('auth stderr');
  });

  it('marks http_health_check as fail when health endpoint is unavailable', async () => {
    const validationResult: DeploymentConfigValidationResult = {
      ok: true,
      errors: [],
      warnings: [],
      results: [],
    };
    mockValidate.mockReturnValue(validationResult);

    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callback(null, 'auth ok', '');
      return {} as any;
    });

    const httpHealthCheck = jest.fn(async (_url: string) => ({ statusCode: null }));

    const options: any = {
      env: 'staging',
      baseUrl: 'http://example.com',
      httpHealthCheck,
    };

    const { report } = await DrillHarness.runSecretsRotationDrill(options);

    const healthCheck = report.checks.find((c) => c.name === 'http_health_check');
    expect(healthCheck).toBeDefined();
    expect(healthCheck?.status).toBe('fail');

    const details = healthCheck?.details as { url: string; statusCode: number | null };
    expect(details.url).toBe('http://example.com/health');
    expect(details.statusCode).toBeNull();
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
      '--baseUrl',
      'http://example.com',
    ]);

    expect(parsed.env).toBe('staging');
    expect(parsed.operator).toBe('alice');
    expect(parsed.output).toBe('results/ops/custom.json');
    expect(parsed.baseUrl).toBe('http://example.com');
  });

  it('supports --base-url as an alias for --baseUrl', () => {
    const parsed = parseArgs([
      'node',
      'script',
      '--env=production',
      '--base-url=https://prod.example.com',
    ]);

    expect(parsed.env).toBe('production');
    expect(parsed.baseUrl).toBe('https://prod.example.com');
  });

  it('throws a helpful error when --env is missing', () => {
    expect(() => parseArgs(['node', 'script'])).toThrow('Missing required --env <env> argument');
  });
});
