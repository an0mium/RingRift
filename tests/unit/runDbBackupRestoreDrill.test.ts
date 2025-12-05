/**
 * DB Backup & Restore Drill Harness Tests
 *
 * Focus on orchestration and CLI parsing only:
 * - Backup and restore shell command wiring
 * - Smoke-test helper integration
 * - JSON report generation under results/ops
 * - CLI arg parsing behaviour
 */

import fs from 'fs';
import { execFile } from 'child_process';

import * as DrillHarness from '../../scripts/run-db-backup-restore-drill';
import { parseArgs } from '../../scripts/run-db-backup-restore-drill';

// Jest module mocks ----------------------------------------------------------------

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

describe('runDbBackupRestoreDrill (orchestration)', () => {
  const mkdirMock = fs.mkdirSync as jest.MockedFunction<typeof fs.mkdirSync>;
  const writeFileMock = fs.writeFileSync as jest.MockedFunction<typeof fs.writeFileSync>;
  const execFileMock = execFile as jest.MockedFunction<typeof execFile>;

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('runs all checks and writes a passing report when everything succeeds', async () => {
    // Mock backup + restore shell commands (two execFile calls).
    let callIndex = 0;
    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callIndex += 1;

      if (callIndex === 1) {
        // Backup command: include sentinel for backup path
        callback(
          null,
          '__BACKUP_FILE__=backups/staging_drill_20251205_101200.sql\nbackup ok\n',
          ''
        );
      } else {
        // Restore command
        callback(null, 'restore ok', '');
      }

      // child_process.execFile returns a ChildProcess; we do not use it in tests.
      return {} as any;
    });

    // Mock smoke test helper to avoid real Prisma/DB work. Inject via options
    // so we do not rely on spying the module export binding.
    const smokeTestFn = jest.fn(async (_url?: string) => ({ ok: true }));

    const { report } = await DrillHarness.runDbBackupRestoreDrill({
      env: 'staging',
      operator: 'alice',
      // Cast to any to pass testing-only option through.
      smokeTestFn: smokeTestFn as any,
    } as any);

    // Overall result
    expect(report.environment).toBe('staging');
    expect(report.operator).toBe('alice');
    expect(report.overallPass).toBe(true);
    expect(report.checks).toHaveLength(3);

    const names = report.checks.map((c) => c.name).sort();
    expect(names).toEqual(['db_backup_create', 'db_restore_smoke_test', 'db_restore_to_scratch']);

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
    expect(String(outputPathArg)).toContain('results/ops/db_backup_restore.staging.');
    const parsed = JSON.parse(String(jsonContent));
    expect(parsed.drillType).toBe('db_backup_restore');
    expect(parsed.environment).toBe('staging');
    expect(parsed.overallPass).toBe(true);
    expect(Array.isArray(parsed.checks)).toBe(true);

    // Smoke test helper invoked with restore URL (may be undefined in this test).
    expect(smokeTestFn).toHaveBeenCalledTimes(1);
  });

  it('marks overallPass=false when backup command fails', async () => {
    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      const err = new Error('backup failed') as any;
      err.code = 1;
      callback(err, 'backup stdout', 'backup stderr');
      return {} as any;
    });

    const smokeTestFn = jest.fn(async (_url?: string) => ({ ok: true }));

    const { report } = await DrillHarness.runDbBackupRestoreDrill({
      env: 'staging',
      smokeTestFn: smokeTestFn as any,
    } as any);

    expect(report.overallPass).toBe(false);
    expect(report.checks).toHaveLength(3);

    const backupCheck = report.checks.find((c) => c.name === 'db_backup_create');
    expect(backupCheck).toBeDefined();
    expect(backupCheck?.status).toBe('fail');

    const restoreCheck = report.checks.find((c) => c.name === 'db_restore_to_scratch');
    expect(restoreCheck).toBeDefined();
    expect(restoreCheck?.status).toBe('fail');

    const smokeCheck = report.checks.find((c) => c.name === 'db_restore_smoke_test');
    expect(smokeCheck).toBeDefined();
    expect(smokeCheck?.status).toBe('fail');

    // When backup fails, smoke test should be skipped and helper not invoked.
    expect(smokeTestFn).not.toHaveBeenCalled();
  });

  it('marks overallPass=false when restore command fails', async () => {
    let callIndex = 0;
    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callIndex += 1;

      if (callIndex === 1) {
        // Backup succeeds
        callback(
          null,
          '__BACKUP_FILE__=backups/staging_drill_20251205_101200.sql\nbackup ok\n',
          ''
        );
      } else {
        // Restore fails
        const err = new Error('restore failed') as any;
        err.code = 1;
        callback(err, 'restore stdout', 'restore stderr');
      }

      return {} as any;
    });

    const smokeTestFn = jest.fn(async (_url?: string) => ({ ok: true }));

    const { report } = await DrillHarness.runDbBackupRestoreDrill({
      env: 'staging',
      smokeTestFn: smokeTestFn as any,
    } as any);

    expect(report.overallPass).toBe(false);

    const backupCheck = report.checks.find((c) => c.name === 'db_backup_create');
    expect(backupCheck).toBeDefined();
    expect(backupCheck?.status).toBe('pass');

    const restoreCheck = report.checks.find((c) => c.name === 'db_restore_to_scratch');
    expect(restoreCheck).toBeDefined();
    expect(restoreCheck?.status).toBe('fail');

    const smokeCheck = report.checks.find((c) => c.name === 'db_restore_smoke_test');
    expect(smokeCheck).toBeDefined();
    expect(smokeCheck?.status).toBe('fail');

    // Smoke test should not run when restore fails.
    expect(smokeTestFn).not.toHaveBeenCalled();
  });

  it('marks overallPass=false when smoke test fails', async () => {
    let callIndex = 0;
    execFileMock.mockImplementation((...args: any[]) => {
      const callback = args[3] as (error: Error | null, stdout: string, stderr: string) => void;
      callIndex += 1;

      if (callIndex === 1) {
        // Backup succeeds
        callback(
          null,
          '__BACKUP_FILE__=backups/staging_drill_20251205_101200.sql\nbackup ok\n',
          ''
        );
      } else {
        // Restore succeeds
        callback(null, 'restore ok', '');
      }

      return {} as any;
    });

    const smokeTestFn = jest.fn(async (_url?: string) => ({
      ok: false,
      error: 'test smoke failure',
    }));

    const { report } = await DrillHarness.runDbBackupRestoreDrill({
      env: 'staging',
      smokeTestFn: smokeTestFn as any,
    } as any);

    expect(report.overallPass).toBe(false);

    const smokeCheck = report.checks.find((c) => c.name === 'db_restore_smoke_test');
    expect(smokeCheck).toBeDefined();
    expect(smokeCheck?.status).toBe('fail');

    const details = smokeCheck?.details as { error?: string };
    expect(details.error).toContain('test smoke failure');
  });
});

describe('parseArgs (CLI parsing)', () => {
  it('parses required --env and optional flags including DB URLs', () => {
    const parsed = parseArgs([
      'node',
      'script',
      '--env',
      'staging',
      '--operator',
      'alice',
      '--output',
      'results/ops/custom.json',
      '--sourceDatabaseUrl',
      'postgresql://user:pass@host:5432/source_db',
      '--restoreDatabaseUrl',
      'postgresql://user:pass@host:5432/restore_db',
    ]);

    expect(parsed.env).toBe('staging');
    expect(parsed.operator).toBe('alice');
    expect(parsed.output).toBe('results/ops/custom.json');
    expect(parsed.sourceDatabaseUrl).toBe('postgresql://user:pass@host:5432/source_db');
    expect(parsed.restoreDatabaseUrl).toBe('postgresql://user:pass@host:5432/restore_db');
  });

  it('supports kebab-case aliases for DB URL flags', () => {
    const parsed = parseArgs([
      'node',
      'script',
      '--env=production',
      '--source-database-url=postgresql://user:pass@host:5432/source_db2',
      '--restore-database-url=postgresql://user:pass@host:5432/restore_db2',
    ]);

    expect(parsed.env).toBe('production');
    expect(parsed.sourceDatabaseUrl).toBe('postgresql://user:pass@host:5432/source_db2');
    expect(parsed.restoreDatabaseUrl).toBe('postgresql://user:pass@host:5432/restore_db2');
  });

  it('throws when --env is missing', () => {
    expect(() => parseArgs(['node', 'script', '--operator', 'alice'])).toThrowError(
      /Missing required --env <env> argument/
    );
  });
});
