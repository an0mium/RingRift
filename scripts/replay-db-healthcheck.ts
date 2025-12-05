#!/usr/bin/env ts-node
/**
 * replay-db-healthcheck.ts
 * ========================
 *
 * Small wrapper around the existing replay DB tooling to give you a
 * one-shot “are my replay databases healthy?” command.
 *
 * It currently:
 * - Invokes the Python cleanup/health script in dry-run mode to classify
 *   each GameReplayDB as having structurally good / mid-snapshot /
 *   inconsistent games, and writes a JSON summary.
 * - Optionally fails the process when any DB is marked_useless so this can
 *   be used as a CI guard.
 *
 * Usage (from repo root):
 *
 *   # Dry run: print summary path and keep exit code 0
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/replay-db-healthcheck.ts
 *
 *   # Treat useless DBs as a failure (e.g. in CI)
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/replay-db-healthcheck.ts --fail-on-useless
 */

/* eslint-disable no-console */

import { spawnSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

interface CleanupSummary {
  total_databases: number;
  databases: Array<{
    db_path: string;
    marked_useless?: boolean;
    structure_counts?: Record<string, number>;
  }>;
}

function main(): void {
  const args = process.argv.slice(2);
  const failOnUseless = args.includes('--fail-on-useless');

  const repoRoot = path.resolve(__dirname, '..');
  const aiServiceDir = path.join(repoRoot, 'ai-service');
  const summaryPath = path.join(aiServiceDir, 'db_health.replay_healthcheck.json');

  if (!fs.existsSync(aiServiceDir)) {
    console.error('ai-service directory not found; replay DB healthcheck cannot run.');
    process.exitCode = 1;
    return;
  }

  const pythonPath = process.env.PYTHON || 'python';

  const cmd = pythonPath;
  const cmdArgs = ['scripts/cleanup_useless_replay_dbs.py', '--summary-json', summaryPath];

  console.log('=== Replay DB healthcheck ===');
  console.log(
    `Running cleanup_useless_replay_dbs.py (dry-run) with summary at ${path.relative(
      repoRoot,
      summaryPath
    )}`
  );

  const result = spawnSync(cmd, cmdArgs, {
    cwd: aiServiceDir,
    stdio: 'inherit',
    env: {
      ...process.env,
      PYTHONPATH: '.',
    },
  });

  if (result.error) {
    console.error(
      `Failed to execute ${cmd} ${cmdArgs.join(' ')}:`,
      result.error.message ?? String(result.error)
    );
    process.exitCode = 1;
    return;
  }

  if (result.status !== 0) {
    console.error(
      `cleanup_useless_replay_dbs.py exited with code ${result.status}. See output above for details.`
    );
    process.exitCode = result.status ?? 1;
    return;
  }

  if (!fs.existsSync(summaryPath)) {
    console.warn('Replay DB health summary JSON was not written; nothing more to do.');
    return;
  }

  let summary: CleanupSummary | null = null;
  try {
    const raw = fs.readFileSync(summaryPath, 'utf-8');
    summary = JSON.parse(raw) as CleanupSummary;
  } catch (err) {
    console.warn(
      `Failed to parse replay DB health summary at ${summaryPath}:`,
      err instanceof Error ? err.message : String(err)
    );
  }

  if (!summary) {
    return;
  }

  const useless = summary.databases.filter((db) => db.marked_useless);
  const total = summary.total_databases ?? summary.databases.length;

  console.log('\nReplay DB health summary:');
  console.log(`  Databases inspected: ${total}`);
  console.log(`  Marked useless:      ${useless.length}`);

  if (useless.length > 0) {
    console.log('  Useless DBs (per cleanup_useless_replay_dbs):');
    for (const db of useless) {
      console.log(`    - ${db.db_path}`);
    }
  }

  console.log(
    `\nFull summary JSON written to ${path.relative(
      repoRoot,
      summaryPath
    )} (suitable for CI artifacts or manual inspection).`
  );

  if (failOnUseless && useless.length > 0) {
    console.error(
      '\nOne or more replay DBs were marked useless. ' +
        'Re-run the cleanup script with --delete or regenerate DBs before proceeding.'
    );
    process.exitCode = 1;
  }
}

main();
