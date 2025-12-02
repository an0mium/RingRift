#!/usr/bin/env ts-node
/**
 * TS↔Python parity wrapper with Prometheus metrics.
 *
 * Runs the Python_vs_TS.traceParity Jest suite and records a single
 * ringrift_parity_checks_total sample via MetricsService to indicate
 * whether the batch passed or failed.
 *
 * Intended for CI/CLIs; unit tests remain focused on correctness and
 * do not emit metrics directly.
 */

import { spawn } from 'child_process';
import { recordParityBatchResult } from '../src/server/utils/parityMetrics';

async function runJestParitySuite(): Promise<boolean> {
  return new Promise((resolve) => {
    const child = spawn('npm', ['test', '--', 'Python_vs_TS', '--passWithNoTests', '--silent'], {
      stdio: 'inherit',
      env: process.env,
      shell: process.platform === 'win32',
    });

    child.on('exit', (code) => {
      resolve(code === 0);
    });

    child.on('error', () => {
      resolve(false);
    });
  });
}

async function main(): Promise<void> {
  const passed = await runJestParitySuite();

  // Emit a single parity check sample for this batch.
  recordParityBatchResult(passed);

  process.exitCode = passed ? 0 : 1;
}

void main();
