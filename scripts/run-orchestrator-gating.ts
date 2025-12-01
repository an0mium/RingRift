#!/usr/bin/env ts-node
/**
 * Orchestrator gating wrapper.
 *
 * Runs the core orchestrator gates in sequence:
 * - TS orchestrator parity tests
 * - Python contract/parity tests (via scripts/run-python-contract-tests.sh)
 * - Short orchestrator soak (ci-short profile, failOnViolation=true)
 *
 * Intended as a single local/CI entrypoint for the gating set described in
 * docs/ORCHESTRATOR_ROLLOUT_PLAN.md §6.9 and the rollout runbook.
 */

import { spawn } from 'child_process';

type StepId = 'ts-parity' | 'python-contracts' | 'short-soak';

interface Step {
  id: StepId;
  description: string;
  command: string;
  args: string[];
  env?: NodeJS.ProcessEnv;
}

const steps: Step[] = [
  {
    id: 'ts-parity',
    description: 'TypeScript orchestrator parity tests',
    command: 'npm',
    args: ['run', 'test:orchestrator-parity:ts'],
  },
  {
    id: 'python-contracts',
    description: 'Python contract/parity tests',
    command: './scripts/run-python-contract-tests.sh',
    args: ['--verbose'],
  },
  {
    id: 'short-soak',
    description: 'Short orchestrator soak (ci-short)',
    command: 'npm',
    args: ['run', 'soak:orchestrator:short'],
    env: {
      // Ensure ts-node uses the server config, matching existing soak scripts.
      ...process.env,
      TS_NODE_PROJECT: process.env.TS_NODE_PROJECT ?? 'tsconfig.server.json',
    },
  },
];

function runStep(step: Step): Promise<void> {
  return new Promise((resolve, reject) => {
    // eslint-disable-next-line no-console
    console.log(`\n=== [${step.id}] ${step.description} ===`);
    // eslint-disable-next-line no-console
    console.log(`$ ${step.command} ${step.args.join(' ')}\n`);

    const child = spawn(step.command, step.args, {
      stdio: 'inherit',
      env: step.env ?? process.env,
      shell: process.platform === 'win32',
    });

    child.on('error', (err) => {
      console.error(`[${step.id}] failed to start:`, err);
      reject(err);
    });

    child.on('exit', (code, signal) => {
      if (code === 0) {
        // eslint-disable-next-line no-console
        console.log(`\n[${step.id}] completed successfully.\n`);
        resolve();
      } else {
        const reason = code !== null ? `exit code ${code}` : `signal ${signal ?? 'unknown'}`;

        console.error(`\n[${step.id}] failed with ${reason}.\n`);
        reject(
          new Error(`Step "${step.id}" failed with ${reason}; aborting orchestrator gating run.`)
        );
      }
    });
  });
}

async function main(): Promise<void> {
  const failed: StepId[] = [];

  for (const step of steps) {
    try {
      await runStep(step);
    } catch (err) {
      failed.push(step.id);
      break;
    }
  }

  if (failed.length > 0) {
    process.exitCode = 1;
  } else {
    // eslint-disable-next-line no-console
    console.log(
      '\nAll orchestrator gating steps completed successfully (TS parity, Python contracts, short soak).\n'
    );
  }
}

main();
