#!/usr/bin/env ts-node
/**
 * Parity Metrics Check
 *
 * Computes contract vector + parity fixture counts and verifies that
 * docs/production/PRODUCTION_READINESS_CHECKLIST.md stays in sync.
 */

import fs from 'fs';
import path from 'path';

type CountResult = {
  contractVectors: number;
  parityVectorSteps: number;
  rulesParityFixtures: number;
  paritySnapshots: number;
  parityFixturesTotal: number;
};

function listFiles(dirPath: string, pattern: RegExp): string[] {
  if (!fs.existsSync(dirPath)) return [];
  const results: string[] = [];
  const entries = fs.readdirSync(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      results.push(...listFiles(fullPath, pattern));
    } else if (pattern.test(entry.name)) {
      results.push(fullPath);
    }
  }
  return results;
}

function readJson(filePath: string): unknown {
  const raw = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(raw);
}

function countVectorsInBundle(bundle: unknown): number {
  if (Array.isArray(bundle)) {
    return bundle.length;
  }
  if (bundle && typeof bundle === 'object') {
    const asRecord = bundle as Record<string, unknown>;
    if (Array.isArray(asRecord.vectors)) {
      return asRecord.vectors.length;
    }
    if (Array.isArray(asRecord.steps)) {
      return asRecord.steps.length;
    }
  }
  return 1;
}

function computeCounts(projectRoot: string): CountResult {
  const contractVectorFiles = listFiles(
    path.join(projectRoot, 'tests', 'fixtures', 'contract-vectors', 'v2'),
    /\.vectors\.json$/
  );
  const contractVectors = contractVectorFiles.reduce((sum, filePath) => {
    return sum + countVectorsInBundle(readJson(filePath));
  }, 0);

  const parityVectorFiles = listFiles(
    path.join(projectRoot, 'ai-service', 'tests', 'parity', 'vectors'),
    /\.json$/
  );
  const parityVectorSteps = parityVectorFiles.reduce((sum, filePath) => {
    return sum + countVectorsInBundle(readJson(filePath));
  }, 0);

  const rulesParityFixtures = listFiles(
    path.join(projectRoot, 'tests', 'fixtures', 'rules-parity'),
    /\.json$/
  ).length;

  const paritySnapshots = listFiles(
    path.join(projectRoot, 'ai-service', 'tests', 'parity'),
    /\.snapshot\.json$/
  ).length;

  return {
    contractVectors,
    parityVectorSteps,
    rulesParityFixtures,
    paritySnapshots,
    parityFixturesTotal: parityVectorSteps + rulesParityFixtures + paritySnapshots,
  };
}

function checkChecklist(projectRoot: string, counts: CountResult): string[] {
  const problems: string[] = [];
  const checklistPath = path.join(
    projectRoot,
    'docs',
    'production',
    'PRODUCTION_READINESS_CHECKLIST.md'
  );

  if (!fs.existsSync(checklistPath)) {
    problems.push('Checklist not found: docs/production/PRODUCTION_READINESS_CHECKLIST.md');
    return problems;
  }

  const content = fs.readFileSync(checklistPath, 'utf8');

  const parityLine = content.match(
    /TS\/Python parity verified[^\n]*?(\d+) contract vectors, (\d+) parity fixtures/
  );
  if (parityLine) {
    const contractCount = Number(parityLine[1]);
    const fixtureCount = Number(parityLine[2]);
    if (contractCount !== counts.contractVectors) {
      problems.push(
        `Checklist parity row contract vectors=${contractCount}, expected ${counts.contractVectors}`
      );
    }
    if (fixtureCount !== counts.parityFixturesTotal) {
      problems.push(
        `Checklist parity row parity fixtures=${fixtureCount}, expected ${counts.parityFixturesTotal}`
      );
    }
  } else {
    problems.push('Checklist parity row not found (TS/Python parity verified).');
  }

  const parityVerification = content.match(
    /Parity verification[^\n]*?(\d+)\/(\d+) contract vectors, (\d+) parity fixtures/
  );
  if (parityVerification) {
    const contractLeft = Number(parityVerification[1]);
    const contractRight = Number(parityVerification[2]);
    const fixtures = Number(parityVerification[3]);
    if (contractLeft !== counts.contractVectors || contractRight !== counts.contractVectors) {
      problems.push(
        `Checklist parity verification row has ${contractLeft}/${contractRight}, expected ${counts.contractVectors}/${counts.contractVectors}`
      );
    }
    if (fixtures !== counts.parityFixturesTotal) {
      problems.push(
        `Checklist parity verification row parity fixtures=${fixtures}, expected ${counts.parityFixturesTotal}`
      );
    }
  } else {
    problems.push('Checklist parity verification row not found.');
  }

  const gateMatches = [...content.matchAll(/Contract vectors (\d+)\/(\d+)/g)];
  if (gateMatches.length > 0) {
    for (const match of gateMatches) {
      const left = Number(match[1]);
      const right = Number(match[2]);
      if (left !== counts.contractVectors || right !== counts.contractVectors) {
        problems.push(
          `Checklist Contract vectors ${left}/${right} mismatch (expected ${counts.contractVectors}/${counts.contractVectors})`
        );
      }
    }
  }

  const noteMatch = content.match(
    /parity fixtures count \((\d+)\) = (\d+) ai-service parity vector steps \+ (\d+) rules-parity JSON fixtures \+ (\d+) snapshot parity fixtures/
  );
  if (noteMatch) {
    const total = Number(noteMatch[1]);
    const paritySteps = Number(noteMatch[2]);
    const rulesParity = Number(noteMatch[3]);
    const snapshots = Number(noteMatch[4]);
    if (total !== counts.parityFixturesTotal) {
      problems.push(
        `Checklist parity fixtures total=${total}, expected ${counts.parityFixturesTotal}`
      );
    }
    if (paritySteps !== counts.parityVectorSteps) {
      problems.push(
        `Checklist parity vector steps=${paritySteps}, expected ${counts.parityVectorSteps}`
      );
    }
    if (rulesParity !== counts.rulesParityFixtures) {
      problems.push(
        `Checklist rules-parity fixtures=${rulesParity}, expected ${counts.rulesParityFixtures}`
      );
    }
    if (snapshots !== counts.paritySnapshots) {
      problems.push(`Checklist snapshot fixtures=${snapshots}, expected ${counts.paritySnapshots}`);
    }
  } else {
    problems.push('Checklist parity fixtures note not found.');
  }

  return problems;
}

function main(): void {
  const projectRoot = process.cwd();
  const counts = computeCounts(projectRoot);
  const problems = checkChecklist(projectRoot, counts);

  if (problems.length > 0) {
    for (const problem of problems) {
      console.error(`[parity-metrics] ${problem}`);
    }

    console.error(
      `[parity-metrics] Expected: ${counts.contractVectors} contract vectors, ${counts.parityFixturesTotal} parity fixtures (steps=${counts.parityVectorSteps}, rules=${counts.rulesParityFixtures}, snapshots=${counts.paritySnapshots}).`
    );
    process.exitCode = 1;
    return;
  }

  // eslint-disable-next-line no-console
  console.log(
    `[parity-metrics] OK: ${counts.contractVectors} contract vectors, ${counts.parityFixturesTotal} parity fixtures (steps=${counts.parityVectorSteps}, rules=${counts.rulesParityFixtures}, snapshots=${counts.paritySnapshots}).`
  );
}

main();
