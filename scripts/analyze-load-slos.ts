// Minimal aggregation helper for k6 load/SLO summaries.
//
// Reads all *.summary.json files under results/load/ (or K6_SUMMARY_DIR if set),
// computes an overall go/no-go decision across scenarios, and writes a compact
// JSON artifact plus a small table to stdout.
//
// Usage (from repo root, after running the k6 scenarios):
//
//   # Using default results/load directory
//   npx ts-node scripts/analyze-load-slos.ts
//
//   # Or with a custom summary directory
//   K6_SUMMARY_DIR=results/load-staging npx ts-node scripts/analyze-load-slos.ts
//

import * as fs from 'fs';
import * as path from 'path';

interface ThresholdStatus {
  metric: string;
  threshold: string;
  statistic: string | null;
  comparison: string | null;
  limit: number | null;
  value: number | null;
  passed: boolean;
  // Allow extra fields from the per-scenario summaries without constraining them here.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

interface ScenarioSloStatus {
  scenario: string;
  environment: string;
  thresholds: ThresholdStatus[];
  overallPass: boolean;
  sourceFile: string;
  // Allow passthrough of additional fields if present.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

interface RunSloSummary {
  runTimestamp: string;
  environment: string;
  scenarios: ScenarioSloStatus[];
  overallPass: boolean;
}

/**
 * Read and parse JSON from a file, with a concise error message on failure.
 */
async function readJsonFile(filePath: string): Promise<any | null> {
  try {
    const raw = await fs.promises.readFile(filePath, 'utf8');
    return JSON.parse(raw);
  } catch (err) {
    console.error(`Failed to read or parse JSON from ${filePath}:`, err);
    return null;
  }
}

/**
 * Derive a ScenarioSloStatus from a single per-scenario summary JSON object.
 */
function toScenarioStatus(filePath: string, json: any): ScenarioSloStatus | null {
  if (!json || typeof json !== 'object') {
    return null;
  }

  const scenario: string =
    json.scenario ?? json.slo?.scenario ?? path.basename(filePath).replace(/\.summary\.json$/, '');

  const environment: string =
    json.environment ?? json.slo?.environment ?? json.thresholdsEnv ?? 'unknown';

  const thresholds: ThresholdStatus[] =
    (Array.isArray(json.thresholds) ? json.thresholds : json.slo?.thresholds) ?? [];

  let overallPass: boolean;
  if (typeof json.overallPass === 'boolean') {
    overallPass = json.overallPass;
  } else if (json.slo && typeof json.slo.overallPass === 'boolean') {
    overallPass = json.slo.overallPass;
  } else if (thresholds.length > 0) {
    overallPass = thresholds.every((t) => t.passed);
  } else {
    overallPass = false;
  }

  return {
    scenario,
    environment,
    thresholds,
    overallPass,
    sourceFile: filePath,
  };
}

async function main(): Promise<void> {
  const baseDir = process.env.K6_SUMMARY_DIR || 'results/load';

  let entries: string[];
  try {
    entries = await fs.promises.readdir(baseDir);
  } catch (err) {
    console.error(
      `Unable to read summary directory "${baseDir}". Have you run the k6 scenarios?`,
      err
    );
    process.exitCode = 1;
    return;
  }

  const summaryFiles = entries
    .filter((name) => name.endsWith('.summary.json'))
    .map((name) => path.join(baseDir, name))
    .sort();

  if (summaryFiles.length === 0) {
    console.error(`No *.summary.json files found under ${baseDir}. Nothing to analyze.`);
    process.exitCode = 1;
    return;
  }

  const scenarios: ScenarioSloStatus[] = [];

  for (const filePath of summaryFiles) {
    const json = await readJsonFile(filePath);
    const status = toScenarioStatus(filePath, json);
    if (status) {
      scenarios.push(status);
    }
  }

  if (scenarios.length === 0) {
    console.error('No valid scenario summaries could be parsed. Aborting.');
    process.exitCode = 1;
    return;
  }

  const envSet = new Set(scenarios.map((s) => s.environment));
  const environment = envSet.size === 1 ? Array.from(envSet)[0] : 'mixed';

  const runTimestamp = new Date().toISOString();
  const overallPass = scenarios.every((s) => s.overallPass);

  const runSummary: RunSloSummary = {
    runTimestamp,
    environment,
    scenarios,
    overallPass,
  };

  const outPath = path.join(baseDir, 'load_slo_summary.json');
  try {
    await fs.promises.writeFile(outPath, JSON.stringify(runSummary, null, 2), 'utf8');
  } catch (err) {
    console.error(`Failed to write aggregated run summary to ${outPath}:`, err);
    process.exitCode = 1;
    return;
  }

  // Human-friendly output for CI logs and local runs.
  // eslint-disable-next-line no-console
  console.log(`Wrote aggregated load SLO summary to ${outPath}`);
  // eslint-disable-next-line no-console
  console.table(
    scenarios.map((s) => ({
      scenario: s.scenario,
      environment: s.environment,
      overallPass: s.overallPass,
    }))
  );
  // eslint-disable-next-line no-console
  console.log(
    `Overall load test SLO result: ${overallPass ? 'GO (all scenarios passed)' : 'NO-GO (one or more scenarios failed)'}`
  );
}

main().catch((err) => {
  console.error('Unexpected error while analyzing load SLO summaries:', err);
  process.exitCode = 1;
});
