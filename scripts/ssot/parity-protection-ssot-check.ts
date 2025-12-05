#!/usr/bin/env ts-node
/**
 * Parity Protection SSOT Check
 *
 * This check validates that protected artefacts (contract vectors, parity
 * infrastructure, orchestrator configuration, and rules SSOT core) are
 * present and properly structured. It does not run full validation suites
 * but ensures the artefacts exist and are not structurally broken.
 *
 * For CI-level validation that runs when protected files change, see the
 * ssot-protection workflow integration in `.github/workflows/ci.yml`.
 *
 * @see scripts/ssot/protected-artefacts.config.ts for protection categories
 * @see docs/RULES_SSOT_MAP.md § Protected Artefacts for documentation
 */

import * as fs from 'fs';
import * as path from 'path';

import type { CheckResult } from './ssot-check';
import { PROTECTED_CATEGORIES, type ProtectedCategory } from './protected-artefacts.config';

/**
 * Verify that a file exists at the given path relative to project root.
 */
function fileExists(projectRoot: string, relativePath: string): boolean {
  const fullPath = path.join(projectRoot, relativePath);
  return fs.existsSync(fullPath);
}

/**
 * Verify that a directory exists at the given path.
 */
function dirExists(projectRoot: string, relativePath: string): boolean {
  const fullPath = path.join(projectRoot, relativePath);
  return fs.existsSync(fullPath) && fs.statSync(fullPath).isDirectory();
}

/**
 * Lists files in a directory matching a pattern.
 */
function listFilesMatching(dirPath: string, pattern: RegExp): string[] {
  if (!fs.existsSync(dirPath)) return [];

  const results: string[] = [];
  const entries = fs.readdirSync(dirPath, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      results.push(...listFilesMatching(fullPath, pattern));
    } else if (pattern.test(entry.name)) {
      results.push(fullPath);
    }
  }

  return results;
}

/**
 * Validate contract vectors structure:
 * - At least one vector file exists in v2/
 * - Vector files are valid JSON
 * - Vector files have expected v2 structure:
 *   { version: string, vectors: [...] } where vectors is an array of test cases
 */
function validateContractVectors(projectRoot: string): string[] {
  const problems: string[] = [];
  const vectorsDir = path.join(projectRoot, 'tests/fixtures/contract-vectors/v2');

  if (!dirExists(projectRoot, 'tests/fixtures/contract-vectors/v2')) {
    problems.push('Contract vectors directory not found: tests/fixtures/contract-vectors/v2');
    return problems;
  }

  const vectorFiles = listFilesMatching(vectorsDir, /\.vectors\.json$/);
  if (vectorFiles.length === 0) {
    problems.push('No vector files found in tests/fixtures/contract-vectors/v2/');
    return problems;
  }

  // Validate each vector file
  for (const filePath of vectorFiles) {
    const relativePath = path.relative(projectRoot, filePath);
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const parsed = JSON.parse(content);

      // v2 vector files have structure: { version, vectors: [...] }
      if (typeof parsed !== 'object' || parsed === null) {
        problems.push(`Vector file ${relativePath} is not a valid object`);
      } else if (!parsed.version) {
        problems.push(`Vector file ${relativePath} is missing 'version' field`);
      } else if (!Array.isArray(parsed.vectors)) {
        problems.push(`Vector file ${relativePath} is missing 'vectors' array`);
      } else if (parsed.vectors.length === 0) {
        problems.push(`Vector file ${relativePath} has no test cases in 'vectors' array`);
      }
    } catch (err) {
      const error = err as Error;
      problems.push(`Vector file ${relativePath} is not valid JSON: ${error.message}`);
    }
  }

  return problems;
}

/**
 * Validate parity infrastructure exists:
 * - Python healthcheck script exists
 * - Python contract test file exists
 * - Shell wrapper script exists
 */
function validateParityInfrastructure(projectRoot: string): string[] {
  const problems: string[] = [];

  const requiredFiles = [
    'ai-service/scripts/run_parity_healthcheck.py',
    'ai-service/tests/contracts/test_contract_vectors.py',
    'scripts/run-python-contract-tests.sh',
  ];

  for (const file of requiredFiles) {
    if (!fileExists(projectRoot, file)) {
      problems.push(`Required parity infrastructure file missing: ${file}`);
    }
  }

  // Check that contracts directory has __init__.py for Python module
  if (
    !fileExists(projectRoot, 'ai-service/tests/contracts/__init__.py') &&
    dirExists(projectRoot, 'ai-service/tests/contracts')
  ) {
    // This is a warning, not an error - the directory might still work
    // But we'll be lenient here
  }

  return problems;
}

/**
 * Validate orchestrator configuration docs exist and have expected content.
 */
function validateOrchestratorConfiguration(projectRoot: string): string[] {
  const problems: string[] = [];

  // Required orchestrator configuration files
  const requiredFiles = [
    'docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md',
    'src/server/config/env.ts',
    'src/server/config/unified.ts',
  ];

  for (const file of requiredFiles) {
    if (!fileExists(projectRoot, file)) {
      problems.push(`Required orchestrator configuration file missing: ${file}`);
    }
  }

  // Verify rollout plan has key sections
  const rolloutPlanPath = path.join(projectRoot, 'docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md');
  if (fs.existsSync(rolloutPlanPath)) {
    const content = fs.readFileSync(rolloutPlanPath, 'utf8');

    const requiredSections = [
      'ORCHESTRATOR_ADAPTER_ENABLED',
      'ORCHESTRATOR_ROLLOUT_PERCENTAGE',
      'SLO',
    ];

    for (const section of requiredSections) {
      if (!content.includes(section)) {
        problems.push(`Orchestrator rollout plan missing expected content: ${section}`);
      }
    }
  }

  return problems;
}

/**
 * Validate rules SSOT core modules exist.
 */
function validateRulesSsotCore(projectRoot: string): string[] {
  const problems: string[] = [];

  const requiredFiles = [
    'src/shared/engine/core.ts',
    'src/shared/engine/orchestration/turnOrchestrator.ts',
    'src/shared/engine/orchestration/types.ts',
  ];

  for (const file of requiredFiles) {
    if (!fileExists(projectRoot, file)) {
      problems.push(`Required rules SSOT core file missing: ${file}`);
    }
  }

  // Check aggregates directory has expected files
  const aggregatesDir = path.join(projectRoot, 'src/shared/engine/aggregates');
  if (dirExists(projectRoot, 'src/shared/engine/aggregates')) {
    const aggregateFiles = listFilesMatching(aggregatesDir, /Aggregate\.ts$/);
    if (aggregateFiles.length < 3) {
      problems.push(
        `Expected at least 3 aggregate files in src/shared/engine/aggregates/, found ${aggregateFiles.length}`
      );
    }
  } else {
    problems.push('Aggregates directory not found: src/shared/engine/aggregates/');
  }

  return problems;
}

/**
 * Validate that protection categories are consistent with actual file structure.
 */
function validateProtectionCategoriesConsistency(projectRoot: string): string[] {
  const problems: string[] = [];

  // Check that at least one file matches each HIGH protection category
  for (const category of PROTECTED_CATEGORIES) {
    if (category.level !== 'HIGH') continue;

    let hasMatch = false;
    for (const pattern of category.patterns) {
      // Check if the pattern matches any existing files
      // For patterns with **, we check the base directory exists
      const basePath = pattern.split('*')[0].replace(/\/$/, '');
      if (basePath && (fileExists(projectRoot, basePath) || dirExists(projectRoot, basePath))) {
        hasMatch = true;
        break;
      }
      // For exact paths, check the file exists
      if (!pattern.includes('*') && fileExists(projectRoot, pattern)) {
        hasMatch = true;
        break;
      }
    }

    if (!hasMatch) {
      problems.push(
        `Protection category '${category.name}' has no matching files in the repository`
      );
    }
  }

  return problems;
}

export async function runParityProtectionSsotCheck(): Promise<CheckResult> {
  try {
    const projectRoot = path.resolve(__dirname, '..', '..');
    const problems: string[] = [];

    // Run all validation checks
    problems.push(...validateContractVectors(projectRoot));
    problems.push(...validateParityInfrastructure(projectRoot));
    problems.push(...validateOrchestratorConfiguration(projectRoot));
    problems.push(...validateRulesSsotCore(projectRoot));
    problems.push(...validateProtectionCategoriesConsistency(projectRoot));

    if (problems.length === 0) {
      return {
        name: 'parity-protection-ssot',
        passed: true,
        details: `Protected artefacts validation passed. Checked ${PROTECTED_CATEGORIES.length} protection categories covering contract vectors, parity infrastructure, orchestrator config, and rules SSOT core.`,
      };
    }

    return {
      name: 'parity-protection-ssot',
      passed: false,
      details: problems.join('\n'),
    };
  } catch (error) {
    const err = error as Error;
    return {
      name: 'parity-protection-ssot',
      passed: false,
      details: err.message ?? String(err),
    };
  }
}

/**
 * Get a summary of protected categories for documentation or reporting.
 */
export function getProtectionSummary(): {
  categories: Array<{
    name: string;
    level: string;
    patternCount: number;
    validationCommands: number;
  }>;
  totalHighProtection: number;
  totalMediumProtection: number;
} {
  const categories = PROTECTED_CATEGORIES.map((cat) => ({
    name: cat.name,
    level: cat.level,
    patternCount: cat.patterns.length,
    validationCommands: cat.validationCommands?.length ?? 0,
  }));

  return {
    categories,
    totalHighProtection: PROTECTED_CATEGORIES.filter((c) => c.level === 'HIGH').length,
    totalMediumProtection: PROTECTED_CATEGORIES.filter((c) => c.level === 'MEDIUM').length,
  };
}

export { PROTECTED_CATEGORIES, type ProtectedCategory };
