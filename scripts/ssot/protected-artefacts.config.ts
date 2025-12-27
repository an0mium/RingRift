/**
 * Protected Artefacts Configuration
 *
 * This file defines the categories of protected artefacts and their validation
 * requirements for SSOT enforcement. Protected artefacts are files that should
 * not be modified without proper review and validation processes.
 *
 * @see docs/RULES_SSOT_MAP.md for detailed documentation
 */

export interface ProtectedCategory {
  /** Category name for reporting */
  name: string;
  /** Description of what this category protects */
  description: string;
  /** Protection level: HIGH requires full validation, MEDIUM requires acknowledgment */
  level: 'HIGH' | 'MEDIUM';
  /** Glob patterns for files in this category */
  patterns: string[];
  /** Validation command(s) to run when these files are modified */
  validationCommands?: string[];
  /** Human-readable explanation of validation requirements */
  validationRequirement: string;
}

/**
 * Protected artefact categories for SSOT enforcement.
 *
 * When files matching these patterns are modified, the corresponding validation
 * should pass or a warning should be issued in CI/pre-commit.
 */
export const PROTECTED_CATEGORIES: ProtectedCategory[] = [
  {
    name: 'contract-vectors',
    description: 'Cross-language contract behaviour vectors (TS↔Python)',
    level: 'HIGH',
    patterns: [
      'tests/fixtures/contract-vectors/**/*.json',
      'tests/fixtures/contract-vectors/v2/*.json',
    ],
    validationCommands: [
      'npm run test:orchestrator-parity',
      './scripts/run-python-contract-tests.sh --verbose',
    ],
    validationRequirement:
      'All contract vector modifications require: (1) TS contract tests pass, (2) Python contract tests pass, (3) parity healthcheck green',
  },
  {
    name: 'parity-infrastructure',
    description: 'Python parity healthcheck and contract test infrastructure',
    level: 'HIGH',
    patterns: [
      'ai-service/scripts/run_parity_healthcheck.py',
      'ai-service/tests/contracts/*.py',
      'ai-service/tests/contracts/**/*.py',
      'scripts/run-python-contract-tests.sh',
    ],
    validationCommands: [
      './scripts/run-python-contract-tests.sh --verbose',
      'cd ai-service && python scripts/run_parity_healthcheck.py --profile parity-healthcheck --fail-on-mismatch',
    ],
    validationRequirement:
      'Parity infrastructure modifications require all Python contract and parity tests to pass',
  },
  {
    name: 'orchestrator-configuration',
    description: 'Orchestrator rollout configuration and feature flag documentation',
    level: 'MEDIUM',
    patterns: [
      'docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md',
      'docs/drafts/ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md',
      'src/server/config/env.ts',
      'src/server/config/unified.ts',
      'src/server/services/OrchestratorRolloutService.ts',
    ],
    validationCommands: ['npm run validate:deployment', 'npm run ssot-check'],
    validationRequirement:
      'Orchestrator config changes require deployment validation and SSOT checks to pass',
  },
  {
    name: 'rules-ssot-core',
    description: 'Core TypeScript shared engine modules (semantic authority)',
    level: 'HIGH',
    patterns: [
      'src/shared/engine/core.ts',
      'src/shared/engine/types.ts',
      'src/shared/engine/orchestration/turnOrchestrator.ts',
      'src/shared/engine/orchestration/types.ts',
      'src/shared/engine/aggregates/*.ts',
    ],
    validationCommands: [
      'npm run test:ts-rules-engine',
      'npm run test:orchestrator-parity',
      './scripts/run-python-contract-tests.sh --verbose',
    ],
    validationRequirement:
      'Rules SSOT core modifications require: (1) TS rules tests pass, (2) orchestrator parity pass, (3) Python parity pass',
  },
  {
    name: 'contract-vector-generation',
    description: 'Scripts that generate contract vectors',
    level: 'HIGH',
    patterns: [
      'scripts/generate-extended-contract-vectors.ts',
      'tests/scripts/generate_rules_parity_fixtures.ts',
    ],
    validationCommands: [
      'npm run test:orchestrator-parity',
      './scripts/run-python-contract-tests.sh --verbose',
    ],
    validationRequirement:
      'Contract vector generation script changes require regeneration and validation of all vectors',
  },
  {
    name: 'python-rules-modules',
    description: 'Python rules engine implementation (parity mirror)',
    level: 'MEDIUM',
    patterns: [
      'ai-service/app/rules/*.py',
      'ai-service/app/rules/mutators/*.py',
      'ai-service/app/rules/validators/*.py',
      'ai-service/app/game_engine.py',
    ],
    validationCommands: [
      'cd ai-service && python -m pytest tests/',
      './scripts/run-python-contract-tests.sh --verbose',
    ],
    validationRequirement:
      'Python rules modifications require all Python tests and contract parity to pass',
  },
];

/**
 * Get the protection category for a given file path.
 * Returns undefined if the file is not in any protected category.
 */
export function getProtectionCategory(filePath: string): ProtectedCategory | undefined {
  // Normalize path separators
  const normalizedPath = filePath.replace(/\\/g, '/');

  for (const category of PROTECTED_CATEGORIES) {
    for (const pattern of category.patterns) {
      if (matchesGlobPattern(normalizedPath, pattern)) {
        return category;
      }
    }
  }
  return undefined;
}

/**
 * Simple glob pattern matching.
 * Supports:
 * - `*` matches any characters except `/`
 * - `**` matches any characters including `/`
 * - Exact path matching
 */
function matchesGlobPattern(path: string, pattern: string): boolean {
  // Convert glob pattern to regex
  const regexPattern = pattern
    .replace(/\./g, '\\.') // Escape dots
    .replace(/\*\*/g, '{{GLOBSTAR}}') // Temporarily replace ** to avoid double replacement
    .replace(/\*/g, '[^/]*') // Single * matches anything except /
    .replace(/\{\{GLOBSTAR\}\}/g, '.*'); // ** matches everything

  const regex = new RegExp(`^${regexPattern}$`);
  return regex.test(path);
}

/**
 * Get all protected categories that apply to a list of file paths.
 */
export function getAffectedCategories(filePaths: string[]): Map<string, ProtectedCategory> {
  const affected = new Map<string, ProtectedCategory>();

  for (const filePath of filePaths) {
    const category = getProtectionCategory(filePath);
    if (category && !affected.has(category.name)) {
      affected.set(category.name, category);
    }
  }

  return affected;
}

export default PROTECTED_CATEGORIES;
