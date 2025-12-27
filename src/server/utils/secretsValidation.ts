/**
 * Secrets validation module for RingRift.
 *
 * This module provides centralized validation of secrets and sensitive
 * configuration values. It runs at startup to fail fast if required
 * secrets are missing or insecure.
 *
 * Security considerations:
 * - Never log secret values, even when masked
 * - Don't include secrets in error messages
 * - Validate format/length without exposing values
 */

/**
 * Known placeholder secrets that are safe for development but
 * MUST NOT be used in production.
 */
export const PLACEHOLDER_SECRETS = new Set<string>([
  // JWT placeholders from .env.example and docker-compose.yml
  'your-super-secret-jwt-key-change-this-in-production',
  'your-super-secret-refresh-key-change-this-in-production',
  'change-this-secret',
  'change-this-refresh-secret',
  'changeme',
  'CHANGEME',
  // Common dev defaults
  'dev-access-token-secret',
  'dev-refresh-token-secret',
  'dev-staging-access-token-secret',
  'dev-staging-refresh-token-secret',
  'secret',
  'password',
  'test',
  'testing',
]);

/**
 * Minimum length requirements for secrets in production.
 */
export const SECRET_MIN_LENGTHS: Record<string, number> = {
  JWT_SECRET: 32,
  JWT_REFRESH_SECRET: 32,
  DB_PASSWORD: 8,
  REDIS_PASSWORD: 8,
};

/**
 * Secret definition for validation.
 */
export interface SecretDefinition {
  /** Environment variable name */
  name: string;
  /** Human-readable description */
  description: string;
  /** Required in production? */
  requiredInProduction: boolean;
  /** Required in development/test? */
  requiredInDevelopment: boolean;
  /** Minimum length for the secret (only enforced in production) */
  minLength?: number;
  /** Whether to check against placeholder values */
  checkPlaceholder: boolean;
  /** Custom validator function */
  customValidator?: (value: string) => { valid: boolean; error?: string };
}

/**
 * All known secrets in the application.
 */
export const SECRET_DEFINITIONS: SecretDefinition[] = [
  {
    name: 'JWT_SECRET',
    description: 'Secret key for signing JWT access tokens',
    requiredInProduction: true,
    requiredInDevelopment: false,
    minLength: 32,
    checkPlaceholder: true,
  },
  {
    name: 'JWT_REFRESH_SECRET',
    description: 'Secret key for signing JWT refresh tokens',
    requiredInProduction: true,
    requiredInDevelopment: false,
    minLength: 32,
    checkPlaceholder: true,
  },
  {
    name: 'DATABASE_URL',
    description: 'PostgreSQL connection string (contains credentials)',
    requiredInProduction: true,
    requiredInDevelopment: false,
    checkPlaceholder: false,
    customValidator: (value: string) => {
      if (!value.startsWith('postgresql://') && !value.startsWith('postgres://')) {
        return { valid: false, error: 'Must be a PostgreSQL connection URL' };
      }
      return { valid: true };
    },
  },
  {
    name: 'REDIS_URL',
    description: 'Redis connection string',
    requiredInProduction: true,
    requiredInDevelopment: false,
    checkPlaceholder: false,
    customValidator: (value: string) => {
      if (!value.startsWith('redis://') && !value.startsWith('rediss://')) {
        return { valid: false, error: 'Must be a Redis connection URL' };
      }
      return { valid: true };
    },
  },
  {
    name: 'REDIS_PASSWORD',
    description: 'Redis authentication password',
    requiredInProduction: false,
    requiredInDevelopment: false,
    minLength: 8,
    checkPlaceholder: true,
  },
  {
    name: 'AI_SERVICE_URL',
    description: 'URL for the AI service (FastAPI)',
    requiredInProduction: true,
    requiredInDevelopment: false,
    checkPlaceholder: false,
    customValidator: (value: string) => {
      try {
        new URL(value);
        return { valid: true };
      } catch {
        return { valid: false, error: 'Must be a valid URL' };
      }
    },
  },
];

/**
 * Result of secret validation.
 */
export interface SecretValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

/**
 * Check if a value is a known placeholder secret.
 */
export function isPlaceholderSecret(value: string | undefined | null): boolean {
  if (!value) return false;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return false;

  // Check exact matches (case-insensitive for common words)
  for (const placeholder of PLACEHOLDER_SECRETS) {
    if (normalized === placeholder.toLowerCase()) {
      return true;
    }
  }

  return false;
}

/**
 * Validate a single secret value.
 */
export function validateSecret(
  definition: SecretDefinition,
  value: string | undefined,
  isProduction: boolean
): { valid: boolean; error?: string; warning?: string } {
  const isEmpty = !value || !value.trim();
  const isRequired = isProduction
    ? definition.requiredInProduction
    : definition.requiredInDevelopment;

  // Check if required secret is missing
  if (isEmpty) {
    if (isRequired) {
      return {
        valid: false,
        error: `${definition.name} is required${isProduction ? ' in production' : ''}`,
      };
    }
    return { valid: true };
  }

  // At this point we know value is defined (isEmpty check returned early if not)
  const trimmedValue = (value ?? '').trim();

  // Check placeholder values (only in production)
  if (isProduction && definition.checkPlaceholder && isPlaceholderSecret(trimmedValue)) {
    return {
      valid: false,
      error: `${definition.name} must not use a placeholder value in production`,
    };
  }

  // Check minimum length (only in production)
  if (isProduction && definition.minLength && trimmedValue.length < definition.minLength) {
    return {
      valid: false,
      error: `${definition.name} must be at least ${definition.minLength} characters in production`,
    };
  }

  // Run custom validator
  if (definition.customValidator) {
    const customResult = definition.customValidator(trimmedValue);
    if (!customResult.valid) {
      return {
        valid: false,
        error: `${definition.name}: ${customResult.error}`,
      };
    }
  }

  // Warn about weak values in development
  if (!isProduction && definition.checkPlaceholder && isPlaceholderSecret(trimmedValue)) {
    return {
      valid: true,
      warning: `${definition.name} is using a placeholder value (acceptable for development)`,
    };
  }

  return { valid: true };
}

/**
 * Validate all secrets based on current environment.
 *
 * @param env - Environment variables to validate (defaults to process.env)
 * @param isProduction - Whether running in production mode
 * @returns Validation result with errors and warnings
 */
export function validateAllSecrets(
  env: Record<string, string | undefined> = process.env as Record<string, string | undefined>,
  isProduction: boolean
): SecretValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  for (const definition of SECRET_DEFINITIONS) {
    const value = env[definition.name];
    const result = validateSecret(definition, value, isProduction);

    if (!result.valid && result.error) {
      errors.push(result.error);
    }
    if (result.warning) {
      warnings.push(result.warning);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate secrets and throw if validation fails.
 * This is the main entry point for startup validation.
 *
 * @param isProduction - Whether running in production mode
 * @param logger - Optional logger for warnings (should mask secrets)
 * @throws Error if required secrets are missing or invalid in production
 */
export function validateSecretsOrThrow(
  isProduction: boolean,
  logger?: { warn: (msg: string) => void }
): void {
  const result = validateAllSecrets(
    process.env as Record<string, string | undefined>,
    isProduction
  );

  // Log warnings in development
  if (logger && result.warnings.length > 0) {
    for (const warning of result.warnings) {
      logger.warn(`[secrets] ${warning}`);
    }
  }

  // Throw if validation failed
  if (!result.valid) {
    const errorMessage = [
      'Secrets validation failed. The following issues must be resolved:',
      ...result.errors.map((e) => `  - ${e}`),
      '',
      'For production deployments, ensure all required secrets are set to strong, unique values.',
      'See docs/operations/SECRETS_MANAGEMENT.md for guidance.',
    ].join('\n');

    throw new Error(errorMessage);
  }
}

/**
 * Get a summary of all secrets (for documentation purposes).
 * Does NOT include actual values.
 */
export function getSecretsDocumentation(): string {
  const lines: string[] = [
    '# RingRift Secrets Reference',
    '',
    '| Name | Description | Required (Prod) | Required (Dev) | Min Length |',
    '|------|-------------|-----------------|----------------|------------|',
  ];

  for (const def of SECRET_DEFINITIONS) {
    const minLen = def.minLength ? `${def.minLength}` : '-';
    lines.push(
      `| ${def.name} | ${def.description} | ${def.requiredInProduction ? 'Yes' : 'No'} | ${def.requiredInDevelopment ? 'Yes' : 'No'} | ${minLen} |`
    );
  }

  return lines.join('\n');
}
