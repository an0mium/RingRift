/**
 * Configuration Module Index
 *
 * This module re-exports the main application configuration and
 * environment utilities for convenient access throughout the codebase.
 *
 * Usage:
 *   import { config } from './config';
 *   // or
 *   import { config, validateSecretsOrThrow } from './config';
 */

// Re-export configuration from the main config module (parent directory)
// This maintains backward compatibility with existing imports
export { config } from '../config';
export type { AppConfig } from '../config';

// Re-export environment validation utilities
export {
  EnvSchema,
  NodeEnvSchema,
  AppTopologySchema,
  RulesModeSchema,
  LogLevelSchema,
  LogFormatSchema,
  parseEnv,
  loadEnvOrExit,
  getEffectiveNodeEnv,
  isProduction,
  isStaging,
  isDevelopment,
  isTest,
  isProductionLike,
} from './env';

export type { RawEnv, EnvValidationResult, NodeEnv, AppTopology, RulesMode, LogLevel, LogFormat } from './env';

// Re-export secrets validation utilities
export {
  validateSecretsOrThrow,
  validateAllSecrets,
  validateSecret,
  isPlaceholderSecret,
  getSecretsDocumentation,
  SECRET_DEFINITIONS,
  SECRET_MIN_LENGTHS,
  PLACEHOLDER_SECRETS,
} from '../utils/secretsValidation';

export type { SecretDefinition, SecretValidationResult } from '../utils/secretsValidation';