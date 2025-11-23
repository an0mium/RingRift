// Centralized application configuration for the Node.js backend.
// Parses and validates process.env once at startup and exposes a
// typed, runtime-validated config object for server code.
//
// This module is intentionally Node-only and should not be imported
// from client/browser bundles.

import dotenv from 'dotenv';
import { z } from 'zod';
import { getRulesMode } from '../shared/utils/envFlags';

// Load .env into process.env before we read anything from it.
dotenv.config();

const NodeEnvSchema = z.enum(['development', 'test', 'production']);
const AppTopologySchema = z.enum(['single', 'multi-unsafe', 'multi-sticky']);

 /**
  * Placeholder or example JWT secrets that are safe for local development but
  * MUST NOT be used in production. These values intentionally mirror the
  * examples in `.env.example` and `docker-compose.yml`.
  */
const PLACEHOLDER_JWT_SECRETS = new Set<string>([
  'your-super-secret-jwt-key-change-this-in-production',
  'your-super-secret-refresh-key-change-this-in-production',
  'change-this-secret',
  'change-this-refresh-secret',
  'changeme',
  'CHANGEME',
]);

function isPlaceholderJwtSecret(value: string | undefined | null): boolean {
  if (!value) return false;
  const normalized = value.trim();
  if (!normalized) return false;
  return PLACEHOLDER_JWT_SECRETS.has(normalized);
}

const EnvSchema = z.object({
  NODE_ENV: NodeEnvSchema.default('development'),
  PORT: z.coerce.number().default(3000),

  CORS_ORIGIN: z.string().default('http://localhost:5173'),
  CLIENT_URL: z.string().default('http://localhost:3000'),
  ALLOWED_ORIGINS: z.string().default('http://localhost:5173,http://localhost:3000'),

  DATABASE_URL: z.string().optional(),

  REDIS_URL: z.string().optional(),
  REDIS_PASSWORD: z.string().optional(),

  JWT_SECRET: z.string().optional(),
  JWT_REFRESH_SECRET: z.string().optional(),
  JWT_EXPIRES_IN: z.string().default('7d'),
  JWT_REFRESH_EXPIRES_IN: z.string().default('30d'),

  AI_SERVICE_URL: z.string().optional(),

  LOG_LEVEL: z.string().default('info'),
  RINGRIFT_APP_TOPOLOGY: AppTopologySchema.default('single'),

  // Exposed by npm during `npm run` invocations; used for health/version
  // endpoints. Optional and safe to default when missing (e.g. direct node
  // invocations).
  npm_package_version: z.string().optional(),
});

// Parse the raw environment with basic type coercion and defaults.
const env = EnvSchema.parse(process.env);

const nodeEnv = env.NODE_ENV;
const isProduction = nodeEnv === 'production';
const isTest = nodeEnv === 'test';
const isDevelopment = nodeEnv === 'development';

// Database URL – required in production, optional elsewhere.
const databaseUrl = env.DATABASE_URL?.trim() || undefined;
if (isProduction && !databaseUrl) {
  throw new Error('DATABASE_URL is required when NODE_ENV=production');
}

// Redis configuration – URL required in production, optional in dev/test
// (defaults to local Redis).
let redisUrl = env.REDIS_URL?.trim() || undefined;
if (!redisUrl && !isProduction) {
  redisUrl = 'redis://localhost:6379';
}
if (isProduction && !redisUrl) {
  throw new Error('REDIS_URL is required when NODE_ENV=production');
}
const redisPassword = env.REDIS_PASSWORD?.trim() || undefined;

// Auth/JWT configuration.
//
// In non-production environments we intentionally fall back to stable
// in-memory secrets so that local development and tests work even when
// JWT env vars are omitted. This mirrors the previous behaviour in
// auth middleware.
const jwtSecretFromEnv = env.JWT_SECRET?.trim() || undefined;
const jwtRefreshFromEnv = env.JWT_REFRESH_SECRET?.trim() || undefined;

let jwtSecret = jwtSecretFromEnv;
if (!jwtSecret && !isProduction) {
  jwtSecret = 'dev-access-token-secret';
}

let jwtRefreshSecret = jwtRefreshFromEnv || jwtSecretFromEnv;
if (!jwtRefreshSecret && !isProduction) {
  jwtRefreshSecret = 'dev-refresh-token-secret';
}

if (isProduction) {
  const missingOrEmpty =
    !jwtSecret ||
    !jwtSecret.trim() ||
    !jwtRefreshSecret ||
    !jwtRefreshSecret.trim();

  const usingPlaceholder =
    isPlaceholderJwtSecret(jwtSecret) || isPlaceholderJwtSecret(jwtRefreshSecret);

  if (missingOrEmpty || usingPlaceholder) {
    const problems: string[] = [];
    if (missingOrEmpty) {
      problems.push('JWT secrets must be non-empty');
    }
    if (usingPlaceholder) {
      problems.push(
        'JWT secrets must not use placeholder values from .env.example or docker-compose.yml'
      );
    }

    throw new Error(
      `Invalid JWT configuration for NODE_ENV=production: ${problems.join(
        '; '
      )}. Please set JWT_SECRET and JWT_REFRESH_SECRET to strong, unique values.`
    );
  }
}

// AI service URL – always defaults to local FastAPI service when unset.
const aiServiceUrl = (env.AI_SERVICE_URL?.trim() || 'http://localhost:8001') as string;

// Application version – driven by npm's injected env var when available.
const appVersion = env.npm_package_version?.trim() || '1.0.0';

// CORS / client origins
const corsOrigin = env.CORS_ORIGIN;
const publicClientUrl = env.CLIENT_URL;
const allowedOrigins = env.ALLOWED_ORIGINS.split(',')
  .map((v) => v.trim())
  .filter(Boolean);

// WebSocket origin follows the same precedence that was previously
// in src/server/websocket/server.ts.
const websocketOrigin =
  env.CLIENT_URL?.trim() ||
  env.CORS_ORIGIN?.trim() ||
  (allowedOrigins[0] ?? 'http://localhost:5173');

// High-level, structured config object.
const ConfigSchema = z.object({
  nodeEnv: NodeEnvSchema,
  isProduction: z.boolean(),
  isDevelopment: z.boolean(),
  isTest: z.boolean(),
  app: z.object({
    version: z.string().min(1),
    topology: AppTopologySchema,
  }),
  server: z.object({
    port: z.number().int().positive(),
    corsOrigin: z.string().min(1),
    publicClientUrl: z.string().min(1),
    allowedOrigins: z.array(z.string().min(1)).nonempty(),
    websocketOrigin: z.string().min(1),
  }),
  database: z.object({
    // Optional outside production; required in production via manual guard above.
    url: z.string().min(1).optional(),
  }),
  redis: z.object({
    url: z.string().min(1),
    password: z.string().optional(),
  }),
  auth: z.object({
    jwtSecret: z.string().min(1),
    jwtRefreshSecret: z.string().min(1),
    accessTokenExpiresIn: z.string().min(1),
    refreshTokenExpiresIn: z.string().min(1),
  }),
  aiService: z.object({
    url: z.string().url(),
  }),
  logging: z.object({
    level: z.string().min(1),
  }),
  rules: z.object({
    mode: z.union([z.literal('ts'), z.literal('python'), z.literal('shadow')]),
  }),
});

const preliminaryConfig = {
  nodeEnv,
  isProduction,
  isDevelopment,
  isTest,
  app: {
    version: appVersion,
    topology: env.RINGRIFT_APP_TOPOLOGY,
  },
  server: {
    port: env.PORT,
    corsOrigin,
    publicClientUrl,
    allowedOrigins,
    websocketOrigin,
  },
  database: {
    url: databaseUrl,
  },
  redis: {
    url: redisUrl as string,
    password: redisPassword,
  },
  auth: {
    jwtSecret: jwtSecret as string,
    jwtRefreshSecret: jwtRefreshSecret as string,
    accessTokenExpiresIn: env.JWT_EXPIRES_IN,
    refreshTokenExpiresIn: env.JWT_REFRESH_EXPIRES_IN,
  },
  aiService: {
    url: aiServiceUrl,
  },
  logging: {
    level: env.LOG_LEVEL,
  },
  rules: {
    mode: getRulesMode(),
  },
};

export type AppConfig = z.infer<typeof ConfigSchema>;

// Parse and freeze the final config so downstream code gets a fully
// validated, immutable view.
export const config: AppConfig = Object.freeze(ConfigSchema.parse(preliminaryConfig));