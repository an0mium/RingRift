/**
 * Tests for the secrets validation module.
 *
 * These tests verify that:
 * 1. Placeholder secrets are detected correctly
 * 2. Minimum length requirements are enforced in production
 * 3. Required secrets fail validation when missing
 * 4. Custom validators work correctly
 * 5. Development mode has appropriate fallbacks
 */

import {
  PLACEHOLDER_SECRETS,
  isPlaceholderSecret,
  validateSecret,
  validateAllSecrets,
  SECRET_DEFINITIONS,
  type SecretDefinition,
} from '../../src/server/utils/secretsValidation';

describe('secretsValidation', () => {
  describe('PLACEHOLDER_SECRETS', () => {
    it('contains known placeholder values from .env.example', () => {
      expect(PLACEHOLDER_SECRETS.has('your-super-secret-jwt-key-change-this-in-production')).toBe(
        true
      );
      expect(
        PLACEHOLDER_SECRETS.has('your-super-secret-refresh-key-change-this-in-production')
      ).toBe(true);
      expect(PLACEHOLDER_SECRETS.has('change-this-secret')).toBe(true);
      expect(PLACEHOLDER_SECRETS.has('changeme')).toBe(true);
    });

    it('contains dev fallback secrets', () => {
      expect(PLACEHOLDER_SECRETS.has('dev-access-token-secret')).toBe(true);
      expect(PLACEHOLDER_SECRETS.has('dev-refresh-token-secret')).toBe(true);
    });
  });

  describe('isPlaceholderSecret', () => {
    it('returns false for null/undefined', () => {
      expect(isPlaceholderSecret(null)).toBe(false);
      expect(isPlaceholderSecret(undefined)).toBe(false);
    });

    it('returns false for empty string', () => {
      expect(isPlaceholderSecret('')).toBe(false);
      expect(isPlaceholderSecret('   ')).toBe(false);
    });

    it('returns true for known placeholders (case-insensitive)', () => {
      expect(isPlaceholderSecret('changeme')).toBe(true);
      expect(isPlaceholderSecret('CHANGEME')).toBe(true);
      expect(isPlaceholderSecret('Changeme')).toBe(true);
      expect(isPlaceholderSecret('secret')).toBe(true);
      expect(isPlaceholderSecret('SECRET')).toBe(true);
    });

    it('returns false for legitimate secrets', () => {
      expect(isPlaceholderSecret('a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6')).toBe(false);
      expect(isPlaceholderSecret('xK9mP2vL8nQ4wR6tY0uI3oA5sD7fG1hJ')).toBe(false);
    });

    it('handles whitespace correctly', () => {
      expect(isPlaceholderSecret('  changeme  ')).toBe(true);
      expect(isPlaceholderSecret('\tpassword\n')).toBe(true);
    });
  });

  describe('validateSecret', () => {
    const jwtSecretDef: SecretDefinition = {
      name: 'JWT_SECRET',
      description: 'Test secret',
      requiredInProduction: true,
      requiredInDevelopment: false,
      minLength: 32,
      checkPlaceholder: true,
    };

    describe('in production mode', () => {
      const isProduction = true;

      it('fails for missing required secret', () => {
        const result = validateSecret(jwtSecretDef, undefined, isProduction);
        expect(result.valid).toBe(false);
        expect(result.error).toContain('JWT_SECRET is required');
      });

      it('fails for empty required secret', () => {
        const result = validateSecret(jwtSecretDef, '', isProduction);
        expect(result.valid).toBe(false);
        expect(result.error).toContain('required');
      });

      it('fails for placeholder values', () => {
        const result = validateSecret(jwtSecretDef, 'changeme', isProduction);
        expect(result.valid).toBe(false);
        expect(result.error).toContain('placeholder');
      });

      it('fails for values under minimum length', () => {
        const result = validateSecret(jwtSecretDef, 'short-secret-only-20ch', isProduction);
        expect(result.valid).toBe(false);
        expect(result.error).toContain('at least 32 characters');
      });

      it('passes for valid secrets', () => {
        const result = validateSecret(
          jwtSecretDef,
          'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8',
          isProduction
        );
        expect(result.valid).toBe(true);
        expect(result.error).toBeUndefined();
      });
    });

    describe('in development mode', () => {
      const isProduction = false;

      it('passes for missing non-required secret', () => {
        const result = validateSecret(jwtSecretDef, undefined, isProduction);
        expect(result.valid).toBe(true);
      });

      it('warns about placeholder values', () => {
        const result = validateSecret(jwtSecretDef, 'changeme', isProduction);
        expect(result.valid).toBe(true);
        expect(result.warning).toContain('placeholder');
      });

      it('does not enforce minimum length', () => {
        const result = validateSecret(jwtSecretDef, 'short', isProduction);
        expect(result.valid).toBe(true);
      });
    });

    describe('custom validators', () => {
      const urlSecretDef: SecretDefinition = {
        name: 'SERVICE_URL',
        description: 'Service URL',
        requiredInProduction: true,
        requiredInDevelopment: false,
        checkPlaceholder: false,
        customValidator: (value: string) => {
          try {
            new URL(value);
            return { valid: true };
          } catch {
            return { valid: false, error: 'Invalid URL format' };
          }
        },
      };

      it('runs custom validator on non-empty values', () => {
        const result = validateSecret(urlSecretDef, 'not-a-url', true);
        expect(result.valid).toBe(false);
        expect(result.error).toContain('Invalid URL format');
      });

      it('passes valid URLs', () => {
        const result = validateSecret(urlSecretDef, 'http://localhost:8001', true);
        expect(result.valid).toBe(true);
      });
    });
  });

  describe('validateAllSecrets', () => {
    describe('in production mode', () => {
      it('fails when required secrets are missing', () => {
        const env = {};
        const result = validateAllSecrets(env, true);

        expect(result.valid).toBe(false);
        expect(result.errors.length).toBeGreaterThan(0);
        expect(result.errors.some((e) => e.includes('JWT_SECRET'))).toBe(true);
        expect(result.errors.some((e) => e.includes('DATABASE_URL'))).toBe(true);
      });

      it('fails for placeholder JWT secrets', () => {
        const env = {
          JWT_SECRET: 'your-super-secret-jwt-key-change-this-in-production',
          JWT_REFRESH_SECRET: 'your-super-secret-refresh-key-change-this-in-production',
          DATABASE_URL: 'postgresql://user:pass@localhost:5432/db',
          REDIS_URL: 'redis://localhost:6379',
          AI_SERVICE_URL: 'http://localhost:8001',
        };
        const result = validateAllSecrets(env, true);

        expect(result.valid).toBe(false);
        expect(result.errors.some((e) => e.includes('placeholder'))).toBe(true);
      });

      it('fails for short JWT secrets', () => {
        const env = {
          JWT_SECRET: 'tooshort123',
          JWT_REFRESH_SECRET: 'tooshort456',
          DATABASE_URL: 'postgresql://user:pass@localhost:5432/db',
          REDIS_URL: 'redis://localhost:6379',
          AI_SERVICE_URL: 'http://localhost:8001',
        };
        const result = validateAllSecrets(env, true);

        expect(result.valid).toBe(false);
        expect(result.errors.some((e) => e.includes('at least 32 characters'))).toBe(true);
      });

      it('passes with all valid secrets', () => {
        const env = {
          JWT_SECRET: 'a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0',
          JWT_REFRESH_SECRET: 'z9y8x7w6v5u4t3s2r1q0p9o8n7m6l5k4j3i2h1g0',
          DATABASE_URL: 'postgresql://user:pass@localhost:5432/db',
          REDIS_URL: 'redis://localhost:6379',
          AI_SERVICE_URL: 'http://localhost:8001',
        };
        const result = validateAllSecrets(env, true);

        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });
    });

    describe('in development mode', () => {
      it('passes with missing secrets', () => {
        const env = {};
        const result = validateAllSecrets(env, false);

        expect(result.valid).toBe(true);
        expect(result.errors).toHaveLength(0);
      });

      it('generates warnings for placeholder values', () => {
        const env = {
          JWT_SECRET: 'changeme',
        };
        const result = validateAllSecrets(env, false);

        expect(result.valid).toBe(true);
        expect(result.warnings.length).toBeGreaterThan(0);
        expect(result.warnings.some((w) => w.includes('placeholder'))).toBe(true);
      });
    });
  });

  describe('SECRET_DEFINITIONS', () => {
    it('includes all critical production secrets', () => {
      const requiredSecrets = ['JWT_SECRET', 'JWT_REFRESH_SECRET', 'DATABASE_URL', 'AI_SERVICE_URL'];

      for (const name of requiredSecrets) {
        const def = SECRET_DEFINITIONS.find((d) => d.name === name);
        expect(def).toBeDefined();
        expect(def?.requiredInProduction).toBe(true);
      }
    });

    it('has JWT secrets with minimum length requirements', () => {
      const jwtSecret = SECRET_DEFINITIONS.find((d) => d.name === 'JWT_SECRET');
      const jwtRefresh = SECRET_DEFINITIONS.find((d) => d.name === 'JWT_REFRESH_SECRET');

      expect(jwtSecret?.minLength).toBeGreaterThanOrEqual(32);
      expect(jwtRefresh?.minLength).toBeGreaterThanOrEqual(32);
    });

    it('has placeholder checking enabled for JWT secrets', () => {
      const jwtSecret = SECRET_DEFINITIONS.find((d) => d.name === 'JWT_SECRET');
      const jwtRefresh = SECRET_DEFINITIONS.find((d) => d.name === 'JWT_REFRESH_SECRET');

      expect(jwtSecret?.checkPlaceholder).toBe(true);
      expect(jwtRefresh?.checkPlaceholder).toBe(true);
    });

    it('has URL validators for connection strings', () => {
      const dbUrl = SECRET_DEFINITIONS.find((d) => d.name === 'DATABASE_URL');
      expect(dbUrl?.customValidator).toBeDefined();

      // Test the validator
      if (dbUrl?.customValidator) {
        expect(dbUrl.customValidator('postgresql://localhost/db').valid).toBe(true);
        expect(dbUrl.customValidator('not-a-url').valid).toBe(false);
      }
    });
  });
});