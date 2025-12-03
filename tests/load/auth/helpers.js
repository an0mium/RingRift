import http from 'k6/http';
import { check } from 'k6';

/**
 * Shared login helper for all RingRift k6 scenarios.
 *
 * This helper:
 * - Calls POST /api/auth/login with the canonical payload shape:
 *     { email, password }
 * - Allows overriding credentials via LOADTEST_EMAIL / LOADTEST_PASSWORD
 * - Returns { token, userId } on success
 *
 * Any scenario using this helper will fail fast if login cannot be
 * established, since meaningful load testing depends on authenticated
 * requests.
 *
 * @param {string} baseUrl - Base HTTP origin, e.g. http://localhost:3001
 * @param {Object} [options]
 * @param {string} [options.apiPrefix='/api'] - API prefix to use
 * @param {Object} [options.tags] - Optional k6 tags to attach to the login request
 * @returns {{ token: string, userId: string | null }}
 */
export function loginAndGetToken(baseUrl, options) {
  const apiPrefix = (options && options.apiPrefix) || '/api';
  const tags = (options && options.tags) || { name: 'auth-login' };

  const email = __ENV.LOADTEST_EMAIL || 'loadtest_user_1@loadtest.local';
  const password = __ENV.LOADTEST_PASSWORD || 'TestPassword123!';

  const res = http.post(
    `${baseUrl}${apiPrefix}/auth/login`,
    JSON.stringify({ email, password }),
    {
      headers: { 'Content-Type': 'application/json' },
      tags,
    }
  );

  const ok = check(res, {
    'login successful': (r) => r.status === 200,
    'access token present': (r) => {
      try {
        const body = JSON.parse(r.body);
        return !!(body && body.data && typeof body.data.accessToken === 'string');
      } catch {
        return false;
      }
    },
  });

  if (!ok) {
    throw new Error(`loginAndGetToken failed: status=${res.status} body=${res.body}`);
  }

  const parsed = JSON.parse(res.body);
  const token = parsed.data.accessToken;
  const userId = parsed.data.user && parsed.data.user.id ? parsed.data.user.id : null;

  return { token, userId };
}