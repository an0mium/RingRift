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
 * It also optionally records classification metrics when provided via
 * options.metrics:
 *   - contractFailures (Counter: contract_failures_total)
 *   - capacityFailures (Counter: capacity_failures_total)
 *
 * Any scenario using this helper will fail fast if login cannot be
 * established, since meaningful load testing depends on authenticated
 * requests.
 *
 * @param {string} baseUrl - Base HTTP origin, e.g. http://localhost:3001
 * @param {Object} [options]
 * @param {string} [options.apiPrefix='/api'] - API prefix to use
 * @param {Object} [options.tags] - Optional k6 tags to attach to the login request
 * @param {{ contractFailures?: any, capacityFailures?: any }} [options.metrics] - Optional
 *   classification counters to record failures against.
 * @returns {{ token: string, userId: string | null }}
 */
export function loginAndGetToken(baseUrl, options) {
  const apiPrefix = (options && options.apiPrefix) || '/api';
  const tags = (options && options.tags) || { name: 'auth-login' };
  const metrics = (options && options.metrics) || {};
  const contractFailures = metrics.contractFailures;
  const capacityFailures = metrics.capacityFailures;

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

  let parsed = null;
  let accessToken = null;
  let userId = null;
  try {
    parsed = JSON.parse(res.body);
    accessToken =
      parsed && parsed.data && typeof parsed.data.accessToken === 'string'
        ? parsed.data.accessToken
        : null;
    userId = parsed.data && parsed.data.user && parsed.data.user.id ? parsed.data.user.id : null;
  } catch (err) {
    parsed = null;
    accessToken = null;
    userId = null;
  }

  const ok = check(res, {
    'login successful': (r) => r.status === 200,
    'access token present': () => typeof accessToken === 'string',
  });

  if (!ok) {
    // Classify login failures so we can distinguish contract vs capacity issues.
    if (!res || res.status === 0) {
      if (capacityFailures) capacityFailures.add(1);
    } else if (res.status >= 400 && res.status < 500 && res.status !== 429) {
      if (contractFailures) contractFailures.add(1);
    } else if (res.status === 429 || res.status >= 500) {
      if (capacityFailures) capacityFailures.add(1);
    } else if (res.status === 200 && typeof accessToken !== 'string') {
      // 200 with a malformed body is a contract failure.
      if (contractFailures) contractFailures.add(1);
    }

    throw new Error(`loginAndGetToken failed: status=${res.status} body=${res.body}`);
  }

  return { token: accessToken, userId };
}