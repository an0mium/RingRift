/**
 * RingRift Remote Smoke Load Test
 *
 * Lightweight, safe-to-run scenario intended for production/staging endpoints.
 * Uses small VU/duration defaults and the shared auth + API helpers.
 *
 * Multi-user mode:
 *   When LOADTEST_USER_POOL_SIZE is set, each VU will authenticate as a
 *   different user from the pool (loadtest_user_1 through loadtest_user_N).
 *   This distributes load across users to avoid per-user rate limits.
 *
 * Usage (single user - legacy):
 *   k6 run --insecure-skip-tls-verify \
 *     --env BASE_URL=https://<public-host-or-ip> \
 *     --env LOADTEST_EMAIL=loadtest@test.local \
 *     --env LOADTEST_PASSWORD=LoadTest123 \
 *     tests/load/scenarios/remote-smoke.js
 *
 * Usage (multi-user pool):
 *   k6 run --insecure-skip-tls-verify \
 *     --env BASE_URL=https://<public-host-or-ip> \
 *     --env LOADTEST_USER_POOL_SIZE=400 \
 *     --env LOADTEST_USER_POOL_PASSWORD=LoadTestK6Pass123 \
 *     --env VUS=100 \
 *     tests/load/scenarios/remote-smoke.js
 */

import { check, sleep } from 'k6';
import { getValidToken } from '../auth/helpers.js';
import { createGame } from '../helpers/api.js';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';

export const options = {
  vus: Number(__ENV.VUS || 10),
  duration: __ENV.DURATION || '1m',
  thresholds: {
    checks: ['rate>0.95'],
  },
  tags: {
    scenario: 'remote-smoke',
    test_type: 'smoke',
    environment: __ENV.THRESHOLD_ENV || 'production',
  },
};

export default function () {
  // Each VU logs in as its own user (when multi-user pool is configured)
  // getValidToken handles caching and token refresh per-VU
  const { token } = getValidToken(BASE_URL, {
    tags: { name: 'auth-login' },
  });

  const { res, success } = createGame(token, {
    boardType: __ENV.BOARD_TYPE || 'square8',
    maxPlayers: Number(__ENV.MAX_PLAYERS || 2),
    isPrivate: true,
    isRated: false,
    tags: { name: 'create-game' },
  });

  check(res, {
    'create-game status 201': (r) => r.status === 201,
    'create-game success envelope': () => success,
  });

  sleep(1);
}
