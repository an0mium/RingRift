/**
 * RingRift Load Test: Game Creation Scenario
 * 
 * Tests the game creation rate and latency under increasing load.
 * Validates production scale assumptions for game lobby operations.
 * 
 * Scenario from STRATEGIC_ROADMAP.md §3.1: Mixed Human vs AI Ladder
 * SLOs from STRATEGIC_ROADMAP.md §2.1: HTTP API targets
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import thresholdsConfig from '../config/thresholds.json';
import { loginAndGetToken } from '../auth/helpers.js';

// Classification metrics shared across scenarios
export const contractFailures = new Counter('contract_failures_total');
export const idLifecycleMismatches = new Counter('id_lifecycle_mismatches_total');
export const capacityFailures = new Counter('capacity_failures_total');

// Custom metrics
const gameCreationErrors = new Counter('game_creation_errors');
const gameCreationSuccess = new Rate('game_creation_success_rate');
const gameCreationLatency = new Trend('game_creation_latency_ms');

// Threshold configuration derived from thresholds.json
const THRESHOLD_ENV = __ENV.THRESHOLD_ENV || 'staging';
const perfEnv =
  thresholdsConfig.environments[THRESHOLD_ENV] || thresholdsConfig.environments.staging;
const loadTestEnv =
  thresholdsConfig.load_tests[THRESHOLD_ENV] || thresholdsConfig.load_tests.staging;
const gameCreationHttp = perfEnv.http_api.game_creation;

// Test configuration aligned with thresholds.json SLOs
export const options = {
  stages: [
    { duration: '30s', target: 10 }, // Warm up: ramp to 10 users
    { duration: '1m', target: 50 }, // Load: ramp to 50 users
    { duration: '2m', target: 50 }, // Sustain: hold at 50 users
    { duration: '30s', target: 0 }, // Ramp down
  ],

  thresholds: {
    // HTTP request duration - env-specific SLOs from thresholds.json
    http_req_duration: [
      `p(95)<${gameCreationHttp.latency_p95_ms}`,
      `p(99)<${gameCreationHttp.latency_p99_ms}`,
    ],

    // Error rate - use environment-specific 5xx error budget
    http_req_failed: [`rate<${gameCreationHttp.error_rate_5xx_percent / 100}`],

    // Custom metrics
    game_creation_success_rate: ['rate>0.99'],
    game_creation_latency_ms: [
      `p(95)<${gameCreationHttp.latency_p95_ms}`,
      `p(99)<${gameCreationHttp.latency_p99_ms}`,
    ],

    // Contract/id-lifecycle/capacity classification
    contract_failures_total: [`count<=${loadTestEnv.contract_failures_total.max}`],
    id_lifecycle_mismatches_total: [
      `count<=${loadTestEnv.id_lifecycle_mismatches_total.max}`,
    ],
    capacity_failures_total: [`rate<${loadTestEnv.capacity_failures_total.rate}`],
  },

  // Test metadata
  tags: {
    scenario: 'game-creation',
    test_type: 'load',
    environment: THRESHOLD_ENV,
  },
};

// Test configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const API_PREFIX = '/api';

/**
 * Safely parse create-game API response and unwrap { success, data: { game } }.
 */
function parseCreateGameResponse(res) {
  try {
    const body = JSON.parse(res.body);
    const game = body && body.data && body.data.game ? body.data.game : null;
    return { body, game };
  } catch (error) {
    return { body: null, game: null };
  }
}

/**
 * Classify failures for POST /api/games when the request payload is expected
 * to be valid according to the CreateGameRequest contract.
 */
function classifyCreateGameFailure(res, parsed) {
  if (!res || res.status === 0 || res.error) {
    capacityFailures.add(1);
    return;
  }

  if (res.status === 400 || res.status === 401 || res.status === 403) {
    contractFailures.add(1);
    return;
  }

  if (res.status === 429 || res.status >= 500) {
    capacityFailures.add(1);
    return;
  }

  // 2xx but missing/malformed body or game object.
  if (res.status >= 200 && res.status < 300 && (!parsed.body || !parsed.game || !parsed.game.id)) {
    contractFailures.add(1);
  }
}

/**
 * Classify failures for GET /api/games/:gameId immediately after creation.
 * A 404 here almost always indicates an ID lifecycle mismatch, since the
 * harness just created the game and has not yet hit any poll budget.
 */
function classifyImmediateGetGameFailure(res, gameId) {
  if (!res || res.status === 0 || res.error) {
    capacityFailures.add(1);
    return;
  }

  if (res.status === 400 || res.status === 401 || res.status === 403) {
    contractFailures.add(1);
    return;
  }

  if (res.status === 404) {
    idLifecycleMismatches.add(1);
    return;
  }

  if (res.status === 429 || res.status >= 500) {
    capacityFailures.add(1);
    return;
  }

  // 2xx but missing or mismatched ID.
  if (res.status >= 200 && res.status < 300) {
    contractFailures.add(1);
  }
}

/**
 * Setup function - runs once before the test and returns shared auth state
 */
export function setup() {
  console.log(`Starting game creation load test against ${BASE_URL}`);
  console.log('Target load: 50 concurrent users creating games');

  // Health check
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200,
  });

  // Login once as a pre-seeded load-test user and share the token with all VUs.
  const { token } = loginAndGetToken(BASE_URL, {
    apiPrefix: API_PREFIX,
    tags: { name: 'auth-login-setup' },
    metrics: {
      contractFailures,
      capacityFailures,
    },
  });

  return { baseUrl: BASE_URL, token };
}

/**
 * Main test function - runs repeatedly for each VU
 */
export default function(data) {
  const baseUrl = data.baseUrl;
  const token = data.token;

  const authHeaders = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`,
  };

  sleep(0.5);

  // Step 3: Create Game (main scenario focus)
  const boardTypes = ['square8', 'square19', 'hexagonal'];
  const boardType = boardTypes[Math.floor(Math.random() * boardTypes.length)];
  const maxPlayersOptions = [2, 3, 4];
  const maxPlayers = maxPlayersOptions[Math.floor(Math.random() * maxPlayersOptions.length)];

  const gameConfig = {
    name: `Load Test Game ${__VU}-${__ITER}`,
    boardType: boardType,
    maxPlayers: maxPlayers,
    isPrivate: false,
    timeControl: {
      initialTime: 300,
      increment: 5,
      type: 'blitz',
    },
    isRated: true,
  };

  const startTime = Date.now();
  const createGameRes = http.post(`${baseUrl}${API_PREFIX}/games`, JSON.stringify(gameConfig), {
    headers: authHeaders,
    tags: { name: 'create-game' },
  });
  const createGameDuration = Date.now() - startTime;

  const parsedCreate = parseCreateGameResponse(createGameRes);
  const gameCreated = check(createGameRes, {
    'status is 201': (r) => r.status === 201,
    'response payload parsed': () => parsedCreate.body !== null && parsedCreate.game !== null,
    'game ID returned': () => Boolean(parsedCreate.game && parsedCreate.game.id),
    'game config matches': () =>
      !!(
        parsedCreate.game &&
        parsedCreate.game.boardType === boardType &&
        parsedCreate.game.maxPlayers === maxPlayers
      ),
    'ai games are unrated when present': () => {
      if (!parsedCreate.game || !parsedCreate.game.aiOpponents) return true;
      const count = parsedCreate.game.aiOpponents.count || 0;
      if (count <= 0) return true;
      return parsedCreate.game.isRated === false;
    },
  });

  // Track metrics
  gameCreationLatency.add(createGameDuration);
  gameCreationSuccess.add(gameCreated);

  if (!gameCreated) {
    gameCreationErrors.add(1);
    classifyCreateGameFailure(createGameRes, parsedCreate);
    console.error(
      `Game creation failed for VU ${__VU}: ${createGameRes.status} - ${createGameRes.body}`
    );
    return;
  }

  const gameId = parsedCreate.game.id;

  sleep(0.5);

  // Step 4: Fetch Game State (validates read path)
  const getGameRes = http.get(`${baseUrl}${API_PREFIX}/games/${gameId}`, {
    headers: authHeaders,
    tags: { name: 'get-game' },
  });

  const gameStateOk = check(getGameRes, {
    'game state retrieved': (r) => r.status === 200,
    'game state valid': (r) => {
      try {
        const body = JSON.parse(r.body);
        const game = body && body.data && body.data.game ? body.data.game : null;
        return !!(game && game.id === gameId);
      } catch {
        return false;
      }
    },
  });

  if (!gameStateOk) {
    classifyImmediateGetGameFailure(getGameRes, gameId);

    const bodySnippet =
      typeof getGameRes.body === 'string' && getGameRes.body.length > 200
        ? `${getGameRes.body.substring(0, 200)}...`
        : getGameRes.body;
    console.error(
      `Game state fetch failed for VU ${__VU}: status=${getGameRes.status} body=${bodySnippet}`
    );
  }

  // Think time - simulates user reviewing game before next action
  sleep(1 + Math.random() * 2); // 1-3 seconds
}

/**
 * Teardown function - runs once after all iterations complete
 */
export function teardown(data) {
  console.log('Game creation load test complete');
  console.log('Review metrics in Grafana or k6 summary output');
}