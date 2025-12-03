/**
 * RingRift Load Test: Concurrent Games Scenario
 * 
 * Tests system behavior with 100+ simultaneous games.
 * Validates production scale assumptions for resource usage and state management.
 * 
 * Scenario from STRATEGIC_ROADMAP.md §3.2: AI-Heavy Concurrent Games
 * Target: 100+ concurrent games with 200-300 players
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Gauge, Trend } from 'k6/metrics';
import { SharedArray } from 'k6/data';
import { loginAndGetToken } from '../auth/helpers.js';

// Custom metrics
const activeGames = new Gauge('concurrent_active_games');
const gameStateErrors = new Counter('game_state_errors');
const gameStateCheckSuccess = new Rate('game_state_check_success');
const resourceOverhead = new Trend('game_resource_overhead_ms');

// Test configuration for production-scale validation
export const options = {
  scenarios: {
    concurrent_games: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up to 50 VUs (50 games)
        { duration: '3m', target: 100 },  // Ramp to 100 VUs (100+ games)
        { duration: '5m', target: 100 },  // Sustain 100+ concurrent games
        { duration: '2m', target: 50 },   // Gradual ramp down
        { duration: '1m', target: 0 }     // Complete shutdown
      ],
      gracefulRampDown: '30s',
    }
  },
  
  thresholds: {
    // Game state retrieval should remain fast even at scale
    'http_req_duration{name:get-game}': [
      'p(95)<400',   // Staging: p95 ≤ 400ms for GET /api/games/:id
      'p(99)<800'    // Staging: p99 ≤ 800ms
    ],
    
    // Game creation overhead at scale
    'http_req_duration{name:create-game}': [
      'p(95)<800',
      'p(99)<1500'
    ],
    
    // Error rate - must remain low even at peak concurrency
    'http_req_failed': ['rate<0.01'],
    
    // Custom thresholds
    'game_state_check_success': ['rate>0.99'],
    'concurrent_active_games': ['value>=100'],  // Confirm we reach target
  },
  
  tags: {
    scenario: 'concurrent-games',
    test_type: 'stress',
    environment: 'staging'
  }
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const API_PREFIX = '/api';

// Track created games per VU for state checking
let myGameId = null;

export function setup() {
  console.log('Starting concurrent games stress test');
  console.log('Target: 100+ simultaneous active games');

  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200,
  });

  // Use the shared auth helper to obtain a canonical JWT for the
  // pre-seeded load-test user. All VUs share this token to avoid
  // (re)registering users during the scenario.
  const { token, userId } = loginAndGetToken(BASE_URL, {
    apiPrefix: API_PREFIX,
    tags: { name: 'auth-login-setup' },
  });

  return { baseUrl: BASE_URL, token, userId };
}

export default function(data) {
  const baseUrl = data.baseUrl;
  const token = data.token;

  // Each VU creates and maintains one game
  if (!myGameId) {
    // Step 1: Create a game (contributes to concurrent count) using the
    // canonical create-game payload shape from the API:
    //   { boardType, maxPlayers, isPrivate, timeControl, isRated, aiOpponents? }
    const boardTypes = ['square8', 'square19', 'hexagonal'];
    const boardType = boardTypes[__VU % boardTypes.length];

    const maxPlayersOptions = [2, 3, 4];
    const maxPlayers = maxPlayersOptions[__VU % maxPlayersOptions.length];

    const aiCount = 1 + (__VU % 2); // 1-2 AI opponents
    const hasAI = aiCount > 0;

    const gameConfig = {
      boardType,
      maxPlayers,
      isPrivate: false,
      timeControl: {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      },
      // AI games must be unrated per backend contract
      isRated: hasAI ? false : true,
      ...(hasAI && {
        aiOpponents: {
          count: aiCount,
          difficulty: Array(aiCount).fill(5),
          mode: 'service',
          aiType: 'heuristic',
        },
      }),
    };

    const createStart = Date.now();
    const createRes = http.post(`${baseUrl}${API_PREFIX}/games`, JSON.stringify(gameConfig), {
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      tags: { name: 'create-game' },
    });
    const createDuration = Date.now() - createStart;

    let createdGameId = null;
    try {
      const body = JSON.parse(createRes.body);
      const game = body && body.data && body.data.game ? body.data.game : null;
      createdGameId = game ? game.id : null;
    } catch {
      createdGameId = null;
    }

    if (createRes.status === 201 && createdGameId) {
      myGameId = createdGameId;
      console.log(`VU ${__VU}: Created game ${myGameId} in ${createDuration}ms`);
    } else {
      console.error(
        `VU ${__VU}: Game creation failed - status=${createRes.status} body=${createRes.body}`
      );
      gameStateErrors.add(1);
      return;
    }
  }

  // Step 3: Continuously monitor game state (validates state management at scale)
  if (myGameId && token) {
    const stateStart = Date.now();
    const stateRes = http.get(`${baseUrl}${API_PREFIX}/games/${myGameId}`, {
      headers: { Authorization: `Bearer ${token}` },
      tags: { name: 'get-game' },
    });
    const stateDuration = Date.now() - stateStart;

    const stateValid = check(stateRes, {
      'game state retrieved': (r) => r.status === 200,
      'game ID matches': (r) => {
        try {
          const body = JSON.parse(r.body);
          const game = body && body.data && body.data.game ? body.data.game : null;
          return !!(game && game.id === myGameId);
        } catch {
          return false;
        }
      },
      'game has players': (r) => {
        try {
          const body = JSON.parse(r.body);
          const game = body && body.data && body.data.game ? body.data.game : null;
          if (!game) return false;
          const playerIds = [
            game.player1Id,
            game.player2Id,
            game.player3Id,
            game.player4Id,
          ].filter(Boolean);
          return playerIds.length > 0;
        } catch {
          return false;
        }
      },
    });
    
    gameStateCheckSuccess.add(stateValid);
    resourceOverhead.add(stateDuration);
    
    if (!stateValid) {
      gameStateErrors.add(1);
      console.error(`VU ${__VU}: Game state check failed for ${myGameId}`);
    }
  }
  
  // Update concurrent games metric
  // Note: This is approximate as VUs may be ramping
  activeGames.add(__VU);
  
  // Simulate realistic polling interval (players checking game state)
  sleep(2 + Math.random() * 3); // 2-5 seconds between checks
}

export function teardown(data) {
  console.log('Concurrent games stress test complete');
  console.log('Check metrics for:');
  console.log('  - Peak concurrent games reached');
  console.log('  - Game state retrieval latency at scale');
  console.log('  - Memory/CPU resource trends (via Prometheus)');
}