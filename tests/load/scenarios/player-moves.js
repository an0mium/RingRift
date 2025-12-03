/**
 * RingRift Load Test: Player Move Submission Scenario
 * 
 * Tests move submission latency and turn processing throughput.
 * Validates production-scale assumptions for real-time gameplay.
 * 
 * Scenario from STRATEGIC_ROADMAP.md §3: Player Moves
 * SLOs from STRATEGIC_ROADMAP.md §2.2: WebSocket gameplay SLOs
 * 
 * NOTE: k6 has limited WebSocket support. For full real-time testing,
 * consider supplementing with socket.io-client or Playwright tests.
 * This scenario focuses on HTTP-based move submission where available.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { loginAndGetToken } from '../auth/helpers.js';

// Custom metrics aligned with STRATEGIC_ROADMAP metrics
const moveSubmissionLatency = new Trend('move_submission_latency_ms');
const moveSubmissionSuccess = new Rate('move_submission_success_rate');
const moveProcessingErrors = new Counter('move_processing_errors');
const turnProcessingLatency = new Trend('turn_processing_latency_ms');
const stalledMoves = new Counter('stalled_moves_total'); // >2s threshold per STRATEGIC_ROADMAP

// Test configuration
export const options = {
  scenarios: {
    realistic_gameplay: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 20 },   // Ramp up to 20 concurrent games (40 players)
        { duration: '3m', target: 40 },   // Increase to 40 games (80 players)
        { duration: '5m', target: 40 },   // Sustain realistic gameplay
        { duration: '1m', target: 0 }     // Ramp down
      ],
      gracefulRampDown: '30s',
    }
  },
  
  thresholds: {
    // Move submission latency - staging SLOs from STRATEGIC_ROADMAP §2.2
    'move_submission_latency_ms': [
      'p(95)<300',   // Staging: 95% ≤ 300ms
      'p(99)<600'    // Staging: 99% ≤ 600ms
    ],
    
    // Stall rate - moves taking >2s should be rare
    'stalled_moves_total': ['count<10'], // <0.5% for staging (assuming ~2000 moves)
    
    // Success rate
    'move_submission_success_rate': ['rate>0.99'],
    
    // Turn processing (includes validation + state update)
    'turn_processing_latency_ms': [
      'p(95)<400',
      'p(99)<800'
    ]
  },
  
  tags: {
    scenario: 'player-moves',
    test_type: 'load',
    environment: 'staging'
  }
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
const API_PREFIX = '/api';

// HTTP move submission is currently **not** part of the production API
// surface; moves are carried exclusively over WebSockets (see
// src/server/websocket/server.ts and shared websocket types). This flag
// guards the legacy HTTP move path so that the scenario can still be
// compiled and iterated on without generating guaranteed 404s. When an
// HTTP move endpoint is introduced, this flag can be flipped and the
// payload adjusted to match the new contract.
const MOVE_HTTP_ENDPOINT_ENABLED = false;

// Game state per VU
let myGameId = null;

export function setup() {
  console.log('Starting player move submission load test');
  console.log('Focus: Move processing latency and turn throughput');

  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200,
  });

  // Use the shared auth helper so this scenario matches the same
  // /api/auth/login contract as game-creation and concurrent-games.
  const { token, userId } = loginAndGetToken(BASE_URL, {
    apiPrefix: API_PREFIX,
    tags: { name: 'auth-login-setup' },
  });

  return { baseUrl: BASE_URL, token, userId };
}

export default function(data) {
  const baseUrl = data.baseUrl;
  const token = data.token;

  // Step 1: Setup - Create game once per VU using the canonical create-game payload
  if (!myGameId) {
    const aiCount = 1; // 1 AI opponent for automated gameplay

    const createPayload = {
      boardType: 'square8', // Smaller board for faster games
      maxPlayers: 2,
      isPrivate: false,
      timeControl: {
        type: 'rapid',
        initialTime: 600,
        increment: 0,
      },
      // AI games must be unrated per backend contract
      isRated: false,
      aiOpponents: {
        count: aiCount,
        difficulty: Array(aiCount).fill(5),
        mode: 'service',
        aiType: 'heuristic',
      },
    };

    const createRes = http.post(
      `${baseUrl}${API_PREFIX}/games`,
      JSON.stringify(createPayload),
      {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      }
    );

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
      console.log(`VU ${__VU}: Created game ${myGameId}`);
    } else {
      console.error(
        `VU ${__VU}: Game creation failed - status=${createRes.status} body=${createRes.body}`
      );
      return;
    }
  }

  // Step 2: Poll game state and (optionally) submit HTTP "moves"
  if (myGameId && token) {
    // Get current game state
    const stateRes = http.get(`${baseUrl}${API_PREFIX}/games/${myGameId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (stateRes.status !== 200) {
      moveProcessingErrors.add(1);
      sleep(2);
      return;
    }
    
    let game = null;
    try {
      const body = JSON.parse(stateRes.body);
      game = body && body.data && body.data.game ? body.data.game : null;
    } catch {
      game = null;
    }

    if (!game) {
      moveProcessingErrors.add(1);
      sleep(2);
      return;
    }

    // Check if game is still active
    if (game.status !== 'active' && game.status !== 'waiting') {
      console.log(`VU ${__VU}: Game ${myGameId} ended with status ${game.status}`);
      // Reset to create a new game next iteration
      myGameId = null;
      sleep(5);
      return;
    }
    
    if (MOVE_HTTP_ENDPOINT_ENABLED) {
      // Simulate move submission via a hypothetical HTTP endpoint. At the
      // time of PASS23, moves are carried exclusively over WebSockets, so
      // this path is kept disabled by default to avoid exercising a
      // non-existent contract. When an HTTP move endpoint is added, this
      // block can be updated to match its payload/response shape.
      const movePayload = {
        gameId: myGameId,
        // Placeholder action payload; real implementation should generate
        // a valid MovePayload compatible with the backend rules engine.
        action: generateRandomMove(game),
      };

      const moveStart = Date.now();
      const moveRes = http.post(
        `${baseUrl}${API_PREFIX}/games/${myGameId}/moves`,
        JSON.stringify(movePayload),
        {
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          tags: { name: 'submit-move' },
        }
      );
      const moveLatency = Date.now() - moveStart;

      // Track metrics
      moveSubmissionLatency.add(moveLatency);

      const moveSuccess = check(moveRes, {
        'move accepted': (r) => r.status === 200 || r.status === 201,
      });

      moveSubmissionSuccess.add(moveSuccess);

      if (moveSuccess) {
        turnProcessingLatency.add(moveLatency);
      } else {
        moveProcessingErrors.add(1);
      }

      // Track stalled moves (>2s per STRATEGIC_ROADMAP stall definition)
      if (moveLatency > 2000) {
        stalledMoves.add(1);
        console.warn(`VU ${__VU}: Stalled move detected - ${moveLatency}ms`);
      }
    }
  }
  
  // Think time between moves - realistic gameplay pacing
  sleep(1 + Math.random() * 3); // 1-4 seconds between moves
}

/**
 * Generate a valid random move based on game state
 * In production, this would analyze the board and generate legal moves
 * For load testing, simplified placeholder logic
 */
function generateRandomMove(gameState) {
  const moveTypes = ['PLACE_RING', 'MOVE_RING', 'PLACE_MARKER'];
  const moveType = moveTypes[Math.floor(Math.random() * moveTypes.length)];
  
  // Simplified move generation - actual implementation needs game rules
  return {
    type: moveType,
    position: {
      q: Math.floor(Math.random() * 8),
      r: Math.floor(Math.random() * 8)
    },
    // Add fields based on move type
    ...(moveType === 'MOVE_RING' && {
      from: {
        q: Math.floor(Math.random() * 8),
        r: Math.floor(Math.random() * 8)
      }
    })
  };
}

export function teardown(data) {
  console.log('Player move submission test complete');
  console.log('Key metrics to review:');
  console.log('  - move_submission_latency_ms (p95, p99)');
  console.log('  - stalled_moves_total (should be <0.5% of total moves)');
  console.log('  - move_submission_success_rate');
}