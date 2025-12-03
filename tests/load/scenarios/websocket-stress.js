/**
 * RingRift Load Test: WebSocket Connection Stress Test
 * 
 * Tests WebSocket connection limits, stability, and reconnection behavior.
 * Validates production-scale assumptions for real-time connection handling.
 * 
 * Scenario from STRATEGIC_ROADMAP.md §3.3: Reconnects and Spectators
 * Alert thresholds from monitoring/prometheus/alerts.yml (>1000 connections)
 * 
 * NOTE: k6 WebSocket support is functional but may require k6 v0.46+
 * for advanced features. See k6.io/docs/using-k6/protocols/websockets
 */

import http from 'k6/http';
import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Counter, Rate, Gauge, Trend } from 'k6/metrics';
import { loginAndGetToken } from '../auth/helpers.js';

// Custom metrics
const wsConnections = new Gauge('websocket_connections_active');
const wsConnectionSuccess = new Rate('websocket_connection_success_rate');
const wsConnectionErrors = new Counter('websocket_connection_errors');
const wsReconnections = new Counter('websocket_reconnections_total');
const wsMessageLatency = new Trend('websocket_message_latency_ms');
const wsConnectionDuration = new Trend('websocket_connection_duration_ms');

// Test configuration - stress test for connection limits
export const options = {
  scenarios: {
    websocket_stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },   // Ramp to 100 connections
        { duration: '2m', target: 300 },   // Ramp to 300 connections
        { duration: '3m', target: 500 },   // Reach 500+ connections
        { duration: '5m', target: 500 },   // Sustain high connection count
        { duration: '2m', target: 100 },   // Gradual ramp down
        { duration: '1m', target: 0 }      // Complete shutdown
      ],
      gracefulRampDown: '30s',
    }
  },
  
  thresholds: {
    // Connection success rate - should remain high even at scale
    'websocket_connection_success_rate': ['rate>0.99'],
    
    // Connection errors should be minimal
    'websocket_connection_errors': ['count<50'], // <10% of peak connections
    
    // Message latency - real-time feel
    'websocket_message_latency_ms': [
      'p(95)<200',  // Most messages arrive quickly
      'p(99)<500'   // Even slow messages acceptable
    ],
    
    // Connection stability - should maintain for 5+ minutes per STRATEGIC_ROADMAP
    'websocket_connection_duration_ms': ['p(50)>300000'], // Median >5 minutes
  },
  
  tags: {
    scenario: 'websocket-stress',
    test_type: 'stress',
    environment: 'staging'
  }
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3001';
/**
 * WebSocket origin used for Socket.IO connections.
 *
 * This may be provided as either:
 *   - WS_URL=ws://host:port
 *   - WS_URL=http://host:port
 *
 * When not set, it falls back to BASE_URL so local runs behave like the
 * browser client, which derives its socket origin from VITE_WS_URL /
 * VITE_API_URL (see src/client/utils/socketBaseUrl.ts and LobbyPage).
 */
const WS_BASE = __ENV.WS_URL || BASE_URL;
const API_PREFIX = '/api';

/**
 * Build a Socket.IO WebSocket endpoint that matches the production
 * WebSocketServer contract:
 *
 *   - Path: /socket.io/
 *   - Query params:
 *       EIO=4&transport=websocket
 *       token=<JWT> (for auth middleware)
 *       vu=<VU id> (diagnostic only)
 *
 * k6 speaks plain WebSocket frames rather than the full Socket.IO
 * protocol, but this is sufficient for:
 *   - Completing the initial handshake (HTTP 101)
 *   - Exercising WebSocketServer authentication and connection metrics
 *
 * The server will likely ignore or close connections that do not send
 * valid Socket.IO frames after handshake, which is acceptable for this
 * scenario: we are primarily interested in connection establishment and
 * handshake throughput, not full game semantics.
 */
function buildSocketIoEndpoint(wsBase, token, vu) {
  let origin = wsBase || '';
  if (origin.startsWith('http://') || origin.startsWith('https://')) {
    origin = origin.replace(/^http/, 'ws');
  }
  origin = origin.replace(/\/$/, '');

  const path = '/socket.io/';
  const params = [];
  // Socket.IO v4 engine and transport selector
  params.push('EIO=4', 'transport=websocket', `vu=${vu}`);
  // WebSocketServer authentication middleware accepts the JWT via
  // either handshake.auth.token (used by the browser client) or
  // handshake.query.token. We use the query-string form here.
  if (token) {
    params.push(`token=${encodeURIComponent(token)}`);
  }

  const query = params.join('&');
  return `${origin}${path}?${query}`;
}

export function setup() {
  console.log('Starting WebSocket connection stress test');
  console.log('Target: 500+ simultaneous WebSocket connections');
  console.log('Duration: 5+ minute sustained connection period');

  // Lightweight health check to match other scenarios and fail fast if
  // the API is not reachable.
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check successful': (r) => r.status === 200,
  });

  // Use the shared auth helper so the WebSocket stress test reuses the
  // canonical /api/auth/login contract with { email, password } and
  // the same pre-seeded load-test user.
  const { token, userId } = loginAndGetToken(BASE_URL, {
    apiPrefix: API_PREFIX,
    tags: { name: 'auth-login-setup' },
  });

  return { wsBase: WS_BASE, baseUrl: BASE_URL, token, userId };
}

export default function(data) {
  const wsBase = data.wsBase;
  const baseUrl = data.baseUrl;
  const token = data.token;

  // Each VU maintains a WebSocket connection
  const connectionStart = Date.now();
  let messagesSent = 0;
  let messagesReceived = 0;
  let reconnectAttempts = 0;

  // WebSocket connection using the same Socket.IO path and auth
  // semantics as the browser client (see SocketGameConnection and
  // LobbyPage). We attach the JWT via the `token` query parameter and
  // use the Socket.IO engine/transport selectors expected by
  // WebSocketServer.
  const wsEndpoint = buildSocketIoEndpoint(wsBase, token, __VU);

  const res = ws.connect(
    wsEndpoint,
    {
      headers: {
        'User-Agent': 'k6-load-test',
      },
      tags: { vu: __VU.toString() },
    },
    function (socket) {
    
    // Track active connections
    wsConnections.add(1);
    
    socket.on('open', () => {
      console.log(`VU ${__VU}: WebSocket connected`);
      wsConnectionSuccess.add(1);
      
      // Send initial ping to test connectivity
      const pingPayload = JSON.stringify({
        type: 'ping',
        timestamp: Date.now(),
        vu: __VU
      });
      socket.send(pingPayload);
      messagesSent++;
    });
    
    socket.on('message', (message) => {
      messagesReceived++;
      
      try {
        const data = JSON.parse(message);
        const latency = Date.now() - (data.timestamp || Date.now());
        wsMessageLatency.add(latency);
        
        // Respond to pings
        if (data.type === 'ping') {
          socket.send(JSON.stringify({
            type: 'pong',
            timestamp: Date.now(),
            vu: __VU
          }));
          messagesSent++;
        }
      } catch (e) {
        // Non-JSON message or parsing error
        console.warn(`VU ${__VU}: Message parse error`);
      }
    });
    
    socket.on('close', (code) => {
      const duration = Date.now() - connectionStart;
      wsConnectionDuration.add(duration);
      wsConnections.add(-1);
      
      console.log(`VU ${__VU}: WebSocket closed after ${duration}ms (code: ${code})`);
      
      // Attempt reconnection if unexpected close
      if (code !== 1000 && reconnectAttempts < 3) {
        wsReconnections.add(1);
        reconnectAttempts++;
        console.log(`VU ${__VU}: Attempting reconnection (${reconnectAttempts}/3)`);
      }
    });
    
    socket.on('error', (e) => {
      wsConnectionErrors.add(1);
      console.error(`VU ${__VU}: WebSocket error - ${e}`);
    });
    
    // Keep connection alive and periodically send messages
    // This simulates spectator activity or game state updates
    let pingInterval = 0;
    socket.setInterval(() => {
      pingInterval++;
      
      // Send periodic heartbeat every ~30 seconds
      if (pingInterval % 30 === 0) {
        socket.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: Date.now(),
          vu: __VU,
          interval: pingInterval
        }));
        messagesSent++;
      }
      
      // Simulate realistic activity - occasional game state requests
      if (pingInterval % 10 === 0 && Math.random() > 0.7) {
        socket.send(JSON.stringify({
          type: 'request_game_state',
          timestamp: Date.now(),
          vu: __VU
        }));
        messagesSent++;
      }
      
    }, 1000); // Check every second
    
    // Maintain connection for test duration
    // Connection will close when VU iteration ends
    socket.setTimeout(() => {
      const duration = Date.now() - connectionStart;
      console.log(`VU ${__VU}: Connection timeout after ${duration}ms. Sent: ${messagesSent}, Received: ${messagesReceived}`);
      socket.close();
    }, 360000); // 6 minutes max per connection (covers 5-minute sustain period)
  });
  
  // Check connection establishment
  check(res, {
    'WebSocket connected': (r) => r && r.status === 101
  });
  
  if (!res || res.status !== 101) {
    wsConnectionErrors.add(1);
    console.error(`VU ${__VU}: Failed to establish WebSocket connection - status ${res?.status}`);
  }
  
  // After WebSocket closes, brief pause before next iteration
  sleep(2 + Math.random() * 3);
}

export function teardown(data) {
  console.log('WebSocket stress test complete');
  console.log('Key metrics to review:');
  console.log('  - Peak websocket_connections_active (should reach 500+)');
  console.log('  - websocket_connection_success_rate');
  console.log('  - websocket_connection_duration_ms (p50 should be >5 minutes)');
  console.log('  - websocket_reconnections_total');
  console.log('Check Prometheus for:');
  console.log('  - ringrift_websocket_connections gauge');
  console.log('  - High connection alert triggers (>1000 threshold)');
}