/**
 * Integration test for real-time lobby updates.
 *
 * IMPORTANT: These tests require a LIVE WebSocket server running at the WS_URL.
 * They are skipped by default in CI environments because they need the full
 * server stack (HTTP server, Socket.IO, database, authentication).
 *
 * To run these tests locally:
 * 1. Start the development server: npm run dev:server
 * 2. In another terminal: ENABLE_LIVE_TESTS=true npm test -- --testPathPattern=LobbyRealtime
 *
 * These tests verify:
 * - WebSocket connection establishment with authentication
 * - Lobby room subscription and event handling
 * - Real-time game creation, join, start, and cancel events
 * - Reconnection behavior
 * - Multiple concurrent subscriber handling
 *
 * For unit testing of WebSocket server logic without a live server, see:
 * - tests/unit/WebSocketServer.authRevocation.test.ts
 * - tests/unit/WebSocketInteractionHandler.test.ts
 * - tests/unit/WebSocketServer.rulesBackend.integration.test.ts
 *
 * These unit tests use mocked Socket.IO and are suitable for CI.
 */

import { io, Socket } from 'socket.io-client';
import { gameApi } from '../../src/client/services/api';

// Skip all tests unless explicitly enabled via environment variable
// This ensures CI doesn't hang waiting for connections to non-existent servers
const ENABLE_LIVE_TESTS = process.env.ENABLE_LIVE_TESTS === 'true';

// Use describe.skip when live tests are disabled
const describeWithLiveServer = ENABLE_LIVE_TESTS ? describe : describe.skip;

describeWithLiveServer('Lobby Real-time Integration (requires live server)', () => {
  let socket: Socket;
  const TEST_TOKEN = 'test-jwt-token';
  const WS_URL = process.env.VITE_WS_URL || 'http://localhost:3000';

  // Reduced timeout for faster failure feedback when server isn't running
  const CONNECTION_TIMEOUT = 5000;

  beforeEach(() => {
    localStorage.setItem('token', TEST_TOKEN);
  });

  afterEach(() => {
    // Ensure socket cleanup to prevent open handles
    if (socket) {
      socket.removeAllListeners();
      if (socket.connected) {
        socket.disconnect();
      }
    }
    localStorage.clear();
  });

  it('should connect to lobby room and receive game_created events', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      timeout: CONNECTION_TIMEOUT,
    });

    const timeoutId = setTimeout(() => {
      socket.disconnect();
      done(new Error('Test timed out waiting for lobby:game_created event'));
    }, CONNECTION_TIMEOUT);

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      socket.on('lobby:game_created', (game) => {
        clearTimeout(timeoutId);
        expect(game).toHaveProperty('id');
        expect(game).toHaveProperty('boardType');
        expect(game).toHaveProperty('status', 'waiting');
        done();
      });

      // Simulate game creation (would normally come from another client)
      setTimeout(() => {
        gameApi.createGame({
          boardType: 'square8',
          maxPlayers: 2,
          isRated: false,
          isPrivate: false,
          timeControl: {
            type: 'blitz',
            initialTime: 300,
            increment: 0,
          },
        });
      }, 100);
    });

    socket.on('connect_error', (err) => {
      clearTimeout(timeoutId);
      done(new Error(`Connection failed: ${err.message}. Is the server running at ${WS_URL}?`));
    });
  }, 10000);

  it('should receive game_joined events when players join', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      timeout: CONNECTION_TIMEOUT,
    });

    let gameId: string;

    const timeoutId = setTimeout(() => {
      socket.disconnect();
      done(new Error('Test timed out waiting for lobby:game_joined event'));
    }, CONNECTION_TIMEOUT);

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      socket.on('lobby:game_created', (game) => {
        gameId = game.id;
      });

      socket.on('lobby:game_joined', ({ gameId: joinedGameId, playerCount }) => {
        clearTimeout(timeoutId);
        expect(joinedGameId).toBe(gameId);
        expect(playerCount).toBeGreaterThan(1);
        done();
      });
    });

    socket.on('connect_error', (err) => {
      clearTimeout(timeoutId);
      done(new Error(`Connection failed: ${err.message}`));
    });
  }, 10000);

  it('should receive game_started events when game begins', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      timeout: CONNECTION_TIMEOUT,
    });

    const timeoutId = setTimeout(() => {
      socket.disconnect();
      done(new Error('Test timed out waiting for lobby:game_started event'));
    }, CONNECTION_TIMEOUT);

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      socket.on('lobby:game_started', ({ gameId }) => {
        clearTimeout(timeoutId);
        expect(gameId).toBeDefined();
        expect(typeof gameId).toBe('string');
        done();
      });
    });

    socket.on('connect_error', (err) => {
      clearTimeout(timeoutId);
      done(new Error(`Connection failed: ${err.message}`));
    });
  }, 10000);

  it('should receive game_cancelled events', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      timeout: CONNECTION_TIMEOUT,
    });

    const timeoutId = setTimeout(() => {
      socket.disconnect();
      done(new Error('Test timed out waiting for lobby:game_cancelled event'));
    }, CONNECTION_TIMEOUT);

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      socket.on('lobby:game_cancelled', ({ gameId }) => {
        clearTimeout(timeoutId);
        expect(gameId).toBeDefined();
        done();
      });
    });

    socket.on('connect_error', (err) => {
      clearTimeout(timeoutId);
      done(new Error(`Connection failed: ${err.message}`));
    });
  }, 10000);

  it('should unsubscribe from lobby updates', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      timeout: CONNECTION_TIMEOUT,
    });

    const timeoutId = setTimeout(() => {
      socket.disconnect();
      done(new Error('Test timed out'));
    }, CONNECTION_TIMEOUT);

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      setTimeout(() => {
        socket.emit('lobby:unsubscribe');

        // After unsubscribe, should not receive events
        let receivedEvent = false;
        socket.on('lobby:game_created', () => {
          receivedEvent = true;
        });

        setTimeout(() => {
          clearTimeout(timeoutId);
          expect(receivedEvent).toBe(false);
          done();
        }, 500);
      }, 100);
    });

    socket.on('connect_error', (err) => {
      clearTimeout(timeoutId);
      done(new Error(`Connection failed: ${err.message}`));
    });
  }, 10000);

  it('should continue receiving lobby events after a disconnect and reconnect', (done) => {
    const reconnectTimeout = setTimeout(() => {
      if (socket && socket.connected) {
        socket.disconnect();
      }
      done(new Error('Test timed out waiting for lobby:game_created after reconnect'));
    }, CONNECTION_TIMEOUT * 2);

    // First connection that will disconnect shortly after subscribing.
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      timeout: CONNECTION_TIMEOUT,
    });

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      // After initial subscription, simulate a disconnect and a fresh reconnect.
      setTimeout(() => {
        socket.disconnect();

        const reconnectSocket = io(WS_URL, {
          auth: { token: TEST_TOKEN },
          transports: ['websocket'],
          timeout: CONNECTION_TIMEOUT,
        });

        reconnectSocket.on('connect', () => {
          reconnectSocket.emit('lobby:subscribe');

          reconnectSocket.on('lobby:game_created', (game) => {
            clearTimeout(reconnectTimeout);
            expect(game).toHaveProperty('id');
            expect(game).toHaveProperty('boardType');
            reconnectSocket.disconnect();
            done();
          });

          // Trigger a game creation after the reconnect has completed.
          setTimeout(() => {
            gameApi.createGame({
              boardType: 'square8',
              maxPlayers: 2,
              isRated: false,
              isPrivate: false,
              timeControl: {
                type: 'blitz',
                initialTime: 300,
                increment: 0,
              },
            });
          }, 100);
        });

        reconnectSocket.on('connect_error', (err) => {
          clearTimeout(reconnectTimeout);
          reconnectSocket.disconnect();
          done(new Error(`Reconnection failed: ${err.message}`));
        });
      }, 100);
    });

    socket.on('connect_error', (err) => {
      clearTimeout(reconnectTimeout);
      socket.disconnect();
      done(new Error(`Initial connection failed: ${err.message}`));
    });
  }, 20000);

  it('should handle multiple concurrent subscribers', (done) => {
    const socket1 = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      timeout: CONNECTION_TIMEOUT,
    });

    const socket2 = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      timeout: CONNECTION_TIMEOUT,
    });

    let socket1Received = false;
    let socket2Received = false;

    const timeoutId = setTimeout(() => {
      socket1.disconnect();
      socket2.disconnect();
      done(new Error('Test timed out waiting for both sockets to receive events'));
    }, CONNECTION_TIMEOUT);

    const cleanup = () => {
      clearTimeout(timeoutId);
      socket1.disconnect();
      socket2.disconnect();
    };

    socket1.on('connect', () => {
      socket1.emit('lobby:subscribe');

      socket1.on('lobby:game_created', () => {
        socket1Received = true;
        checkBothReceived();
      });
    });

    socket1.on('connect_error', (err) => {
      cleanup();
      done(new Error(`Socket1 connection failed: ${err.message}`));
    });

    socket2.on('connect', () => {
      socket2.emit('lobby:subscribe');

      socket2.on('lobby:game_created', () => {
        socket2Received = true;
        checkBothReceived();
      });
    });

    socket2.on('connect_error', (err) => {
      cleanup();
      done(new Error(`Socket2 connection failed: ${err.message}`));
    });

    function checkBothReceived() {
      if (socket1Received && socket2Received) {
        cleanup();
        done();
      }
    }
  }, 10000);

  it('should reconnect and resubscribe to lobby', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      timeout: CONNECTION_TIMEOUT,
      reconnection: true,
      reconnectionDelay: 100,
    });

    let reconnected = false;

    const timeoutId = setTimeout(() => {
      socket.disconnect();
      done(new Error('Test timed out waiting for reconnection'));
    }, CONNECTION_TIMEOUT);

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      if (!reconnected) {
        // Force disconnect to test reconnection
        setTimeout(() => {
          socket.disconnect();
          socket.connect();
        }, 200);
      }
    });

    socket.on('reconnect', () => {
      reconnected = true;
      socket.emit('lobby:subscribe');

      socket.on('lobby:game_created', () => {
        clearTimeout(timeoutId);
        expect(reconnected).toBe(true);
        done();
      });
    });

    socket.on('connect_error', (err) => {
      clearTimeout(timeoutId);
      done(new Error(`Connection failed: ${err.message}`));
    });
  }, 10000);
});

// Placeholder test to ensure file is recognized as a test suite when live tests are skipped
describe('Lobby Real-time Integration (placeholder for CI)', () => {
  it('skips live server tests in CI - set ENABLE_LIVE_TESTS=true to run', () => {
    expect(ENABLE_LIVE_TESTS).toBe(false);
    // This documents that live tests are intentionally skipped
    // For WebSocket unit tests that run in CI, see:
    // - tests/unit/WebSocketServer.authRevocation.test.ts
    // - tests/unit/WebSocketInteractionHandler.test.ts
    // - tests/unit/WebSocketServer.rulesBackend.integration.test.ts
  });
});
