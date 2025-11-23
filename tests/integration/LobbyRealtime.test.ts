/**
 * Integration test for real-time lobby updates.
 * 
 * Note: These tests require @testing-library/react to be installed.
 * Run: npm install --save-dev @testing-library/react @testing-library/jest-dom
 */

import { io, Socket } from 'socket.io-client';
import { gameApi } from '../../src/client/services/api';

describe('Lobby Real-time Integration', () => {
  let socket: Socket;
  const TEST_TOKEN = 'test-jwt-token';
  const WS_URL = process.env.VITE_WS_URL || 'http://localhost:3000';

  beforeEach(() => {
    localStorage.setItem('token', TEST_TOKEN);
  });

  afterEach(() => {
    if (socket && socket.connected) {
      socket.disconnect();
    }
    localStorage.clear();
  });

  it('should connect to lobby room and receive game_created events', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
    });

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      socket.on('lobby:game_created', (game) => {
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
      done(err);
    });
  });

  it('should receive game_joined events when players join', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
    });

    let gameId: string;

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      socket.on('lobby:game_created', (game) => {
        gameId = game.id;
      });

      socket.on('lobby:game_joined', ({ gameId: joinedGameId, playerCount }) => {
        expect(joinedGameId).toBe(gameId);
        expect(playerCount).toBeGreaterThan(1);
        done();
      });
    });
  });

  it('should receive game_started events when game begins', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
    });

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      socket.on('lobby:game_started', ({ gameId }) => {
        expect(gameId).toBeDefined();
        expect(typeof gameId).toBe('string');
        done();
      });
    });
  });

  it('should receive game_cancelled events', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
    });

    socket.on('connect', () => {
      socket.emit('lobby:subscribe');

      socket.on('lobby:game_cancelled', ({ gameId }) => {
        expect(gameId).toBeDefined();
        done();
      });
    });
  });

  it('should unsubscribe from lobby updates', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
    });

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
          expect(receivedEvent).toBe(false);
          done();
        }, 500);
      }, 100);
    });
  });

  it('should handle multiple concurrent subscribers', (done) => {
    const socket1 = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
    });

    const socket2 = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
    });

    let socket1Received = false;
    let socket2Received = false;

    socket1.on('connect', () => {
      socket1.emit('lobby:subscribe');

      socket1.on('lobby:game_created', () => {
        socket1Received = true;
        checkBothReceived();
      });
    });

    socket2.on('connect', () => {
      socket2.emit('lobby:subscribe');

      socket2.on('lobby:game_created', () => {
        socket2Received = true;
        checkBothReceived();
      });
    });

    function checkBothReceived() {
      if (socket1Received && socket2Received) {
        socket1.disconnect();
        socket2.disconnect();
        done();
      }
    }
  });

  it('should reconnect and resubscribe to lobby', (done) => {
    socket = io(WS_URL, {
      auth: { token: TEST_TOKEN },
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 100,
    });

    let reconnected = false;

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
        expect(reconnected).toBe(true);
        done();
      });
    });
  });
});