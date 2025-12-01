/**
 * Multi-Player Coordination E2E Tests
 * ============================================================================
 *
 * This test suite demonstrates the usage of MultiClientCoordinator for
 * coordinating multiple WebSocket clients in E2E test scenarios.
 *
 * The MultiClientCoordinator provides:
 * - Multiple WebSocket connection management
 * - Promise-based event waiting
 * - Sequential and parallel action coordination
 * - Message queue inspection per client
 *
 * RUN COMMAND: npx playwright test multiPlayer.coordination.test.ts
 *
 * NOTE: These tests require:
 * - PostgreSQL running (for user accounts and game persistence)
 * - Redis running (for WebSocket session management)
 * - Backend server running on http://localhost:3000
 */

import { test, expect } from '@playwright/test';
import {
  MultiClientCoordinator,
  createMultiClientCoordinator,
  isGameStateMessage,
  isGameOverMessage,
} from '../helpers/MultiClientCoordinator';
import {
  generateTestUser,
  registerUser,
  waitForApiReady,
  type TestUser,
  createFixtureGame,
} from './helpers/test-utils';
import type { GameOverMessage } from '../../src/shared/types/websocket';

/**
 * Helper to register a user and extract their JWT token.
 *
 * In a real E2E test, you would typically:
 * 1. Register/login via the UI or API
 * 2. Extract the JWT from localStorage or a cookie
 *
 * For this example, we use the test API to create users and return tokens.
 */
async function createUserAndGetToken(
  page: import('@playwright/test').Page,
  user: TestUser
): Promise<string> {
  // Register the user via the UI
  await registerUser(page, user.username, user.email, user.password);

  // Extract token from localStorage
  // Note: This assumes the app stores the token in localStorage after login
  const token = await page.evaluate(() => {
    return localStorage.getItem('auth_token') ?? localStorage.getItem('token');
  });

  if (!token) {
    throw new Error('Failed to get auth token after registration');
  }

  return token;
}

test.describe('Multi-Player Coordination with MultiClientCoordinator', () => {
  // Mark all tests as slow since they involve multiple WebSocket connections
  test.slow();
  test.setTimeout(120_000); // 2 minutes per test

  const serverUrl = process.env.E2E_API_BASE_URL || 'http://localhost:3000';

  test.beforeEach(async ({ page }) => {
    // Ensure backend is ready before running tests
    await waitForApiReady(page);
  });

  test.describe('Basic Coordination Patterns', () => {
    test('demonstrates connecting and coordinating two players', async ({ browser }) => {
      // Create browser contexts for each player
      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();

      // Generate unique users (demonstrate the pattern, even if not used in this minimal test)
      const _user1 = generateTestUser();
      const _user2 = generateTestUser();

      // Create the coordinator
      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        // Register users and get tokens
        await test.step('Navigate to app', async () => {
          await page1.goto('/');
          await page2.goto('/');
        });

        // For this example, we'll demonstrate the coordinator API
        // In a real scenario, you would get actual JWT tokens after login
        await test.step('Verify coordinator created', async () => {
          expect(coordinator).toBeTruthy();
          expect(coordinator.getConnectedClientIds()).toEqual([]);
        });

        // Note: The actual connection would require valid JWT tokens
        // This demonstrates the intended usage pattern

        await test.step('Cleanup browser contexts', async () => {
          await context1.close();
          await context2.close();
        });
      } finally {
        await coordinator.cleanup();
      }
    });

    test('demonstrates the executeSequence pattern for turn-based actions', async () => {
      /**
       * This test demonstrates how executeSequence would be used
       * to coordinate alternating actions between players.
       *
       * The pattern:
       * 1. Player 1 makes a move
       * 2. Wait for Player 2 to see the update
       * 3. Player 2 makes a move
       * 4. Wait for Player 1 to see the update
       */

      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        // In a real test with valid tokens:
        /*
        await coordinator.connect('player1', {
          playerId: 'p1',
          token: player1Token,
        });
        await coordinator.connect('player2', {
          playerId: 'p2',
          token: player2Token,
        });

        // Join the same game
        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);

        // Execute alternating turns
        await coordinator.executeSequence([
          {
            clientId: 'player1',
            action: async () => {
              await coordinator.sendMoveById('player1', gameId, 'move-id-1');
            },
            waitFor: {
              type: 'gameState',
              predicate: (state) => state.currentPlayer === 2,
            },
            waitOnClientId: 'player2',
          },
          {
            clientId: 'player2',
            action: async () => {
              await coordinator.sendMoveById('player2', gameId, 'move-id-2');
            },
            waitFor: {
              type: 'gameState',
              predicate: (state) => state.currentPlayer === 1,
            },
            waitOnClientId: 'player1',
          },
        ]);
        */

        // Verify coordinator was created
        expect(coordinator).toBeTruthy();
        expect(coordinator.getConnectedClientIds()).toEqual([]);
      } finally {
        await coordinator.cleanup();
      }
    });

    test('demonstrates waitForAll pattern for synchronized events', async () => {
      /**
       * This test demonstrates how waitForAll would be used to wait
       * for multiple clients to receive the same event.
       *
       * Use case: Waiting for all players to see a game state update
       * after a single action.
       */

      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        // In a real test:
        /*
        // Connect all players
        await coordinator.connect('player1', { playerId: 'p1', token: token1 });
        await coordinator.connect('player2', { playerId: 'p2', token: token2 });
        await coordinator.connect('player3', { playerId: 'p3', token: token3 });

        // All join the same game
        await Promise.all([
          coordinator.joinGame('player1', gameId),
          coordinator.joinGame('player2', gameId),
          coordinator.joinGame('player3', gameId),
        ]);

        // Player 1 makes a move that all should see
        await coordinator.sendMoveById('player1', gameId, moveId);

        // Wait for all players to see the updated state
        const results = await coordinator.waitForAll(
          ['player1', 'player2', 'player3'],
          {
            type: 'gameState',
            predicate: (data) => {
              const msg = data as GameStateUpdateMessage;
              return msg.data.gameState.moveHistory.length > 0;
            },
            timeout: 10000,
          }
        );

        // Verify all received the same state
        expect(results.size).toBe(3);
        for (const [clientId, data] of results) {
          expect(isGameStateMessage(data)).toBe(true);
        }
        */

        expect(coordinator).toBeTruthy();
      } finally {
        await coordinator.cleanup();
      }
    });

    test('demonstrates message queue inspection', async () => {
      /**
       * This test demonstrates how to inspect the message queue
       * to verify which events a client has received.
       *
       * Useful for:
       * - Debugging test failures
       * - Verifying event ordering
       * - Checking for unexpected events
       */

      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        // In a real test:
        /*
        await coordinator.connect('observer', { playerId: 'obs', token: token });
        await coordinator.joinGame('observer', gameId);

        // Let some events happen...
        await page.waitForTimeout(2000);

        // Inspect received messages
        const messages = coordinator.getMessages('observer');
        console.log('Received messages:', messages);

        // Filter for specific events
        const gameStateMessages = coordinator.getMessagesMatching(
          'observer',
          (msg) => msg.eventName === 'game_state'
        );

        // Get the last known game state
        const lastState = coordinator.getLastGameState('observer');

        // Clear messages if needed (e.g., between test phases)
        coordinator.clearMessages('observer');
        */

        expect(coordinator).toBeTruthy();
      } finally {
        await coordinator.cleanup();
      }
    });
  });

  test.describe('Game Phase Waiting', () => {
    test('demonstrates waiting for specific game phases', async () => {
      /**
       * This test demonstrates how to wait for specific game phases
       * using the convenience methods.
       *
       * Phases: 'ring_placement', 'movement', 'capture', 'chain_capture',
       *         'line_processing', 'territory_processing'
       */

      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        // In a real test:
        /*
        await coordinator.connect('player1', { playerId: 'p1', token: token });
        await coordinator.joinGame('player1', gameId);

        // Wait for game to transition to movement phase
        const movePhaseState = await coordinator.waitForPhase(
          'player1',
          'movement',
          30000
        );

        expect(movePhaseState.data.gameState.currentPhase).toBe('movement');

        // Wait for a specific player's turn
        const turnState = await coordinator.waitForTurn('player1', 1, 15000);
        expect(turnState.data.gameState.currentPlayer).toBe(1);
        */

        expect(coordinator).toBeTruthy();
      } finally {
        await coordinator.cleanup();
      }
    });

    test('demonstrates waiting for game over', async () => {
      /**
       * This test demonstrates waiting for a game to complete.
       */

      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        // In a real test:
        /*
        // Set up game near victory...
        await coordinator.connect('player1', { playerId: 'p1', token: token });
        await coordinator.joinGame('player1', gameId);

        // Make the winning move
        await coordinator.sendMoveById('player1', gameId, winningMoveId);

        // Wait for game over
        const gameOverMsg = await coordinator.waitForGameOver('player1', 30000);

        expect(gameOverMsg.data.gameResult.winner).toBeDefined();
        expect(gameOverMsg.data.gameState.gameStatus).toBe('finished');
        */

        expect(coordinator).toBeTruthy();
      } finally {
        await coordinator.cleanup();
      }
    });

    test('territory near-victory fixture emits territory_control game_over', async ({
      browser,
    }) => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      // Create browser contexts to register users and obtain JWT tokens.
      const context1 = await browser.newContext();
      const context2 = await browser.newContext();
      const page1 = await context1.newPage();
      const page2 = await context2.newPage();

      const user1 = generateTestUser();
      const user2 = generateTestUser();

      // Helper to register and extract auth token from localStorage.
      const createUserAndGetToken = async (
        page: import('@playwright/test').Page,
        user: TestUser
      ): Promise<string> => {
        await registerUser(page, user.username, user.email, user.password);
        const token = await page.evaluate(() => {
          return localStorage.getItem('auth_token') ?? localStorage.getItem('token');
        });
        if (!token) {
          throw new Error('Failed to get auth token after registration');
        }
        return token;
      };

      try {
        // Register users and obtain tokens via UI.
        const token1 = await createUserAndGetToken(page1, user1);
        const token2 = await createUserAndGetToken(page2, user2);

        // Create a near-victory territory fixture game as user1.
        const { gameId } = await createFixtureGame(page1, {
          scenario: 'near_victory_territory',
          isRated: false,
        });

        // Connect both players via WebSocket.
        await coordinator.connect('player1', {
          playerId: user1.username,
          userId: user1.username,
          token: token1,
        });
        await coordinator.connect('player2', {
          playerId: user2.username,
          userId: user2.username,
          token: token2,
        });

        // Join the same game room.
        await coordinator.joinGame('player1', gameId);
        await coordinator.joinGame('player2', gameId);

        // Wait for game_over on both clients.
        const results = await coordinator.waitForAll(['player1', 'player2'], {
          type: 'gameOver',
          predicate: (data) => isGameOverMessage(data),
          timeout: 30_000,
        });

        const p1Msg = results.get('player1') as GameOverMessage | undefined;
        const p2Msg = results.get('player2') as GameOverMessage | undefined;

        expect(p1Msg).toBeDefined();
        expect(p2Msg).toBeDefined();
        expect(p1Msg!.data.gameResult.reason).toBe('territory_control');
        expect(p2Msg!.data.gameResult.reason).toBe('territory_control');
      } finally {
        await coordinator.cleanup();
        await context1.close();
        await context2.close();
      }
    });
  });

  test.describe('Error Handling', () => {
    test('handles connection timeout gracefully', async () => {
      const coordinator = new MultiClientCoordinator(serverUrl, 1000); // 1 second timeout

      try {
        // Attempting to connect with an invalid token should fail
        // The server should reject the connection during auth
        await expect(
          coordinator.connect('invalid-client', {
            playerId: 'test',
            token: 'invalid-jwt-token-that-will-be-rejected',
          })
        ).rejects.toThrow();
      } finally {
        await coordinator.cleanup();
      }
    });

    test('handles multiple disconnection calls gracefully', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      // These should not throw even though nothing is connected
      await coordinator.disconnect('nonexistent');
      await coordinator.disconnectAll();
      await coordinator.cleanup();

      // Multiple cleanup calls should be safe
      await coordinator.cleanup();
      await coordinator.cleanup();
    });

    test('throws when sending to disconnected client', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        await expect(
          coordinator.send('not-connected', 'join_game', { gameId: 'test' })
        ).rejects.toThrow("Client 'not-connected' is not connected");

        await expect(coordinator.joinGame('not-connected', 'test')).rejects.toThrow(
          "Client 'not-connected' is not connected"
        );
      } finally {
        await coordinator.cleanup();
      }
    });

    test('throws when getting messages from disconnected client', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        expect(() => coordinator.getMessages('not-connected')).toThrow(
          "Client 'not-connected' is not connected"
        );

        expect(() => coordinator.clearMessages('not-connected')).toThrow(
          "Client 'not-connected' is not connected"
        );
      } finally {
        await coordinator.cleanup();
      }
    });
  });

  test.describe('Utility Methods', () => {
    test('getConnectedClientIds returns empty array initially', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        const clientIds = coordinator.getConnectedClientIds();
        expect(clientIds).toEqual([]);
      } finally {
        await coordinator.cleanup();
      }
    });

    test('isConnected returns false for unknown clients', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        expect(coordinator.isConnected('unknown')).toBe(false);
      } finally {
        await coordinator.cleanup();
      }
    });

    test('getSocket returns null for unknown clients', async () => {
      const coordinator = createMultiClientCoordinator(serverUrl);

      try {
        expect(coordinator.getSocket('unknown')).toBeNull();
      } finally {
        await coordinator.cleanup();
      }
    });
  });
});

/**
 * Example: Full Multiplayer Game Flow
 * ============================================================================
 *
 * This is a comprehensive example showing how a complete multiplayer game
 * test would look using the MultiClientCoordinator.
 *
 * Note: This is commented out because it requires a running backend and
 * valid JWT tokens. It serves as documentation for the intended usage.
 */

/*
test('full multiplayer game flow example', async ({ browser }) => {
  const coordinator = createMultiClientCoordinator('http://localhost:3000');

  try {
    // Create users and get tokens (via UI or API)
    const context1 = await browser.newContext();
    const context2 = await browser.newContext();
    const page1 = await context1.newPage();
    const page2 = await context2.newPage();

    const user1 = generateTestUser();
    const user2 = generateTestUser();

    const token1 = await createUserAndGetToken(page1, user1);
    const token2 = await createUserAndGetToken(page2, user2);

    // Connect both players via WebSocket
    await coordinator.connect('player1', {
      playerId: user1.id,
      userId: user1.id,
      token: token1,
    });
    await coordinator.connect('player2', {
      playerId: user2.id,
      userId: user2.id,
      token: token2,
    });

    // Player 1 creates a game (via API)
    const gameId = 'created-game-id';

    // Both players join the game room
    await coordinator.joinGame('player1', gameId);
    await coordinator.joinGame('player2', gameId);

    // Wait for both to receive initial game state
    const initialStates = await coordinator.waitForAll(['player1', 'player2'], {
      type: 'gameState',
      predicate: (data) => isGameStateMessage(data),
      timeout: 10000,
    });

    expect(initialStates.size).toBe(2);

    // Get valid moves for player 1
    const p1State = coordinator.getLastGameState('player1');
    expect(p1State?.currentPlayer).toBe(1);

    // Execute ring placement turns
    await coordinator.executeSequence([
      {
        clientId: 'player1',
        action: async () => {
          // In real test, get moveId from validMoves
          const moves = await getValidMoves('player1');
          await coordinator.sendMoveById('player1', gameId, moves[0].id);
        },
        waitFor: {
          type: 'turn',
          predicate: (data) => {
            const msg = data as GameStateUpdateMessage;
            return msg.data.gameState.currentPlayer === 2;
          },
        },
        waitOnClientId: 'player2',
      },
      {
        clientId: 'player2',
        action: async () => {
          const moves = await getValidMoves('player2');
          await coordinator.sendMoveById('player2', gameId, moves[0].id);
        },
        waitFor: {
          type: 'turn',
          predicate: (data) => {
            const msg = data as GameStateUpdateMessage;
            return msg.data.gameState.currentPlayer === 1;
          },
        },
        waitOnClientId: 'player1',
      },
    ]);

    // Verify both players see consistent state
    const p1FinalState = coordinator.getLastGameState('player1');
    const p2FinalState = coordinator.getLastGameState('player2');

    expect(p1FinalState?.moveHistory.length).toBe(p2FinalState?.moveHistory.length);

    // Cleanup
    await coordinator.leaveGame('player1', gameId);
    await coordinator.leaveGame('player2', gameId);

    await context1.close();
    await context2.close();
  } finally {
    await coordinator.cleanup();
  }
});
*/
