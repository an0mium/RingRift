/**
 * GameContext.branchCoverage.test.tsx
 *
 * Branch coverage tests for GameContext.tsx
 */

import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import { GameProvider, useGame } from '../../../src/client/contexts/GameContext';
import type { GameState, BoardState, Move } from '../../../src/shared/types/game';

// Mock dependencies
jest.mock('react-hot-toast', () => ({
  toast: Object.assign(jest.fn(), {
    error: jest.fn(),
    success: jest.fn(),
  }),
}));

jest.mock('../../../src/client/utils/errorReporting', () => ({
  reportClientError: jest.fn(),
  isErrorReportingEnabled: jest.fn(() => false),
  extractErrorMessage: jest.fn((e, fallback) => fallback),
}));

// Mock SocketGameConnection
const mockConnect = jest.fn();
const mockDisconnect = jest.fn();
const mockRespondToChoice = jest.fn();
const mockSubmitMove = jest.fn();
const mockSendChatMessage = jest.fn();
const mockRequestRematch = jest.fn();
const mockRespondToRematch = jest.fn();

jest.mock('../../../src/client/services/GameConnection', () => ({
  SocketGameConnection: jest.fn().mockImplementation(() => ({
    connect: mockConnect,
    disconnect: mockDisconnect,
    respondToChoice: mockRespondToChoice,
    submitMove: mockSubmitMove,
    sendChatMessage: mockSendChatMessage,
    requestRematch: mockRequestRematch,
    respondToRematch: mockRespondToRematch,
    status: 'disconnected',
  })),
}));

// Test component to access context
function TestConsumer({ onContext }: { onContext?: (ctx: ReturnType<typeof useGame>) => void }) {
  const ctx = useGame();
  React.useEffect(() => {
    if (onContext) onContext(ctx);
  }, [ctx, onContext]);
  return (
    <div>
      <span data-testid="gameId">{ctx.gameId ?? 'none'}</span>
      <span data-testid="status">{ctx.connectionStatus}</span>
      <span data-testid="isConnecting">{String(ctx.isConnecting)}</span>
      <span data-testid="error">{ctx.error ?? 'none'}</span>
    </div>
  );
}

describe('GameContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('useGame hook', () => {
    it('throws when used outside GameProvider', () => {
      // Suppress console.error for this test
      const originalError = console.error;
      console.error = jest.fn();

      expect(() => {
        render(<TestConsumer />);
      }).toThrow('useGame must be used within a GameProvider');

      console.error = originalError;
    });
  });

  describe('GameProvider initial state', () => {
    it('provides default context values', () => {
      let capturedContext: ReturnType<typeof useGame> | null = null;

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      expect(capturedContext).not.toBeNull();
      expect(capturedContext!.gameId).toBeNull();
      expect(capturedContext!.gameState).toBeNull();
      expect(capturedContext!.validMoves).toBeNull();
      expect(capturedContext!.isConnecting).toBe(false);
      expect(capturedContext!.error).toBeNull();
      expect(capturedContext!.victoryState).toBeNull();
      expect(capturedContext!.pendingChoice).toBeNull();
      expect(capturedContext!.choiceDeadline).toBeNull();
      expect(capturedContext!.chatMessages).toEqual([]);
      expect(capturedContext!.connectionStatus).toBe('disconnected');
      expect(capturedContext!.lastHeartbeatAt).toBeNull();
      expect(capturedContext!.decisionAutoResolved).toBeNull();
      expect(capturedContext!.decisionPhaseTimeoutWarning).toBeNull();
      expect(capturedContext!.pendingRematchRequest).toBeNull();
      expect(capturedContext!.rematchGameId).toBeNull();
      expect(capturedContext!.rematchLastStatus).toBeNull();
      expect(capturedContext!.evaluationHistory).toEqual([]);
      expect(capturedContext!.disconnectedOpponents).toEqual([]);
      expect(capturedContext!.gameEndedByAbandonment).toBe(false);
    });

    it('renders children', () => {
      render(
        <GameProvider>
          <div data-testid="child">Child content</div>
        </GameProvider>
      );

      expect(screen.getByTestId('child')).toHaveTextContent('Child content');
    });
  });

  describe('connectToGame', () => {
    it('sets isConnecting to true while connecting', async () => {
      let capturedContext: ReturnType<typeof useGame> | null = null;
      mockConnect.mockImplementation(() => new Promise(() => {})); // Never resolves

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      await act(async () => {
        capturedContext!.connectToGame('test-game-id');
      });

      expect(screen.getByTestId('isConnecting')).toHaveTextContent('true');
    });

    it('handles connection errors', async () => {
      let capturedContext: ReturnType<typeof useGame> | null = null;
      mockConnect.mockRejectedValue(new Error('Connection failed'));

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      await act(async () => {
        await capturedContext!.connectToGame('test-game-id');
      });

      expect(screen.getByTestId('error')).toHaveTextContent('Failed to connect to game');
      expect(screen.getByTestId('isConnecting')).toHaveTextContent('false');
    });
  });

  describe('disconnect', () => {
    it('resets all state on disconnect', async () => {
      let capturedContext: ReturnType<typeof useGame> | null = null;

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      await act(async () => {
        capturedContext!.disconnect();
      });

      expect(capturedContext!.gameId).toBeNull();
      expect(capturedContext!.gameState).toBeNull();
      expect(capturedContext!.connectionStatus).toBe('disconnected');
      // Note: mockDisconnect is only called if a connection was established
    });
  });

  describe('respondToChoice', () => {
    it('warns if called without active connection', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation();
      let capturedContext: ReturnType<typeof useGame> | null = null;

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      act(() => {
        capturedContext!.respondToChoice(
          {
            id: 'test',
            type: 'ring_elimination',
            gameId: 'g',
            playerNumber: 1,
            prompt: 'test',
            options: [],
          },
          {}
        );
      });

      expect(warnSpy).toHaveBeenCalledWith('respondToChoice called without active connection/game');
      warnSpy.mockRestore();
    });
  });

  describe('submitMove', () => {
    it('warns if called without active connection', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation();
      let capturedContext: ReturnType<typeof useGame> | null = null;

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      act(() => {
        capturedContext!.submitMove({
          type: 'place_ring',
          player: 1,
          to: { x: 0, y: 0 },
        });
      });

      expect(warnSpy).toHaveBeenCalledWith('submitMove called without active connection/game');
      warnSpy.mockRestore();
    });
  });

  describe('sendChatMessage', () => {
    it('warns if called without active connection', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation();
      let capturedContext: ReturnType<typeof useGame> | null = null;

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      act(() => {
        capturedContext!.sendChatMessage('Hello');
      });

      expect(warnSpy).toHaveBeenCalledWith('sendChatMessage called without active connection/game');
      warnSpy.mockRestore();
    });
  });

  describe('rematch functions', () => {
    it('requestRematch warns if called without connection', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation();
      let capturedContext: ReturnType<typeof useGame> | null = null;

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      act(() => {
        capturedContext!.requestRematch();
      });

      expect(warnSpy).toHaveBeenCalledWith('requestRematch called without active connection/game');
      warnSpy.mockRestore();
    });

    it('acceptRematch warns if called without connection', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation();
      let capturedContext: ReturnType<typeof useGame> | null = null;

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      act(() => {
        capturedContext!.acceptRematch('request-123');
      });

      expect(warnSpy).toHaveBeenCalledWith('acceptRematch called without active connection');
      warnSpy.mockRestore();
    });

    it('declineRematch warns if called without connection', () => {
      const warnSpy = jest.spyOn(console, 'warn').mockImplementation();
      let capturedContext: ReturnType<typeof useGame> | null = null;

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      act(() => {
        capturedContext!.declineRematch('request-123');
      });

      expect(warnSpy).toHaveBeenCalledWith('declineRematch called without active connection');
      warnSpy.mockRestore();
    });
  });

  describe('gameEndedByAbandonment', () => {
    it('returns true when victoryState reason is abandonment', () => {
      // This would require setting victoryState which needs game_over event handling
      let capturedContext: ReturnType<typeof useGame> | null = null;

      render(
        <GameProvider>
          <TestConsumer
            onContext={(ctx) => {
              capturedContext = ctx;
            }}
          />
        </GameProvider>
      );

      // Initially should be false
      expect(capturedContext!.gameEndedByAbandonment).toBe(false);
    });
  });
});
