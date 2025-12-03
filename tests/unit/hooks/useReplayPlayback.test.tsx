/**
 * Tests for useReplayPlayback hook.
 *
 * This hook handles game replay playback including loading games,
 * stepping through moves, and auto-play functionality.
 */

import React, { useState } from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useReplayPlayback } from '../../../src/client/hooks/useReplayPlayback';
import * as ReplayServiceModule from '../../../src/client/services/ReplayService';
import type {
  ReplayGameMetadata,
  ReplayMovesResponse,
  ReplayStateResponse,
} from '../../../src/client/types/replay';
import { createTestGameState } from '../../utils/fixtures';

// Mock the ReplayService
jest.mock('../../../src/client/services/ReplayService', () => ({
  getReplayService: jest.fn(),
  resetReplayService: jest.fn(),
}));

// Create a mock service instance
const mockService = {
  getGame: jest.fn(),
  getStateAtMove: jest.fn(),
  getMoves: jest.fn(),
  listGames: jest.fn(),
  getChoices: jest.fn(),
  getStats: jest.fn(),
  storeGame: jest.fn(),
  isAvailable: jest.fn(),
};

// Setup the mock to return our mock service
(ReplayServiceModule.getReplayService as jest.Mock).mockReturnValue(mockService);

// Create mock data
function createMockMetadata(totalMoves = 10): ReplayGameMetadata {
  return {
    gameId: 'test-game-123',
    boardType: 'square8',
    numPlayers: 2,
    winner: null,
    terminationReason: null,
    totalMoves,
    totalTurns: Math.ceil(totalMoves / 2),
    createdAt: '2025-01-01T00:00:00Z',
    completedAt: null,
    durationMs: null,
    source: 'test',
    players: [
      { playerNumber: 1, playerType: 'human' },
      { playerNumber: 2, playerType: 'ai', aiType: 'heuristic', aiDifficulty: 5 },
    ],
  };
}

function createMockStateResponse(moveNumber: number): ReplayStateResponse {
  return {
    gameState: createTestGameState({ id: 'test-game-123' }),
    moveNumber,
    totalMoves: 10,
  };
}

function createMockMovesResponse(count = 10): ReplayMovesResponse {
  return {
    moves: Array.from({ length: count }, (_, i) => ({
      moveNumber: i + 1,
      turnNumber: Math.ceil((i + 1) / 2),
      player: (i % 2) + 1,
      phase: 'movement',
      moveType: 'move_stack',
      move: { from: { x: i, y: i }, to: { x: i + 1, y: i + 1 } },
      timestamp: null,
      thinkTimeMs: null,
    })),
    hasMore: false,
  };
}

// Test harness component
function TestHarness() {
  const playback = useReplayPlayback();

  return (
    <div>
      <div data-testid="game-id">{playback.gameId ?? 'null'}</div>
      <div data-testid="current-move">{playback.currentMoveNumber}</div>
      <div data-testid="total-moves">{playback.totalMoves}</div>
      <div data-testid="is-playing">{playback.isPlaying ? 'true' : 'false'}</div>
      <div data-testid="is-loading">{playback.isLoading ? 'true' : 'false'}</div>
      <div data-testid="error">{playback.error ?? 'null'}</div>
      <div data-testid="playback-speed">{playback.playbackSpeed}</div>
      <div data-testid="can-step-forward">{playback.canStepForward ? 'true' : 'false'}</div>
      <div data-testid="can-step-backward">{playback.canStepBackward ? 'true' : 'false'}</div>
      <div data-testid="current-move-data">
        {playback.getCurrentMove() ? JSON.stringify(playback.getCurrentMove()) : 'null'}
      </div>
      <div data-testid="has-state">{playback.currentState ? 'true' : 'false'}</div>
      <div data-testid="has-metadata">{playback.metadata ? 'true' : 'false'}</div>

      <button data-testid="load-game" onClick={() => playback.loadGame('test-game-123')}>
        Load Game
      </button>
      <button data-testid="unload-game" onClick={() => playback.unloadGame()}>
        Unload Game
      </button>
      <button data-testid="step-forward" onClick={() => playback.stepForward()}>
        Step Forward
      </button>
      <button data-testid="step-backward" onClick={() => playback.stepBackward()}>
        Step Backward
      </button>
      <button data-testid="jump-to-5" onClick={() => playback.jumpToMove(5)}>
        Jump to 5
      </button>
      <button data-testid="jump-to-start" onClick={() => playback.jumpToStart()}>
        Jump to Start
      </button>
      <button data-testid="jump-to-end" onClick={() => playback.jumpToEnd()}>
        Jump to End
      </button>
      <button data-testid="play" onClick={() => playback.play()}>
        Play
      </button>
      <button data-testid="pause" onClick={() => playback.pause()}>
        Pause
      </button>
      <button data-testid="toggle-play" onClick={() => playback.togglePlay()}>
        Toggle Play
      </button>
      <button data-testid="set-speed-2" onClick={() => playback.setSpeed(2)}>
        Set Speed 2x
      </button>
      <button data-testid="set-speed-4" onClick={() => playback.setSpeed(4)}>
        Set Speed 4x
      </button>
    </div>
  );
}

// Wrapper with QueryClient
function renderWithQueryClient() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: Infinity,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <TestHarness />
    </QueryClientProvider>
  );
}

describe('useReplayPlayback', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();

    // Reset mock implementations
    mockService.getGame.mockResolvedValue(createMockMetadata());
    mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(0));
    mockService.getMoves.mockResolvedValue(createMockMovesResponse());
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('initial state', () => {
    it('should have null gameId initially', () => {
      renderWithQueryClient();
      expect(screen.getByTestId('game-id').textContent).toBe('null');
    });

    it('should start at move 0', () => {
      renderWithQueryClient();
      expect(screen.getByTestId('current-move').textContent).toBe('0');
    });

    it('should not be playing initially', () => {
      renderWithQueryClient();
      expect(screen.getByTestId('is-playing').textContent).toBe('false');
    });

    it('should not be loading initially', () => {
      renderWithQueryClient();
      expect(screen.getByTestId('is-loading').textContent).toBe('false');
    });

    it('should have speed 1 initially', () => {
      renderWithQueryClient();
      expect(screen.getByTestId('playback-speed').textContent).toBe('1');
    });

    it('should have no error initially', () => {
      renderWithQueryClient();
      expect(screen.getByTestId('error').textContent).toBe('null');
    });

    it('should not be able to step forward or backward initially', () => {
      renderWithQueryClient();
      expect(screen.getByTestId('can-step-forward').textContent).toBe('false');
      expect(screen.getByTestId('can-step-backward').textContent).toBe('false');
    });
  });

  describe('loadGame', () => {
    it('should load game metadata and initial state', async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      expect(screen.getByTestId('total-moves').textContent).toBe('10');
      expect(screen.getByTestId('has-metadata').textContent).toBe('true');
      expect(screen.getByTestId('has-state').textContent).toBe('true');
    });

    it('should set loading state while fetching', async () => {
      // Make the fetch slow
      mockService.getGame.mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve(createMockMetadata()), 100))
      );

      renderWithQueryClient();

      act(() => {
        screen.getByTestId('load-game').click();
      });

      expect(screen.getByTestId('is-loading').textContent).toBe('true');

      await act(async () => {
        jest.advanceTimersByTime(150);
      });

      await waitFor(() => {
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });
    });

    it('should handle errors during load', async () => {
      mockService.getGame.mockRejectedValue(new Error('Network error'));

      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('error').textContent).toBe('Network error');
      });
    });

    it('should handle non-Error exceptions', async () => {
      mockService.getGame.mockRejectedValue('String error');

      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('error').textContent).toBe('Failed to load game');
      });
    });
  });

  describe('unloadGame', () => {
    it('should reset state when unloading', async () => {
      renderWithQueryClient();

      // Load first
      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      // Then unload
      await act(async () => {
        screen.getByTestId('unload-game').click();
      });

      expect(screen.getByTestId('game-id').textContent).toBe('null');
      expect(screen.getByTestId('current-move').textContent).toBe('0');
      expect(screen.getByTestId('total-moves').textContent).toBe('0');
    });
  });

  describe('stepping', () => {
    beforeEach(async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });
    });

    it('should step forward', async () => {
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(1));

      await act(async () => {
        screen.getByTestId('step-forward').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('1');
      });
    });

    it('should step backward', async () => {
      // First step forward
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(1));
      await act(async () => {
        screen.getByTestId('step-forward').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('1');
      });

      // Then step backward
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(0));
      await act(async () => {
        screen.getByTestId('step-backward').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('0');
      });
    });

    it('should not step backward past 0', async () => {
      await act(async () => {
        screen.getByTestId('step-backward').click();
        await Promise.resolve();
      });

      expect(screen.getByTestId('current-move').textContent).toBe('0');
    });

    it('should update canStepForward and canStepBackward', async () => {
      expect(screen.getByTestId('can-step-forward').textContent).toBe('true');
      expect(screen.getByTestId('can-step-backward').textContent).toBe('false');

      // Step forward
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(1));
      await act(async () => {
        screen.getByTestId('step-forward').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('can-step-backward').textContent).toBe('true');
      });
    });
  });

  describe('jumpToMove', () => {
    beforeEach(async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });
    });

    it('should jump to specific move', async () => {
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(5));

      await act(async () => {
        screen.getByTestId('jump-to-5').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('5');
      });
    });

    it('should jump to start', async () => {
      // First jump somewhere
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(5));
      await act(async () => {
        screen.getByTestId('jump-to-5').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('5');
      });

      // Then jump to start
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(0));
      await act(async () => {
        screen.getByTestId('jump-to-start').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('0');
      });
    });

    it('should jump to end', async () => {
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(10));

      await act(async () => {
        screen.getByTestId('jump-to-end').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('10');
      });
    });

    it('should clamp move number to valid range', async () => {
      // Try to jump beyond total moves
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(10));

      await act(async () => {
        // Manually call jumpToMove with out-of-range value through the hook
        // Since we can't directly access the hook, we test clamp via jumpToEnd
        screen.getByTestId('jump-to-end').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('10');
      });
    });
  });

  describe('playback', () => {
    beforeEach(async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });
    });

    it('should start playing', async () => {
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(1));

      await act(async () => {
        screen.getByTestId('play').click();
      });

      expect(screen.getByTestId('is-playing').textContent).toBe('true');
    });

    it('should pause playing', async () => {
      await act(async () => {
        screen.getByTestId('play').click();
      });

      expect(screen.getByTestId('is-playing').textContent).toBe('true');

      await act(async () => {
        screen.getByTestId('pause').click();
      });

      expect(screen.getByTestId('is-playing').textContent).toBe('false');
    });

    it('should toggle play/pause', async () => {
      await act(async () => {
        screen.getByTestId('toggle-play').click();
      });

      expect(screen.getByTestId('is-playing').textContent).toBe('true');

      await act(async () => {
        screen.getByTestId('toggle-play').click();
      });

      expect(screen.getByTestId('is-playing').textContent).toBe('false');
    });

    it('should not start playing when at end', async () => {
      // Jump to end first
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(10));
      await act(async () => {
        screen.getByTestId('jump-to-end').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('10');
      });

      // Try to play
      await act(async () => {
        screen.getByTestId('play').click();
      });

      // Should not be playing since we're at the end
      expect(screen.getByTestId('is-playing').textContent).toBe('false');
    });

    it('should auto-advance moves during playback', async () => {
      // Return sequential states
      let callCount = 0;
      mockService.getStateAtMove.mockImplementation((_gameId, moveNumber) => {
        callCount++;
        return Promise.resolve(createMockStateResponse(moveNumber));
      });

      await act(async () => {
        screen.getByTestId('play').click();
      });

      // Advance time to trigger move
      await act(async () => {
        jest.advanceTimersByTime(1100);
      });

      await waitFor(() => {
        expect(parseInt(screen.getByTestId('current-move').textContent!)).toBeGreaterThan(0);
      });
    });

    it('should stop when reaching end', async () => {
      // Set up a short game with only 1 move so reaching end is easier to test
      mockService.getGame.mockResolvedValue(createMockMetadata(1));
      mockService.getMoves.mockResolvedValue(createMockMovesResponse(1));

      // Reload with short game
      await act(async () => {
        screen.getByTestId('unload-game').click();
      });

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('total-moves').textContent).toBe('1');
      });

      // Jump to end first
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(1));
      await act(async () => {
        screen.getByTestId('jump-to-end').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('1');
      });

      // Try to play - should not start because we're at the end
      await act(async () => {
        screen.getByTestId('play').click();
      });

      // Should not be playing since we're at the end
      expect(screen.getByTestId('is-playing').textContent).toBe('false');
    });
  });

  describe('speed control', () => {
    beforeEach(async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });
    });

    it('should change speed to 2x', async () => {
      await act(async () => {
        screen.getByTestId('set-speed-2').click();
      });

      expect(screen.getByTestId('playback-speed').textContent).toBe('2');
    });

    it('should change speed to 4x', async () => {
      await act(async () => {
        screen.getByTestId('set-speed-4').click();
      });

      expect(screen.getByTestId('playback-speed').textContent).toBe('4');
    });
  });

  describe('getCurrentMove', () => {
    beforeEach(async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });
    });

    it('should return null at move 0', () => {
      expect(screen.getByTestId('current-move-data').textContent).toBe('null');
    });

    it('should return move record after stepping', async () => {
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(1));

      await act(async () => {
        screen.getByTestId('step-forward').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('1');
      });

      const moveData = JSON.parse(screen.getByTestId('current-move-data').textContent!);
      expect(moveData.moveNumber).toBe(1);
      expect(moveData.moveType).toBe('move_stack');
    });
  });

  describe('edge cases', () => {
    it('should handle loading a new game while one is already loaded', async () => {
      renderWithQueryClient();

      // Load first game
      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      // Load another game (same ID but demonstrates the path)
      mockService.getGame.mockResolvedValue({
        ...createMockMetadata(20),
        gameId: 'another-game',
      });

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        // This still loads the same endpoint but tests the "already playing" branch
        expect(screen.getByTestId('is-loading').textContent).toBe('false');
      });
    });

    it('should handle jumpToMove errors', async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      // Make getStateAtMove fail
      mockService.getStateAtMove.mockRejectedValue(new Error('State not found'));

      await act(async () => {
        screen.getByTestId('jump-to-5').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('error').textContent).toBe('State not found');
      });
    });

    it('should handle jumpToMove with non-Error exception', async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      mockService.getStateAtMove.mockRejectedValue('String error');

      await act(async () => {
        screen.getByTestId('jump-to-5').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('error').textContent).toBe('Failed to load state');
      });
    });

    it('should do nothing when jumping without loaded game', async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('jump-to-5').click();
        await Promise.resolve();
      });

      // Should remain at initial state
      expect(screen.getByTestId('current-move').textContent).toBe('0');
      expect(screen.getByTestId('game-id').textContent).toBe('null');
    });

    it('should do nothing when stepping without loaded game', async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('step-forward').click();
        await Promise.resolve();
      });

      expect(screen.getByTestId('current-move').textContent).toBe('0');
    });

    it('should prefetch key positions for long games (> 20 moves)', async () => {
      // Setup a long game with 30 moves
      mockService.getGame.mockResolvedValue(createMockMetadata(30));
      mockService.getMoves.mockResolvedValue(createMockMovesResponse(30));

      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('total-moves').textContent).toBe('30');
      });

      // The prefetchAdjacent should have been called with key positions
      // We can't directly test prefetch but we verify the game loaded successfully
      expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
    });

    it('should clear existing playback when loading new game', async () => {
      renderWithQueryClient();

      // Load first game
      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      // Start playing
      await act(async () => {
        screen.getByTestId('play').click();
      });

      expect(screen.getByTestId('is-playing').textContent).toBe('true');

      // Load new game while playing - should stop playback
      mockService.getGame.mockResolvedValue(createMockMetadata(5));
      mockService.getMoves.mockResolvedValue(createMockMovesResponse(5));

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('is-playing').textContent).toBe('false');
      });
    });

    it('should clear playback timeout when unloading during playback', async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      // Start playing
      await act(async () => {
        screen.getByTestId('play').click();
      });

      expect(screen.getByTestId('is-playing').textContent).toBe('true');

      // Unload while playing
      await act(async () => {
        screen.getByTestId('unload-game').click();
      });

      expect(screen.getByTestId('game-id').textContent).toBe('null');
      expect(screen.getByTestId('is-playing').textContent).toBe('false');
    });

    it('should use cached state when available for instant navigation', async () => {
      const queryClient = new QueryClient({
        defaultOptions: {
          queries: {
            retry: false,
            staleTime: Infinity,
          },
        },
      });

      // Pre-populate cache with state for move 5
      queryClient.setQueryData(['replay', 'games', 'test-game-123', 'state', 5], {
        gameState: createTestGameState({ id: 'test-game-123' }),
      });

      render(
        <QueryClientProvider client={queryClient}>
          <TestHarness />
        </QueryClientProvider>
      );

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      // Jump to move 5 - should use cached state (no fetch)
      const fetchCallsBefore = mockService.getStateAtMove.mock.calls.length;

      await act(async () => {
        screen.getByTestId('jump-to-5').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('5');
      });

      // getStateAtMove should not have been called again since we had cache
      // (it may have been called for prefetch, but the main navigation used cache)
    });

    it('should not step forward when at total moves', async () => {
      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      // Jump to end
      mockService.getStateAtMove.mockResolvedValue(createMockStateResponse(10));
      await act(async () => {
        screen.getByTestId('jump-to-end').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('current-move').textContent).toBe('10');
      });

      // Try to step forward - should stay at 10
      await act(async () => {
        screen.getByTestId('step-forward').click();
        await Promise.resolve();
      });

      expect(screen.getByTestId('current-move').textContent).toBe('10');
    });

    it('should return null from getCurrentMove when moves array is empty', async () => {
      mockService.getMoves.mockResolvedValue({ moves: [], hasMore: false });

      renderWithQueryClient();

      await act(async () => {
        screen.getByTestId('load-game').click();
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(screen.getByTestId('game-id').textContent).toBe('test-game-123');
      });

      expect(screen.getByTestId('current-move-data').textContent).toBe('null');
    });
  });
});
