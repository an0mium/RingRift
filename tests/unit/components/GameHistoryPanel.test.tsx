import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameHistoryPanel } from '../../../src/client/components/GameHistoryPanel';
import type { GameHistoryResponse, GameHistoryMove } from '../../../src/client/services/api';

// Mock the game API so we can control the history payload returned to the panel
const mockGetGameHistory = jest.fn<Promise<GameHistoryResponse>, [string]>();

jest.mock('../../../src/client/services/api', () => {
  const actual = jest.requireActual('../../../src/client/services/api');
  return {
    ...actual,
    gameApi: {
      ...actual.gameApi,
      getGameHistory: (gameId: string) => mockGetGameHistory(gameId),
    },
  };
});

describe('GameHistoryPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  function createMove(overrides: Partial<GameHistoryMove> = {}): GameHistoryMove {
    return {
      moveNumber: 1,
      playerId: 'player-1',
      playerName: 'Player One',
      moveType: 'place_ring',
      moveData: {
        type: 'place_ring',
        player: 1,
        to: { x: 3, y: 3 },
      },
      timestamp: new Date('2024-01-15T10:05:30Z').toISOString(),
      ...overrides,
    };
  }

  it('renders an auto-resolve badge for moves with autoResolved metadata', async () => {
    const autoResolvedMove = createMove({
      moveNumber: 1,
      autoResolved: {
        reason: 'timeout',
        choiceKind: 'line',
        choiceType: 'reward',
      },
    });

    const normalMove = createMove({
      moveNumber: 2,
      playerId: 'player-2',
      playerName: 'Player Two',
      autoResolved: undefined,
    });

    const history: GameHistoryResponse = {
      gameId: 'game-1',
      moves: [autoResolvedMove, normalMove],
      totalMoves: 2,
    };

    mockGetGameHistory.mockResolvedValue(history);

    render(<GameHistoryPanel gameId="game-1" />);

    // Wait for history to load and the badge to appear
    const badge = await waitFor(() => screen.getByTestId('auto-resolved-badge'));

    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent('Auto-resolved (timeout)');

    // There should be exactly one auto-resolve badge even though we have two moves
    const allBadges = screen.getAllByTestId('auto-resolved-badge');
    expect(allBadges).toHaveLength(1);
  });

  it.each([
    { reason: 'timeout', expected: 'Auto-resolved (timeout)' },
    { reason: 'disconnected', expected: 'Auto-resolved (disconnect)' },
    { reason: 'fallback', expected: 'Auto-resolved (fallback move)' },
    // Unknown reasons should be surfaced verbatim.
    { reason: 'custom_reason' as any, expected: 'Auto-resolved (custom_reason)' },
  ])('maps auto-resolve reason %s to a readable label', async ({ reason, expected }) => {
    const move = createMove({
      autoResolved: {
        reason,
        choiceKind: 'line',
        choiceType: 'reward',
      } as any,
    });

    const history: GameHistoryResponse = {
      gameId: 'game-auto-reasons',
      moves: [move],
      totalMoves: 1,
    };

    mockGetGameHistory.mockResolvedValue(history);

    render(<GameHistoryPanel gameId="game-auto-reasons" />);

    const badge = await waitFor(() => screen.getByTestId('auto-resolved-badge'));
    expect(badge).toHaveTextContent(expected);
  });

  it('does not render an auto-resolve badge when no moves are auto-resolved', async () => {
    const normalMove = createMove({
      moveNumber: 1,
      autoResolved: undefined,
    });

    const history: GameHistoryResponse = {
      gameId: 'game-2',
      moves: [normalMove],
      totalMoves: 1,
    };

    mockGetGameHistory.mockResolvedValue(history);

    render(<GameHistoryPanel gameId="game-2" />);

    // Wait for loading to complete by asserting that the move row appears
    await waitFor(() => {
      expect(screen.getByText('Player One')).toBeInTheDocument();
    });

    expect(screen.queryByTestId('auto-resolved-badge')).not.toBeInTheDocument();
  });

  it.each([
    { reason: 'timeout', winner: 1, expectedLabel: 'Result: Timeout', expectWinner: true },
    {
      reason: 'resignation',
      winner: 2,
      expectedLabel: 'Result: Resignation',
      expectWinner: true,
    },
    {
      reason: 'abandonment',
      winner: null,
      expectedLabel: 'Result: Abandonment',
      expectWinner: false,
    },
  ] as const)(
    'renders terminal result banner for $reason games',
    async ({ reason, winner, expectedLabel, expectWinner }) => {
      const history: GameHistoryResponse = {
        gameId: 'game-terminal',
        moves: [createMove()],
        totalMoves: 1,
        result: {
          // Cast needed because reason is a narrowed string literal in this test
          reason: reason as GameHistoryResponse['result'] extends { reason: infer R } ? R : never,
          winner,
        },
      };

      mockGetGameHistory.mockResolvedValue(history);

      render(<GameHistoryPanel gameId="game-terminal" />);

      await waitFor(() => {
        expect(screen.getByText('Player One')).toBeInTheDocument();
      });

      expect(screen.getByText(expectedLabel)).toBeInTheDocument();

      const winnerText = screen.queryByText(/Winner: P/i);
      if (expectWinner) {
        expect(winnerText).toBeInTheDocument();
      } else {
        expect(winnerText).not.toBeInTheDocument();
      }
    }
  );

  it('formats unknown move types using fallback spacing', async () => {
    const history: GameHistoryResponse = {
      gameId: 'game-custom-move',
      moves: [
        createMove({
          moveType: 'custom_move_type',
          moveData: {
            type: 'custom_move_type',
            player: 1,
            to: { x: 1, y: 2 },
          },
        }),
      ],
      totalMoves: 1,
    };

    mockGetGameHistory.mockResolvedValue(history);

    render(<GameHistoryPanel gameId="game-custom-move" />);

    await waitFor(() => {
      // formatMoveType should fall back to replacing underscores with spaces.
      expect(screen.getByText('custom move type')).toBeInTheDocument();
    });
  });

  it('shows loading spinner and message while fetching history', async () => {
    // Return a promise that we resolve manually so we can observe the loading state.
    let resolveHistory: (value: GameHistoryResponse) => void;
    const historyPromise = new Promise<GameHistoryResponse>((resolve) => {
      resolveHistory = resolve;
    });

    mockGetGameHistory.mockReturnValue(historyPromise);

    render(<GameHistoryPanel gameId="loading-game" />);

    // Immediately after render, the loading state should be visible.
    expect(screen.getByText('Loading history...')).toBeInTheDocument();

    // Now resolve the history and wait for the move row to appear.
    resolveHistory!({
      gameId: 'loading-game',
      moves: [createMove()],
      totalMoves: 1,
    } satisfies GameHistoryResponse);

    await waitFor(() => {
      expect(screen.getByText('Player One')).toBeInTheDocument();
    });
  });

  it('renders an error state and allows retry when the history request fails', async () => {
    mockGetGameHistory.mockRejectedValueOnce(new Error('Network failure'));

    render(<GameHistoryPanel gameId="error-game" />);

    await waitFor(() => {
      expect(screen.getByText(/Network failure/)).toBeInTheDocument();
    });

    // Retry button should clear error and trigger a new fetch.
    const retryButton = screen.getByRole('button', { name: /Retry/i });
    expect(retryButton).toBeInTheDocument();

    const successHistory: GameHistoryResponse = {
      gameId: 'error-game',
      moves: [createMove({ moveNumber: 1 })],
      totalMoves: 1,
    };
    mockGetGameHistory.mockResolvedValueOnce(successHistory);

    fireEvent.click(retryButton);

    await waitFor(() => {
      expect(screen.getByText('Player One')).toBeInTheDocument();
    });
  });

  it('renders an empty state message when there are no moves', async () => {
    const history: GameHistoryResponse = {
      gameId: 'empty-game',
      moves: [],
      totalMoves: 0,
    };

    mockGetGameHistory.mockResolvedValue(history);

    render(<GameHistoryPanel gameId="empty-game" />);

    await waitFor(() => {
      expect(screen.getByText('No moves recorded yet.')).toBeInTheDocument();
    });
  });

  it('does not fetch history when initially collapsed until expanded', async () => {
    mockGetGameHistory.mockResolvedValue({
      gameId: 'lazy-game',
      moves: [createMove()],
      totalMoves: 1,
    });

    render(<GameHistoryPanel gameId="lazy-game" defaultCollapsed={true} />);

    // Initially collapsed: useEffect early-returns; no fetch yet.
    expect(mockGetGameHistory).not.toHaveBeenCalled();

    // Click header to expand, which should trigger fetch.
    const headerButton = screen.getByRole('button', { name: /Move History/i });
    fireEvent.click(headerButton);

    await waitFor(() => {
      expect(mockGetGameHistory).toHaveBeenCalledWith('lazy-game');
      expect(screen.getByText('Player One')).toBeInTheDocument();
    });
  });

  it('renders from/to position description including z-coordinate when present', async () => {
    const history: GameHistoryResponse = {
      gameId: 'game-positions',
      moves: [
        createMove({
          moveData: {
            type: 'move_stack',
            player: 1,
            from: { x: 0, y: 0, z: 0 },
            to: { x: 1, y: -1, z: 0 },
          },
        }),
      ],
      totalMoves: 1,
    };

    mockGetGameHistory.mockResolvedValue(history);

    render(<GameHistoryPanel gameId="game-positions" />);

    await waitFor(() => {
      // getPositionDescription builds a single string with both endpoints and
      // includes z when it is defined.
      expect(screen.getByText('from (0,0,0) → to (1,-1,0)')).toBeInTheDocument();
    });
  });
});
