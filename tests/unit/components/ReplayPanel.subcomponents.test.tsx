import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { GameList } from '../../../src/client/components/ReplayPanel/GameList';
import { GameFilters } from '../../../src/client/components/ReplayPanel/GameFilters';
import type { ReplayGameMetadata, ReplayGameQueryParams } from '../../../src/client/types/replay';

describe('ReplayPanel subcomponents', () => {
  describe('GameList', () => {
    function createGame(overrides: Partial<ReplayGameMetadata> = {}): ReplayGameMetadata {
      return {
        gameId: 'game-1',
        boardType: 'square8',
        numPlayers: 2,
        winner: null,
        terminationReason: null,
        totalMoves: 10,
        totalTurns: 5,
        createdAt: '2025-01-01T12:00:00Z',
        completedAt: null,
        durationMs: null,
        source: 'sandbox',
        ...overrides,
      } as ReplayGameMetadata;
    }

    it('renders loading state when loading and no games yet', () => {
      render(
        <GameList
          games={[]}
          selectedGameId={null}
          onSelectGame={jest.fn()}
          isLoading={true}
          error={null}
          total={0}
          offset={0}
          limit={10}
          hasMore={false}
          onPageChange={jest.fn()}
        />
      );

      expect(screen.getByText(/Loading games.../i)).toBeInTheDocument();
    });

    it('renders empty state when there are no games', () => {
      render(
        <GameList
          games={[]}
          selectedGameId={null}
          onSelectGame={jest.fn()}
          isLoading={false}
          error={null}
          total={0}
          offset={0}
          limit={10}
          hasMore={false}
          onPageChange={jest.fn()}
        />
      );

      expect(
        screen.getByText(
          /No games found\. Try adjusting your filters or run some self-play games\./i
        )
      ).toBeInTheDocument();
    });

    it('renders error message when error is provided', () => {
      render(
        <GameList
          games={[]}
          selectedGameId={null}
          onSelectGame={jest.fn()}
          isLoading={false}
          error="Backend unavailable"
          total={0}
          offset={0}
          limit={10}
          hasMore={false}
          onPageChange={jest.fn()}
        />
      );

      expect(screen.getByText(/Error: Backend unavailable/i)).toBeInTheDocument();
    });

    it('renders non-empty list with metadata and supports selection', () => {
      const onSelectGame = jest.fn();
      const games: ReplayGameMetadata[] = [
        createGame({
          gameId: 'game-1',
          boardType: 'square8',
          numPlayers: 2,
          winner: 1,
          terminationReason: 'ring_elimination',
          totalMoves: 42,
        }),
        createGame({
          gameId: 'game-2',
          boardType: 'hexagonal',
          numPlayers: 4,
          winner: null,
          terminationReason: null,
          totalMoves: 10,
        }),
      ];

      render(
        <GameList
          games={games}
          selectedGameId="game-2"
          onSelectGame={onSelectGame}
          isLoading={false}
          error={null}
          total={25}
          offset={0}
          limit={10}
          hasMore={true}
          onPageChange={jest.fn()}
        />
      );

      // Board type formatting: square8 -> 8×8, hexagonal -> Hex
      expect(screen.getByText('8×8')).toBeInTheDocument();
      expect(screen.getByText('Hex')).toBeInTheDocument();

      // Winner metadata
      expect(screen.getByText('P1 won')).toBeInTheDocument();
      expect(screen.getByText('42 moves')).toBeInTheDocument();

      // Termination reason mapping
      expect(screen.getByText('Ring Elim.')).toBeInTheDocument();

      // Pagination summary
      expect(screen.getByText('1–2 of 25')).toBeInTheDocument();

      // Clicking a game row selects it
      const secondRow = screen.getByText('Hex').closest('button');
      expect(secondRow).not.toBeNull();
      if (secondRow) {
        fireEvent.click(secondRow);
      }
      expect(onSelectGame).toHaveBeenCalledWith('game-2');
    });

    it('handles pagination buttons and disabled states', () => {
      const onPageChange = jest.fn();
      const games: ReplayGameMetadata[] = [createGame({ gameId: 'game-1' })];

      const { rerender } = render(
        <GameList
          games={games}
          selectedGameId={null}
          onSelectGame={jest.fn()}
          isLoading={false}
          error={null}
          total={30}
          offset={10}
          limit={10}
          hasMore={true}
          onPageChange={onPageChange}
        />
      );

      const prevButton = screen.getByRole('button', { name: 'Previous page' });
      const nextButton = screen.getByRole('button', { name: 'Next page' });

      expect(prevButton).not.toBeDisabled();
      expect(nextButton).not.toBeDisabled();

      fireEvent.click(prevButton);
      fireEvent.click(nextButton);

      expect(onPageChange).toHaveBeenNthCalledWith(1, 0);
      expect(onPageChange).toHaveBeenNthCalledWith(2, 20);

      // When at the first page and no more results, buttons should be disabled
      rerender(
        <GameList
          games={games}
          selectedGameId={null}
          onSelectGame={jest.fn()}
          isLoading={false}
          error={null}
          total={1}
          offset={0}
          limit={10}
          hasMore={false}
          onPageChange={onPageChange}
        />
      );

      expect(screen.getByRole('button', { name: 'Previous page' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Next page' })).toBeDisabled();
    });
  });

  describe('GameFilters', () => {
    function renderWithFilters(
      filters: ReplayGameQueryParams,
      onFilterChange: jest.Mock = jest.fn()
    ) {
      render(<GameFilters filters={filters} onFilterChange={onFilterChange} />);
      return onFilterChange;
    }

    it('clears board type filter and resets offset when selecting "All Boards"', () => {
      const onFilterChange = renderWithFilters({
        board_type: 'square8',
        offset: 20,
      });

      fireEvent.change(screen.getByLabelText('Filter by board type'), {
        target: { value: '' },
      });

      expect(onFilterChange).toHaveBeenCalledTimes(1);
      const nextFilters = onFilterChange.mock.calls[0][0] as ReplayGameQueryParams;
      expect(nextFilters.board_type).toBeUndefined();
      expect(nextFilters.offset).toBe(0);
    });

    it('parses num_players as a number and resets offset', () => {
      const onFilterChange = renderWithFilters({
        num_players: 2,
        offset: 10,
      });

      fireEvent.change(screen.getByLabelText('Filter by player count'), {
        target: { value: '3' },
      });

      expect(onFilterChange).toHaveBeenCalledTimes(1);
      const nextFilters = onFilterChange.mock.calls[0][0] as ReplayGameQueryParams;
      expect(nextFilters.num_players).toBe(3);
      expect(nextFilters.offset).toBe(0);
    });

    it('sets termination_reason and source as strings', () => {
      // Simulate controlled parent state: first update outcome, then source.
      let filters: ReplayGameQueryParams = {};
      const onFilterChange = jest.fn();
      const { rerender } = render(
        <GameFilters filters={filters} onFilterChange={onFilterChange} />
      );

      // First, set termination_reason
      fireEvent.change(screen.getByLabelText('Filter by outcome'), {
        target: { value: 'territory' },
      });

      expect(onFilterChange).toHaveBeenCalledTimes(1);
      filters = onFilterChange.mock.calls[0][0] as ReplayGameQueryParams;
      expect(filters.termination_reason).toBe('territory');
      expect(filters.offset).toBe(0);

      // Rerender with updated filters and set source
      onFilterChange.mockClear();
      rerender(<GameFilters filters={filters} onFilterChange={onFilterChange} />);

      fireEvent.change(screen.getByLabelText('Filter by source'), {
        target: { value: 'self_play' },
      });

      expect(onFilterChange).toHaveBeenCalledTimes(1);
      const nextFilters = onFilterChange.mock.calls[0][0] as ReplayGameQueryParams;
      expect(nextFilters.termination_reason).toBe('territory');
      expect(nextFilters.source).toBe('self_play');
      expect(nextFilters.offset).toBe(0);
    });
  });
});
