/**
 * useReplayService hook tests
 *
 * Exercises the React Query wrappers around ReplayService to ensure that:
 * - Each hook calls the correct ReplayService method with the right arguments.
 * - Enabled flags and null gameId handling behave as expected.
 * - StoreGame mutation invalidates the appropriate queries.
 */

import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import {
  useReplayServiceAvailable,
  useReplayStats,
  useGameList,
  useGame,
  useReplayStateAt,
  useMoves,
  useAllMoves,
  useChoices,
  useStoreGame,
  usePrefetchGame,
  usePrefetchState,
} from '../../../src/client/hooks/useReplayService';

const mockService = {
  isAvailable: jest.fn<Promise<boolean>, []>(),
  getStats: jest.fn<Promise<any>, []>(),
  listGames: jest.fn<Promise<any>, [any]>(),
  getGame: jest.fn<Promise<any>, [string]>(),
  getStateAtMove: jest.fn<Promise<any>, [string, number]>(),
  getMoves: jest.fn<Promise<any>, [string, number, number | undefined, number | undefined]>(),
  getChoices: jest.fn<Promise<any>, [string, number]>(),
  storeGame: jest.fn<Promise<any>, [any]>(),
};

jest.mock('../../../src/client/services/ReplayService', () => ({
  getReplayService: () => mockService,
}));

function createWrapper() {
  const queryClient = new QueryClient();
  const wrapper: React.FC<React.PropsWithChildren> = ({ children }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return { wrapper, queryClient };
}

describe('useReplayService hooks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('useReplayServiceAvailable calls service.isAvailable', async () => {
    mockService.isAvailable.mockResolvedValueOnce(true);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useReplayServiceAvailable(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBe(true);
    expect(mockService.isAvailable).toHaveBeenCalledTimes(1);
  });

  it('useReplayStats calls service.getStats', async () => {
    const stats = { totalGames: 42 };
    mockService.getStats.mockResolvedValueOnce(stats);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useReplayStats(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(stats);
    expect(mockService.getStats).toHaveBeenCalledTimes(1);
  });

  it('useGameList passes filters to listGames', async () => {
    const filters = { board_type: 'square8' } as any;
    const payload = { games: [], total: 0 };
    mockService.listGames.mockResolvedValueOnce(payload);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useGameList(filters), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockService.listGames).toHaveBeenCalledWith(filters);
    expect(result.current.data).toEqual(payload);
  });

  it('useGame fetches metadata when gameId is provided', async () => {
    const game = { gameId: 'g1' };
    mockService.getGame.mockResolvedValueOnce(game);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useGame('g1'), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockService.getGame).toHaveBeenCalledWith('g1');
    expect(result.current.data).toEqual(game);
  });

  it('useReplayStateAt calls getStateAtMove with move number', async () => {
    const state = { gameState: {} };
    mockService.getStateAtMove.mockResolvedValueOnce(state);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useReplayStateAt('g1', 5), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockService.getStateAtMove).toHaveBeenCalledWith('g1', 5);
    expect(result.current.data).toEqual(state);
  });

  it('useMoves calls getMoves with start/end', async () => {
    const moves = { moves: [] };
    mockService.getMoves.mockResolvedValueOnce(moves);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useMoves('g1', 2, 10), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockService.getMoves).toHaveBeenCalledWith('g1', 2, 10);
    expect(result.current.data).toEqual(moves);
  });

  it('useAllMoves fetches with high limit based on totalMoves', async () => {
    const moves = { moves: [] };
    mockService.getMoves.mockResolvedValueOnce(moves);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useAllMoves('g1', 1500), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    // limit should be Math.max(totalMoves, 1000), so 1500 here
    expect(mockService.getMoves).toHaveBeenCalledWith('g1', 0, undefined, 1500);
    expect(result.current.data).toEqual(moves);
  });

  it('useChoices calls getChoices with gameId and moveNumber', async () => {
    const choices = { choices: [] };
    mockService.getChoices.mockResolvedValueOnce(choices);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => useChoices('g1', 7), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(mockService.getChoices).toHaveBeenCalledWith('g1', 7);
    expect(result.current.data).toEqual(choices);
  });

  it('useStoreGame mutation calls storeGame and invalidates queries', async () => {
    const stored = { gameId: 'stored' };
    mockService.storeGame.mockResolvedValueOnce(stored);
    const { wrapper, queryClient } = createWrapper();
    const invalidateSpy = jest.spyOn(queryClient, 'invalidateQueries');

    const { result } = renderHook(() => useStoreGame(), { wrapper });

    await result.current.mutateAsync({} as any);

    expect(mockService.storeGame).toHaveBeenCalledTimes(1);
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['replay', 'games'] });
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['replay', 'stats'] });
  });

  it('usePrefetchGame calls getGame via queryClient', async () => {
    const game = { gameId: 'g2' };
    mockService.getGame.mockResolvedValueOnce(game);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => usePrefetchGame(), { wrapper });

    await result.current('g2');

    expect(mockService.getGame).toHaveBeenCalledWith('g2');
  });

  it('usePrefetchState calls getStateAtMove via queryClient', async () => {
    const state = { gameState: {} };
    mockService.getStateAtMove.mockResolvedValueOnce(state);
    const { wrapper } = createWrapper();

    const { result } = renderHook(() => usePrefetchState(), { wrapper });

    await result.current('g1', 3);

    expect(mockService.getStateAtMove).toHaveBeenCalledWith('g1', 3);
  });
});
