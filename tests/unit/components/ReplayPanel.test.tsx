import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ReplayPanel } from '../../../src/client/components/ReplayPanel';

jest.mock('../../../src/client/hooks/useReplayService', () => ({
  useReplayServiceAvailable: jest.fn(),
  useGameList: jest.fn(),
}));

jest.mock('../../../src/client/hooks/useReplayPlayback', () => ({
  useReplayPlayback: jest.fn(),
}));

jest.mock('../../../src/client/hooks/useReplayAnimation', () => ({
  useReplayAnimation: jest.fn(() => ({ pendingAnimation: null })),
}));

// Mock components - use function references that return React elements lazily
const MockGameFilters = jest.fn(({ className }: { className?: string }) =>
  require('react').createElement('div', {
    'data-testid': 'game-filters',
    className: className ?? '',
  })
);
const MockGameList = jest.fn(() =>
  require('react').createElement('div', { 'data-testid': 'game-list' })
);
const MockPlaybackControls = jest.fn(() =>
  require('react').createElement('div', { 'data-testid': 'playback-controls' })
);
const MockMoveInfo = jest.fn(() =>
  require('react').createElement('div', { 'data-testid': 'move-info' })
);

jest.mock('../../../src/client/components/ReplayPanel/GameFilters', () => ({
  GameFilters: (props: any) => MockGameFilters(props),
}));

jest.mock('../../../src/client/components/ReplayPanel/GameList', () => ({
  GameList: () => MockGameList(),
}));

jest.mock('../../../src/client/components/ReplayPanel/PlaybackControls', () => ({
  PlaybackControls: () => MockPlaybackControls(),
}));

jest.mock('../../../src/client/components/ReplayPanel/MoveInfo', () => ({
  MoveInfo: () => MockMoveInfo(),
}));

describe('ReplayPanel', () => {
  const useReplayServiceAvailable =
    require('../../../src/client/hooks/useReplayService').useReplayServiceAvailable;
  const useGameList = require('../../../src/client/hooks/useReplayService').useGameList;
  const useReplayPlayback =
    require('../../../src/client/hooks/useReplayPlayback').useReplayPlayback;

  beforeEach(() => {
    jest.clearAllMocks();

    useReplayServiceAvailable.mockReturnValue({
      data: true,
      isLoading: false,
    });

    useGameList.mockReturnValue({
      data: { games: [], total: 0, hasMore: false },
      isLoading: false,
      error: null,
    });

    useReplayPlayback.mockReturnValue({
      gameId: null,
      currentState: null,
      currentMoveNumber: 0,
      totalMoves: 0,
      isPlaying: false,
      playbackSpeed: 1,
      isLoading: false,
      canStepForward: false,
      canStepBackward: false,
      moves: [],
      metadata: null,
      error: null,
      getCurrentMove: () => null,
      loadGame: jest.fn(),
      unloadGame: jest.fn(),
      stepForward: jest.fn(),
      stepBackward: jest.fn(),
      togglePlay: jest.fn(),
      jumpToStart: jest.fn(),
      jumpToEnd: jest.fn(),
      jumpToMove: jest.fn(),
      setSpeed: jest.fn(),
    });
  });

  it('renders collapsed state by default and expands on click', () => {
    render(<ReplayPanel />);

    expect(screen.getByText('Game Database')).toBeInTheDocument();
    expect(screen.queryByTestId('game-filters')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /game database/i }));
    expect(screen.getByTestId('game-filters')).toBeInTheDocument();
  });

  it('renders service-unavailable copy when replay service is not available', () => {
    useReplayServiceAvailable.mockReturnValue({
      data: false,
      isLoading: false,
    });

    render(<ReplayPanel defaultCollapsed={false} />);

    expect(
      screen.getByText(
        /Replay service unavailable\. Start the AI service to browse stored games\./i
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText(/cd ai-service && uvicorn app\.main:app --port 8001/i)
    ).toBeInTheDocument();
  });

  it('renders replay mode layout when a game is loaded', () => {
    useReplayPlayback.mockReturnValue({
      gameId: 'replay-game-1',
      currentState: null,
      currentMoveNumber: 3,
      totalMoves: 10,
      isPlaying: false,
      playbackSpeed: 1,
      isLoading: false,
      canStepForward: true,
      canStepBackward: true,
      moves: [],
      metadata: {
        gameId: 'replay-game-1',
        boardType: 'square8',
        numPlayers: 2,
        winner: 1,
      },
      error: 'Test error',
      getCurrentMove: () => ({}),
      loadGame: jest.fn(),
      unloadGame: jest.fn(),
      stepForward: jest.fn(),
      stepBackward: jest.fn(),
      togglePlay: jest.fn(),
      jumpToStart: jest.fn(),
      jumpToEnd: jest.fn(),
      jumpToMove: jest.fn(),
      setSpeed: jest.fn(),
    });

    render(<ReplayPanel defaultCollapsed={false} />);

    expect(screen.getByText('Replay Mode')).toBeInTheDocument();
    expect(screen.getByTestId('playback-controls')).toBeInTheDocument();
    expect(screen.getByTestId('move-info')).toBeInTheDocument();
    expect(screen.getByText('Test error')).toBeInTheDocument();

    expect(screen.getByText(/← → Step • Space Play\/Pause • \[ \] Speed/i)).toBeInTheDocument();
  });
});
