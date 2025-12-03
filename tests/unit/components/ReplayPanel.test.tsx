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
const MockGameList = jest.fn((props?: any) =>
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
  GameList: (props: any) => MockGameList(props),
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

  it('shows checking state while verifying replay service availability', () => {
    useReplayServiceAvailable.mockReturnValue({
      data: undefined,
      isLoading: true,
    });

    render(<ReplayPanel defaultCollapsed={false} />);

    expect(screen.getByText(/Checking replay service.../i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Collapse/i })).toBeInTheDocument();
  });

  it('invokes parent callbacks with initial playback state and animation', () => {
    const onStateChange = jest.fn();
    const onReplayModeChange = jest.fn();
    const onAnimationChange = jest.fn();

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
      gameId: 'replay-callback-game',
      currentState: { id: 'state-1' },
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

    render(
      <ReplayPanel
        defaultCollapsed={false}
        onStateChange={onStateChange}
        onReplayModeChange={onReplayModeChange}
        onAnimationChange={onAnimationChange}
      />
    );

    expect(onStateChange).toHaveBeenCalledWith(expect.objectContaining({ id: 'state-1' }));
    expect(onReplayModeChange).toHaveBeenCalledWith(true);
    // useReplayAnimation is mocked to always return { pendingAnimation: null }
    expect(onAnimationChange).toHaveBeenCalledWith(null);
  });

  it('wires keyboard shortcuts to playback controls in replay mode', () => {
    const stepForward = jest.fn();
    const stepBackward = jest.fn();
    const togglePlay = jest.fn();
    const jumpToStart = jest.fn();
    const jumpToEnd = jest.fn();
    const jumpToMove = jest.fn();
    const setSpeed = jest.fn();
    const unloadGame = jest.fn();
    const onForkFromPosition = jest.fn();

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
      gameId: 'keyboard-game',
      currentState: { id: 'state-for-fork' },
      currentMoveNumber: 3,
      totalMoves: 10,
      isPlaying: false,
      playbackSpeed: 1,
      isLoading: false,
      canStepForward: true,
      canStepBackward: true,
      moves: [],
      metadata: null,
      error: null,
      getCurrentMove: () => null,
      loadGame: jest.fn(),
      unloadGame,
      stepForward,
      stepBackward,
      togglePlay,
      jumpToStart,
      jumpToEnd,
      jumpToMove,
      setSpeed,
    });

    render(<ReplayPanel defaultCollapsed={false} onForkFromPosition={onForkFromPosition} />);

    // Arrow keys and h/l
    fireEvent.keyDown(window, { key: 'ArrowRight' });
    fireEvent.keyDown(window, { key: 'l' });
    fireEvent.keyDown(window, { key: 'ArrowLeft' });
    fireEvent.keyDown(window, { key: 'h' });

    expect(stepForward).toHaveBeenCalledTimes(2);
    expect(stepBackward).toHaveBeenCalledTimes(2);

    // Space toggles play / pause
    fireEvent.keyDown(window, { key: ' ' });
    expect(togglePlay).toHaveBeenCalledTimes(1);

    // Home / 0 -> jumpToStart
    fireEvent.keyDown(window, { key: 'Home' });
    fireEvent.keyDown(window, { key: '0' });
    expect(jumpToStart).toHaveBeenCalledTimes(2);

    // End / $ -> jumpToEnd
    fireEvent.keyDown(window, { key: 'End' });
    fireEvent.keyDown(window, { key: '$' });
    expect(jumpToEnd).toHaveBeenCalledTimes(2);

    // Speed controls
    fireEvent.keyDown(window, { key: '[' });
    expect(setSpeed).toHaveBeenCalledWith(0.5);
    fireEvent.keyDown(window, { key: ']' });
    expect(setSpeed).toHaveBeenCalledWith(2);

    // Fork shortcut (f) should call onForkFromPosition and unloadGame
    fireEvent.keyDown(window, { key: 'f' });
    expect(onForkFromPosition).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'state-for-fork' })
    );
    expect(unloadGame).toHaveBeenCalled();

    // Escape should close replay (unload game)
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(unloadGame).toHaveBeenCalledTimes(2);

    // Events originating from inputs should be ignored
    const input = document.createElement('input');
    document.body.appendChild(input);
    fireEvent.keyDown(input, { key: 'ArrowRight' });
    // No additional stepForward calls from input-targeted events
    expect(stepForward).toHaveBeenCalledTimes(2);
  });
});
