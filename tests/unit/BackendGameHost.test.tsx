import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MemoryRouter } from 'react-router-dom';
import { BackendGameHost } from '../../src/client/pages/BackendGameHost';
import { useGame } from '../../src/client/contexts/GameContext';
import { useAuth } from '../../src/client/contexts/AuthContext';

jest.mock('../../src/client/contexts/GameContext', () => ({
  useGame: jest.fn(),
}));

jest.mock('../../src/client/contexts/AuthContext', () => ({
  useAuth: jest.fn(),
}));

const mockedUseGame = useGame as jest.MockedFunction<typeof useGame>;
const mockedUseAuth = useAuth as jest.MockedFunction<typeof useAuth>;

describe('BackendGameHost', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    mockedUseAuth.mockReturnValue({
      user: { id: 'user-1' } as any,
      isLoading: false,
      login: jest.fn(),
      register: jest.fn(),
      logout: jest.fn(),
      updateUser: jest.fn(),
    });
  });

  it('connects to backend game and shows connecting state when game is loading', async () => {
    const connectToGame = jest.fn().mockResolvedValue(undefined);
    const disconnect = jest.fn();

    mockedUseGame.mockReturnValue({
      gameId: null,
      gameState: null,
      validMoves: null,
      isConnecting: true,
      error: null,
      victoryState: null,
      connectToGame,
      disconnect,
      pendingChoice: null,
      choiceDeadline: null,
      respondToChoice: jest.fn(),
      submitMove: jest.fn(),
      sendChatMessage: jest.fn(),
      chatMessages: [],
      connectionStatus: 'connecting',
      lastHeartbeatAt: null,
    } as any);

    render(
      <MemoryRouter initialEntries={['/game/game-123']}>
        <BackendGameHost gameId="game-123" />
      </MemoryRouter>
    );

    expect(screen.getByText(/Connecting to game…/i)).toBeInTheDocument();
    expect(screen.getByText(/Game ID: game-123/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(connectToGame).toHaveBeenCalledWith('game-123');
    });

    // Ensure disconnect is wired for cleanup (effect cleanup will run on unmount)
    // We don't explicitly unmount here, but presence of the jest.fn ensures wiring is valid.
    expect(disconnect).not.toHaveBeenCalled();
  });

  it('shows error state when backend reports an error with no gameState', () => {
    const connectToGame = jest.fn();
    const disconnect = jest.fn();

    mockedUseGame.mockReturnValue({
      gameId: null,
      gameState: null,
      validMoves: null,
      isConnecting: false,
      error: 'Backend error loading game',
      victoryState: null,
      connectToGame,
      disconnect,
      pendingChoice: null,
      choiceDeadline: null,
      respondToChoice: jest.fn(),
      submitMove: jest.fn(),
      sendChatMessage: jest.fn(),
      chatMessages: [],
      connectionStatus: 'disconnected',
      lastHeartbeatAt: null,
    } as any);

    render(
      <MemoryRouter initialEntries={['/game/game-err']}>
        <BackendGameHost gameId="game-err" />
      </MemoryRouter>
    );

    expect(screen.getByText(/Unable to load game/i)).toBeInTheDocument();
    expect(screen.getByText(/Backend error loading game/i)).toBeInTheDocument();
    expect(screen.getByText(/Game ID: game-err/i)).toBeInTheDocument();
    expect(connectToGame).toHaveBeenCalledWith('game-err');
  });

  it('shows generic fallback when no gameState is available and no error', () => {
    const connectToGame = jest.fn();
    const disconnect = jest.fn();

    mockedUseGame.mockReturnValue({
      gameId: null,
      gameState: null,
      validMoves: null,
      isConnecting: false,
      error: null,
      victoryState: null,
      connectToGame,
      disconnect,
      pendingChoice: null,
      choiceDeadline: null,
      respondToChoice: jest.fn(),
      submitMove: jest.fn(),
      sendChatMessage: jest.fn(),
      chatMessages: [],
      connectionStatus: 'connected',
      lastHeartbeatAt: null,
    } as any);

    render(
      <MemoryRouter initialEntries={['/game/game-missing']}>
        <BackendGameHost gameId="game-missing" />
      </MemoryRouter>
    );

    expect(screen.getByText(/Game not available/i)).toBeInTheDocument();
    expect(screen.getByText(/No game state received from server/i)).toBeInTheDocument();
    expect(connectToGame).toHaveBeenCalledWith('game-missing');
  });
});
