import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import LobbyPage from '../../src/client/pages/LobbyPage';
import { gameApi } from '../../src/client/services/api';
import { io } from 'socket.io-client';

// Mock dependencies
jest.mock('../../src/client/services/api');
jest.mock('socket.io-client');

const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

// Mock socket
const mockSocket = {
  on: jest.fn(),
  emit: jest.fn(),
  disconnect: jest.fn(),
};

(io as jest.Mock).mockReturnValue(mockSocket);

// Helper to create mock games
function createMockGame(overrides = {}) {
  return {
    id: 'game-1',
    boardType: 'square8',
    maxPlayers: 2,
    isRated: true,
    allowSpectators: true,
    status: 'waiting',
    timeControl: {
      type: 'blitz',
      initialTime: 600,
      increment: 0,
    },
    player1: {
      id: 'user-1',
      username: 'Player1',
      rating: 1500,
    },
    player1Id: 'user-1',
    player2Id: null,
    player3Id: null,
    player4Id: null,
    createdAt: new Date(),
    updatedAt: new Date(),
    ...overrides,
  };
}

describe('LobbyPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.setItem('token', btoa(JSON.stringify({ userId: 'test-user' })));
  });

  afterEach(() => {
    localStorage.clear();
  });

  describe('Game Display', () => {
    it('should display available games', async () => {
      const games = [
        createMockGame({
          id: 'game-1',
          player1: { id: 'user-1', username: 'Alice', rating: 1500 },
        }),
        createMockGame({ id: 'game-2', player1: { id: 'user-2', username: 'Bob', rating: 1600 } }),
      ];

      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Alice's Game/i)).toBeInTheDocument();
        expect(screen.getByText(/Bob's Game/i)).toBeInTheDocument();
      });
    });

    it('should show loading spinner while fetching games', () => {
      (gameApi.getAvailableGames as jest.Mock).mockImplementation(
        () => new Promise(() => {}) // Never resolves
      );

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      expect(screen.getByRole('status')).toBeInTheDocument();
    });

    it('should show empty state when no games available', async () => {
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games: [] });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/No games available/i)).toBeInTheDocument();
        expect(screen.getByText(/Be the first to create a game/i)).toBeInTheDocument();
      });
    });
  });

  describe('Filtering', () => {
    it('should filter games by board type', async () => {
      const games = [
        createMockGame({
          id: 'game-1',
          boardType: 'square8',
          player1: { id: 'u1', username: 'Alice' },
        }),
        createMockGame({
          id: 'game-2',
          boardType: 'hexagonal',
          player1: { id: 'u2', username: 'Bob' },
        }),
      ];

      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Alice's Game/i)).toBeInTheDocument();
        expect(screen.getByText(/Bob's Game/i)).toBeInTheDocument();
      });

      // Filter by hexagonal
      const boardTypeSelect = screen.getByLabelText(/Board Type/i);
      fireEvent.change(boardTypeSelect, { target: { value: 'hexagonal' } });

      await waitFor(() => {
        expect(screen.queryByText(/Alice's Game/i)).not.toBeInTheDocument();
        expect(screen.getByText(/Bob's Game/i)).toBeInTheDocument();
      });
    });

    it('should filter games by rated status', async () => {
      const games = [
        createMockGame({ id: 'game-1', isRated: true, player1: { id: 'u1', username: 'Alice' } }),
        createMockGame({ id: 'game-2', isRated: false, player1: { id: 'u2', username: 'Bob' } }),
      ];

      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Alice's Game/i)).toBeInTheDocument();
      });

      // Filter by unrated
      const gameTypeSelect = screen.getByLabelText(/Game Type/i);
      fireEvent.change(gameTypeSelect, { target: { value: 'false' } });

      await waitFor(() => {
        expect(screen.queryByText(/Alice's Game/i)).not.toBeInTheDocument();
        expect(screen.getByText(/Bob's Game/i)).toBeInTheDocument();
      });
    });

    it('should filter games by search term', async () => {
      const games = [
        createMockGame({ id: 'game-alice', player1: { id: 'u1', username: 'Alice' } }),
        createMockGame({ id: 'game-bob', player1: { id: 'u2', username: 'Bob' } }),
      ];

      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Alice's Game/i)).toBeInTheDocument();
      });

      // Search for "bob"
      const searchInput = screen.getByPlaceholderText(/Creator name or game ID/i);
      fireEvent.change(searchInput, { target: { value: 'bob' } });

      await waitFor(() => {
        expect(screen.queryByText(/Alice's Game/i)).not.toBeInTheDocument();
        expect(screen.getByText(/Bob's Game/i)).toBeInTheDocument();
      });
    });

    it('should clear all filters', async () => {
      const games = [
        createMockGame({
          id: 'game-1',
          boardType: 'square8',
          player1: { id: 'u1', username: 'Alice' },
        }),
        createMockGame({
          id: 'game-2',
          boardType: 'hexagonal',
          player1: { id: 'u2', username: 'Bob' },
        }),
      ];

      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Alice's Game/i)).toBeInTheDocument();
      });

      // Apply filter
      const boardTypeSelect = screen.getByLabelText(/Board Type/i);
      fireEvent.change(boardTypeSelect, { target: { value: 'hexagonal' } });

      await waitFor(() => {
        expect(screen.queryByText(/Alice's Game/i)).not.toBeInTheDocument();
      });

      // Clear all filters
      const clearButton = screen.getByText(/Clear All Filters/i);
      fireEvent.click(clearButton);

      await waitFor(() => {
        expect(screen.getByText(/Alice's Game/i)).toBeInTheDocument();
        expect(screen.getByText(/Bob's Game/i)).toBeInTheDocument();
      });
    });
  });

  describe('Sorting', () => {
    it('should sort games by creation date (newest first)', async () => {
      const games = [
        createMockGame({
          id: 'game-1',
          player1: { id: 'u1', username: 'Alice' },
          createdAt: new Date('2024-01-01'),
        }),
        createMockGame({
          id: 'game-2',
          player1: { id: 'u2', username: 'Bob' },
          createdAt: new Date('2024-01-02'),
        }),
      ];

      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        const gameHeadings = screen.getAllByRole('heading', { name: /'s Game/i });
        expect(gameHeadings[0]).toHaveTextContent("Bob's Game");
      });
    });

    it('should sort games by board type', async () => {
      const games = [
        createMockGame({
          id: 'game-1',
          boardType: 'square8',
          player1: { id: 'u1', username: 'Alice' },
        }),
        createMockGame({
          id: 'game-2',
          boardType: 'hexagonal',
          player1: { id: 'u2', username: 'Bob' },
        }),
      ];

      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Alice's Game/i)).toBeInTheDocument();
      });

      // Change sort to board type
      const sortSelect = screen.getByDisplayValue(/Newest First/i);
      fireEvent.change(sortSelect, { target: { value: 'board_type' } });

      await waitFor(() => {
        const gameHeadings = screen.getAllByRole('heading', { name: /'s Game/i });
        // hexagonal (Bob's game) should come before square8 (Alice's game)
        expect(gameHeadings[0]).toHaveTextContent("Bob's Game");
      });
    });
  });

  describe('WebSocket Real-time Updates', () => {
    it('should subscribe to lobby updates on mount', async () => {
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games: [] });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      // Simulate a successful WebSocket connection so the component's
      // `connect` handler can emit the subscribe event.
      const connectHandler = (mockSocket.on as jest.Mock).mock.calls.find(
        ([event]) => event === 'connect'
      )?.[1];

      if (connectHandler) {
        connectHandler();
      }

      await waitFor(() => {
        expect(mockSocket.emit).toHaveBeenCalledWith('lobby:subscribe');
      });
    });

    it('should add new games when lobby:game_created event is received', async () => {
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games: [] });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/No games available/i)).toBeInTheDocument();
      });

      // Simulate WebSocket event
      const onHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'lobby:game_created'
      )?.[1];
      if (onHandler) {
        const newGame = createMockGame({
          id: 'new-game',
          player1: { id: 'u1', username: 'NewPlayer' },
        });
        onHandler(newGame);
      }

      await waitFor(() => {
        expect(screen.getByText(/NewPlayer's Game/i)).toBeInTheDocument();
      });
    });

    it('should remove games when lobby:game_started event is received', async () => {
      const games = [createMockGame({ id: 'game-1', player1: { id: 'u1', username: 'Alice' } })];
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Alice's Game/i)).toBeInTheDocument();
      });

      // Simulate game started event
      const onHandler = mockSocket.on.mock.calls.find(
        (call) => call[0] === 'lobby:game_started'
      )?.[1];
      if (onHandler) {
        onHandler({ gameId: 'game-1' });
      }

      await waitFor(() => {
        expect(screen.queryByText(/Alice's Game/i)).not.toBeInTheDocument();
      });
    });

    it('should unsubscribe from lobby on unmount', async () => {
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games: [] });

      const { unmount } = render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      unmount();

      await waitFor(() => {
        expect(mockSocket.emit).toHaveBeenCalledWith('lobby:unsubscribe');
        expect(mockSocket.disconnect).toHaveBeenCalled();
      });
    });
  });

  describe('Game Actions', () => {
    it('should join game when join button is clicked', async () => {
      const games = [
        createMockGame({ id: 'game-1', player1: { id: 'other-user', username: 'Alice' } }),
      ];
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });
      (gameApi.joinGame as jest.Mock).mockResolvedValue({});

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Join Game/i })).toBeInTheDocument();
      });

      const joinButton = screen.getByRole('button', { name: /Join Game/i });
      fireEvent.click(joinButton);

      await waitFor(() => {
        expect(gameApi.joinGame).toHaveBeenCalledWith('game-1');
        expect(mockNavigate).toHaveBeenCalledWith('/game/game-1');
      });
    });

    it('should navigate to game when spectate button is clicked', async () => {
      const games = [createMockGame({ id: 'game-1', player1: { id: 'u1', username: 'Alice' } })];
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Watch/i)).toBeInTheDocument();
      });

      const spectateButton = screen.getByText(/Watch/i);
      fireEvent.click(spectateButton);

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/spectate/game-1');
      });
    });

    it('should show error when join fails', async () => {
      const games = [createMockGame({ id: 'game-1', player1: { id: 'u1', username: 'Alice' } })];
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });
      (gameApi.joinGame as jest.Mock).mockRejectedValue(new Error('Game is full'));

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Join Game/i })).toBeInTheDocument();
      });

      const joinButton = screen.getByRole('button', { name: /Join Game/i });
      fireEvent.click(joinButton);

      await waitFor(() => {
        expect(screen.getByText(/Game is full/i)).toBeInTheDocument();
      });
    });
  });

  describe('Empty States', () => {
    it('should show filtered empty state with clear filters button', async () => {
      const games = [
        createMockGame({ boardType: 'square8', player1: { id: 'u1', username: 'Alice' } }),
      ];
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/Alice's Game/i)).toBeInTheDocument();
      });

      // Apply filter that matches nothing
      const boardTypeSelect = screen.getByLabelText(/Board Type/i);
      fireEvent.change(boardTypeSelect, { target: { value: 'hexagonal' } });

      await waitFor(() => {
        expect(screen.getByText(/No games match your filters/i)).toBeInTheDocument();
        expect(screen.getByText(/Clear Filters/i)).toBeInTheDocument();
      });
    });

    it('should show create game prompt in empty lobby', async () => {
      (gameApi.getAvailableGames as jest.Mock).mockResolvedValue({ games: [] });

      render(
        <BrowserRouter>
          <LobbyPage />
        </BrowserRouter>
      );

      await waitFor(() => {
        expect(screen.getByText(/No games available/i)).toBeInTheDocument();

        // Scope the query to the empty-lobby prompt so we don't fail when
        // multiple "Create Game" buttons exist (e.g. header + empty state).
        const emptyPrompt = screen.getByText(/Be the first to create a game!/i).closest('div');

        expect(emptyPrompt).not.toBeNull();
        expect(within(emptyPrompt as HTMLElement).getByText(/Create Game/i)).toBeInTheDocument();
      });
    });
  });
});
