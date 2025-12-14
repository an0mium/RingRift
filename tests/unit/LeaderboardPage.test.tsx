import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock the API
const mockGetLeaderboard = jest.fn();
jest.mock('../../src/client/services/api', () => ({
  userApi: {
    getLeaderboard: (...args: unknown[]) => mockGetLeaderboard(...args),
  },
}));

// Mock LoadingSpinner at module scope with mock prefix
const mockLoadingSpinner = ({ size }: { size: string }) =>
  React.createElement('div', {
    'data-testid': 'loading-spinner',
    'data-size': size,
    children: 'Loading...',
  });

jest.mock('../../src/client/components/LoadingSpinner', () => ({
  __esModule: true,
  default: (props: { size: string }) => mockLoadingSpinner(props),
}));

// Import after mocks
import LeaderboardPage from '../../src/client/pages/LeaderboardPage';

describe('LeaderboardPage', () => {
  const mockUsers = [
    {
      id: 'user1',
      username: 'TopPlayer',
      rating: 2100,
      gamesPlayed: 100,
      gamesWon: 75,
    },
    {
      id: 'user2',
      username: 'SecondBest',
      rating: 1950,
      gamesPlayed: 80,
      gamesWon: 52,
    },
    {
      id: 'user3',
      username: 'Newcomer',
      rating: 1200,
      gamesPlayed: 0,
      gamesWon: 0,
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows loading spinner while fetching data', () => {
    mockGetLeaderboard.mockReturnValue(new Promise(() => {})); // Never resolves

    render(<LeaderboardPage />);

    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
    expect(screen.getByTestId('loading-spinner')).toHaveAttribute('data-size', 'lg');
  });

  it('displays leaderboard data after successful fetch', async () => {
    mockGetLeaderboard.mockResolvedValue({ users: mockUsers });

    render(<LeaderboardPage />);

    await waitFor(() => {
      expect(screen.getByText('TopPlayer')).toBeInTheDocument();
    });

    expect(screen.getByText('SecondBest')).toBeInTheDocument();
    expect(screen.getByText('Newcomer')).toBeInTheDocument();
  });

  it('displays ratings for each user', async () => {
    mockGetLeaderboard.mockResolvedValue({ users: mockUsers });

    render(<LeaderboardPage />);

    await waitFor(() => {
      expect(screen.getByText('2100')).toBeInTheDocument();
    });

    expect(screen.getByText('1950')).toBeInTheDocument();
    expect(screen.getByText('1200')).toBeInTheDocument();
  });

  it('calculates and displays win rates correctly', async () => {
    mockGetLeaderboard.mockResolvedValue({ users: mockUsers });

    render(<LeaderboardPage />);

    await waitFor(() => {
      // TopPlayer: 75/100 = 75%
      expect(screen.getByText('75%')).toBeInTheDocument();
    });

    // SecondBest: 52/80 = 65%
    expect(screen.getByText('65%')).toBeInTheDocument();
    // Newcomer: 0/0 = 0%
    expect(screen.getByText('0%')).toBeInTheDocument();
  });

  it('displays games played for each user', async () => {
    mockGetLeaderboard.mockResolvedValue({ users: mockUsers });

    render(<LeaderboardPage />);

    await waitFor(() => {
      expect(screen.getByText('100')).toBeInTheDocument();
    });

    expect(screen.getByText('80')).toBeInTheDocument();
  });

  it('displays error message when fetch fails', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    mockGetLeaderboard.mockRejectedValue(new Error('Network error'));

    render(<LeaderboardPage />);

    await waitFor(() => {
      expect(screen.getByText('Failed to load leaderboard data')).toBeInTheDocument();
    });

    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });

  it('renders page header with title', async () => {
    mockGetLeaderboard.mockResolvedValue({ users: mockUsers });

    render(<LeaderboardPage />);

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Leaderboard/i })).toBeInTheDocument();
    });

    expect(screen.getByText('Top players ranked by rating')).toBeInTheDocument();
  });

  it('renders table headers correctly', async () => {
    mockGetLeaderboard.mockResolvedValue({ users: mockUsers });

    render(<LeaderboardPage />);

    await waitFor(() => {
      expect(screen.getByText('Rank')).toBeInTheDocument();
    });

    expect(screen.getByText('Player')).toBeInTheDocument();
    expect(screen.getByText('Rating')).toBeInTheDocument();
    expect(screen.getByText('Win Rate')).toBeInTheDocument();
    expect(screen.getByText('Games')).toBeInTheDocument();
  });

  it('displays correct rank numbers', async () => {
    mockGetLeaderboard.mockResolvedValue({ users: mockUsers });

    render(<LeaderboardPage />);

    await waitFor(() => {
      expect(screen.getByText('TopPlayer')).toBeInTheDocument();
    });

    // Ranks 1, 2, 3 should be shown
    const rows = screen.getAllByRole('row');
    // First row is header, subsequent rows are data
    expect(rows[1]).toHaveTextContent('1');
    expect(rows[2]).toHaveTextContent('2');
    expect(rows[3]).toHaveTextContent('3');
  });

  it('calls getLeaderboard with limit of 50', async () => {
    mockGetLeaderboard.mockResolvedValue({ users: [] });

    render(<LeaderboardPage />);

    await waitFor(() => {
      expect(mockGetLeaderboard).toHaveBeenCalledWith({ limit: 50 });
    });
  });

  it('handles empty leaderboard gracefully', async () => {
    mockGetLeaderboard.mockResolvedValue({ users: [] });

    render(<LeaderboardPage />);

    await waitFor(() => {
      expect(screen.getByRole('table')).toBeInTheDocument();
    });

    // Table should exist but have no data rows
    const rows = screen.getAllByRole('row');
    expect(rows).toHaveLength(1); // Just the header row
  });
});
