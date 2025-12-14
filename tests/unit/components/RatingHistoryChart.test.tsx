import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { RatingHistoryChart } from '../../../src/client/components/RatingHistoryChart';

describe('RatingHistoryChart', () => {
  const mockHistory = [
    { date: '2024-12-01T10:00:00Z', rating: 1000, change: 0, gameId: null },
    { date: '2024-12-02T10:00:00Z', rating: 1015, change: 15, gameId: 'game-1' },
    { date: '2024-12-03T10:00:00Z', rating: 1008, change: -7, gameId: 'game-2' },
    { date: '2024-12-04T10:00:00Z', rating: 1025, change: 17, gameId: 'game-3' },
    { date: '2024-12-05T10:00:00Z', rating: 1042, change: 17, gameId: 'game-4' },
  ];

  const defaultProps = {
    history: mockHistory,
    currentRating: 1042,
  };

  describe('empty states', () => {
    it('renders empty state when no history', () => {
      render(<RatingHistoryChart history={[]} currentRating={1000} />);

      expect(screen.getByText('No rating history yet')).toBeInTheDocument();
    });

    it('renders insufficient data state when only one entry', () => {
      render(
        <RatingHistoryChart
          history={[{ date: '2024-12-01T10:00:00Z', rating: 1000, change: 0, gameId: null }]}
          currentRating={1000}
        />
      );

      expect(screen.getByText('Play more games to see your rating trend')).toBeInTheDocument();
    });
  });

  describe('header display', () => {
    it('renders Rating History heading', () => {
      render(<RatingHistoryChart {...defaultProps} />);

      expect(screen.getByText('Rating History')).toBeInTheDocument();
    });

    it('displays current rating', () => {
      render(<RatingHistoryChart {...defaultProps} />);

      expect(screen.getByText('1042')).toBeInTheDocument();
    });

    it('displays positive rating change with plus sign', () => {
      render(<RatingHistoryChart {...defaultProps} />);

      expect(screen.getByText('+17')).toBeInTheDocument();
    });

    it('displays negative rating change', () => {
      const historyWithNegativeChange = [
        ...mockHistory.slice(0, -1),
        { date: '2024-12-05T10:00:00Z', rating: 1020, change: -5, gameId: 'game-4' },
      ];

      render(<RatingHistoryChart history={historyWithNegativeChange} currentRating={1020} />);

      expect(screen.getByText('-5')).toBeInTheDocument();
    });

    it('does not display change badge when change is zero', () => {
      const historyWithZeroChange = [
        { date: '2024-12-01T10:00:00Z', rating: 1000, change: 0, gameId: null },
        { date: '2024-12-02T10:00:00Z', rating: 1000, change: 0, gameId: 'game-1' },
      ];

      render(<RatingHistoryChart history={historyWithZeroChange} currentRating={1000} />);

      expect(screen.queryByText('+0')).not.toBeInTheDocument();
      expect(screen.queryByText('-0')).not.toBeInTheDocument();
    });
  });

  describe('chart rendering', () => {
    it('renders SVG element', () => {
      const { container } = render(<RatingHistoryChart {...defaultProps} />);

      expect(container.querySelector('svg')).toBeInTheDocument();
    });

    it('renders chart with correct viewBox', () => {
      const { container } = render(<RatingHistoryChart {...defaultProps} />);

      const svg = container.querySelector('svg');
      expect(svg).toHaveAttribute('viewBox', '0 0 400 120');
    });

    it('renders gradient definition', () => {
      const { container } = render(<RatingHistoryChart {...defaultProps} />);

      const gradient = container.querySelector('#ratingGradient');
      expect(gradient).toBeInTheDocument();
    });

    it('renders line path', () => {
      const { container } = render(<RatingHistoryChart {...defaultProps} />);

      const linePath = container.querySelector('path[stroke="rgb(16, 185, 129)"]');
      expect(linePath).toBeInTheDocument();
    });

    it('renders area path with gradient fill', () => {
      const { container } = render(<RatingHistoryChart {...defaultProps} />);

      const areaPath = container.querySelector('path[fill="url(#ratingGradient)"]');
      expect(areaPath).toBeInTheDocument();
    });

    it('renders data points as circles', () => {
      const { container } = render(<RatingHistoryChart {...defaultProps} />);

      const circles = container.querySelectorAll('circle');
      expect(circles.length).toBe(mockHistory.length);
    });
  });

  describe('axis labels', () => {
    it('renders Y-axis max rating label', () => {
      const { container } = render(<RatingHistoryChart {...defaultProps} />);

      // Max rating + 50 padding, rounded
      const yAxisLabels = container.querySelectorAll('text.fill-slate-500');
      const maxLabel = Array.from(yAxisLabels).find((el) => Number(el.textContent) > 1000);
      expect(maxLabel).toBeInTheDocument();
    });

    it('renders Y-axis min rating label', () => {
      const { container } = render(<RatingHistoryChart {...defaultProps} />);

      // Min rating - 50 padding, rounded
      const yAxisLabels = container.querySelectorAll('text.fill-slate-500');
      const minLabel = Array.from(yAxisLabels).find((el) => Number(el.textContent) < 1000);
      expect(minLabel).toBeInTheDocument();
    });

    it('renders X-axis date labels', () => {
      const { container } = render(<RatingHistoryChart {...defaultProps} />);

      // Should have first and last date labels
      const xAxisLabels = container.querySelectorAll('text');
      const dateLabels = Array.from(xAxisLabels).filter((el) => el.textContent?.includes('Dec'));
      expect(dateLabels.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('legend', () => {
    it('displays game count', () => {
      render(<RatingHistoryChart {...defaultProps} />);

      expect(screen.getByText('5 games')).toBeInTheDocument();
    });

    it('displays time range', () => {
      render(<RatingHistoryChart {...defaultProps} />);

      expect(screen.getByText('Last 30 days')).toBeInTheDocument();
    });
  });

  describe('styling', () => {
    it('applies positive change styling', () => {
      render(<RatingHistoryChart {...defaultProps} />);

      const changeBadge = screen.getByText('+17');
      expect(changeBadge).toHaveClass('text-emerald-400');
      expect(changeBadge).toHaveClass('bg-emerald-900/30');
    });

    it('applies negative change styling', () => {
      const historyWithNegativeChange = [
        ...mockHistory.slice(0, -1),
        { date: '2024-12-05T10:00:00Z', rating: 1020, change: -5, gameId: 'game-4' },
      ];

      render(<RatingHistoryChart history={historyWithNegativeChange} currentRating={1020} />);

      const changeBadge = screen.getByText('-5');
      expect(changeBadge).toHaveClass('text-red-400');
      expect(changeBadge).toHaveClass('bg-red-900/30');
    });
  });

  describe('data sorting', () => {
    it('sorts history by date', () => {
      const unsortedHistory = [
        { date: '2024-12-05T10:00:00Z', rating: 1050, change: 10, gameId: 'game-3' },
        { date: '2024-12-01T10:00:00Z', rating: 1000, change: 0, gameId: null },
        { date: '2024-12-03T10:00:00Z', rating: 1030, change: 15, gameId: 'game-2' },
      ];

      const { container } = render(
        <RatingHistoryChart history={unsortedHistory} currentRating={1050} />
      );

      // Chart should render without errors
      expect(container.querySelector('svg')).toBeInTheDocument();
    });
  });

  describe('edge cases', () => {
    it('handles very large rating values', () => {
      const highRatingHistory = [
        { date: '2024-12-01T10:00:00Z', rating: 2500, change: 0, gameId: null },
        { date: '2024-12-02T10:00:00Z', rating: 2520, change: 20, gameId: 'game-1' },
      ];

      render(<RatingHistoryChart history={highRatingHistory} currentRating={2520} />);

      expect(screen.getByText('2520')).toBeInTheDocument();
    });

    it('handles very low rating values', () => {
      const lowRatingHistory = [
        { date: '2024-12-01T10:00:00Z', rating: 100, change: 0, gameId: null },
        { date: '2024-12-02T10:00:00Z', rating: 85, change: -15, gameId: 'game-1' },
      ];

      render(<RatingHistoryChart history={lowRatingHistory} currentRating={85} />);

      expect(screen.getByText('85')).toBeInTheDocument();
    });

    it('handles identical ratings', () => {
      const flatHistory = [
        { date: '2024-12-01T10:00:00Z', rating: 1000, change: 0, gameId: null },
        { date: '2024-12-02T10:00:00Z', rating: 1000, change: 0, gameId: 'game-1' },
        { date: '2024-12-03T10:00:00Z', rating: 1000, change: 0, gameId: 'game-2' },
      ];

      const { container } = render(
        <RatingHistoryChart history={flatHistory} currentRating={1000} />
      );

      expect(container.querySelector('svg')).toBeInTheDocument();
    });

    it('handles large rating swings', () => {
      const volatileHistory = [
        { date: '2024-12-01T10:00:00Z', rating: 1000, change: 0, gameId: null },
        { date: '2024-12-02T10:00:00Z', rating: 1200, change: 200, gameId: 'game-1' },
        { date: '2024-12-03T10:00:00Z', rating: 800, change: -400, gameId: 'game-2' },
      ];

      const { container } = render(
        <RatingHistoryChart history={volatileHistory} currentRating={800} />
      );

      expect(container.querySelector('svg')).toBeInTheDocument();
    });
  });

  describe('custom className', () => {
    it('applies custom className to container', () => {
      const { container } = render(
        <RatingHistoryChart {...defaultProps} className="custom-class" />
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });

    it('applies custom className to empty state', () => {
      const { container } = render(
        <RatingHistoryChart history={[]} currentRating={1000} className="custom-class" />
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });
});
