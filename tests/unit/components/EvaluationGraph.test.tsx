import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { EvaluationGraph } from '../../../src/client/components/EvaluationGraph';
import type { Player } from '../../../src/shared/types/game';
import type { PositionEvaluationPayload } from '../../../src/shared/types/websocket';

describe('EvaluationGraph', () => {
  const mockPlayers: Player[] = [
    {
      id: '1',
      username: 'Alice',
      playerNumber: 0,
      ringsInHand: 5,
      isAI: false,
      isEliminated: false,
    },
    { id: '2', username: 'Bob', playerNumber: 1, ringsInHand: 5, isAI: false, isEliminated: false },
  ];

  const mockEvaluationHistory: PositionEvaluationPayload['data'][] = [
    {
      moveNumber: 1,
      perPlayer: {
        0: { totalEval: 0.5, territoryEval: 0.2, ringEval: 0.3 },
        1: { totalEval: -0.3, territoryEval: -0.1, ringEval: -0.2 },
      },
    },
    {
      moveNumber: 2,
      perPlayer: {
        0: { totalEval: 1.2, territoryEval: 0.6, ringEval: 0.6 },
        1: { totalEval: -1.0, territoryEval: -0.4, ringEval: -0.6 },
      },
    },
    {
      moveNumber: 3,
      perPlayer: {
        0: { totalEval: 2.5, territoryEval: 1.0, ringEval: 1.5 },
        1: { totalEval: -2.0, territoryEval: -0.8, ringEval: -1.2 },
      },
    },
    {
      moveNumber: 4,
      perPlayer: {
        0: { totalEval: 1.8, territoryEval: 0.8, ringEval: 1.0 },
        1: { totalEval: -1.5, territoryEval: -0.6, ringEval: -0.9 },
      },
    },
    {
      moveNumber: 5,
      perPlayer: {
        0: { totalEval: 3.0, territoryEval: 1.2, ringEval: 1.8 },
        1: { totalEval: -2.5, territoryEval: -1.0, ringEval: -1.5 },
      },
    },
  ];

  const defaultProps = {
    evaluationHistory: mockEvaluationHistory,
    players: mockPlayers,
  };

  describe('empty state', () => {
    it('renders empty state when no evaluation history', () => {
      render(<EvaluationGraph evaluationHistory={[]} players={mockPlayers} />);

      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
      expect(screen.getByText('Evaluation Timeline')).toBeInTheDocument();
      expect(screen.getByText('No evaluation data available yet.')).toBeInTheDocument();
    });

    it('renders empty state when evaluationHistory is undefined', () => {
      render(
        <EvaluationGraph
          evaluationHistory={undefined as unknown as PositionEvaluationPayload['data'][]}
          players={mockPlayers}
        />
      );

      expect(screen.getByText('No evaluation data available yet.')).toBeInTheDocument();
    });
  });

  describe('header and legend', () => {
    it('renders Evaluation Timeline heading', () => {
      render(<EvaluationGraph {...defaultProps} />);

      expect(screen.getByText('Evaluation Timeline')).toBeInTheDocument();
    });

    it('renders player legend with names', () => {
      render(<EvaluationGraph {...defaultProps} />);

      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByText('Bob')).toBeInTheDocument();
    });

    it('shows fallback player names when not found', () => {
      render(<EvaluationGraph {...defaultProps} players={[]} />);

      expect(screen.getByText('P0')).toBeInTheDocument();
      expect(screen.getByText('P1')).toBeInTheDocument();
    });

    it('renders color indicators for each player', () => {
      render(<EvaluationGraph {...defaultProps} />);

      // Each player legend entry should have a color dot
      const legendItems = screen.getAllByText(/Alice|Bob/);
      legendItems.forEach((item) => {
        const colorDot = item.previousElementSibling;
        expect(colorDot).toHaveClass('rounded-full');
      });
    });
  });

  describe('graph rendering', () => {
    it('renders SVG element', () => {
      const { container } = render(<EvaluationGraph {...defaultProps} />);

      expect(container.querySelector('svg')).toBeInTheDocument();
    });

    it('renders path elements for each player', () => {
      const { container } = render(<EvaluationGraph {...defaultProps} />);

      const paths = container.querySelectorAll('path[stroke]');
      expect(paths.length).toBeGreaterThanOrEqual(2);
    });

    it('renders zero line', () => {
      const { container } = render(<EvaluationGraph {...defaultProps} />);

      const zeroLine = container.querySelector('.border-dashed');
      expect(zeroLine).toBeInTheDocument();
    });

    it('renders Y-axis labels', () => {
      render(<EvaluationGraph {...defaultProps} />);

      // Should have labels for max, zero, and min
      expect(screen.getByText('0')).toBeInTheDocument();
    });

    it('renders X-axis labels', () => {
      render(<EvaluationGraph {...defaultProps} />);

      expect(screen.getByText('Move 1')).toBeInTheDocument();
      expect(screen.getByText('Move 5')).toBeInTheDocument();
    });
  });

  describe('current move indicator', () => {
    it('renders current move indicator when provided', () => {
      const { container } = render(<EvaluationGraph {...defaultProps} currentMoveIndex={2} />);

      // Current move indicator is a vertical line
      const indicator = container.querySelector('line[stroke="#3b82f6"]');
      expect(indicator).toBeInTheDocument();
    });

    it('does not render indicator when currentMoveIndex is undefined', () => {
      const { container } = render(<EvaluationGraph {...defaultProps} />);

      const indicator = container.querySelector('line[stroke="#3b82f6"]');
      expect(indicator).not.toBeInTheDocument();
    });
  });

  describe('click interaction', () => {
    it('renders clickable areas when onMoveClick provided', () => {
      const { container } = render(<EvaluationGraph {...defaultProps} onMoveClick={jest.fn()} />);

      // Should have rect elements for click targets
      const clickTargets = container.querySelectorAll('rect[fill="transparent"]');
      expect(clickTargets.length).toBe(mockEvaluationHistory.length);
    });

    it('calls onMoveClick with move number when clicked', () => {
      const onMoveClick = jest.fn();
      const { container } = render(<EvaluationGraph {...defaultProps} onMoveClick={onMoveClick} />);

      const clickTargets = container.querySelectorAll('rect[fill="transparent"]');
      fireEvent.click(clickTargets[2]);

      expect(onMoveClick).toHaveBeenCalledWith(3);
    });

    it('does not render click targets when onMoveClick not provided', () => {
      const { container } = render(<EvaluationGraph {...defaultProps} />);

      const clickTargets = container.querySelectorAll('rect[fill="transparent"]');
      expect(clickTargets.length).toBe(0);
    });
  });

  describe('height prop', () => {
    it('uses default height of 120', () => {
      const { container } = render(<EvaluationGraph {...defaultProps} />);

      const graphContainer = container.querySelector('[style*="height"]');
      expect(graphContainer).toHaveStyle({ height: '80px' }); // 120 - 40 for labels
    });

    it('uses custom height when provided', () => {
      const { container } = render(<EvaluationGraph {...defaultProps} height={200} />);

      const graphContainer = container.querySelector('[style*="height"]');
      expect(graphContainer).toHaveStyle({ height: '160px' }); // 200 - 40 for labels
    });
  });

  describe('multi-player support', () => {
    it('handles 4 players', () => {
      const fourPlayerHistory: PositionEvaluationPayload['data'][] = [
        {
          moveNumber: 1,
          perPlayer: {
            0: { totalEval: 1.0 },
            1: { totalEval: 0.5 },
            2: { totalEval: -0.5 },
            3: { totalEval: -1.0 },
          },
        },
        {
          moveNumber: 2,
          perPlayer: {
            0: { totalEval: 1.5 },
            1: { totalEval: 0.8 },
            2: { totalEval: -0.8 },
            3: { totalEval: -1.5 },
          },
        },
      ];

      const fourPlayers: Player[] = [
        {
          id: '1',
          username: 'Alice',
          playerNumber: 0,
          ringsInHand: 5,
          isAI: false,
          isEliminated: false,
        },
        {
          id: '2',
          username: 'Bob',
          playerNumber: 1,
          ringsInHand: 5,
          isAI: false,
          isEliminated: false,
        },
        {
          id: '3',
          username: 'Carol',
          playerNumber: 2,
          ringsInHand: 5,
          isAI: false,
          isEliminated: false,
        },
        {
          id: '4',
          username: 'Dave',
          playerNumber: 3,
          ringsInHand: 5,
          isAI: false,
          isEliminated: false,
        },
      ];

      render(<EvaluationGraph evaluationHistory={fourPlayerHistory} players={fourPlayers} />);

      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByText('Bob')).toBeInTheDocument();
      expect(screen.getByText('Carol')).toBeInTheDocument();
      expect(screen.getByText('Dave')).toBeInTheDocument();
    });
  });

  describe('evaluation range calculation', () => {
    it('handles all positive evaluations', () => {
      const positiveHistory: PositionEvaluationPayload['data'][] = [
        { moveNumber: 1, perPlayer: { 0: { totalEval: 5.0 }, 1: { totalEval: 3.0 } } },
        { moveNumber: 2, perPlayer: { 0: { totalEval: 8.0 }, 1: { totalEval: 4.0 } } },
      ];

      render(<EvaluationGraph evaluationHistory={positiveHistory} players={mockPlayers} />);

      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });

    it('handles all negative evaluations', () => {
      const negativeHistory: PositionEvaluationPayload['data'][] = [
        { moveNumber: 1, perPlayer: { 0: { totalEval: -5.0 }, 1: { totalEval: -3.0 } } },
        { moveNumber: 2, perPlayer: { 0: { totalEval: -8.0 }, 1: { totalEval: -4.0 } } },
      ];

      render(<EvaluationGraph evaluationHistory={negativeHistory} players={mockPlayers} />);

      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });

    it('handles zero evaluations', () => {
      const zeroHistory: PositionEvaluationPayload['data'][] = [
        { moveNumber: 1, perPlayer: { 0: { totalEval: 0 }, 1: { totalEval: 0 } } },
        { moveNumber: 2, perPlayer: { 0: { totalEval: 0 }, 1: { totalEval: 0 } } },
      ];

      render(<EvaluationGraph evaluationHistory={zeroHistory} players={mockPlayers} />);

      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });
  });

  describe('single data point', () => {
    it('handles single evaluation entry', () => {
      const singleHistory: PositionEvaluationPayload['data'][] = [
        { moveNumber: 1, perPlayer: { 0: { totalEval: 1.0 }, 1: { totalEval: -0.5 } } },
      ];

      render(<EvaluationGraph evaluationHistory={singleHistory} players={mockPlayers} />);

      // Should still render without errors (may not show path with single point)
      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });
  });

  describe('custom className', () => {
    it('applies custom className to container', () => {
      render(<EvaluationGraph {...defaultProps} className="custom-class" />);

      expect(screen.getByTestId('evaluation-graph')).toHaveClass('custom-class');
    });

    it('applies custom className to empty state', () => {
      render(
        <EvaluationGraph evaluationHistory={[]} players={mockPlayers} className="custom-class" />
      );

      expect(screen.getByTestId('evaluation-graph')).toHaveClass('custom-class');
    });
  });
});
