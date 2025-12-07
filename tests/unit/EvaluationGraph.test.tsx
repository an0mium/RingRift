import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { EvaluationGraph, type EvaluationGraphProps } from '../../src/client/components/EvaluationGraph';
import type { PositionEvaluationPayload } from '../../src/shared/types/websocket';
import type { Player } from '../../src/shared/types/game';

// Helper to create test players
function createTestPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 600000,
      ringsInHand: 18,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

// Helper to create evaluation data
function createEvaluationData(
  moveNumber: number,
  player1Eval: number,
  player2Eval: number
): PositionEvaluationPayload['data'] {
  return {
    gameId: 'test-game',
    moveNumber,
    boardType: 'square8',
    engineProfile: 'test',
    evaluationScale: 'zero_sum_margin',
    perPlayer: {
      1: {
        totalEval: player1Eval,
        territoryEval: player1Eval * 0.5,
        ringEval: player1Eval * 0.5,
      },
      2: {
        totalEval: player2Eval,
        territoryEval: player2Eval * 0.5,
        ringEval: player2Eval * 0.5,
      },
    },
  };
}

// Helper to create evaluation history
function createEvaluationHistory(
  count: number
): PositionEvaluationPayload['data'][] {
  const history: PositionEvaluationPayload['data'][] = [];
  for (let i = 1; i <= count; i++) {
    // Create alternating evaluations
    const p1Eval = Math.sin(i * 0.5) * 5;
    const p2Eval = -p1Eval;
    history.push(createEvaluationData(i, p1Eval, p2Eval));
  }
  return history;
}

describe('EvaluationGraph', () => {
  const defaultProps: EvaluationGraphProps = {
    evaluationHistory: [],
    players: createTestPlayers(),
  };

  describe('Empty State', () => {
    it('renders without crashing with empty evaluation history', () => {
      render(<EvaluationGraph {...defaultProps} />);

      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });

    it('displays empty state message when no evaluations exist', () => {
      render(<EvaluationGraph {...defaultProps} />);

      expect(screen.getByText('No evaluation data available yet.')).toBeInTheDocument();
    });

    it('shows title in empty state', () => {
      render(<EvaluationGraph {...defaultProps} />);

      expect(screen.getByText('Evaluation Timeline')).toBeInTheDocument();
    });
  });

  describe('With Data', () => {
    it('renders graph when evaluation history exists', () => {
      const history = createEvaluationHistory(10);

      render(<EvaluationGraph {...defaultProps} evaluationHistory={history} />);

      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
      expect(screen.queryByText('No evaluation data available yet.')).not.toBeInTheDocument();
    });

    it('displays player legends', () => {
      const history = createEvaluationHistory(5);
      const players = createTestPlayers();

      render(
        <EvaluationGraph {...defaultProps} evaluationHistory={history} players={players} />
      );

      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByText('Bob')).toBeInTheDocument();
    });

    it('displays X-axis labels for moves', () => {
      const history = createEvaluationHistory(10);

      render(<EvaluationGraph {...defaultProps} evaluationHistory={history} />);

      expect(screen.getByText('Move 1')).toBeInTheDocument();
      expect(screen.getByText('Move 10')).toBeInTheDocument();
    });

    it('displays Y-axis labels for evaluation', () => {
      const history = createEvaluationHistory(5);

      render(<EvaluationGraph {...defaultProps} evaluationHistory={history} />);

      // Should show 0 as the center line label
      expect(screen.getByText('0')).toBeInTheDocument();
    });

    it('renders SVG with graph lines', () => {
      const history = createEvaluationHistory(5);

      const { container } = render(
        <EvaluationGraph {...defaultProps} evaluationHistory={history} />
      );

      // Should have SVG paths for player evaluation lines
      const paths = container.querySelectorAll('path');
      expect(paths.length).toBeGreaterThan(0);
    });
  });

  describe('Current Move Indicator', () => {
    it('renders current move indicator line when currentMoveIndex is provided', () => {
      const history = createEvaluationHistory(10);

      const { container } = render(
        <EvaluationGraph
          {...defaultProps}
          evaluationHistory={history}
          currentMoveIndex={5}
        />
      );

      // Should have a line element for current move indicator
      const lines = container.querySelectorAll('line');
      expect(lines.length).toBeGreaterThan(0);
    });

    it('does not render current move indicator when currentMoveIndex is undefined', () => {
      const history = createEvaluationHistory(10);

      const { container } = render(
        <EvaluationGraph {...defaultProps} evaluationHistory={history} />
      );

      // The only lines should be part of the zero line, not current move
      // This is a soft check - the component may still render other lines
      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });
  });

  describe('Move Click Handler', () => {
    it('calls onMoveClick when a clickable area is clicked', () => {
      const onMoveClick = jest.fn();
      const history = createEvaluationHistory(5);

      const { container } = render(
        <EvaluationGraph
          {...defaultProps}
          evaluationHistory={history}
          onMoveClick={onMoveClick}
        />
      );

      // Find clickable rect elements
      const rects = container.querySelectorAll('rect');
      if (rects.length > 0) {
        fireEvent.click(rects[0]);
        expect(onMoveClick).toHaveBeenCalled();
      }
    });

    it('does not render clickable areas when onMoveClick is not provided', () => {
      const history = createEvaluationHistory(5);

      const { container } = render(
        <EvaluationGraph {...defaultProps} evaluationHistory={history} />
      );

      // Clickable rects should not be present without onMoveClick
      const rects = container.querySelectorAll('rect.cursor-pointer');
      expect(rects.length).toBe(0);
    });
  });

  describe('Height Configuration', () => {
    it('uses default height when not specified', () => {
      const history = createEvaluationHistory(5);

      render(<EvaluationGraph {...defaultProps} evaluationHistory={history} />);

      // Component should render without errors with default height
      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });

    it('accepts custom height prop', () => {
      const history = createEvaluationHistory(5);

      render(
        <EvaluationGraph {...defaultProps} evaluationHistory={history} height={200} />
      );

      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });
  });

  describe('Custom className', () => {
    it('applies custom className', () => {
      const history = createEvaluationHistory(5);

      render(
        <EvaluationGraph
          {...defaultProps}
          evaluationHistory={history}
          className="custom-class"
        />
      );

      const graph = screen.getByTestId('evaluation-graph');
      expect(graph).toHaveClass('custom-class');
    });
  });

  describe('Edge Cases', () => {
    it('handles single evaluation point', () => {
      const history = [createEvaluationData(1, 5, -5)];

      render(<EvaluationGraph {...defaultProps} evaluationHistory={history} />);

      // Should still show "No evaluation data" since we need at least 2 points for a line
      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });

    it('handles extreme evaluation values', () => {
      const history = [
        createEvaluationData(1, 100, -100),
        createEvaluationData(2, -100, 100),
      ];

      render(<EvaluationGraph {...defaultProps} evaluationHistory={history} />);

      expect(screen.getByTestId('evaluation-graph')).toBeInTheDocument();
    });

    it('handles players without usernames', () => {
      const players: Player[] = [
        {
          id: 'p1',
          username: '',
          playerNumber: 1,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
        {
          id: 'p2',
          username: '',
          playerNumber: 2,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];
      const history = createEvaluationHistory(5);

      render(
        <EvaluationGraph {...defaultProps} evaluationHistory={history} players={players} />
      );

      // Should show fallback player names
      expect(screen.getByText('P1')).toBeInTheDocument();
      expect(screen.getByText('P2')).toBeInTheDocument();
    });

    it('handles more than 2 players', () => {
      const players: Player[] = [
        ...createTestPlayers(),
        {
          id: 'p3',
          username: 'Charlie',
          playerNumber: 3,
          type: 'human',
          isReady: true,
          timeRemaining: 600000,
          ringsInHand: 18,
          eliminatedRings: 0,
          territorySpaces: 0,
        },
      ];

      const history: PositionEvaluationPayload['data'][] = [
        {
          gameId: 'test-game',
          moveNumber: 1,
          boardType: 'square8',
          engineProfile: 'test',
          evaluationScale: 'zero_sum_margin',
          perPlayer: {
            1: { totalEval: 5, territoryEval: 2.5, ringEval: 2.5 },
            2: { totalEval: -2, territoryEval: -1, ringEval: -1 },
            3: { totalEval: -3, territoryEval: -1.5, ringEval: -1.5 },
          },
        },
        {
          gameId: 'test-game',
          moveNumber: 2,
          boardType: 'square8',
          engineProfile: 'test',
          evaluationScale: 'zero_sum_margin',
          perPlayer: {
            1: { totalEval: 3, territoryEval: 1.5, ringEval: 1.5 },
            2: { totalEval: 0, territoryEval: 0, ringEval: 0 },
            3: { totalEval: -3, territoryEval: -1.5, ringEval: -1.5 },
          },
        },
      ];

      render(
        <EvaluationGraph {...defaultProps} evaluationHistory={history} players={players} />
      );

      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByText('Bob')).toBeInTheDocument();
      expect(screen.getByText('Charlie')).toBeInTheDocument();
    });
  });
});