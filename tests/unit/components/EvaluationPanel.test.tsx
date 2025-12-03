import React from 'react';
import { render, screen } from '@testing-library/react';
import { EvaluationPanel } from '../../../src/client/components/EvaluationPanel';
import type { PositionEvaluationPayload } from '../../../src/shared/types/websocket';

describe('EvaluationPanel', () => {
  const baseSnapshot: PositionEvaluationPayload['data'] = {
    gameId: 'game-1',
    moveNumber: 10,
    boardType: 'square8',
    engineProfile: 'heuristic_v1_d5',
    evaluationScale: 'zero_sum_margin',
    perPlayer: {
      1: { totalEval: 3.2, territoryEval: 2.0, ringEval: 1.2 },
      2: { totalEval: -3.2, territoryEval: -2.0, ringEval: -1.2 },
    },
  };

  it('renders placeholder when history is empty', () => {
    render(<EvaluationPanel evaluationHistory={[]} players={[]} />);

    const panel = screen.getByTestId('evaluation-panel');
    expect(panel).toBeInTheDocument();
    expect(screen.getByText(/No evaluation data has been received yet/i)).toBeInTheDocument();
  });

  it('renders current evaluation summary for players when history is present', () => {
    const players = [
      { id: 'p1', username: 'Alice', playerNumber: 1 } as any,
      { id: 'p2', username: 'Bob', playerNumber: 2 } as any,
    ];

    render(<EvaluationPanel evaluationHistory={[baseSnapshot]} players={players} />);

    expect(screen.getByTestId('evaluation-panel')).toBeInTheDocument();
    expect(screen.getByText(/AI Evaluation/i)).toBeInTheDocument();
    expect(screen.getByText(/Move 10/i)).toBeInTheDocument();

    // Player names should appear
    expect(screen.getByText('Alice')).toBeInTheDocument();
    expect(screen.getByText('Bob')).toBeInTheDocument();

    // Advantage for the leading player should be rendered with sign.
    expect(screen.getByText(/\+3\.2/)).toBeInTheDocument();
  });

  it('skips players that have no evaluation entry in perPlayer', () => {
    const snapshot: PositionEvaluationPayload['data'] = {
      ...baseSnapshot,
      perPlayer: {
        // Missing / undefined entry for player 1 should be ignored.
        1: undefined as any,
        2: { totalEval: -1.0, territoryEval: -0.5, ringEval: -0.5 },
      },
    };

    const players = [
      { id: 'p1', username: 'Alice', playerNumber: 1 } as any,
      { id: 'p2', username: 'Bob', playerNumber: 2 } as any,
    ];

    render(<EvaluationPanel evaluationHistory={[snapshot]} players={players} />);

    // Only the player with an evaluation entry should appear.
    expect(screen.queryByText('Alice')).toBeNull();
    expect(screen.getByText('Bob')).toBeInTheDocument();
  });

  it('falls back to generic player labels and default color for unknown player numbers', () => {
    const snapshot: PositionEvaluationPayload['data'] = {
      ...baseSnapshot,
      perPlayer: {
        5: { totalEval: 1.0, territoryEval: 0.5, ringEval: 0.5 },
      } as any,
    };

    // No matching Player entries; labels and colors should fall back.
    const { container } = render(<EvaluationPanel evaluationHistory={[snapshot]} players={[]} />);

    // Name should default to P5
    expect(screen.getByText('P5')).toBeInTheDocument();

    // The color dot should use the default bg-slate-300 class from the fallback colors.
    const dots = container.querySelectorAll('span.inline-block.w-2.h-2.rounded-full');
    expect(dots.length).toBeGreaterThan(0);
    expect(dots[0].className).toContain('bg-slate-300');
  });
});
