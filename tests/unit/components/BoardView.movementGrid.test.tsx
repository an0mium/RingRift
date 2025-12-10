import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardView } from '../../../src/client/components/BoardView';
import type { BoardState } from '../../../src/shared/types/game';

function createBoard(type: 'square8' | 'square19' | 'hexagonal', size: number): BoardState {
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size,
    type,
  };
}

describe('BoardView movement grid overlay', () => {
  it('renders movement grid SVG when showMovementGrid is true (square8)', () => {
    const board = createBoard('square8', 8);
    const { container } = render(
      <BoardView boardType="square8" board={board} showMovementGrid={true} />
    );

    // Movement overlay is an SVG with line/circle elements layered over the board.
    const overlay = container.querySelector('svg');
    expect(overlay).toBeInTheDocument();
    expect(overlay?.querySelectorAll('circle').length).toBeGreaterThan(0);
    expect(overlay?.querySelectorAll('line').length).toBeGreaterThan(0);
  });

  it('renders movement grid for hex boards using cube coordinates', () => {
    const board = createBoard('hexagonal', 3);
    render(<BoardView boardType="hexagonal" board={board} showMovementGrid={true} />);

    const grid = screen.getByTestId('board-view').querySelector('svg');
    expect(grid).toBeInTheDocument();
    expect(grid?.querySelectorAll('circle').length).toBeGreaterThan(0);
  });
});
