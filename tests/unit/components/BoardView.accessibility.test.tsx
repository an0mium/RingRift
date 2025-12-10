import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardView } from '../../../src/client/components/BoardView';
import type { BoardState, Position } from '../../../src/shared/types/game';

function emptyBoard(): BoardState {
  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: 8,
    type: 'square8',
  };
}

describe('BoardView accessibility announcements', () => {
  it('announces valid move count when a piece is selected', () => {
    const selected: Position = { x: 1, y: 1 };
    const validTargets: Position[] = [
      { x: 1, y: 2 },
      { x: 2, y: 1 },
    ];

    render(
      <BoardView
        boardType="square8"
        board={emptyBoard()}
        selectedPosition={selected}
        validTargets={validTargets}
      />
    );

    const announcement = screen.getByRole('status');
    expect(announcement).toHaveTextContent('Piece selected. 2 valid moves available');
  });
});
