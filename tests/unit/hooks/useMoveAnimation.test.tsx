import React, { useState } from 'react';
import { render, screen, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { useAutoMoveAnimation } from '../../../src/client/hooks/useMoveAnimation';
import type { GameState, Move, Position } from '../../../src/shared/types/game';
import { createTestBoard, createTestGameState, posStr } from '../../utils/fixtures';

describe('useAutoMoveAnimation', () => {
  function createMovementMove(from: Position, to: Position, overrides: Partial<Move> = {}): Move {
    return {
      id: 'move-1',
      type: 'move_stack',
      player: 1,
      from,
      to,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
      ...overrides,
    };
  }

  it('clears stale pending animation when game id changes', async () => {
    const board1 = createTestBoard('hexagonal');
    const from: Position = { x: 0, y: 0, z: 0 };
    const to: Position = { x: 0, y: 1, z: -1 };
    const move = createMovementMove(from, to, {
      // Minimal extra fields so history helpers remain happy in case they are used
      stackMoved: {
        position: from,
        rings: [1],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 1,
      },
    } as Partial<Move>);

    const game1Before: GameState = createTestGameState({
      id: 'test-game-1',
      boardType: 'hexagonal',
      board: board1,
      currentPlayer: 1,
      moveHistory: [],
      history: [],
    });

    const game1After: GameState = {
      ...game1Before,
      moveHistory: [move],
      history: [
        {
          moveNumber: 1,
          action: move,
          actor: 1,
          phaseBefore: game1Before.currentPhase,
          phaseAfter: game1Before.currentPhase,
          statusBefore: game1Before.gameStatus,
          statusAfter: game1Before.gameStatus,
          progressBefore: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
          progressAfter: { markers: 0, collapsed: 0, eliminated: 0, S: 0 },
        },
      ],
    };

    const board2 = createTestBoard('hexagonal');
    const game2: GameState = createTestGameState({
      id: 'test-game-2',
      boardType: 'hexagonal',
      board: board2,
      currentPlayer: 1,
      moveHistory: [],
      history: [],
    });

    let setGameState: React.Dispatch<React.SetStateAction<GameState | null>>;

    const Harness: React.FC = () => {
      const [state, internalSetState] = useState<GameState | null>(game1Before);
      setGameState = internalSetState;
      const { pendingAnimation } = useAutoMoveAnimation(state);
      const label = pendingAnimation
        ? posStr(pendingAnimation.to.x, pendingAnimation.to.y, (pendingAnimation.to as any).z)
        : 'none';
      return <div data-testid="pending-animation-label">{label}</div>;
    };

    render(<Harness />);

    // First, apply a state update that appends a move; this should
    // create a pending animation rooted at the move's destination.
    await act(async () => {
      setGameState(game1After);
    });

    expect(screen.getByTestId('pending-animation-label').textContent).toBe('0,1,-1');

    // Now switch to a brand-new game with a different id and no moves.
    // The hook should clear the stale animation so the board renders cleanly.
    await act(async () => {
      setGameState(game2);
    });

    expect(screen.getByTestId('pending-animation-label').textContent).toBe('none');
  });
});
