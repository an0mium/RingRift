import React, { useState } from 'react';
import { render, screen, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { useMoveAnimation, useAutoMoveAnimation } from '../../../src/client/hooks/useMoveAnimation';
import type { GameState, Move, Position, Stack, Board } from '../../../src/shared/types/game';
import { createTestBoard, createTestGameState, posStr } from '../../utils/fixtures';

describe('useMoveAnimation', () => {
  function createMove(overrides: Partial<Move> = {}): Move {
    return {
      id: 'move-1',
      type: 'move_stack',
      player: 1,
      from: { x: 0, y: 0 },
      to: { x: 1, y: 1 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
      ...overrides,
    };
  }

  const Harness: React.FC<{
    onResult: (result: ReturnType<typeof useMoveAnimation>) => void;
  }> = ({ onResult }) => {
    const result = useMoveAnimation();
    onResult(result);
    return (
      <div>
        <div data-testid="animation-type">{result.pendingAnimation?.type ?? 'none'}</div>
        <div data-testid="animation-to">
          {result.pendingAnimation
            ? `${result.pendingAnimation.to.x},${result.pendingAnimation.to.y}`
            : 'none'}
        </div>
      </div>
    );
  };

  it('should trigger animation for move_stack type', async () => {
    let hookResult: ReturnType<typeof useMoveAnimation> | null = null;

    render(<Harness onResult={(r) => (hookResult = r)} />);

    await act(async () => {
      hookResult!.triggerAnimation(createMove(), 1);
    });

    expect(screen.getByTestId('animation-type').textContent).toBe('move');
    expect(screen.getByTestId('animation-to').textContent).toBe('1,1');
  });

  it('should trigger animation for place_ring type', async () => {
    let hookResult: ReturnType<typeof useMoveAnimation> | null = null;

    render(<Harness onResult={(r) => (hookResult = r)} />);

    await act(async () => {
      hookResult!.triggerAnimation(createMove({ type: 'place_ring', from: undefined }), 1);
    });

    expect(screen.getByTestId('animation-type').textContent).toBe('place');
  });

  it('should trigger animation for overtaking_capture type without chain path', async () => {
    let hookResult: ReturnType<typeof useMoveAnimation> | null = null;

    render(<Harness onResult={(r) => (hookResult = r)} />);

    await act(async () => {
      hookResult!.triggerAnimation(createMove({ type: 'overtaking_capture' }), 1);
    });

    expect(screen.getByTestId('animation-type').textContent).toBe('capture');
  });

  it('should trigger animation for overtaking_capture type with chain path', async () => {
    let hookResult: ReturnType<typeof useMoveAnimation> | null = null;

    render(<Harness onResult={(r) => (hookResult = r)} />);

    await act(async () => {
      hookResult!.triggerAnimation(createMove({ type: 'overtaking_capture' }), 1, {
        chainCapturePath: [{ x: 2, y: 2 }],
      });
    });

    expect(screen.getByTestId('animation-type').textContent).toBe('chain_capture');
  });

  it('should clear animation', async () => {
    let hookResult: ReturnType<typeof useMoveAnimation> | null = null;

    render(<Harness onResult={(r) => (hookResult = r)} />);

    await act(async () => {
      hookResult!.triggerAnimation(createMove(), 1);
    });

    expect(screen.getByTestId('animation-type').textContent).toBe('move');

    await act(async () => {
      hookResult!.clearAnimation();
    });

    expect(screen.getByTestId('animation-type').textContent).toBe('none');
  });

  it('should include stack height and cap height when provided', async () => {
    let hookResult: ReturnType<typeof useMoveAnimation> | null = null;
    let capturedAnimation: ReturnType<typeof useMoveAnimation>['pendingAnimation'] = null;

    const DetailedHarness: React.FC = () => {
      const result = useMoveAnimation();
      hookResult = result;
      capturedAnimation = result.pendingAnimation;
      return null;
    };

    render(<DetailedHarness />);

    await act(async () => {
      hookResult!.triggerAnimation(createMove(), 1, {
        stackHeight: 5,
        capHeight: 3,
      });
    });

    expect(capturedAnimation?.stackHeight).toBe(5);
    expect(capturedAnimation?.capHeight).toBe(3);
  });

  it('should include intermediate positions for chain capture', async () => {
    let capturedAnimation: ReturnType<typeof useMoveAnimation>['pendingAnimation'] = null;

    const DetailedHarness: React.FC = () => {
      const result = useMoveAnimation();
      capturedAnimation = result.pendingAnimation;
      return (
        <button
          onClick={() =>
            result.triggerAnimation(createMove({ type: 'overtaking_capture' }), 1, {
              chainCapturePath: [
                { x: 2, y: 2 },
                { x: 3, y: 3 },
              ],
            })
          }
        >
          trigger
        </button>
      );
    };

    render(<DetailedHarness />);

    await act(async () => {
      screen.getByText('trigger').click();
    });

    expect(capturedAnimation?.intermediatePositions).toHaveLength(2);
    expect(capturedAnimation?.intermediatePositions?.[0]).toEqual({ x: 2, y: 2 });
  });
});

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

  it('clears animation when gameState becomes null', async () => {
    const board = createTestBoard('hexagonal');
    const from: Position = { x: 0, y: 0, z: 0 };
    const to: Position = { x: 0, y: 1, z: -1 };
    const move = createMovementMove(from, to);

    const gameWithMove: GameState = createTestGameState({
      id: 'test-game-null',
      boardType: 'hexagonal',
      board,
      currentPlayer: 1,
      moveHistory: [move],
      history: [],
    });

    let setGameState: React.Dispatch<React.SetStateAction<GameState | null>>;

    const Harness: React.FC = () => {
      const [state, internalSetState] = useState<GameState | null>(null);
      setGameState = internalSetState;
      const { pendingAnimation } = useAutoMoveAnimation(state);
      const label = pendingAnimation
        ? posStr(pendingAnimation.to.x, pendingAnimation.to.y, (pendingAnimation.to as any).z)
        : 'none';
      return <div data-testid="pending-animation-label">{label}</div>;
    };

    render(<Harness />);

    // Initially null - should show no animation
    expect(screen.getByTestId('pending-animation-label').textContent).toBe('none');

    // Set to a game with a move
    await act(async () => {
      setGameState(gameWithMove);
    });

    // Now go back to null - animation should be cleared
    await act(async () => {
      setGameState(null);
    });

    expect(screen.getByTestId('pending-animation-label').textContent).toBe('none');
  });

  it('detects chain capture path from move history', async () => {
    const board = createTestBoard('hexagonal');
    const from1: Position = { x: 0, y: 0, z: 0 };
    const to1: Position = { x: 0, y: 1, z: -1 };
    const to2: Position = { x: 0, y: 2, z: -2 };

    // First capture in chain
    const capture1 = createMovementMove(from1, to1, {
      id: 'capture-1',
      type: 'overtaking_capture',
      player: 1,
      moveNumber: 1,
    });

    // Second capture in same chain (same moveNumber)
    const capture2 = createMovementMove(to1, to2, {
      id: 'capture-2',
      type: 'overtaking_capture',
      player: 1,
      moveNumber: 1,
    });

    const gameBeforeChain: GameState = createTestGameState({
      id: 'test-chain-game',
      boardType: 'hexagonal',
      board,
      currentPlayer: 1,
      moveHistory: [capture1],
      history: [],
    });

    const gameAfterChain: GameState = {
      ...gameBeforeChain,
      moveHistory: [capture1, capture2],
    };

    let setGameState: React.Dispatch<React.SetStateAction<GameState | null>>;
    let capturedAnimation: ReturnType<typeof useAutoMoveAnimation>['pendingAnimation'] = null;

    const Harness: React.FC = () => {
      const [state, internalSetState] = useState<GameState | null>(gameBeforeChain);
      setGameState = internalSetState;
      const result = useAutoMoveAnimation(state);
      capturedAnimation = result.pendingAnimation;
      const label = result.pendingAnimation
        ? `${result.pendingAnimation.type}:${posStr(
            result.pendingAnimation.to.x,
            result.pendingAnimation.to.y,
            (result.pendingAnimation.to as any).z
          )}`
        : 'none';
      return <div data-testid="pending-animation-label">{label}</div>;
    };

    render(<Harness />);

    // Wait for initial render
    await act(async () => {
      // Initial state has the first capture already
    });

    // Add the second capture
    await act(async () => {
      setGameState(gameAfterChain);
    });

    // The animation should be triggered for the chain capture
    expect(screen.getByTestId('pending-animation-label').textContent).toContain('0,2,-2');
  });

  it('handles moves without explicit from position using board diff', async () => {
    // Create two boards that differ by one stack position
    const board1 = createTestBoard('hexagonal');
    const stack: Stack = {
      position: { x: 1, y: 1, z: -2 },
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    };

    // Add a stack to board1
    const board1WithStack = {
      ...board1,
      stacks: new Map(board1.stacks).set('1,1,-2', stack),
    };

    // Board2 has the stack moved to a different position
    const board2 = {
      ...board1,
      stacks: new Map(board1.stacks).set('2,2,-4', {
        ...stack,
        position: { x: 2, y: 2, z: -4 },
      }),
    };

    // Create a move without explicit 'from' position
    const moveWithoutFrom = createMovementMove(
      undefined as unknown as Position, // No from
      { x: 2, y: 2, z: -4 },
      { id: 'move-no-from' }
    );

    const gameBefore: GameState = createTestGameState({
      id: 'test-game-diff',
      boardType: 'hexagonal',
      board: board1WithStack,
      currentPlayer: 1,
      moveHistory: [],
      history: [],
    });

    const gameAfter: GameState = {
      ...gameBefore,
      board: board2,
      moveHistory: [moveWithoutFrom],
    };

    let setGameState: React.Dispatch<React.SetStateAction<GameState | null>>;

    const Harness: React.FC = () => {
      const [state, internalSetState] = useState<GameState | null>(gameBefore);
      setGameState = internalSetState;
      const { pendingAnimation } = useAutoMoveAnimation(state);
      const label = pendingAnimation
        ? posStr(pendingAnimation.to.x, pendingAnimation.to.y, (pendingAnimation.to as any).z)
        : 'none';
      return <div data-testid="pending-animation-label">{label}</div>;
    };

    render(<Harness />);

    await act(async () => {
      setGameState(gameAfter);
    });

    // Animation should be triggered at the destination
    expect(screen.getByTestId('pending-animation-label').textContent).toBe('2,2,-4');
  });
});
