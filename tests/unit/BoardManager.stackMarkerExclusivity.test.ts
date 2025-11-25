import { BoardManager } from '../../src/server/game/BoardManager';
import {
  BoardState,
  Position,
  RingStack,
  positionToString,
} from '../../src/shared/types/game';

describe('BoardManager stack/marker exclusivity invariants', () => {
  const boardType = 'square8' as const;

  function createManagerAndBoard(): { manager: BoardManager; board: BoardState } {
    const manager = new BoardManager(boardType);
    const board = manager.createBoard();
    return { manager, board };
  }

  it('setMarker on empty cell creates marker without stack', () => {
    const { manager, board } = createManagerAndBoard();
    const pos: Position = { x: 3, y: 3 };
    const key = positionToString(pos);

    expect(board.stacks.has(key)).toBe(false);
    expect(board.markers.has(key)).toBe(false);

    manager.setMarker(pos, 1, board);

    expect(board.markers.has(key)).toBe(true);
    expect(board.stacks.has(key)).toBe(false);

    // Pure marker placement via the legal mutator must never trigger backend
    // board repairs; any repair here would indicate a hidden invariant defect.
    expect(manager.getRepairCountForTesting()).toBe(0);
  });

  it('setMarker on cell with stack removes stack and leaves only marker', () => {
    const { manager, board } = createManagerAndBoard();
    const pos: Position = { x: 4, y: 4 };
    const key = positionToString(pos);

    const stack: RingStack = {
      position: pos,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 1,
    };

    manager.setStack(pos, stack, board);

    expect(board.stacks.has(key)).toBe(true);
    expect(board.markers.has(key)).toBe(false);

    manager.setMarker(pos, 1, board);

    expect(board.stacks.has(key)).toBe(false);
    expect(board.markers.has(key)).toBe(true);

    // This exclusivity behaviour is part of the core mutator semantics
    // (marker replaces stack) rather than a defensive repair; the repair
    // counter must therefore remain zero.
    expect(manager.getRepairCountForTesting()).toBe(0);
  });

  it('setMarker on collapsed cell is ignored and leaves collapsed space intact', () => {
    const { manager, board } = createManagerAndBoard();
    const pos: Position = { x: 5, y: 5 };
    const key = positionToString(pos);

    manager.setCollapsedSpace(pos, 1, board);

    expect(board.collapsedSpaces.has(key)).toBe(true);
    expect(board.markers.has(key)).toBe(false);
    expect(board.stacks.has(key)).toBe(false);

    manager.setMarker(pos, 1, board);

    expect(board.collapsedSpaces.has(key)).toBe(true);
    expect(board.markers.has(key)).toBe(false);
    expect(board.stacks.has(key)).toBe(false);

    // Ignoring an illegal marker placement on collapsed territory is defined
    // behaviour for setMarker itself and must not be implemented via a repair.
    expect(manager.getRepairCountForTesting()).toBe(0);
  });
});