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
  });
});