import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import type {
  BoardState,
  BoardType,
  GameState,
  Move,
  Position,
  Territory,
} from '../../src/shared/types/game';
import { createTestBoard, createTestGameState, pos, addStack } from '../utils/fixtures';

/**
 * Targeted edge-case tests for RuleEngine validation branches that are
 * not covered by the existing movement/placement/skip suites. These are
 * intended to exercise negative paths (invalid phases, missing regions,
 * inconsistent elimination metadata) for branch-coverage rather than to
 * restate rules semantics already anchored in shared-engine tests.
 */

class FakeBoardManagerForValidation {
  constructor(
    public boardType: BoardType,
    private regions: Territory[] = []
  ) {}

  isValidPosition(_pos: Position): boolean {
    return true;
  }

  isCollapsedSpace(_pos: Position, _board: BoardState): boolean {
    return false;
  }

  getMarker(_pos: Position, _board: BoardState): number | undefined {
    return undefined;
  }

  getAllPositions(): Position[] {
    return [];
  }

  findAllLines(_board: BoardState): Array<{ player: number; positions: Position[] }> {
    return [];
  }

  findTerritories(_player: number, _board: BoardState): Territory[] {
    return [];
  }

  findDisconnectedRegions(_board: BoardState, _player: number): Territory[] {
    return this.regions;
  }
}

describe('RuleEngine validation edge branches', () => {
  it('rejects moves for non-active player turn', () => {
    const boardType: BoardType = 'square8';
    const boardManager = new BoardManager(boardType);
    const engine = new RuleEngine(boardManager, boardType);

    const state: GameState = createTestGameState({
      boardType,
      currentPlayer: 1,
    });

    const move: Move = {
      id: 'm1',
      type: 'place_ring',
      player: 2, // valid player but not currentPlayer
      to: pos(0, 0),
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    expect(engine.validateMove(move, state)).toBe(false);
  });

  it('rejects stack movement outside movement/capture phases', () => {
    const boardType: BoardType = 'square8';
    const boardManager = new BoardManager(boardType);
    const engine = new RuleEngine(boardManager, boardType);

    const board = createTestBoard(boardType);
    const from = pos(0, 0);
    const to = pos(1, 1);
    addStack(board, from, 1, 2);

    const state: GameState = createTestGameState({
      boardType,
      board,
      currentPhase: 'ring_placement', // illegal for movement
      currentPlayer: 1,
    });

    const move: Move = {
      id: 'move-stack-1',
      type: 'move_stack',
      player: 1,
      from,
      to,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    expect(engine.validateMove(move, state)).toBe(false);
  });

  it('rejects territory-processing move when no disconnected regions are present', () => {
    const boardType: BoardType = 'square8';
    const fakeManager = new FakeBoardManagerForValidation(boardType, []);
    const engine = new RuleEngine(fakeManager as unknown as BoardManager, boardType);

    const board = createTestBoard(boardType);
    const state: GameState = createTestGameState({
      boardType,
      board,
      currentPhase: 'territory_processing',
      currentPlayer: 1,
    });

    const bogusRegion: Territory = {
      spaces: [pos(0, 0)],
      controllingPlayer: 1,
      isDisconnected: true,
    };

    const move: Move = {
      id: 'territory-1',
      type: 'choose_territory_option',
      player: 1,
      to: pos(0, 0),
      disconnectedRegions: [bogusRegion],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    expect(engine.validateMove(move, state)).toBe(false);
  });

  it('rejects eliminate_rings_from_stack when eliminatedRings metadata is inconsistent with cap height', () => {
    const boardType: BoardType = 'square8';
    const boardManager = new BoardManager(boardType);
    const engine = new RuleEngine(boardManager, boardType);

    const board = createTestBoard(boardType);
    const stackPos = pos(2, 2);
    // Stack of height 2 → capHeight 2
    addStack(board, stackPos, 1, 2);

    const state: GameState = createTestGameState({
      boardType,
      board,
      currentPhase: 'territory_processing',
      currentPlayer: 1,
    });

    const move: Move = {
      id: 'elim-1',
      type: 'eliminate_rings_from_stack',
      player: 1,
      to: stackPos,
      eliminatedRings: [
        {
          player: 1,
          count: 3, // greater than capHeight, should be rejected
        },
      ],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as Move;

    expect(engine.validateMove(move, state)).toBe(false);
  });
});
