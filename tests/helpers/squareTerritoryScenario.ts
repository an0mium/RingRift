import {
  type GameState,
  type Player,
  type Position,
  type Territory,
  type TimeControl,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';

export interface SquareTerritoryScenario {
  /** Initial GameState before any territory decisions are applied. */
  initialState: GameState;
  /** Disconnected region to be processed by choose_territory_option. */
  region: Territory;
  /** Stack position used for mandatory self-elimination after region processing. */
  outsideStackPosition: Position;
}

export interface SquareTwoRegionTerritoryScenario {
  /** Initial GameState before any territory decisions are applied. */
  initialState: GameState;
  /** First disconnected region (Region A) to be processed by choose_territory_option. */
  regionA: Territory;
  /** Second disconnected region (Region B) to be processed by choose_territory_option. */
  regionB: Territory;
  /**
   * Outside stack positions for the moving player used for mandatory
   * self-elimination after region processing.
   */
  outsideStackPositions: Position[];
}

function createPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
    {
      id: 'p2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 0,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

function createTimeControl(): TimeControl {
  return {
    initialTime: 600,
    increment: 5,
    type: 'blitz',
  };
}

/**
 * Build a compact square8 territory scenario inspired by the Q23
 * mini-region geometry:
 *
 * - Board type: square8.
 * - Region spaces: 2×2 block at (2,2), (2,3), (3,2), (3,3).
 * - Stacks:
 *   - Inside region: Player 2 stacks of height 2 on each region space.
 *   - Outside region: a single Player 1 stack of height 3 at (0,0)
 *     used for mandatory self-elimination.
 *
 * The resulting GameState is in `territory_processing` for Player 1,
 * with gameStatus 'active' and zero prior eliminations.
 */
export function createSquareTerritoryRegionScenario(
  gameId: string = 'contract-square-territory-seq'
): SquareTerritoryScenario {
  const boardType = 'square8' as const;
  const players = createPlayers();
  const timeControl = createTimeControl();

  const baseState = createInitialGameState(gameId, boardType, players, timeControl, true, 1234);

  const state: GameState = {
    ...baseState,
    currentPlayer: 1,
    currentPhase: 'territory_processing',
    gameStatus: 'active',
    moveHistory: [],
    history: [],
    totalRingsEliminated: 0,
  };

  const board = state.board;
  const config = BOARD_CONFIGS[boardType];

  // Ensure board metadata is consistent with square8 configuration.
  board.type = boardType;
  board.size = config.size;

  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();
  board.territories.clear();
  board.formedLines = [];

  // Initialise eliminatedRings map for both players.
  board.eliminatedRings = {
    1: 0,
    2: 0,
  };

  // 2×2 region for Player 1 with victim stacks for Player 2 inside.
  const regionSpaces: Position[] = [
    { x: 2, y: 2 },
    { x: 2, y: 3 },
    { x: 3, y: 2 },
    { x: 3, y: 3 },
  ];

  for (const pos of regionSpaces) {
    const key = positionToString(pos);
    board.stacks.set(key, {
      position: pos,
      rings: [2, 2],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 2,
    } as any);
  }

  // Outside stack for Player 1 used for mandatory self-elimination.
  const outsideStackPosition: Position = { x: 0, y: 0 };
  const outsideKey = positionToString(outsideStackPosition);
  board.stacks.set(outsideKey, {
    position: outsideStackPosition,
    rings: [1, 1, 1],
    stackHeight: 3,
    capHeight: 3,
    controllingPlayer: 1,
  } as any);

  const region: Territory = {
    spaces: regionSpaces,
    controllingPlayer: 1,
    isDisconnected: true,
  };

  return {
    initialState: state,
    region,
    outsideStackPosition,
  };
}

/**
 * Build a two-region square8 territory scenario inspired by the
 * Rules_12_3_region_order_choice_two_regions_square8 matrix entry and the
 * buildMultiRegionFixture helper used by TerritoryDecisions parity tests:
 *
 * - Board type: square8.
 * - Region A spaces: (1,1), (1,2) with victim stacks for Player 2.
 * - Region B spaces: (5,5), (5,6) with victim stacks for Player 2.
 * - Moving player: Player 1.
 * - Outside stacks for Player 1 used for mandatory self-elimination:
 *     - One at (0,0) of height 2,
 *     - One at (7,7) of height 2.
 *
 * The resulting GameState is in `territory_processing` for Player 1,
 * with gameStatus 'active' and zero prior eliminations. Both regions are
 * disconnected and credited to Player 1.
 */
export function createSquareTwoRegionTerritoryScenario(
  gameId: string = 'contract-square-two-region-territory-seq'
): SquareTwoRegionTerritoryScenario {
  const boardType = 'square8' as const;
  const players = createPlayers();
  const timeControl = createTimeControl();

  const baseState = createInitialGameState(gameId, boardType, players, timeControl, true, 5678);

  const state: GameState = {
    ...baseState,
    currentPlayer: 1,
    currentPhase: 'territory_processing',
    gameStatus: 'active',
    moveHistory: [],
    history: [],
    totalRingsEliminated: 0,
  };

  const board = state.board;
  const config = BOARD_CONFIGS[boardType];

  board.type = boardType;
  board.size = config.size;

  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();
  board.territories.clear();
  board.formedLines = [];

  board.eliminatedRings = {
    1: 0,
    2: 0,
  };

  const regionASpaces: Position[] = [
    { x: 1, y: 1 },
    { x: 1, y: 2 },
  ];
  const regionBSpaces: Position[] = [
    { x: 5, y: 5 },
    { x: 5, y: 6 },
  ];

  for (const pos of regionASpaces) {
    const key = positionToString(pos);
    board.stacks.set(key, {
      position: pos,
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2,
    } as any);
  }

  for (const pos of regionBSpaces) {
    const key = positionToString(pos);
    board.stacks.set(key, {
      position: pos,
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2,
    } as any);
  }

  const outsideStackPositions: Position[] = [
    { x: 0, y: 0 },
    { x: 7, y: 7 },
  ];

  for (const pos of outsideStackPositions) {
    const key = positionToString(pos);
    board.stacks.set(key, {
      position: pos,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as any);
  }

  const regionA: Territory = {
    spaces: regionASpaces,
    controllingPlayer: 1,
    isDisconnected: true,
  };

  const regionB: Territory = {
    spaces: regionBSpaces,
    controllingPlayer: 1,
    isDisconnected: true,
  };

  return {
    initialState: state,
    regionA,
    regionB,
    outsideStackPositions,
  };
}

/**
 * Build a two-region square19 territory scenario derived from the Q23
 * square19 geometry:
 *
 * - Board type: square19.
 * - Region A spaces: 3×3 block at (5,5)–(7,7) with victim stacks for Player 2.
 * - Region B spaces: 3×3 block at (11,11)–(13,13) with victim stacks for Player 2.
 * - Moving player: Player 1.
 * - Outside stacks for Player 1 used for mandatory self-elimination:
 *     - One at (0,1) of height 2 (mirroring the Q23 square19 scenario),
 *     - One at (18,18) of height 2.
 *
 * The resulting GameState is in `territory_processing` for Player 1,
 * with gameStatus 'active' and zero prior eliminations. Both regions are
 * disconnected and credited to Player 1.
 */
export function createSquare19TwoRegionTerritoryScenario(
  gameId: string = 'contract-square19-two-region-territory-seq'
): SquareTwoRegionTerritoryScenario {
  const boardType = 'square19' as const;
  const players = createPlayers();
  const timeControl = createTimeControl();

  const baseState = createInitialGameState(gameId, boardType, players, timeControl, true, 9012);

  const state: GameState = {
    ...baseState,
    currentPlayer: 1,
    currentPhase: 'territory_processing',
    gameStatus: 'active',
    moveHistory: [],
    history: [],
    totalRingsEliminated: 0,
  };

  const board = state.board;
  const config = BOARD_CONFIGS[boardType];

  board.type = boardType;
  board.size = config.size;

  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();
  board.territories.clear();
  board.formedLines = [];

  board.eliminatedRings = {
    1: 0,
    2: 0,
  };

  const regionASpaces: Position[] = [];
  const regionBSpaces: Position[] = [];

  for (let x = 5; x <= 7; x += 1) {
    for (let y = 5; y <= 7; y += 1) {
      regionASpaces.push({ x, y });
    }
  }

  for (let x = 11; x <= 13; x += 1) {
    for (let y = 11; y <= 13; y += 1) {
      regionBSpaces.push({ x, y });
    }
  }

  for (const pos of regionASpaces) {
    const key = positionToString(pos);
    board.stacks.set(key, {
      position: pos,
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2,
    } as any);
  }

  for (const pos of regionBSpaces) {
    const key = positionToString(pos);
    board.stacks.set(key, {
      position: pos,
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2,
    } as any);
  }

  const outsideStackPositions: Position[] = [
    { x: 0, y: 1 },
    { x: 18, y: 18 },
  ];

  for (const pos of outsideStackPositions) {
    const key = positionToString(pos);
    board.stacks.set(key, {
      position: pos,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    } as any);
  }

  const regionA: Territory = {
    spaces: regionASpaces,
    controllingPlayer: 1,
    isDisconnected: true,
  };

  const regionB: Territory = {
    spaces: regionBSpaces,
    controllingPlayer: 1,
    isDisconnected: true,
  };

  return {
    initialState: state,
    regionA,
    regionB,
    outsideStackPositions,
  };
}
