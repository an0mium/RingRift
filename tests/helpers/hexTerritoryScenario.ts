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

export interface HexTerritoryScenario {
  /** Initial GameState before any territory decisions are applied. */
  initialState: GameState;
  /** Disconnected region to be processed by process_territory_region. */
  region: Territory;
  /** Stack position used for mandatory self-elimination after region processing. */
  outsideStackPosition: Position;
}

export interface HexTwoRegionTerritoryScenario {
  /** Initial GameState before any territory decisions are applied. */
  initialState: GameState;
  /** First disconnected region (Region A) to be processed by process_territory_region. */
  regionA: Territory;
  /** Second disconnected region (Region B) to be processed by process_territory_region. */
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
 * Build a small hexagonal territory scenario:
 *
 * - Board type: hexagonal (size from BOARD_CONFIGS.hexagonal).
 * - Region spaces: a compact three-cell region around the origin.
 * - Stacks:
 *   - Inside region: Player 2 stacks of height 1 on each region space.
 *   - Outside region: a single Player 1 stack of height 1 used for
 *     mandatory self-elimination.
 *
 * The resulting GameState is in `territory_processing` for Player 1,
 * with gameStatus 'active' and zero prior eliminations.
 */
export function createHexTerritoryRegionScenario(
  gameId: string = 'contract-hex-territory-seq'
): HexTerritoryScenario {
  const boardType = 'hexagonal' as const;
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

  // Ensure board metadata is consistent with hex configuration.
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

  // Compact three-cell region around the origin.
  const regionSpaces: Position[] = [
    { x: 0, y: 0, z: 0 },
    { x: 1, y: -1, z: 0 },
    { x: 0, y: -1, z: 1 },
  ];

  // Victim stacks for Player 2 inside the region.
  for (const pos of regionSpaces) {
    const key = positionToString(pos);
    board.stacks.set(key, {
      position: pos,
      rings: [2],
      stackHeight: 1,
      capHeight: 1,
      controllingPlayer: 2,
    } as any);
  }

  // Outside stack for Player 1 used for mandatory self-elimination.
  const outsideStackPosition: Position = { x: 2, y: -2, z: 0 };
  const outsideKey = positionToString(outsideStackPosition);
  board.stacks.set(outsideKey, {
    position: outsideStackPosition,
    rings: [1],
    stackHeight: 1,
    capHeight: 1,
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
 * Build a two-region hexagon territory scenario mirroring the
 * square multi-region fixtures:
 *
 * - Board type: hexagonal.
 * - Region A spaces: compact three-cell region around origin:
 *     (0,0,0), (1,-1,0), (0,-1,1).
 * - Region B spaces: a translated three-cell region:
 *     (3,-3,0), (4,-4,0), (3,-4,1).
 * - Moving player: Player 1.
 * - Outside stacks for Player 1 used for mandatory self-elimination:
 *     - One at (2,-2,0),
 *     - One at (-3,3,0).
 *
 * The resulting GameState is in `territory_processing` for Player 1,
 * with gameStatus 'active' and zero prior eliminations. Both regions
 * are disconnected and credited to Player 1.
 */
export function createHexTwoRegionTerritoryScenario(
  gameId: string = 'contract-hex-two-region-territory-seq'
): HexTwoRegionTerritoryScenario {
  const boardType = 'hexagonal' as const;
  const players = createPlayers();
  const timeControl = createTimeControl();

  const baseState = createInitialGameState(gameId, boardType, players, timeControl, true, 2468);

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
    { x: 0, y: 0, z: 0 },
    { x: 1, y: -1, z: 0 },
    { x: 0, y: -1, z: 1 },
  ];

  const regionBSpaces: Position[] = [
    { x: 3, y: -3, z: 0 },
    { x: 4, y: -4, z: 0 },
    { x: 3, y: -4, z: 1 },
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
    { x: 2, y: -2, z: 0 },
    { x: -3, y: 3, z: 0 },
  ];

  for (const pos of outsideStackPositions) {
    const key = positionToString(pos);
    board.stacks.set(key, {
      position: pos,
      rings: [1],
      stackHeight: 1,
      capHeight: 1,
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
