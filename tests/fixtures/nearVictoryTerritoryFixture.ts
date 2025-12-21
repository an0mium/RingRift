/**
 * Near-Victory Territory Fixture
 *
 * Provides programmatic access to game states where Player 1 is one
 * territory region resolution away from winning by territory control.
 *
 * Territory victory threshold: > 50% of board spaces (33 for square8).
 * These fixtures set up Player 1 at (threshold - 1) collapsed spaces and a
 * pending region that, when processed, crosses the territory victory line.
 */

import type {
  GameState,
  BoardState,
  Player,
  Position,
  Territory,
  Move,
  RingStack,
  BoardType,
} from '../../src/shared/types/game';
import { BOARD_CONFIGS, positionToString } from '../../src/shared/types/game';

/**
 * Result of creating a near-victory territory fixture.
 */
export interface NearVictoryTerritoryFixture {
  /** Complete game state ready for engine use */
  gameState: GameState;
  /** Move that triggers territory victory when applied */
  winningMove: Move;
  /** Expected winner (should be 1) */
  expectedWinner: number;
  /** Victory type (always 'territory') */
  victoryType: 'territory';
  /** Number of territory spaces Player 1 has before the winning move */
  initialTerritorySpaces: number;
  /** Territory victory threshold (33 for square8) */
  territoryVictoryThreshold: number;
}

/**
 * Configuration options for creating near-victory territory fixtures.
 */
export interface NearVictoryTerritoryOptions {
  /** Board type (default: 'square8') */
  boardType?: BoardType;
  /** Custom game ID (default: auto-generated) */
  gameId?: string;
  /** How many spaces below threshold to start (default: 1) */
  spacesbelowThreshold?: number;
  /** Number of spaces in the pending region (default: 1) */
  pendingRegionSize?: number;
}

function listBoardPositions(boardType: BoardType, size: number): Position[] {
  const positions: Position[] = [];

  if (boardType === 'hexagonal' || boardType === 'hex8') {
    const radius = size - 1;
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        positions.push({ x: q, y: r, z: -q - r });
      }
    }
    return positions;
  }

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      positions.push({ x, y });
    }
  }

  return positions;
}

function buildPendingRegionSpaces(
  boardType: BoardType,
  size: number,
  pendingRegionSize: number
): Position[] {
  if (boardType === 'hexagonal' || boardType === 'hex8') {
    const spaces: Position[] = [];
    for (let i = 0; i < pendingRegionSize; i++) {
      spaces.push({ x: i, y: -i, z: 0 });
    }
    return spaces;
  }

  const centerX = Math.floor(size / 2);
  const centerY = Math.floor(size / 2);
  return Array.from({ length: pendingRegionSize }, (_, i) => ({
    x: centerX,
    y: centerY + i,
  }));
}

/**
 * Creates a near-victory territory fixture for the requested board.
 *
 * Sets up:
 * - Player 1 at (threshold - 1) collapsed territory spaces
 * - A pending territory region near the center with 1+ spaces
 * - Player 1 has a stack on the edge to maintain game validity
 * - Player 2 has a nearby stack to keep an active opponent
 * - Game in 'territory_processing' phase with pending decision
 *
 * When the territory region is processed, Player 1 crosses the territory
 * threshold and triggers territory_control victory.
 */
export function createNearVictoryTerritoryFixture(
  options: NearVictoryTerritoryOptions = {}
): NearVictoryTerritoryFixture {
  const boardType = options.boardType ?? 'square8';
  const boardConfig = BOARD_CONFIGS[boardType];
  const gameId = options.gameId ?? `near-victory-territory-${Date.now()}`;
  const spacesbelowThreshold = options.spacesbelowThreshold ?? 1;
  const pendingRegionSize = options.pendingRegionSize ?? 1;

  // Board dimensions
  const boardSize = boardConfig.size;
  const totalSpaces = boardConfig.totalSpaces;

  // Territory victory threshold is > 50% of board spaces
  const territoryVictoryThreshold = Math.floor(totalSpaces / 2) + 1;

  // Initial territory spaces (just below threshold)
  const initialTerritorySpaces = territoryVictoryThreshold - spacesbelowThreshold;

  // Create collapsed spaces (Player 1 territory)
  const collapsedSpaces = new Map<string, number>();
  let placedSpaces = 0;
  const skipPositions = new Set<string>();

  // Reserve the pending region positions
  const pendingRegionSpaces = buildPendingRegionSpaces(boardType, boardSize, pendingRegionSize);
  for (const pos of pendingRegionSpaces) {
    skipPositions.add(positionToString(pos));
  }

  // Also reserve stack positions
  const p1StackPos: Position =
    boardType === 'hexagonal' || boardType === 'hex8'
      ? { x: boardSize - 1, y: 0, z: -(boardSize - 1) }
      : { x: boardSize - 1, y: boardSize - 1 };
  const p2StackPos: Position =
    boardType === 'hexagonal' || boardType === 'hex8'
      ? { x: boardSize - 2, y: 0, z: -(boardSize - 2) }
      : { x: boardSize - 2, y: boardSize - 1 };
  skipPositions.add(positionToString(p1StackPos));
  skipPositions.add(positionToString(p2StackPos));

  // Fill collapsed spaces up to initialTerritorySpaces
  const allPositions = listBoardPositions(boardType, boardSize);
  for (const pos of allPositions) {
    if (placedSpaces >= initialTerritorySpaces) {
      break;
    }
    const key = positionToString(pos);
    if (!skipPositions.has(key)) {
      collapsedSpaces.set(key, 1);
      placedSpaces++;
    }
  }

  if (placedSpaces < initialTerritorySpaces) {
    throw new Error(
      `Near-victory fixture could not allocate ${initialTerritorySpaces} territory spaces for ${boardType}.`
    );
  }

  // Create stacks map
  const stacks = new Map<string, RingStack>();

  // Player 1 stack at corner
  stacks.set(positionToString(p1StackPos), {
    position: p1StackPos,
    rings: [1, 1],
    stackHeight: 2,
    capHeight: 2,
    controllingPlayer: 1,
  });

  // Player 2 stack adjacent
  stacks.set(positionToString(p2StackPos), {
    position: p2StackPos,
    rings: [2],
    stackHeight: 1,
    capHeight: 1,
    controllingPlayer: 2,
  });

  // Create pending territory
  const territoryId = 'pending_victory_region';
  const territories = new Map<string, Territory>();
  territories.set(territoryId, {
    spaces: pendingRegionSpaces,
    controllingPlayer: 1,
    isDisconnected: false,
  });

  // Create board state
  const board: BoardState = {
    type: boardType,
    size: boardSize,
    stacks,
    markers: new Map(),
    collapsedSpaces,
    territories,
    formedLines: [],
    eliminatedRings: { 1: 0, 2: 0 },
  };

  // Create players
  const players: Player[] = [
    {
      id: 'player-1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 10,
      eliminatedRings: 0,
      territorySpaces: initialTerritorySpaces,
    },
    {
      id: 'player-2',
      username: 'Player 2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: 600,
      ringsInHand: 10,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  // Create game state
  const gameState: GameState = {
    id: gameId,
    boardType,
    board,
    players,
    currentPhase: 'territory_processing',
    currentPlayer: 1,
    moveHistory: [],
    history: [],
    timeControl: {
      initialTime: 600,
      increment: 0,
      type: 'blitz',
    },
    spectators: [],
    gameStatus: 'active',
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: boardConfig.ringsPerPlayer * 2,
    totalRingsEliminated: 0,
    victoryThreshold: boardConfig.ringsPerPlayer,
    territoryVictoryThreshold,
  };

  // Add pending territory decision to state
  (gameState as unknown as Record<string, unknown>).pendingTerritoryDecision = {
    territories: [territoryId],
    currentIndex: 0,
  };

  // Create the winning move
  const winningMove: Move = {
    id: `process-victory-region-${Date.now()}`,
    type: 'choose_territory_option',
    player: 1,
    to: pendingRegionSpaces[0],
    disconnectedRegions: [
      {
        spaces: pendingRegionSpaces,
        controllingPlayer: 1,
        isDisconnected: false,
      },
    ],
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  };

  return {
    gameState,
    winningMove,
    expectedWinner: 1,
    victoryType: 'territory',
    initialTerritorySpaces,
    territoryVictoryThreshold,
  };
}

/**
 * Creates a variant fixture where the pending region has multiple cells,
 * ensuring the final territory count exceeds the threshold by more than 1.
 */
export function createNearVictoryTerritoryFixtureMultiRegion(
  options: NearVictoryTerritoryOptions = {}
): NearVictoryTerritoryFixture {
  return createNearVictoryTerritoryFixture({
    ...options,
    spacesbelowThreshold: options.spacesbelowThreshold ?? 2,
    pendingRegionSize: options.pendingRegionSize ?? 3, // 3 cells ensures we pass the threshold
    gameId: options.gameId ?? `near-victory-territory-multi-${Date.now()}`,
  });
}

/**
 * Serializes the fixture's game state to a JSON-compatible format
 * suitable for contract vectors or test snapshots.
 */
export function serializeNearVictoryTerritoryFixture(fixture: NearVictoryTerritoryFixture): {
  gameState: Record<string, unknown>;
  winningMove: Record<string, unknown>;
} {
  const state = fixture.gameState;

  // Convert Maps to plain objects for JSON serialization
  const stacksObj: Record<string, unknown> = {};
  for (const [key, stack] of state.board.stacks) {
    stacksObj[key] = {
      position: stack.position,
      rings: stack.rings,
      stackHeight: stack.stackHeight,
      capHeight: stack.capHeight,
      controllingPlayer: stack.controllingPlayer,
    };
  }

  const markersObj: Record<string, unknown> = {};
  for (const [key, marker] of state.board.markers) {
    markersObj[key] = {
      player: marker.player,
      position: marker.position,
      type: marker.type,
    };
  }

  const collapsedObj: Record<string, number> = {};
  for (const [key, owner] of state.board.collapsedSpaces) {
    collapsedObj[key] = owner;
  }

  const territoriesObj: Record<string, unknown> = {};
  for (const [key, territory] of state.board.territories) {
    territoriesObj[key] = {
      spaces: territory.spaces,
      controllingPlayer: territory.controllingPlayer,
      isDisconnected: territory.isDisconnected,
    };
  }

  return {
    gameState: {
      gameId: state.id,
      boardType: state.boardType,
      board: {
        type: state.board.type,
        size: state.board.size,
        stacks: stacksObj,
        markers: markersObj,
        collapsedSpaces: collapsedObj,
        territories: territoriesObj,
        eliminatedRings: state.board.eliminatedRings,
        formedLines: state.board.formedLines,
      },
      players: state.players.map((p) => ({
        playerNumber: p.playerNumber,
        ringsInHand: p.ringsInHand,
        eliminatedRings: p.eliminatedRings,
        territorySpaces: p.territorySpaces,
        isActive: true,
      })),
      currentPlayer: state.currentPlayer,
      currentPhase: state.currentPhase,
      turnNumber: 20,
      moveHistory: [],
      gameStatus: state.gameStatus,
      victoryThreshold: state.victoryThreshold,
      territoryVictoryThreshold: state.territoryVictoryThreshold,
      totalRingsEliminated: state.totalRingsEliminated,
    },
    winningMove: {
      id: fixture.winningMove.id,
      type: fixture.winningMove.type,
      player: fixture.winningMove.player,
      to: fixture.winningMove.to,
      disconnectedRegions: fixture.winningMove.disconnectedRegions,
      timestamp: fixture.winningMove.timestamp.toISOString(),
      thinkTime: fixture.winningMove.thinkTime,
      moveNumber: fixture.winningMove.moveNumber,
    },
  };
}
