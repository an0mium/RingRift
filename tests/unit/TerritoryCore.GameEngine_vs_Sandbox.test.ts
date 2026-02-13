import { BoardState, BoardType, GameState, Player, Position } from '../../src/shared/types/game';
import {
  pos,
  addStack,
  addMarker,
  createTestBoard,
  createTestPlayer,
  createTestGameState,
} from '../utils/fixtures';
import { processDisconnectedRegionCoreOnBoard } from '../../src/client/sandbox/sandboxTerritory';
import { applyTerritoryRegion } from '../../src/shared/engine/territoryProcessing';
import {
  snapshotFromGameState,
  snapshotsEqual,
  diffSnapshots,
  ComparableSnapshot,
} from '../utils/stateSnapshots';

/**
 * Territory core-processing parity tests (C3)
 *
 * These tests compare the shared applyTerritoryRegion helper against the
 * sandboxTerritory.processDisconnectedRegionCoreOnBoard helper in isolation.
 * Both paths start from identical board + player
 * fixtures and apply the same geometric operation:
 *
 *   - Eliminate all rings inside the disconnected region,
 *   - Collapse region spaces and border markers to the moving player's color,
 *   - Credit all eliminations and territorySpaces to the moving player.
 *
 * For each fixture we assert parity on:
 *   - collapsedSpaces keys and owners,
 *   - marker sets (especially border markers),
 *   - per-player eliminatedRings and territorySpaces,
 *   - totalRingsEliminated.
 */

describe('Territory core processing parity (GameEngine vs Sandbox)', () => {
  interface TerritoryCoreFixture {
    name: string;
    boardType: BoardType;
    movingPlayer: number;
    initialBoard: BoardState;
    initialPlayers: Player[];
    regionSpaces: Position[];
  }

  function cloneBoard(board: BoardState): BoardState {
    return {
      ...board,
      stacks: new Map(board.stacks),
      markers: new Map(board.markers),
      collapsedSpaces: new Map(board.collapsedSpaces),
      territories: new Map(board.territories),
      formedLines: [...board.formedLines],
      eliminatedRings: { ...board.eliminatedRings },
    };
  }

  function clonePlayers(players: Player[]): Player[] {
    return players.map((p) => ({ ...p }));
  }

  /**
   * Fixture A:
   * - square8 / 2p
   * - Region: a 2x2 interior block around (2,2):
   *     (2,2), (2,3), (3,2), (3,3)
   *   all containing stacks for Player 2.
   * - Border markers: a "ring" of Player 1 markers around that block,
   *   including diagonals, so border detection must:
   *     - seed via von_neumann territory adjacency,
   *     - expand via Moore adjacency across connected markers.
   */
  function buildFixtureA(): TerritoryCoreFixture {
    const boardType: BoardType = 'square8';
    const movingPlayer = 1;
    const victimPlayer = 2;

    const board = createTestBoard(boardType);
    const players: Player[] = [createTestPlayer(1), createTestPlayer(2)];

    const regionSpaces: Position[] = [pos(2, 2), pos(2, 3), pos(3, 2), pos(3, 3)];

    // Victim stacks inside region.
    regionSpaces.forEach((p) => addStack(board, p, victimPlayer, 2));

    // Marker ring for moving player around the region.
    const borderMarkers: Position[] = [
      // North row
      pos(1, 1),
      pos(2, 1),
      pos(3, 1),
      pos(4, 1),
      // West / East
      pos(1, 2),
      pos(1, 3),
      pos(4, 2),
      pos(4, 3),
      // South row
      pos(1, 4),
      pos(2, 4),
      pos(3, 4),
      pos(4, 4),
    ];
    borderMarkers.forEach((p) => addMarker(board, p, movingPlayer));

    return {
      name: 'square8 interior 2x2 region with marker ring border',
      boardType,
      movingPlayer,
      initialBoard: board,
      initialPlayers: players,
      regionSpaces,
    };
  }

  /**
   * Fixture B:
   * - square8 / 2p
   * - Region: an L-shaped cluster near the left edge:
   *     (1,2), (1,3), (2,2)
   *   all containing stacks for Player 2.
   * - Border markers: Player 1 markers form a partial ring that touches
   *   the left edge and includes both orthogonal and diagonal neighbors.
   *
   * This stresses border detection in the presence of board edges and
   * ensures that neither engine accidentally steps "through" the edge
   * while flooding the border marker set.
   */
  function buildFixtureB(): TerritoryCoreFixture {
    const boardType: BoardType = 'square8';
    const movingPlayer = 1;
    const victimPlayer = 2;

    const board = createTestBoard(boardType);
    const players: Player[] = [createTestPlayer(1), createTestPlayer(2)];

    const regionSpaces: Position[] = [pos(1, 2), pos(1, 3), pos(2, 2)];

    regionSpaces.forEach((p) => addStack(board, p, victimPlayer, 1));

    const borderMarkers: Position[] = [
      // Above region
      pos(0, 1),
      pos(1, 1),
      pos(2, 1),
      // Left/right flanks
      pos(0, 2),
      pos(0, 3),
      pos(2, 3),
      // Below region
      pos(0, 4),
      pos(1, 4),
      pos(2, 4),
    ];
    borderMarkers.forEach((p) => addMarker(board, p, movingPlayer));

    // Add a pre-existing collapsed space elsewhere on the board to ensure
    // it does not interfere with border enumeration.
    addMarker(board, pos(6, 6), movingPlayer);

    return {
      name: 'square8 edge-adjacent L region with partial marker ring border',
      boardType,
      movingPlayer,
      initialBoard: board,
      initialPlayers: players,
      regionSpaces,
    };
  }

  function runCoreParityForFixture(fixture: TerritoryCoreFixture): {
    backendSnapshot: ComparableSnapshot;
    sandboxSnapshot: ComparableSnapshot;
  } {
    const { boardType, movingPlayer, regionSpaces } = fixture;

    // Clone baseline board + players for backend and sandbox so that each
    // engine starts from an identical but independent view.
    const backendBoard = cloneBoard(fixture.initialBoard);
    const backendPlayers = clonePlayers(fixture.initialPlayers);
    const sandboxBoard = cloneBoard(fixture.initialBoard);
    const sandboxPlayers = clonePlayers(fixture.initialPlayers);

    // --- Backend core path via shared applyTerritoryRegion ---
    const backendState: GameState = createTestGameState({
      boardType,
      board: backendBoard,
      players: backendPlayers,
      currentPlayer: movingPlayer,
      totalRingsEliminated: 0,
    });

    const region = {
      spaces: regionSpaces,
      controllingPlayer: movingPlayer,
      isDisconnected: true,
    };

    // Call the shared territory helper directly (previously went through
    // GameEngine.processDisconnectedRegionCore, which was a thin wrapper).
    const outcome = applyTerritoryRegion(backendState.board, region, { player: movingPlayer });

    const backendAfterState: GameState = {
      ...backendState,
      board: outcome.board,
    };

    const territoryGain = outcome.territoryGainedByPlayer[movingPlayer] ?? 0;
    if (territoryGain > 0) {
      const player = backendAfterState.players.find((p) => p.playerNumber === movingPlayer);
      if (player) player.territorySpaces += territoryGain;
    }

    const internalElims = outcome.eliminatedRingsByPlayer[movingPlayer] ?? 0;
    if (internalElims > 0) {
      backendAfterState.totalRingsEliminated += internalElims;
      const player = backendAfterState.players.find((p) => p.playerNumber === movingPlayer);
      if (player) player.eliminatedRings += internalElims;
    }

    const backendSnapshot = snapshotFromGameState('backend-territory-core', backendAfterState);

    // --- Sandbox core path via processDisconnectedRegionCoreOnBoard ---
    const sandboxInitial: GameState = createTestGameState({
      boardType,
      board: sandboxBoard,
      players: sandboxPlayers,
      currentPlayer: movingPlayer,
      totalRingsEliminated: 0,
    });

    const sandboxCoreResult = processDisconnectedRegionCoreOnBoard(
      sandboxInitial.board,
      sandboxInitial.players,
      movingPlayer,
      regionSpaces
    );

    const sandboxAfter: GameState = {
      ...sandboxInitial,
      board: sandboxCoreResult.board,
      players: sandboxCoreResult.players,
      totalRingsEliminated:
        sandboxInitial.totalRingsEliminated + sandboxCoreResult.totalRingsEliminatedDelta,
    };

    const sandboxSnapshot = snapshotFromGameState('sandbox-territory-core', sandboxAfter);

    return { backendSnapshot, sandboxSnapshot };
  }

  const fixtures: TerritoryCoreFixture[] = [buildFixtureA(), buildFixtureB()];

  test.each(fixtures)('%s', (fixture) => {
    const { backendSnapshot, sandboxSnapshot } = runCoreParityForFixture(fixture);

    if (!snapshotsEqual(backendSnapshot, sandboxSnapshot)) {
      const diff = diffSnapshots(backendSnapshot, sandboxSnapshot);

      console.error('[TerritoryCore.GameEngine_vs_Sandbox] snapshot mismatch', {
        fixture: fixture.name,
        diff,
      });
    }

    expect(snapshotsEqual(backendSnapshot, sandboxSnapshot)).toBe(true);
  });
});
