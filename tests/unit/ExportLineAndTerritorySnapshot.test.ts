import fs from 'fs';
import path from 'path';
import type { BoardType, GameState, Position } from '../../src/shared/types/game';
import { snapshotFromGameState } from '../utils/stateSnapshots';
import {
  createOrchestratorBackendEngine,
  seedOverlengthLineForPlayer,
  seedTerritoryRegionWithOutsideStack,
} from '../helpers/orchestratorTestUtils';
import { lineAndTerritoryRuleScenarios } from '../scenarios/rulesMatrix';

/**
 * Utility Jest suite to export canonical ComparableSnapshot JSON fixtures
 * for the combined line+territory scenario on each supported board type.
 *
 * These snapshots are consumed by the Python parity tests in:
 *   ai-service/tests/parity/test_line_and_territory_scenario_parity.py
 *
 * The exporter is intentionally opt-in and is not part of normal CI runs.
 * To (re)generate the fixtures locally, run:
 *
 *   RINGRIFT_EXPORT_PARITY_SNAPSHOTS=1 \\
 *   npx jest tests/unit/ExportLineAndTerritorySnapshot.test.ts --runInBand
 *
 * The snapshots will be written to:
 *   ai-service/tests/parity/line_territory_scenario_square8.snapshot.json
 *   ai-service/tests/parity/line_territory_scenario_square19.snapshot.json
 *   ai-service/tests/parity/line_territory_scenario_hexagonal.snapshot.json
 */

const EXPORT_ENABLED =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_EXPORT_PARITY_SNAPSHOTS ?? '');

const maybeTest = EXPORT_ENABLED ? test : test.skip;

interface SnapshotConfig {
  boardType: BoardType;
  fileName: string;
}

const SNAPSHOT_CONFIGS: SnapshotConfig[] = [
  {
    boardType: 'square8',
    fileName: 'line_territory_scenario_square8.snapshot.json',
  },
  {
    boardType: 'square19',
    fileName: 'line_territory_scenario_square19.snapshot.json',
  },
  {
    boardType: 'hexagonal',
    fileName: 'line_territory_scenario_hexagonal.snapshot.json',
  },
];

function buildLineTerritorySnapshotState(boardType: BoardType): GameState {
  const scenario = lineAndTerritoryRuleScenarios.find(
    (s) => s.boardType === boardType && s.kind === 'line-and-territory'
  );

  if (!scenario) {
    throw new Error(`No lineAndTerritoryRuleScenario found for boardType=${boardType}`);
  }

  const engine = createOrchestratorBackendEngine(`line-territory-${boardType}`, boardType);
  const engineAny = engine as any;
  const state: GameState = engineAny.gameState as GameState;
  const board = state.board;

  // Clear any initial geometry and realise the scenario directly:
  // an overlength line for the controlling player plus a single-cell
  // disconnected region and outside stack, mirroring the Q7/Q20
  // lineAndTerritoryRuleScenarios definitions.
  board.stacks.clear();
  board.markers.clear();
  board.collapsedSpaces.clear();

  const { line, territoryRegion } = scenario;
  const controllingPlayer = territoryRegion.controllingPlayer;
  const victimPlayer = territoryRegion.victimPlayer;
  const regionSpaces = territoryRegion.spaces as Position[];
  const outsideStackPos = territoryRegion.outsideStackPosition as Position;
  const outsideHeight = territoryRegion.selfEliminationStackHeight;

  // Seed an overlength line for the controlling player using the
  // same helper as the orchestrator multi-phase tests.
  seedOverlengthLineForPlayer(engine, controllingPlayer, line.rowIndex, line.overlengthBy);

  // Seed the region + outside stack geometry from the scenario.
  seedTerritoryRegionWithOutsideStack(engine, {
    regionSpaces,
    controllingPlayer,
    victimPlayer,
    outsideStackPosition: outsideStackPos,
    outsideStackHeight: outsideHeight,
  });

  // Recompute simple ring counts for invariants.
  const ringsOnBoardByPlayer = new Map<number, number>();
  for (const stack of board.stacks.values() as Iterable<{ rings: number[] }>) {
    for (const ring of stack.rings) {
      ringsOnBoardByPlayer.set(ring, (ringsOnBoardByPlayer.get(ring) ?? 0) + 1);
    }
  }

  state.totalRingsInPlay = Array.from(ringsOnBoardByPlayer.values()).reduce(
    (sum, count) => sum + count,
    0
  );
  state.totalRingsEliminated = 0;

  state.players.forEach((player) => {
    const onBoard = ringsOnBoardByPlayer.get(player.playerNumber) ?? 0;
    player.ringsInHand = Math.max(0, player.ringsInHand - onBoard);
    player.eliminatedRings = 0;
    player.territorySpaces = 0;
  });

  state.currentPlayer = controllingPlayer;
  state.currentPhase = 'line_processing';
  state.gameStatus = 'active';

  // Return a fresh copy from the engine API for snapshotting.
  return engine.getGameState() as GameState;
}

maybeTest('export line+territory ComparableSnapshots for square8/square19/hex', () => {
  const outDir = path.join(process.cwd(), 'ai-service', 'tests', 'parity');
  fs.mkdirSync(outDir, { recursive: true });

  for (const { boardType, fileName } of SNAPSHOT_CONFIGS) {
    const state = buildLineTerritorySnapshotState(boardType);
    const snapshot = snapshotFromGameState(`line-territory-${boardType}`, state);

    const outPath = path.join(outDir, fileName);
    fs.writeFileSync(outPath, JSON.stringify(snapshot, null, 2), 'utf8');
  }
});
