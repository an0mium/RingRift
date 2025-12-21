import fs from 'fs';
import path from 'path';
import type { GameState, Move, Position } from '../../src/shared/types/game';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import { serializeGameState } from '../../src/shared/engine/contracts/serialization';
import { snapshotFromGameState } from '../utils/stateSnapshots';
import { createSquareTwoRegionTerritoryScenario } from '../helpers/squareTerritoryScenario';

/**
 * Exporter for a mixed line+multi-region territory snapshot on square8.
 *
 * This mirrors the combined line + multi-region parity scenario used in
 * Backend_vs_Sandbox.CaptureAndTerritoryParity tests, but runs directly
 * against the shared orchestrator via processTurn. The resulting snapshot
 * is consumed by Python parity tests as a TS→Python contract fixture.
 *
 * To regenerate the fixture, run:
 *
 *   RINGRIFT_EXPORT_PARITY_SNAPSHOTS=1 \\
 *   npx jest tests/unit/ExportLineAndTerritoryMultiRegionSnapshot.test.ts --runInBand
 *
 * Output:
 *   ai-service/tests/parity/line_territory_multi_region_square8.snapshot.json
 */

const EXPORT_ENABLED =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_EXPORT_PARITY_SNAPSHOTS ?? '');

const maybeTest = EXPORT_ENABLED ? test : test.skip;

function seedOverlengthLineOnRow(
  state: GameState,
  playerNumber: number,
  rowIndex: number,
  overlengthBy: number
): Position[] {
  const requiredLength = getEffectiveLineLengthThreshold(
    state.boardType,
    state.players.length,
    state.rulesOptions
  );
  const length = requiredLength + overlengthBy;

  const positions: Position[] = [];
  for (let x = 0; x < length; x += 1) {
    const pos: Position = { x, y: rowIndex };
    positions.push(pos);
    const key = `${pos.x},${pos.y}`;
    state.board.markers.set(key, {
      position: pos,
      player: playerNumber,
      type: 'regular',
    } as any);
  }

  return positions;
}

maybeTest('export mixed line+multi-region territory snapshot for Python parity', () => {
  const { initialState, regionA, regionB } = createSquareTwoRegionTerritoryScenario(
    'square-multi-region-line-then-territory-snapshot'
  );

  const state0: GameState = {
    ...initialState,
    currentPlayer: 1,
    currentPhase: 'line_processing',
  };

  // STEP 1: Overlength line on row 3, Option 2 reward.
  const linePositions = seedOverlengthLineOnRow(state0, 1, 3, 1);
  const requiredLength = getEffectiveLineLengthThreshold(
    state0.boardType,
    state0.players.length,
    state0.rulesOptions
  );

  const lineMove: Move = {
    id: 'square-multi-region-line',
    type: 'choose_line_option',
    player: 1,
    thinkTime: 0,
    timestamp: new Date('2025-12-03T00:00:00.000Z'),
    moveNumber: 1,
    formedLines: [
      {
        positions: linePositions,
        player: 1,
        length: linePositions.length,
        direction: { x: 1, y: 0 },
      },
    ],
    collapsedMarkers: linePositions.slice(0, requiredLength),
  } as Move;

  const afterLine = processTurn(state0, lineMove);
  const state1 = afterLine.nextState;

  // STEP 2: Explicitly process Region B.
  const regionBMove: Move = {
    id: 'square-multi-region-regionB',
    type: 'choose_territory_option',
    player: 1,
    disconnectedRegions: [
      {
        spaces: regionB.spaces,
        controllingPlayer: 1,
        isDisconnected: true,
      },
    ],
    to: regionB.spaces[0],
    timestamp: new Date('2025-12-03T00:00:01.000Z'),
    thinkTime: 0,
    moveNumber: 2,
  } as Move;

  const afterRegionB = processTurn(state1, regionBMove);
  const state2 = afterRegionB.nextState;

  // STEP 3: Explicitly process Region A.
  const regionAMove: Move = {
    id: 'square-multi-region-regionA',
    type: 'choose_territory_option',
    player: 1,
    disconnectedRegions: [
      {
        spaces: regionA.spaces,
        controllingPlayer: 1,
        isDisconnected: true,
      },
    ],
    to: regionA.spaces[0],
    timestamp: new Date('2025-12-03T00:00:02.000Z'),
    thinkTime: 0,
    moveNumber: 3,
  } as Move;

  const afterRegionA = processTurn(state2, regionAMove);
  const finalState = afterRegionA.nextState;

  // Sanity: ensure final state is still active and not in a decision phase.
  expect(finalState.gameStatus).toBe('active');

  const snapshot = snapshotFromGameState('line-territory-multi-region-square8', finalState);

  const outDir = path.join(process.cwd(), 'ai-service', 'tests', 'parity');
  fs.mkdirSync(outDir, { recursive: true });

  const outPath = path.join(outDir, 'line_territory_multi_region_square8.snapshot.json');

  const exportPayload = {
    state: serializeGameState(finalState),
    snapshot,
  };

  fs.writeFileSync(outPath, JSON.stringify(exportPayload, null, 2), 'utf8');
});
