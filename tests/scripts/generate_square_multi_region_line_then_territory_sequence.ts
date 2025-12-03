import { serializeGameState } from '../../src/shared/engine/contracts/serialization';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import { type GameState, type Move, type Position } from '../../src/shared/types/game';
import { getEffectiveLineLengthThreshold } from '../../src/shared/engine';
import { createSquareTwoRegionTerritoryScenario } from '../helpers/squareTerritoryScenario';

function seedOverlengthLineMarkers(
  state: GameState,
  playerNumber: number,
  rowIndex: number,
  overlengthBy: number
): Position[] {
  const board = state.board;
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
    board.markers.set(key, {
      position: pos,
      player: playerNumber,
      type: 'regular',
    } as any);
  }

  return positions;
}

function main() {
  const { initialState, regionA, regionB, outsideStackPositions } =
    createSquareTwoRegionTerritoryScenario('contract-square-multi-region-line-then-territory-seq');

  const state0: GameState = {
    ...initialState,
    currentPlayer: 1,
    currentPhase: 'line_processing',
  };

  // Seed an overlength horizontal line for Player 1 on row 0 that does not
  // interfere with the two Q23-style regions.
  const linePositions = seedOverlengthLineMarkers(state0, 1, 3, 1);

  const requiredLength = getEffectiveLineLengthThreshold(
    state0.boardType,
    state0.players.length,
    state0.rulesOptions
  );

  const lineMove: Move = {
    id: 'square-multi-region-line',
    type: 'choose_line_reward',
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

  const resultAfterLine = processTurn(state0, lineMove);
  const state1 = resultAfterLine.nextState;

  // After line processing, we assume both regions from the underlying
  // two-region scenario are available for explicit territory decisions.
  const regionBMove: Move = {
    id: 'square-multi-region-regionB',
    type: 'process_territory_region',
    player: 1,
    disconnectedRegions: [regionB],
    to: regionB.spaces[0],
    timestamp: new Date('2025-12-03T00:00:01.000Z'),
    thinkTime: 0,
    moveNumber: 2,
  } as Move;

  const resultAfterRegionB = processTurn(state1, regionBMove);
  const state2 = resultAfterRegionB.nextState;

  const regionAMove: Move = {
    id: 'square-multi-region-regionA',
    type: 'process_territory_region',
    player: 1,
    disconnectedRegions: [regionA],
    to: regionA.spaces[0],
    timestamp: new Date('2025-12-03T00:00:02.000Z'),
    thinkTime: 0,
    moveNumber: 3,
  } as Move;

  const resultAfterRegionA = processTurn(state2, regionAMove);
  const state3 = resultAfterRegionA.nextState;

  const serialized0 = serializeGameState(state0);
  const serialized1 = serializeGameState(state1);
  const serialized2 = serializeGameState(state2);
  const serialized3 = serializeGameState(state3);

  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify(
      {
        initialState: serialized0,
        postLineState: serialized1,
        postRegionBState: serialized2,
        postRegionAState: serialized3,
        statusAfterLine: resultAfterLine.status,
        statusAfterRegionB: resultAfterRegionB.status,
        statusAfterRegionA: resultAfterRegionA.status,
      },
      null,
      2
    )
  );
}

main();
