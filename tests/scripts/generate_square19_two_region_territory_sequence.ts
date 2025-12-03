import { serializeGameState } from '../../src/shared/engine/contracts/serialization';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { GameState, Move } from '../../src/shared/types/game';
import { createSquare19TwoRegionTerritoryScenario } from '../helpers/squareTerritoryScenario';

function main() {
  const { initialState, regionA, regionB, outsideStackPositions } =
    createSquare19TwoRegionTerritoryScenario('contract-square19-two-region-territory-seq');

  const state0: GameState = initialState;

  const regionBMove: Move = {
    id: 'square19-two-region-B',
    type: 'process_territory_region',
    player: 1,
    disconnectedRegions: [regionB],
    to: regionB.spaces[0],
    timestamp: new Date('2025-12-02T00:00:00.000Z'),
    thinkTime: 0,
    moveNumber: 1,
  } as Move;

  const resultAfterRegionB = processTurn(state0, regionBMove);
  const state1 = resultAfterRegionB.nextState;

  const regionAMove: Move = {
    id: 'square19-two-region-A',
    type: 'process_territory_region',
    player: 1,
    disconnectedRegions: [regionA],
    to: regionA.spaces[0],
    timestamp: new Date('2025-12-02T00:00:01.000Z'),
    thinkTime: 0,
    moveNumber: 2,
  } as Move;

  const resultAfterRegionA = processTurn(state1, regionAMove);
  const state2 = resultAfterRegionA.nextState;

  const eliminateMove: Move = {
    id: 'square19-two-region-elim',
    type: 'eliminate_rings_from_stack',
    player: 1,
    to: outsideStackPositions[0],
    eliminatedRings: [
      {
        player: 1,
        count: 2,
      },
    ],
    eliminationFromStack: {
      position: outsideStackPositions[0],
      capHeight: 2,
      totalHeight: 2,
    },
    timestamp: new Date('2025-12-02T00:00:02.000Z'),
    thinkTime: 0,
    moveNumber: 3,
  } as Move;

  const resultAfterElim = processTurn(state2, eliminateMove);
  const state3 = resultAfterElim.nextState;

  const serialized0 = serializeGameState(state0);
  const serialized1 = serializeGameState(state1);
  const serialized2 = serializeGameState(state2);
  const serialized3 = serializeGameState(state3);

  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify(
      {
        initialState: serialized0,
        postRegionBState: serialized1,
        postRegionAState: serialized2,
        postEliminationState: serialized3,
        statusAfterRegionB: resultAfterRegionB.status,
        statusAfterRegionA: resultAfterRegionA.status,
        statusAfterElimination: resultAfterElim.status,
      },
      null,
      2
    )
  );
}

main();
