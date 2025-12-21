import { serializeGameState } from '../../src/shared/engine/contracts/serialization';
import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import type { GameState, Move } from '../../src/shared/types/game';
import { createSquareTerritoryRegionScenario } from '../helpers/squareTerritoryScenario';

function main() {
  const { initialState, region, outsideStackPosition } = createSquareTerritoryRegionScenario(
    'contract-square-territory-seq'
  );

  const state0: GameState = initialState;

  const regionMove: Move = {
    id: 'square-region-1',
    type: 'choose_territory_option',
    player: 1,
    disconnectedRegions: [region],
    to: region.spaces[0],
    timestamp: new Date('2025-12-02T00:00:00.000Z'),
    thinkTime: 0,
    moveNumber: 1,
  } as Move;

  const resultAfterRegion = processTurn(state0, regionMove);
  const state1 = resultAfterRegion.nextState;

  const eliminateMove: Move = {
    id: 'square-elim-1',
    type: 'eliminate_rings_from_stack',
    player: 1,
    to: outsideStackPosition,
    eliminatedRings: [
      {
        player: 1,
        count: 3,
      },
    ],
    eliminationFromStack: {
      position: outsideStackPosition,
      capHeight: 3,
      totalHeight: 3,
    },
    timestamp: new Date('2025-12-02T00:00:01.000Z'),
    thinkTime: 0,
    moveNumber: 2,
  } as Move;

  const resultAfterElim = processTurn(state1, eliminateMove);
  const state2 = resultAfterElim.nextState;

  const serialized0 = serializeGameState(state0);
  const serialized1 = serializeGameState(state1);
  const serialized2 = serializeGameState(state2);

  // eslint-disable-next-line no-console
  console.log(
    JSON.stringify(
      {
        initialState: serialized0,
        postRegionState: serialized1,
        postEliminationState: serialized2,
        statusAfterRegion: resultAfterRegion.status,
        statusAfterElimination: resultAfterElim.status,
      },
      null,
      2
    )
  );
}

main();
