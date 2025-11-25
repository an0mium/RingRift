import { GameEngine as SharedGameEngine } from '../../src/shared/engine/GameEngine';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import {
  Player,
  BoardType,
  GameState as WireGameState,
} from '../../src/shared/types/game';
import {
  computeProgressSnapshot,
  summarizeBoard,
  hashGameState,
} from '../../src/shared/engine/core';
import {
  GameState as SharedEngineGameState,
  PlaceRingAction,
  MoveStackAction,
} from '../../src/shared/engine/types';

type Snapshot = {
  boardType: BoardType;
  currentPlayer: number;
  currentPhase: WireGameState['currentPhase'];
  gameStatus: WireGameState['gameStatus'];
  totalRingsInPlay: number;
  totalRingsEliminated: number;
  victoryThreshold: number;
  territoryVictoryThreshold: number;
  progress: ReturnType<typeof computeProgressSnapshot>;
  boardSummary: ReturnType<typeof summarizeBoard>;
  players: Array<{
    playerNumber: number;
    ringsInHand: number;
    eliminatedRings: number;
    territorySpaces: number;
  }>;
};

function createPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: 600,
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
      timeRemaining: 600,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

const timeControl = { initialTime: 600, increment: 0, type: 'blitz' as const };

function snapshotFromState(state: WireGameState): Snapshot {
  const progress = computeProgressSnapshot(state);
  const boardSummary = summarizeBoard(state.board);

  const players = state.players
    .map((p) => ({
      playerNumber: p.playerNumber,
      ringsInHand: p.ringsInHand,
      eliminatedRings: p.eliminatedRings,
      territorySpaces: p.territorySpaces,
    }))
    .sort((a, b) => a.playerNumber - b.playerNumber);

  return {
    boardType: state.boardType,
    currentPlayer: state.currentPlayer,
    currentPhase: state.currentPhase,
    gameStatus: state.gameStatus,
    totalRingsInPlay: state.totalRingsInPlay,
    totalRingsEliminated: state.totalRingsEliminated,
    victoryThreshold: state.victoryThreshold,
    territoryVictoryThreshold: state.territoryVictoryThreshold,
    progress,
    boardSummary,
    players,
  };
}

function runDeterministicScript(): { snapshot: Snapshot; hash: string } {
  const players = createPlayers();

  const initial = createInitialGameState(
    'shared-engine-determinism',
    'square8',
    players,
    timeControl,
    false,
    123456
  );

  const engine = new SharedGameEngine(initial as unknown as SharedEngineGameState);

  const actions: Array<PlaceRingAction | MoveStackAction> = [
    {
      type: 'PLACE_RING',
      playerId: 1,
      position: { x: 0, y: 0 },
      count: 1,
    },
    {
      type: 'MOVE_STACK',
      playerId: 1,
      from: { x: 0, y: 0 },
      to: { x: 0, y: 1 },
    },
  ];

  for (const action of actions) {
    const event = engine.processAction(action);
    expect(event.type).toBe('ACTION_PROCESSED');
  }

  const finalState = engine.getGameState() as unknown as WireGameState;
  return {
    snapshot: snapshotFromState(finalState),
    hash: hashGameState(finalState),
  };
}

describe('Shared engine determinism', () => {
  it('replaying the same action script twice yields identical final snapshot and hash', () => {
    const first = runDeterministicScript();
    const second = runDeterministicScript();

    expect(second.snapshot).toEqual(first.snapshot);
    expect(second.hash).toBe(first.hash);
  });
});