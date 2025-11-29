import { processTurn } from '../../src/shared/engine/orchestration/turnOrchestrator';
import {
  type BoardType,
  type Player,
  type TimeControl,
  type Move,
} from '../../src/shared/types/game';
import { createInitialGameState } from '../../src/shared/engine/initialState';
import { computeProgressSnapshot } from '../../src/shared/engine/core';
import { serializeGameState, validateProcessTurnResponse } from '../../src/shared/engine/contracts';

describe('turnOrchestrator – ProcessingMetadata S-invariant alignment', () => {
  const boardType: BoardType = 'square8';

  const timeControl: TimeControl = {
    initialTime: 600,
    increment: 0,
    type: 'blitz',
  };

  const players: Player[] = [
    {
      id: 'p1',
      username: 'Player 1',
      type: 'human',
      playerNumber: 1,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
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
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];

  it('populates sInvariantBefore/After from computeProgressSnapshot and matches contracts schema', () => {
    const initial = createInitialGameState(
      'orchestrator-metadata',
      boardType,
      players,
      timeControl
    );

    const move: Move = {
      id: 'move-1',
      type: 'place_ring',
      player: 1,
      to: { x: 3, y: 3 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    };

    const beforeS = computeProgressSnapshot(initial as any).S;

    const result = processTurn(initial as any, move);

    const afterS = computeProgressSnapshot(result.nextState as any).S;

    expect(result.metadata.sInvariantBefore).toBe(beforeS);
    expect(result.metadata.sInvariantAfter).toBe(afterS);
    expect(result.metadata.sInvariantAfter).toBeGreaterThanOrEqual(
      result.metadata.sInvariantBefore
    );

    const contractResponse = {
      nextState: serializeGameState(result.nextState as any),
      status: result.status,
      pendingDecision: result.pendingDecision,
      victoryResult: result.victoryResult,
      metadata: result.metadata,
    };

    const validation = validateProcessTurnResponse(contractResponse);
    expect(validation.success).toBe(true);
  });
});
