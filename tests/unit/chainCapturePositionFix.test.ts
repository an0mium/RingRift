import { deserializeGameState } from '../../src/shared/engine/contracts/serialization';
import { getValidMoves } from '../../src/shared/engine/orchestration/turnOrchestrator';
import { positionToString } from '../../src/shared/types/game';

describe('chain_capture position fix', () => {
  it('should only return captures from chainCapturePosition, not all stacks', () => {
    // This reproduces the bug from sandbox fixture where game was stuck in chain_capture
    // but no valid moves were shown because getValidMoves returned captures from ALL stacks
    // instead of just the chainCapturePosition
    const fixture = {
      board: {
        type: 'square8',
        size: 8,
        stacks: {
          '6,6': {
            position: { x: 6, y: 6 },
            rings: [2, 1],
            stackHeight: 2,
            capHeight: 1,
            controllingPlayer: 2,
          },
          '5,5': {
            position: { x: 5, y: 5 },
            rings: [2],
            stackHeight: 1,
            capHeight: 1,
            controllingPlayer: 2,
          },
          '6,2': {
            position: { x: 6, y: 2 },
            rings: [2, 2, 2, 1, 1],
            stackHeight: 5,
            capHeight: 3,
            controllingPlayer: 2,
          },
          '3,6': {
            position: { x: 3, y: 6 },
            rings: [1, 1, 1, 1],
            stackHeight: 4,
            capHeight: 4,
            controllingPlayer: 1,
          },
          '1,2': {
            position: { x: 1, y: 2 },
            rings: [1, 1],
            stackHeight: 2,
            capHeight: 2,
            controllingPlayer: 1,
          },
          '0,0': {
            position: { x: 0, y: 0 },
            rings: [1, 1, 1, 1, 2],
            stackHeight: 5,
            capHeight: 4,
            controllingPlayer: 1,
          },
        },
        markers: {
          '2,5': { position: { x: 2, y: 5 }, player: 1, type: 'regular' },
          '7,1': { position: { x: 7, y: 1 }, player: 2, type: 'regular' },
          '1,0': { position: { x: 1, y: 0 }, player: 2, type: 'regular' },
          '1,7': { position: { x: 1, y: 7 }, player: 2, type: 'regular' },
          '4,6': { position: { x: 4, y: 6 }, player: 1, type: 'regular' },
          '0,6': { position: { x: 0, y: 6 }, player: 1, type: 'regular' },
          '5,2': { position: { x: 5, y: 2 }, player: 2, type: 'regular' },
          '4,4': { position: { x: 4, y: 4 }, player: 2, type: 'regular' },
          '3,3': { position: { x: 3, y: 3 }, player: 2, type: 'regular' },
        },
        collapsedSpaces: {
          '3,4': 1,
          '2,1': 2,
          '1,3': 2,
          '4,5': 2,
          '5,4': 2,
          '6,3': 2,
          '5,3': 2,
          '3,1': 2,
          '4,1': 2,
          '5,1': 2,
          '1,1': 1,
        },
        eliminatedRings: {},
        formedLines: [],
      },
      players: [
        { playerNumber: 1, ringsInHand: 5, eliminatedRings: 0, territorySpaces: 0, isActive: true },
        {
          playerNumber: 2,
          ringsInHand: 12,
          eliminatedRings: 0,
          territorySpaces: 6,
          isActive: true,
        },
      ],
      currentPlayer: 2,
      currentPhase: 'chain_capture',
      chainCapturePosition: { x: 6, y: 6 },
      turnNumber: 37,
      moveHistory: [],
      gameStatus: 'active',
      victoryThreshold: 19,
      territoryVictoryThreshold: 33,
      totalRingsEliminated: 0,
    };

    const state = deserializeGameState(fixture as any);

    expect(state.currentPhase).toBe('chain_capture');
    expect(state.chainCapturePosition).toEqual({ x: 6, y: 6 });

    const moves = getValidMoves(state);

    console.log(
      'Valid moves:',
      moves.map((m) => ({
        type: m.type,
        from: m.from ? positionToString(m.from) : undefined,
        target: m.captureTarget ? positionToString(m.captureTarget) : undefined,
        to: positionToString(m.to),
      }))
    );

    // Should have at least one capture from (6,6) to (5,5)
    expect(moves.length).toBeGreaterThan(0);

    // All moves should originate from chainCapturePosition (6,6)
    for (const move of moves) {
      expect(move.from).toBeDefined();
      expect(positionToString(move.from!)).toBe('6,6');
    }

    // Should find the capture from (6,6) capturing (5,5)
    const captureToFiveFive = moves.find(
      (m) => m.captureTarget && positionToString(m.captureTarget) === '5,5'
    );
    expect(captureToFiveFive).toBeDefined();
  });
});
