/**
 * GameEngine LPS Tracking Broadcast Tests
 *
 * Tests that the server GameEngine correctly includes lpsTracking in
 * getGameState() output per RR-CANON-R172 for client display.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import { BoardType, Player, TimeControl } from '../../src/shared/types/game';

const boardType: BoardType = 'square8';
const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

function createTwoPlayerConfig(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Player1',
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
      username: 'Player2',
      type: 'human',
      playerNumber: 2,
      isReady: true,
      timeRemaining: timeControl.initialTime * 1000,
      ringsInHand: 0,
      eliminatedRings: 0,
      territorySpaces: 0,
    },
  ];
}

describe('GameEngine.lpsTracking', () => {
  describe('getGameState includes lpsTracking', () => {
    it('returns lpsTracking with initial values on new game', () => {
      const players = createTwoPlayerConfig();
      const engine = new GameEngine('test-game', boardType, players, timeControl);

      const state = engine.getGameState();

      expect(state.lpsTracking).toBeDefined();
      expect(state.lpsTracking).toEqual({
        roundIndex: 0,
        consecutiveExclusiveRounds: 0,
        consecutiveExclusivePlayer: null,
      });
    });

    it('returns lpsTracking after game starts', () => {
      const players = createTwoPlayerConfig();
      const engine = new GameEngine('test-game', boardType, players, timeControl);
      engine.startGame();

      const state = engine.getGameState();

      expect(state.lpsTracking).toBeDefined();
      expect(typeof state.lpsTracking?.roundIndex).toBe('number');
      expect(typeof state.lpsTracking?.consecutiveExclusiveRounds).toBe('number');
    });

    it('returns fresh lpsTracking copy on each getGameState call', () => {
      const players = createTwoPlayerConfig();
      const engine = new GameEngine('test-game', boardType, players, timeControl);

      const state1 = engine.getGameState();
      const state2 = engine.getGameState();

      // Should be equal values but different object references
      expect(state1.lpsTracking).toEqual(state2.lpsTracking);
      expect(state1.lpsTracking).not.toBe(state2.lpsTracking);
    });
  });

  describe('getLpsTrackingSummary', () => {
    it('returns lightweight summary for client display', () => {
      const players = createTwoPlayerConfig();
      const engine = new GameEngine('test-game', boardType, players, timeControl);

      const summary = engine.getLpsTrackingSummary();

      expect(summary).toBeDefined();
      expect(summary).toHaveProperty('roundIndex');
      expect(summary).toHaveProperty('consecutiveExclusiveRounds');
      expect(summary).toHaveProperty('consecutiveExclusivePlayer');
      // Should only have these 3 properties (no currentRoundActorMask Map)
      expect(Object.keys(summary!)).toHaveLength(3);
    });
  });
});
