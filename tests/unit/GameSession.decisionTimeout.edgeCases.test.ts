import { Server as SocketIOServer } from 'socket.io';
import { GameSession } from '../../src/server/game/GameSession';
import { PythonRulesClient } from '../../src/server/services/PythonRulesClient';

jest.mock('../../src/server/utils/logger', () => ({
  logger: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn(),
  },
}));

describe('GameSession edge-case branches', () => {
  const { logger } = require('../../src/server/utils/logger');

  const createSession = () => {
    const io = {} as unknown as SocketIOServer;
    const pythonClient = {} as unknown as PythonRulesClient;
    const userSockets = new Map<string, string>();
    const session = new GameSession('test-game', io, pythonClient, userSockets) as any;
    session.gameEngine = {
      makeMove: jest.fn(),
      getValidMoves: jest.fn().mockReturnValue([]),
    };
    return session;
  };

  describe('replayMove error handling', () => {
    it('logs and skips historical move with no destination', () => {
      const session = createSession();

      const move = {
        id: 'm-1',
        position: { from: { x: 0, y: 0 } },
        moveType: 'move_stack',
        playerId: '1',
      } as any;

      session.replayMove(move);

      expect(session.gameEngine.makeMove).not.toHaveBeenCalled();
      expect(logger.warn).toHaveBeenCalledWith(
        'Skipping historical move with no destination',
        expect.objectContaining({
          gameId: 'test-game',
          moveId: 'm-1',
        })
      );
    });
  });

  describe('decision-phase timeout scheduling early returns', () => {
    const baseState = {
      gameStatus: 'active',
      currentPhase: 'movement',
      currentPlayer: 1,
      players: [
        {
          playerNumber: 1,
          type: 'human',
        },
      ],
    } as any;

    it('does not schedule timeout when game is not active', () => {
      const session = createSession();
      const state = { ...baseState, gameStatus: 'finished' };

      (session as any).scheduleDecisionPhaseTimeout(state);

      expect(session.getDecisionPhaseRemainingMs()).toBeNull();
    });

    it('does not schedule timeout when current player is AI', () => {
      const session = createSession();
      const state = {
        ...baseState,
        gameStatus: 'active',
        currentPhase: 'line_processing',
        players: [
          {
            playerNumber: 1,
            type: 'ai',
          },
        ],
      };

      (session as any).scheduleDecisionPhaseTimeout(state);

      expect(session.getDecisionPhaseRemainingMs()).toBeNull();
      expect(session.gameEngine.getValidMoves).not.toHaveBeenCalled();
    });

    it('does not schedule timeout when classifyDecisionSurface returns null', () => {
      const session = createSession();
      const state = {
        ...baseState,
        gameStatus: 'active',
        currentPhase: 'line_processing',
      };

      session.gameEngine.getValidMoves = jest.fn().mockReturnValue([
        {
          type: 'process_line',
          id: 'move-1',
        },
      ]);

      (session as any).classifyDecisionSurface = jest.fn(() => null);

      (session as any).scheduleDecisionPhaseTimeout(state);

      expect(session.gameEngine.getValidMoves).toHaveBeenCalledWith(1);
      expect((session as any).classifyDecisionSurface).toHaveBeenCalled();
      expect(session.getDecisionPhaseRemainingMs()).toBeNull();
    });
  });
});
