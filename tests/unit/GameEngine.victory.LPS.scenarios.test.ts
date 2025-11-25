import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Move,
  Player,
  TimeControl,
} from '../../src/shared/types/game';

describe('GameEngine R172 last-player-standing (LPS) scenarios', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createThreePlayerConfig(): Player[] {
    return [
      {
        id: 'p1',
        username: 'Player1',
        type: 'human',
        playerNumber: 1,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 1,
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
        ringsInHand: 1,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
      {
        id: 'p3',
        username: 'Player3',
        type: 'human',
        playerNumber: 3,
        isReady: true,
        timeRemaining: timeControl.initialTime * 1000,
        ringsInHand: 1,
        eliminatedRings: 0,
        territorySpaces: 0,
      },
    ];
  }

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

  function createEngineWithPlayers(
    players: Player[]
  ): { engine: GameEngine; engineAny: any; gameState: GameState } {
    const engine = new GameEngine('lps-test', boardType, players, timeControl, false);
    const engineAny: any = engine as any;
    const gameState: GameState = engineAny.gameState as GameState;
    gameState.gameStatus = 'active';
    gameState.currentPhase = 'ring_placement';
    return { engine, engineAny, gameState };
  }

  function startInteractiveTurn(engineAny: any, gameState: GameState, playerNumber: number) {
    gameState.currentPlayer = playerNumber;
    gameState.currentPhase = 'ring_placement';
    engineAny.updateLpsTrackingForCurrentTurn();
    return engineAny.maybeEndGameByLastPlayerStanding();
  }

  it('LPS_3P_unique_actor_full_round_then_win', () => {
    const { engineAny, gameState } = createEngineWithPlayers(createThreePlayerConfig());

    const realActionByPlayer: Record<number, boolean> = { 1: true, 2: false, 3: false };
    engineAny.hasAnyRealActionForPlayer = jest.fn(
      (_state: GameState, playerNumber: number) => {
        return !!realActionByPlayer[playerNumber];
      }
    );

    let result = startInteractiveTurn(engineAny, gameState, 1);
    expect(result).toBeUndefined();

    result = startInteractiveTurn(engineAny, gameState, 2);
    expect(result).toBeUndefined();

    result = startInteractiveTurn(engineAny, gameState, 3);
    expect(result).toBeUndefined();

    result = startInteractiveTurn(engineAny, gameState, 1);

    expect(engineAny.lpsExclusivePlayerForCompletedRound).toBe(1);
    expect(result).not.toBeNull();
    expect(result.winner).toBe(1);
    expect(result.reason).toBe('last_player_standing');

    // Canonical LPS terminal snapshot invariants:
    // - game is completed
    // - winner is the player whose interactive turn triggered LPS
    // - currentPhase is normalised to 'ring_placement'
    // - currentPlayer remains that winning player (the one who would
    //   have taken the next real action if the game had not ended)
    expect(gameState.gameStatus).toBe('completed');
    expect(gameState.winner).toBe(1);
    expect(gameState.currentPhase).toBe('ring_placement');
    expect(gameState.currentPlayer).toBe(1);
  });

  it('LPS_resets_when_another_player_regains_real_action_mid_round', () => {
    const { engineAny, gameState } = createEngineWithPlayers(createThreePlayerConfig());

    let mode: 'mixed' | 'exclusive' = 'mixed';
    engineAny.hasAnyRealActionForPlayer = jest.fn(
      (_state: GameState, playerNumber: number) => {
        if (mode === 'mixed') {
          return playerNumber === 1 || playerNumber === 2;
        }
        return playerNumber === 1;
      }
    );

    let result = startInteractiveTurn(engineAny, gameState, 1);
    expect(result).toBeUndefined();

    result = startInteractiveTurn(engineAny, gameState, 2);
    expect(result).toBeUndefined();

    result = startInteractiveTurn(engineAny, gameState, 3);
    expect(result).toBeUndefined();

    result = startInteractiveTurn(engineAny, gameState, 1);
    expect(result).toBeUndefined();
    expect(engineAny.lpsExclusivePlayerForCompletedRound).toBeNull();

    mode = 'exclusive';

    result = startInteractiveTurn(engineAny, gameState, 2);
    expect(result).toBeUndefined();

    result = startInteractiveTurn(engineAny, gameState, 3);
    expect(result).toBeUndefined();

    result = startInteractiveTurn(engineAny, gameState, 1);
    expect(engineAny.lpsExclusivePlayerForCompletedRound).toBe(1);
    expect(result).not.toBeNull();
    expect(result.winner).toBe(1);
    expect(result.reason).toBe('last_player_standing');

    // LPS win should drive the same canonical terminal invariants as above.
    expect(gameState.gameStatus).toBe('completed');
    expect(gameState.winner).toBe(1);
    expect(gameState.currentPhase).toBe('ring_placement');
    expect(gameState.currentPlayer).toBe(1);
  });

  it('hasAnyRealActionForPlayer_ignores_forced_elimination_only_states', () => {
    const { engineAny, gameState } = createEngineWithPlayers(createTwoPlayerConfig());

    const forcedMove: Move = {
      id: 'elim-1',
      type: 'eliminate_rings_from_stack',
      player: 1,
      to: { x: 0, y: 0 },
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: 1,
    } as any;

    engineAny.ruleEngine.getValidMoves = jest.fn(() => [forcedMove]);

    const p1HasAction = engineAny.hasAnyRealActionForPlayer(gameState, 1);
    const p2HasAction = engineAny.hasAnyRealActionForPlayer(gameState, 2);

    expect(p1HasAction).toBe(false);
    expect(p2HasAction).toBe(false);
  });
});
