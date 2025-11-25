import { GameEngine } from '../../src/server/game/GameEngine';
import { BoardType, GameState, Player, TimeControl } from '../../src/shared/types/game';
import { victoryRuleScenarios, VictoryRuleScenario } from './rulesMatrix';

/**
 * RulesMatrix → GameEngine victory scenarios
 *
 * Data-driven backend checks for §13.1 (ring-elimination) and §13.2
 * (territory-control) using victoryRuleScenarios defined in rulesMatrix.ts.
 */

describe('RulesMatrix → GameEngine victory scenarios (backend)', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };

  function createPlayers(): Player[] {
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

  function createEngineWithPlayers(
    players: Player[]
  ): { engine: GameEngine; engineAny: any; gameState: GameState } {
    const engine = new GameEngine(
      'rules-matrix-victory-lps',
      boardType,
      players,
      timeControl,
      false
    );
    const engineAny: any = engine as any;
    const gameState: GameState = engineAny.gameState as GameState;
    gameState.gameStatus = 'active';
    gameState.currentPhase = 'ring_placement';
    return { engine, engineAny, gameState };
  }

  function startInteractiveTurn(
    engineAny: any,
    gameState: GameState,
    playerNumber: number
  ): any {
    gameState.currentPlayer = playerNumber;
    gameState.currentPhase = 'ring_placement';
    engineAny.updateLpsTrackingForCurrentTurn();
    return engineAny.maybeEndGameByLastPlayerStanding();
  }

  const scenarios: VictoryRuleScenario[] = victoryRuleScenarios.filter(
    (s) =>
      s.ref.id === 'Rules_13_1_ring_elimination_threshold_square8' ||
      s.ref.id === 'Rules_13_2_territory_control_threshold_square8' ||
      s.ref.id === 'Rules_13_3_last_player_standing_3p_unique_actor_square8'
  );

  test.each<VictoryRuleScenario>(scenarios)(
    '%s → backend victory semantics match rules/FAQ expectations',
    (scenario) => {
      if (scenario.ref.id === 'Rules_13_1_ring_elimination_threshold_square8') {
        const engine = new GameEngine(
          `rules-matrix-victory-${scenario.ref.id}`,
          boardType,
          createPlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState: GameState = engineAny.gameState as GameState;

        const player1 = gameState.players.find((p) => p.playerNumber === 1)!;
        const threshold = gameState.victoryThreshold;
        player1.eliminatedRings = threshold;
        gameState.totalRingsEliminated = threshold;
        gameState.board.eliminatedRings[1] = threshold;

        const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);
        expect(endCheck.isGameOver).toBe(true);
        expect(endCheck.winner).toBe(1);
        expect(endCheck.reason).toBe('ring_elimination');
      } else if (scenario.ref.id === 'Rules_13_2_territory_control_threshold_square8') {
        const engine = new GameEngine(
          `rules-matrix-victory-${scenario.ref.id}`,
          boardType,
          createPlayers(),
          timeControl,
          false
        );
        const engineAny: any = engine;
        const gameState: GameState = engineAny.gameState as GameState;

        const player1 = gameState.players.find((p) => p.playerNumber === 1)!;
        const threshold = gameState.territoryVictoryThreshold;
        player1.territorySpaces = threshold;

        const endCheck = engineAny.ruleEngine.checkGameEnd(gameState);
        expect(endCheck.isGameOver).toBe(true);
        expect(endCheck.winner).toBe(1);
        expect(endCheck.reason).toBe('territory_control');
      } else if (
        scenario.ref.id === 'Rules_13_3_last_player_standing_3p_unique_actor_square8'
      ) {
        const players = createThreePlayerConfig();
        const { engineAny, gameState } = createEngineWithPlayers(players);

        const realActionByPlayer: Record<number, boolean> = { 1: true, 2: false, 3: false };
        engineAny.hasAnyRealActionForPlayer = jest.fn(
          (_state: GameState, playerNumber: number) => !!realActionByPlayer[playerNumber]
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

        const finalState: GameState = engineAny.gameState as GameState;
        expect(finalState.gameStatus).toBe('completed');
        expect(finalState.winner).toBe(1);
      } else {
        throw new Error(`Unhandled victory scenario id: ${scenario.ref.id}`);
      }
    }
  );
});
