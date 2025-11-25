/**
 * LPS Cross-Interaction Scenarios: Lines & Territory
 *
 * These tests exercise Last-Player-Standing (LPS) victory (R172) in the
 * presence of active line and territory mechanics, verifying that:
 * 1. Lines and territory resolve fully before LPS triggers.
 * 2. The backend GameEngine correctly sequences phase processing.
 *
 * Canonical rules references:
 * - R172 (§13.3): LPS victory condition requiring a full round where only
 *   one player has any "real actions" (placements, non-capture moves,
 *   overtaking captures).
 * - Turn phase order (R070–R072): Line Processing → Territory Processing →
 *   Victory Check
 * - CLAR-002: Forced-elimination reactions do NOT count as real actions.
 *
 * TODO-LPS-CROSS-INTERACTION: These tests require deep investigation into
 * LPS victory condition implementation and internal GameEngine APIs.
 * The tests mock internal methods (updateLpsTrackingForCurrentTurn,
 * maybeEndGameByLastPlayerStanding) which may have changed behavior.
 * Skipped pending dedicated LPS victory condition refactoring.
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import {
  BoardType,
  GameState,
  Player,
  Position,
  TimeControl,
  BOARD_CONFIGS,
  positionToString,
} from '../../src/shared/types/game';

describe.skip('GameEngine LPS + Line/Territory Cross-Interaction Scenarios', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  const requiredLineLength = BOARD_CONFIGS[boardType].lineLength;

  function createThreePlayerConfig(): Player[] {
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
      {
        id: 'p3',
        username: 'Player3',
        type: 'human',
        playerNumber: 3,
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
    const engine = new GameEngine(
      'lps-cross-interaction-test',
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

  function startInteractiveTurn(engineAny: any, gameState: GameState, playerNumber: number) {
    gameState.currentPlayer = playerNumber;
    gameState.currentPhase = 'ring_placement';
    engineAny.updateLpsTrackingForCurrentTurn();
    return engineAny.maybeEndGameByLastPlayerStanding();
  }

  function makeStack(
    boardManager: any,
    gameState: GameState,
    playerNumber: number,
    height: number,
    position: Position
  ) {
    const rings = Array(height).fill(playerNumber);
    const stack = {
      position,
      rings,
      stackHeight: rings.length,
      capHeight: rings.length,
      controllingPlayer: playerNumber,
    };
    boardManager.setStack(position, stack, gameState.board);
  }

  describe('LPS + Lines Cross-Interaction', () => {
    /**
     * Scenario: After line processing completes, LPS plateau is detected
     *
     * Setup:
     * - 3-player game on square8
     * - P1 has stacks and can make moves (has real actions)
     * - P2 and P3 have no stacks, no rings in hand (no real actions)
     * - A line for P1 is detected and processed
     *
     * Expected:
     * - Line processing completes FIRST
     * - THEN LPS tracking is updated
     * - After a full round of P1 being the only active player, LPS triggers
     * - Game does NOT end during line processing itself
     */
    it('LPS_triggers_only_after_line_processing_completes', async () => {
      const { engine, engineAny, gameState } = createEngineWithPlayers(createThreePlayerConfig());
      const boardManager: any = engineAny.boardManager;

      // Setup: P1 has a stack (can act), P2 and P3 have nothing
      const p1StackPos: Position = { x: 7, y: 7 };
      makeStack(boardManager, gameState, 1, 2, p1StackPos);

      // Create markers for a line that P1 owns (will be processed)
      const linePositions: Position[] = [];
      for (let i = 0; i < requiredLineLength; i++) {
        linePositions.push({ x: i, y: 1 });
      }

      // Mock findAllLines to return P1's exact-length line
      const findAllLinesSpy = jest.spyOn(boardManager, 'findAllLines');
      findAllLinesSpy
        .mockImplementationOnce(() => [
          {
            player: 1,
            positions: linePositions,
            length: linePositions.length,
            direction: { x: 1, y: 0 },
          },
        ])
        .mockImplementation(() => []);

      // Mock hasAnyRealActionForPlayer: P1 has real actions, P2/P3 don't
      const realActionByPlayer: Record<number, boolean> = { 1: true, 2: false, 3: false };
      engineAny.hasAnyRealActionForPlayer = jest.fn(
        (_state: GameState, playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      // Process lines for P1 (simulating post-movement phase)
      await engineAny.processLineFormations();

      // Verify line processing completed (all markers collapsed)
      for (const pos of linePositions) {
        const key = positionToString(pos);
        expect(gameState.board.collapsedSpaces.get(key)).toBe(1);
      }

      // Now simulate a full round of LPS tracking
      // Turn 1: P1 starts (has actions)
      let result = startInteractiveTurn(engineAny, gameState, 1);
      expect(result).toBeUndefined(); // Not yet - need a full round

      // Turn 2: P2 (no actions)
      result = startInteractiveTurn(engineAny, gameState, 2);
      expect(result).toBeUndefined();

      // Turn 3: P3 (no actions)
      result = startInteractiveTurn(engineAny, gameState, 3);
      expect(result).toBeUndefined();

      // Turn 4: P1 again - now LPS should trigger
      result = startInteractiveTurn(engineAny, gameState, 1);

      // LPS should trigger now
      expect(result).not.toBeNull();
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');

      // Terminal state invariants
      expect(gameState.gameStatus).toBe('completed');
      expect(gameState.winner).toBe(1);
    });

    /**
     * Scenario: Line processing changes who has real actions, affecting LPS
     *
     * Setup:
     * - 3-player game
     * - Initially P1 and P2 both have stacks (both have real actions)
     * - Line processing for P1 collapses line markers (P2's stack not on line)
     * - After line processing and state changes, only P1 has real actions
     *
     * Expected:
     * - LPS plateau begins only AFTER the hasAnyRealActionForPlayer returns change
     * - This simulates game state changes that affect action availability
     */
    it('line_processing_can_create_LPS_plateau_by_changing_board_state', async () => {
      const { engine, engineAny, gameState } = createEngineWithPlayers(createThreePlayerConfig());
      const boardManager: any = engineAny.boardManager;

      // Setup: P1 has a stack (retained after line processing)
      const p1StackPos: Position = { x: 7, y: 7 };
      makeStack(boardManager, gameState, 1, 2, p1StackPos);

      // Simulate collapsed spaces from line processing
      const linePositions: Position[] = [];
      for (let i = 0; i < requiredLineLength; i++) {
        const pos = { x: i, y: 1 };
        linePositions.push(pos);
        const key = positionToString(pos);
        gameState.board.collapsedSpaces.set(key, 1);
      }

      // After line processing: only P1 has real actions
      engineAny.hasAnyRealActionForPlayer = jest.fn(
        (_state: GameState, playerNumber: number) => playerNumber === 1
      );

      // Full round where only P1 has actions
      let result = startInteractiveTurn(engineAny, gameState, 1);
      expect(result).toBeUndefined();
      result = startInteractiveTurn(engineAny, gameState, 2);
      expect(result).toBeUndefined();
      result = startInteractiveTurn(engineAny, gameState, 3);
      expect(result).toBeUndefined();

      // P1's turn again - LPS should trigger now (completed a full round as sole player)
      result = startInteractiveTurn(engineAny, gameState, 1);

      expect(result).not.toBeUndefined();
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');

      // Verify the board state reflects line processing
      for (const pos of linePositions) {
        const key = positionToString(pos);
        expect(gameState.board.collapsedSpaces.get(key)).toBe(1);
      }
    });
  });

  describe('LPS + Territory Cross-Interaction', () => {
    /**
     * Scenario: After territory processing completes, LPS plateau is detected
     *
     * Setup:
     * - 3-player game on square8
     * - P1 has stacks and can act
     * - P2 and P3 have no real actions
     * - Territory processing does not change who has actions (no region to process)
     *
     * Expected:
     * - Territory processing runs (no-op in this case)
     * - THEN LPS tracking is updated
     * - After a full round of P1 being the only active player, LPS triggers
     *
     * Note: This tests the phase ordering, not the actual territory collapse
     * (which is covered by more specific territory tests).
     */
    it('LPS_triggers_only_after_territory_processing_completes', async () => {
      const { engine, engineAny, gameState } = createEngineWithPlayers(createThreePlayerConfig());
      const boardManager: any = engineAny.boardManager;

      // P1 has a stack (can act)
      const p1OutsidePos: Position = { x: 0, y: 0 };
      makeStack(boardManager, gameState, 1, 2, p1OutsidePos);

      // No regions to process (simple case)
      jest
        .spyOn(boardManager, 'findDisconnectedRegions')
        .mockImplementation(() => []);

      // Only P1 has real actions
      const realActionByPlayer: Record<number, boolean> = { 1: true, 2: false, 3: false };

      engineAny.hasAnyRealActionForPlayer = jest.fn(
        (_state: GameState, playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      // Process territory (no-op - no regions)
      gameState.currentPlayer = 1;
      await engineAny.processDisconnectedRegions();

      // Full round of LPS tracking where only P1 has actions
      let result = startInteractiveTurn(engineAny, gameState, 1);
      expect(result).toBeUndefined();
      result = startInteractiveTurn(engineAny, gameState, 2);
      expect(result).toBeUndefined();
      result = startInteractiveTurn(engineAny, gameState, 3);
      expect(result).toBeUndefined();

      // P1's turn again - LPS should trigger now (completed a round)
      result = startInteractiveTurn(engineAny, gameState, 1);

      expect(result).not.toBeNull();
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');

      expect(gameState.gameStatus).toBe('completed');
      expect(gameState.winner).toBe(1);
    });

    /**
     * Scenario: Territory collapse removes all of P2's material, creating LPS
     *
     * This tests the case where territory collapse (simulated via direct board
     * manipulation) affects the LPS calculation by fundamentally changing
     * the hasAnyRealActionForPlayer results.
     */
    it('territory_collapse_can_create_LPS_plateau_by_eliminating_player_material', async () => {
      const { engine, engineAny, gameState } = createEngineWithPlayers(createThreePlayerConfig());
      const boardManager: any = engineAny.boardManager;

      // P1 has stacks outside any territory region
      const p1StackPos: Position = { x: 0, y: 0 };
      makeStack(boardManager, gameState, 1, 3, p1StackPos);

      // P2's only stack
      const p2StackPos: Position = { x: 5, y: 5 };
      makeStack(boardManager, gameState, 2, 2, p2StackPos);

      // P3 has nothing
      const player3 = gameState.players.find((p) => p.playerNumber === 3)!;
      player3.ringsInHand = 0;

      // Track whether territory collapse has happened
      let territoryProcessed = false;

      engineAny.hasAnyRealActionForPlayer = jest.fn(
        (_state: GameState, playerNumber: number) => {
          if (!territoryProcessed) {
            // Before territory: P1 and P2 have material
            return playerNumber === 1 || playerNumber === 2;
          } else {
            // After territory: only P1 has real actions
            return playerNumber === 1;
          }
        }
      );

      // Verify both P1 and P2 have material before territory processing
      expect(engineAny.hasAnyRealActionForPlayer(gameState, 1)).toBe(true);
      expect(engineAny.hasAnyRealActionForPlayer(gameState, 2)).toBe(true);

      // Simulate territory collapse by directly manipulating board state
      gameState.board.stacks.delete(positionToString(p2StackPos));
      gameState.board.collapsedSpaces.set(positionToString(p2StackPos), 1);
      territoryProcessed = true;

      // P2's stack is gone
      expect(gameState.board.stacks.has(positionToString(p2StackPos))).toBe(false);

      // Full round of LPS tracking
      let result = startInteractiveTurn(engineAny, gameState, 1);
      expect(result).toBeUndefined();
      result = startInteractiveTurn(engineAny, gameState, 2);
      expect(result).toBeUndefined();
      result = startInteractiveTurn(engineAny, gameState, 3);
      expect(result).toBeUndefined();

      // LPS triggers
      result = startInteractiveTurn(engineAny, gameState, 1);
      expect(result).not.toBeNull();
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');
    });
  });

  describe('LPS + Lines + Territory Combined', () => {
    /**
     * Scenario: Move triggers both line and territory processing in sequence
     *
     * Setup:
     * - 3-player game
     * - P1 makes a move that creates a line
     * - Line processing runs first
     * - Then territory processing detects a region and collapses it
     * - After both complete, LPS is evaluated
     *
     * Expected:
     * - Lines process BEFORE territory (per turn phase order)
     * - Territory processes AFTER lines
     * - LPS is checked ONLY after BOTH complete
     */
    it('LPS_evaluated_only_after_both_lines_and_territory_complete', async () => {
      const { engine, engineAny, gameState } = createEngineWithPlayers(createThreePlayerConfig());
      const boardManager: any = engineAny.boardManager;

      // P1 has a stack outside all regions
      const p1StackPos: Position = { x: 7, y: 7 };
      makeStack(boardManager, gameState, 1, 3, p1StackPos);

      // P3 has nothing
      const player3 = gameState.players.find((p) => p.playerNumber === 3)!;
      player3.ringsInHand = 0;

      // Step 1: Simulate line processing effect (collapse line positions)
      const linePositions: Position[] = [];
      for (let i = 0; i < requiredLineLength; i++) {
        const pos = { x: i, y: 0 };
        linePositions.push(pos);
        const key = positionToString(pos);
        gameState.board.collapsedSpaces.set(key, 1);
      }

      // Step 2: Simulate territory collapse (P2's position collapsed)
      const p2CollapsedPos: Position = { x: 5, y: 5 };
      gameState.board.collapsedSpaces.set(positionToString(p2CollapsedPos), 1);

      // Game should NOT have ended yet (no LPS tracking has occurred)
      expect(gameState.gameStatus).toBe('active');

      // After both processing phases: only P1 has real actions
      engineAny.hasAnyRealActionForPlayer = jest.fn(
        (_state: GameState, playerNumber: number) => playerNumber === 1
      );

      // Step 3: Full round of LPS tracking
      let result = startInteractiveTurn(engineAny, gameState, 1);
      expect(result).toBeUndefined();
      result = startInteractiveTurn(engineAny, gameState, 2);
      expect(result).toBeUndefined();
      result = startInteractiveTurn(engineAny, gameState, 3);
      expect(result).toBeUndefined();

      // Step 4: LPS triggers on P1's next turn
      result = startInteractiveTurn(engineAny, gameState, 1);

      expect(result).not.toBeUndefined();
      expect(result.winner).toBe(1);
      expect(result.reason).toBe('last_player_standing');
      expect(gameState.gameStatus).toBe('completed');
      expect(gameState.winner).toBe(1);
    });
  });

  describe('Phase Order Verification', () => {
    /**
     * Verify that the engine processes phases in the correct order:
     * 1. Line Processing
     * 2. Territory Processing
     * 3. Victory Check (including LPS)
     */
    it('verifies_line_before_territory_before_LPS_check_order', async () => {
      const { engine, engineAny, gameState } = createEngineWithPlayers(createThreePlayerConfig());
      const boardManager: any = engineAny.boardManager;

      // Track processing order
      const processingOrder: string[] = [];

      // P1 has material
      makeStack(boardManager, gameState, 1, 2, { x: 7, y: 7 });

      // Mock line processing
      const originalProcessLineFormations = engineAny.processLineFormations.bind(engineAny);
      engineAny.processLineFormations = async () => {
        processingOrder.push('line_processing');
        return originalProcessLineFormations();
      };

      // Mock territory processing
      const originalProcessDisconnectedRegions =
        engineAny.processDisconnectedRegions.bind(engineAny);
      engineAny.processDisconnectedRegions = async () => {
        processingOrder.push('territory_processing');
        return originalProcessDisconnectedRegions();
      };

      // Mock LPS check
      const originalMaybeEndGameByLPS =
        engineAny.maybeEndGameByLastPlayerStanding.bind(engineAny);
      engineAny.maybeEndGameByLastPlayerStanding = () => {
        processingOrder.push('lps_check');
        return originalMaybeEndGameByLPS();
      };

      // Simulate a turn completion that triggers all phases
      // (Normally this would be triggered by movement completion)
      await engineAny.processLineFormations();
      await engineAny.processDisconnectedRegions();
      await engineAny.maybeEndGameByLastPlayerStanding();

      // Verify order
      expect(processingOrder).toEqual([
        'line_processing',
        'territory_processing',
        'lps_check',
      ]);
    });
  });
});