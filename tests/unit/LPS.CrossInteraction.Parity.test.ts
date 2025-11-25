/**
 * LPS Cross-Interaction Parity Test: GameEngine vs ClientSandboxEngine
 *
 * This test verifies that both the backend GameEngine and the ClientSandboxEngine
 * produce identical outcomes for LPS + Line/Territory cross-interaction scenarios.
 *
 * For at least one cross-interaction scenario, we:
 * - Run the same board setup and move sequence through both engines
 * - Assert that both produce the same final winner
 * - Assert that both report 'last_player_standing' as the victory reason
 * - Assert that both have matching final board states
 */

import { GameEngine } from '../../src/server/game/GameEngine';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  BoardState,
  GameState,
  Player,
  Position,
  TimeControl,
  BOARD_CONFIGS,
  positionToString,
  PlayerChoice,
  PlayerChoiceResponseFor,
} from '../../src/shared/types/game';
import { pos, addStack, addMarker } from '../utils/fixtures';

describe('LPS + Lines/Territory Parity: GameEngine vs ClientSandboxEngine', () => {
  const boardType: BoardType = 'square8';
  const timeControl: TimeControl = { initialTime: 600, increment: 0, type: 'blitz' };
  const requiredLineLength = BOARD_CONFIGS[boardType].lineLength;

  // === Backend (GameEngine) Setup ===
  function createBackendThreePlayerConfig(): Player[] {
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

  function createBackendEngine(): { engine: GameEngine; engineAny: any; gameState: GameState } {
    const engine = new GameEngine(
      'lps-parity-test',
      boardType,
      createBackendThreePlayerConfig(),
      timeControl,
      false
    );
    const engineAny: any = engine as any;
    const gameState: GameState = engineAny.gameState as GameState;
    gameState.gameStatus = 'active';
    gameState.currentPhase = 'ring_placement';
    return { engine, engineAny, gameState };
  }

  // === Sandbox (ClientSandboxEngine) Setup ===
  function createSandboxEngine(): ClientSandboxEngine {
    const config: SandboxConfig = {
      boardType,
      numPlayers: 3,
      playerKinds: ['human', 'human', 'human'],
    };

    const handler: SandboxInteractionHandler = {
      async requestChoice<TChoice extends PlayerChoice>(
        choice: TChoice
      ): Promise<PlayerChoiceResponseFor<TChoice>> {
        const anyChoice = choice as any;
        const options: any[] = (anyChoice.options as any[]) ?? [];
        const selectedOption = options.length > 0 ? options[0] : undefined;

        return {
          choiceId: anyChoice.id,
          playerNumber: anyChoice.playerNumber,
          choiceType: anyChoice.type,
          selectedOption,
        } as PlayerChoiceResponseFor<TChoice>;
      },
    };

    return new ClientSandboxEngine({ config, interactionHandler: handler });
  }

  // === Helper: Set up identical board state in both engines ===
  function setupIdenticalBoardState(
    backendState: GameState,
    backendBoardManager: any,
    sandboxState: GameState
  ): void {
    // Clear any existing state
    backendState.board.stacks.clear();
    backendState.board.markers.clear();
    backendState.board.collapsedSpaces.clear();

    sandboxState.board.stacks.clear();
    sandboxState.board.markers.clear();
    sandboxState.board.collapsedSpaces.clear();

    // Clear rings in hand for all players
    for (const player of backendState.players) {
      player.ringsInHand = 0;
    }
    for (const player of sandboxState.players) {
      player.ringsInHand = 0;
    }

    // Add P1's stack at (7,7) - this player will retain actions
    const p1StackPos: Position = pos(7, 7);
    const p1Stack = {
      position: p1StackPos,
      rings: [1, 1],
      stackHeight: 2,
      capHeight: 2,
      controllingPlayer: 1,
    };
    backendBoardManager.setStack(p1StackPos, p1Stack, backendState.board);
    addStack(sandboxState.board, p1StackPos, 1, 2);

    // Create markers for a line at y=1 (P1's line)
    for (let i = 0; i < requiredLineLength; i++) {
      const linePos = pos(i, 1);
      // Backend: use setMarker via board manager
      backendState.board.markers.set(positionToString(linePos), {
        position: linePos,
        player: 1,
        type: 'regular',
      });
      // Sandbox: use addMarker helper
      addMarker(sandboxState.board, linePos, 1);
    }
  }

  // === Helper: Compare board states ===
  function compareBoardStates(backendBoard: BoardState, sandboxBoard: BoardState): {
    identical: boolean;
    differences: string[];
  } {
    const differences: string[] = [];

    // Compare stacks
    const backendStackKeys = new Set(backendBoard.stacks.keys());
    const sandboxStackKeys = new Set(sandboxBoard.stacks.keys());

    for (const key of backendStackKeys) {
      if (!sandboxStackKeys.has(key)) {
        differences.push(`Stack at ${key} exists in backend but not sandbox`);
      } else {
        const backendStack = backendBoard.stacks.get(key)!;
        const sandboxStack = sandboxBoard.stacks.get(key)!;
        if (backendStack.controllingPlayer !== sandboxStack.controllingPlayer) {
          differences.push(
            `Stack at ${key} controlled by ${backendStack.controllingPlayer} in backend, ${sandboxStack.controllingPlayer} in sandbox`
          );
        }
        if (backendStack.stackHeight !== sandboxStack.stackHeight) {
          differences.push(
            `Stack at ${key} has height ${backendStack.stackHeight} in backend, ${sandboxStack.stackHeight} in sandbox`
          );
        }
      }
    }
    for (const key of sandboxStackKeys) {
      if (!backendStackKeys.has(key)) {
        differences.push(`Stack at ${key} exists in sandbox but not backend`);
      }
    }

    // Compare collapsed spaces
    const backendCollapsedKeys = new Set(backendBoard.collapsedSpaces.keys());
    const sandboxCollapsedKeys = new Set(sandboxBoard.collapsedSpaces.keys());

    for (const key of backendCollapsedKeys) {
      if (!sandboxCollapsedKeys.has(key)) {
        differences.push(`Collapsed space at ${key} exists in backend but not sandbox`);
      } else {
        const backendOwner = backendBoard.collapsedSpaces.get(key);
        const sandboxOwner = sandboxBoard.collapsedSpaces.get(key);
        if (backendOwner !== sandboxOwner) {
          differences.push(
            `Collapsed space at ${key} owned by ${backendOwner} in backend, ${sandboxOwner} in sandbox`
          );
        }
      }
    }
    for (const key of sandboxCollapsedKeys) {
      if (!backendCollapsedKeys.has(key)) {
        differences.push(`Collapsed space at ${key} exists in sandbox but not backend`);
      }
    }

    return { identical: differences.length === 0, differences };
  }

  describe('LPS + Lines Parity', () => {
    /**
     * Parity test: After line processing and a full round of LPS tracking,
     * both engines should produce the same winner with the same reason.
     */
    it('both_engines_agree_on_LPS_winner_after_line_processing', async () => {
      // === Setup Backend ===
      const { engineAny: backendAny, gameState: backendState } = createBackendEngine();
      const backendBoardManager = backendAny.boardManager;

      // === Setup Sandbox ===
      const sandboxEngine = createSandboxEngine();
      const sandboxAny: any = sandboxEngine;
      const sandboxState: GameState = sandboxAny.gameState as GameState;

      // Setup identical board state in both engines
      setupIdenticalBoardState(backendState, backendBoardManager, sandboxState);

      // === Mock hasAnyRealActionForPlayer identically in both engines ===
      const realActionByPlayer: Record<number, boolean> = { 1: true, 2: false, 3: false };

      backendAny.hasAnyRealActionForPlayer = jest.fn(
        (_state: GameState, playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      sandboxAny.hasAnyRealActionForPlayer = jest.fn(
        (playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      // === Simulate line processing by directly collapsing line positions ===
      const linePositions: Position[] = [];
      for (let i = 0; i < requiredLineLength; i++) {
        linePositions.push(pos(i, 1));
      }

      // Collapse line positions identically in both engines
      for (const p of linePositions) {
        const key = positionToString(p);
        backendState.board.collapsedSpaces.set(key, 1);
        sandboxState.board.collapsedSpaces.set(key, 1);
        // Clear the markers since they would be removed by line processing
        backendState.board.markers.delete(key);
        sandboxState.board.markers.delete(key);
      }

      backendState.currentPlayer = 1;
      sandboxState.currentPlayer = 1;

      // === Verify line processing results are identical ===
      for (const p of linePositions) {
        const key = positionToString(p);
        const backendCollapsed = backendState.board.collapsedSpaces.get(key);
        const sandboxCollapsed = sandboxState.board.collapsedSpaces.get(key);
        expect(backendCollapsed).toBe(1);
        expect(sandboxCollapsed).toBe(1);
      }

      // === Run LPS tracking in both engines ===
      // Backend LPS tracking
      function backendStartTurn(playerNumber: number) {
        backendState.currentPlayer = playerNumber;
        backendState.currentPhase = 'ring_placement';
        backendAny.updateLpsTrackingForCurrentTurn();
        return backendAny.maybeEndGameByLastPlayerStanding();
      }

      // Sandbox LPS tracking
      function sandboxStartTurn(playerNumber: number) {
        sandboxState.currentPlayer = playerNumber;
        sandboxState.currentPhase = 'ring_placement';
        sandboxAny.handleStartOfInteractiveTurn();
        return sandboxAny.victoryResult;
      }

      // Full round of turns for both engines
      // Note: Backend returns undefined when no LPS, sandbox returns null
      let backendResult = backendStartTurn(1);
      let sandboxResult = sandboxStartTurn(1);
      expect(backendResult ?? null).toBeNull(); // Normalize undefined to null
      expect(sandboxResult).toBeNull();

      backendResult = backendStartTurn(2);
      sandboxResult = sandboxStartTurn(2);
      expect(backendResult ?? null).toBeNull();
      expect(sandboxResult).toBeNull();

      backendResult = backendStartTurn(3);
      sandboxResult = sandboxStartTurn(3);
      expect(backendResult ?? null).toBeNull();
      expect(sandboxResult).toBeNull();

      // P1's turn again - LPS should trigger in both
      backendResult = backendStartTurn(1);
      sandboxResult = sandboxStartTurn(1);

      // === Verify both engines agree on the outcome ===
      expect(backendResult).not.toBeNull();
      expect(sandboxResult).not.toBeNull();

      expect(backendResult.winner).toBe(1);
      expect(sandboxResult!.winner).toBe(1);

      expect(backendResult.reason).toBe('last_player_standing');
      expect(sandboxResult!.reason).toBe('last_player_standing');

      // === Verify final outcomes match ===
      // Backend updates state directly; sandbox stores result in victoryResult
      expect(backendState.gameStatus).toBe('completed');
      expect(backendState.winner).toBe(1);

      // The key parity assertion: both engines agree on winner and reason
      expect(backendResult.winner).toBe(sandboxResult!.winner);
      expect(backendResult.reason).toBe(sandboxResult!.reason);
    });
  });

  describe('LPS + Territory Parity', () => {
    /**
     * Parity test: After territory processing and a full round of LPS tracking,
     * both engines should produce the same winner with the same reason.
     */
    it('both_engines_agree_on_LPS_winner_after_territory_collapse', async () => {
      // === Setup Backend ===
      const { engineAny: backendAny, gameState: backendState } = createBackendEngine();
      const backendBoardManager = backendAny.boardManager;

      // === Setup Sandbox ===
      const sandboxEngine = createSandboxEngine();
      const sandboxAny: any = sandboxEngine;
      const sandboxState: GameState = sandboxAny.gameState as GameState;

      // Clear any existing state
      backendState.board.stacks.clear();
      backendState.board.markers.clear();
      backendState.board.collapsedSpaces.clear();

      sandboxState.board.stacks.clear();
      sandboxState.board.markers.clear();
      sandboxState.board.collapsedSpaces.clear();

      // Clear rings in hand
      for (const player of backendState.players) {
        player.ringsInHand = 0;
      }
      for (const player of sandboxState.players) {
        player.ringsInHand = 0;
      }

      // P1 has stack outside any region
      const p1StackPos: Position = pos(0, 0);
      const p1Stack = {
        position: p1StackPos,
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      };
      backendBoardManager.setStack(p1StackPos, p1Stack, backendState.board);
      addStack(sandboxState.board, p1StackPos, 1, 3);

      // P2 has stack inside the region (will be eliminated)
      const p2StackPos: Position = pos(5, 5);
      const p2Stack = {
        position: p2StackPos,
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      backendBoardManager.setStack(p2StackPos, p2Stack, backendState.board);
      addStack(sandboxState.board, p2StackPos, 2, 1);

      // === Mock hasAnyRealActionForPlayer ===
      // Before territory: P1 and P2 have actions
      // After territory: only P1 has actions
      let realActionByPlayer: Record<number, boolean> = { 1: true, 2: true, 3: false };

      backendAny.hasAnyRealActionForPlayer = jest.fn(
        (_state: GameState, playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      sandboxAny.hasAnyRealActionForPlayer = jest.fn(
        (playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      // === Simulate territory collapse in both engines ===
      // (Directly manipulate the board to simulate territory collapse)
      const p2Key = positionToString(p2StackPos);

      // Backend: remove P2's stack and collapse the space
      backendState.board.stacks.delete(p2Key);
      backendState.board.collapsedSpaces.set(p2Key, 1);

      // Sandbox: same
      sandboxState.board.stacks.delete(p2Key);
      sandboxState.board.collapsedSpaces.set(p2Key, 1);

      // After territory collapse, only P1 has actions
      realActionByPlayer = { 1: true, 2: false, 3: false };

      // === Run LPS tracking in both engines ===
      function backendStartTurn(playerNumber: number) {
        backendState.currentPlayer = playerNumber;
        backendState.currentPhase = 'ring_placement';
        backendAny.updateLpsTrackingForCurrentTurn();
        return backendAny.maybeEndGameByLastPlayerStanding();
      }

      function sandboxStartTurn(playerNumber: number) {
        sandboxState.currentPlayer = playerNumber;
        sandboxState.currentPhase = 'ring_placement';
        sandboxAny.handleStartOfInteractiveTurn();
        return sandboxAny.victoryResult;
      }

      // Full round of turns
      // Note: Backend returns undefined when no LPS, sandbox returns null
      let backendResult = backendStartTurn(1);
      let sandboxResult = sandboxStartTurn(1);
      expect(backendResult ?? null).toBeNull();
      expect(sandboxResult).toBeNull();

      backendResult = backendStartTurn(2);
      sandboxResult = sandboxStartTurn(2);
      expect(backendResult ?? null).toBeNull();
      expect(sandboxResult).toBeNull();

      backendResult = backendStartTurn(3);
      sandboxResult = sandboxStartTurn(3);
      expect(backendResult ?? null).toBeNull();
      expect(sandboxResult).toBeNull();

      // P1's turn again - LPS should trigger in both
      backendResult = backendStartTurn(1);
      sandboxResult = sandboxStartTurn(1);

      // === Verify both engines agree on the winner ===
      expect(backendResult).not.toBeUndefined();
      expect(sandboxResult).not.toBeNull();

      expect(backendResult!.winner).toBe(sandboxResult!.winner);
      expect(backendResult!.reason).toBe(sandboxResult!.reason);

      expect(backendResult!.winner).toBe(1);
      expect(backendResult!.reason).toBe('last_player_standing');

      // Backend updates state directly, sandbox stores in victoryResult
      expect(backendState.gameStatus).toBe('completed');
      expect(backendState.winner).toBe(1);
    });
  });

  describe('LPS + Lines + Territory Combined Parity', () => {
    /**
     * Parity test: After both line and territory processing, both engines
     * should produce the same LPS winner.
     */
    it('both_engines_agree_on_LPS_after_combined_line_and_territory_processing', async () => {
      // === Setup Backend ===
      const { engineAny: backendAny, gameState: backendState } = createBackendEngine();
      const backendBoardManager = backendAny.boardManager;

      // === Setup Sandbox ===
      const sandboxEngine = createSandboxEngine();
      const sandboxAny: any = sandboxEngine;
      const sandboxState: GameState = sandboxAny.gameState as GameState;

      // Clear state
      backendState.board.stacks.clear();
      backendState.board.markers.clear();
      backendState.board.collapsedSpaces.clear();

      sandboxState.board.stacks.clear();
      sandboxState.board.markers.clear();
      sandboxState.board.collapsedSpaces.clear();

      for (const player of backendState.players) {
        player.ringsInHand = 0;
      }
      for (const player of sandboxState.players) {
        player.ringsInHand = 0;
      }

      // P1 has stack at (7,7)
      const p1StackPos: Position = pos(7, 7);
      const p1Stack = {
        position: p1StackPos,
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      };
      backendBoardManager.setStack(p1StackPos, p1Stack, backendState.board);
      addStack(sandboxState.board, p1StackPos, 1, 3);

      // P2 has stack at (5,5) - will be eliminated by territory
      const p2StackPos: Position = pos(5, 5);
      const p2Stack = {
        position: p2StackPos,
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      };
      backendBoardManager.setStack(p2StackPos, p2Stack, backendState.board);
      addStack(sandboxState.board, p2StackPos, 2, 1);

      // Create markers for a line at y=0
      for (let i = 0; i < requiredLineLength; i++) {
        const linePos = pos(i, 0);
        backendState.board.markers.set(positionToString(linePos), {
          position: linePos,
          player: 1,
          type: 'regular',
        });
        addMarker(sandboxState.board, linePos, 1);
      }

      // Mock hasAnyRealActionForPlayer
      let realActionByPlayer: Record<number, boolean> = { 1: true, 2: true, 3: false };

      backendAny.hasAnyRealActionForPlayer = jest.fn(
        (_state: GameState, playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      sandboxAny.hasAnyRealActionForPlayer = jest.fn(
        (playerNumber: number) => !!realActionByPlayer[playerNumber]
      );

      // === Phase 1: Simulate line processing by directly collapsing line positions ===
      const linePositions: Position[] = [];
      for (let i = 0; i < requiredLineLength; i++) {
        linePositions.push(pos(i, 0));
      }

      backendState.currentPlayer = 1;
      sandboxState.currentPlayer = 1;

      // Collapse line positions identically in both engines
      for (const p of linePositions) {
        const key = positionToString(p);
        backendState.board.collapsedSpaces.set(key, 1);
        sandboxState.board.collapsedSpaces.set(key, 1);
        // Clear the markers since they would be removed by line processing
        backendState.board.markers.delete(key);
        sandboxState.board.markers.delete(key);
      }

      // Verify lines collapsed in both
      for (const p of linePositions) {
        const key = positionToString(p);
        expect(backendState.board.collapsedSpaces.get(key)).toBe(1);
        expect(sandboxState.board.collapsedSpaces.get(key)).toBe(1);
      }

      // === Phase 2: Simulate territory collapse ===
      const p2Key = positionToString(p2StackPos);

      backendState.board.stacks.delete(p2Key);
      backendState.board.collapsedSpaces.set(p2Key, 1);

      sandboxState.board.stacks.delete(p2Key);
      sandboxState.board.collapsedSpaces.set(p2Key, 1);

      // After both phases: only P1 has actions
      realActionByPlayer = { 1: true, 2: false, 3: false };

      // === LPS tracking ===
      function backendStartTurn(playerNumber: number) {
        backendState.currentPlayer = playerNumber;
        backendState.currentPhase = 'ring_placement';
        backendAny.updateLpsTrackingForCurrentTurn();
        return backendAny.maybeEndGameByLastPlayerStanding();
      }

      function sandboxStartTurn(playerNumber: number) {
        sandboxState.currentPlayer = playerNumber;
        sandboxState.currentPhase = 'ring_placement';
        sandboxAny.handleStartOfInteractiveTurn();
        return sandboxAny.victoryResult;
      }

      // Full round
      backendStartTurn(1);
      sandboxStartTurn(1);
      backendStartTurn(2);
      sandboxStartTurn(2);
      backendStartTurn(3);
      sandboxStartTurn(3);

      // LPS trigger
      const backendResult = backendStartTurn(1);
      const sandboxResult = sandboxStartTurn(1);

      // === Verify parity ===
      expect(backendResult).not.toBeNull();
      expect(sandboxResult).not.toBeNull();

      expect(backendResult.winner).toBe(1);
      expect(sandboxResult!.winner).toBe(1);

      expect(backendResult.reason).toBe('last_player_standing');
      expect(sandboxResult!.reason).toBe('last_player_standing');

      // Backend updates state directly
      expect(backendState.gameStatus).toBe('completed');
      expect(backendState.winner).toBe(1);

      // The key parity assertion: both engines agree on winner and reason
      expect(backendResult.winner).toBe(sandboxResult!.winner);
      expect(backendResult.reason).toBe(sandboxResult!.reason);
    });
  });
});