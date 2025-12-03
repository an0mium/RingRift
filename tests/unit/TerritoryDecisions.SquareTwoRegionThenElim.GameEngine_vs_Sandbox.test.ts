import { GameEngine } from '../../src/server/game/GameEngine';
import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  type GameState,
  type Move,
  type Player,
  type Territory,
  type Position,
  positionToString,
} from '../../src/shared/types/game';
import {
  enumerateProcessTerritoryRegionMoves,
  applyProcessTerritoryRegionDecision,
} from '../../src/shared/engine/territoryDecisionHelpers';
import { createSquareTwoRegionTerritoryScenario } from '../helpers/squareTerritoryScenario';
import { snapshotFromGameState, snapshotsEqual, diffSnapshots } from '../utils/stateSnapshots';

// Skip by default when the orchestrator adapter is enabled, unless an explicit
// override is provided. This keeps these suites as non-blocking diagnostics for
// orchestrator parity work while still allowing targeted runs under
// TERRITORY_PARITY_ALLOW_ORCHESTRATOR=true.
const skipWithOrchestrator =
  process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true' &&
  process.env.TERRITORY_PARITY_ALLOW_ORCHESTRATOR !== 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'Square8 two-region territory + elimination parity (GameEngine vs ClientSandboxEngine)',
  () => {
    function clonePlayers(players: Player[]): Player[] {
      return players.map((p) => ({ ...p }));
    }

    function keyFromSpaces(spaces: Position[]): string {
      return spaces
        .map((p) => positionToString(p))
        .sort()
        .join('|');
    }

    function findMoveForRegion(moves: Move[], region: Territory): Move {
      const targetKey = keyFromSpaces(region.spaces);
      const match = moves.find((m) => {
        const spaces = (m.disconnectedRegions && m.disconnectedRegions[0]?.spaces) || [];
        const key = keyFromSpaces(spaces);
        return key === targetKey;
      });
      if (!match) {
        throw new Error(`No move found for region with key ${targetKey}`);
      }
      return match;
    }

    test('backend and sandbox stay in parity through two-region processing (B then A) and self-elimination', async () => {
      const { initialState, regionA, regionB, outsideStackPositions } =
        createSquareTwoRegionTerritoryScenario('square-two-region-territory-region-then-elim');

      const basePlayers = initialState.players;
      const boardType = initialState.board.type;
      const movingPlayer = initialState.currentPlayer;

      // --- Backend setup ---
      const backendPlayers = clonePlayers(basePlayers);
      const backendTimeControl = initialState.timeControl;

      const backendEngine = new GameEngine(
        'square-two-region-territory-region-then-elim-backend',
        boardType,
        backendPlayers,
        backendTimeControl,
        true
      );
      backendEngine.enableMoveDrivenDecisionPhases();

      const backendAny: any = backendEngine;
      const backendState0: GameState = backendEngine.getGameState();

      backendAny.gameState = {
        ...backendState0,
        board: {
          ...initialState.board,
          stacks: new Map(initialState.board.stacks),
          markers: new Map(initialState.board.markers),
          collapsedSpaces: new Map(initialState.board.collapsedSpaces),
          territories: new Map(initialState.board.territories),
          formedLines: [...initialState.board.formedLines],
          eliminatedRings: { ...initialState.board.eliminatedRings },
        },
        players: backendPlayers,
        currentPlayer: movingPlayer,
        currentPhase: 'territory_processing',
        gameStatus: 'active',
        history: [],
        moveHistory: [],
        totalRingsEliminated: initialState.totalRingsEliminated,
      } as GameState;

      // --- Sandbox setup ---
      const sandboxConfig: SandboxConfig = {
        boardType,
        numPlayers: basePlayers.length,
        playerKinds: basePlayers.map((p) => p.type as 'human' | 'ai'),
      };

      const sandboxHandler: SandboxInteractionHandler = {
        async requestChoice(choice: any) {
          const options = ((choice as any).options as any[]) ?? [];
          const selectedOption = options.length > 0 ? options[0] : undefined;
          return {
            choiceId: (choice as any).id,
            playerNumber: (choice as any).playerNumber,
            choiceType: (choice as any).type,
            selectedOption,
          } as any;
        },
      };

      const sandboxEngine = new ClientSandboxEngine({
        config: sandboxConfig,
        interactionHandler: sandboxHandler,
        traceMode: true,
      });
      const sandboxAny: any = sandboxEngine;
      const sandboxState0: GameState = sandboxEngine.getGameState();
      const sandboxPlayers = clonePlayers(basePlayers);

      sandboxAny.gameState = {
        ...sandboxState0,
        board: {
          ...initialState.board,
          stacks: new Map(initialState.board.stacks),
          markers: new Map(initialState.board.markers),
          collapsedSpaces: new Map(initialState.board.collapsedSpaces),
          territories: new Map(initialState.board.territories),
          formedLines: [...initialState.board.formedLines],
          eliminatedRings: { ...initialState.board.eliminatedRings },
        },
        players: sandboxPlayers,
        currentPlayer: movingPlayer,
        currentPhase: 'territory_processing',
        gameStatus: 'active',
        history: [],
        moveHistory: [],
        totalRingsEliminated: initialState.totalRingsEliminated,
      } as GameState;

      // Sanity: initial snapshots identical.
      const backendInitialSnap = snapshotFromGameState(
        'backend-initial',
        backendEngine.getGameState()
      );
      const sandboxInitialSnap = snapshotFromGameState(
        'sandbox-initial',
        sandboxEngine.getGameState()
      );
      expect(snapshotsEqual(backendInitialSnap, sandboxInitialSnap)).toBe(true);

      const regions: Territory[] = [regionA, regionB];

      // --- Step 1: enumerate and apply Region B first in both engines ---
      const backendRegionMovesInitial: Move[] = enumerateProcessTerritoryRegionMoves(
        backendEngine.getGameState(),
        movingPlayer,
        { testOverrideRegions: regions }
      );

      const sandboxRegionMovesInitial: Move[] = enumerateProcessTerritoryRegionMoves(
        sandboxEngine.getGameState(),
        movingPlayer,
        { testOverrideRegions: regions }
      );

      expect(backendRegionMovesInitial.length).toBe(2);
      expect(sandboxRegionMovesInitial.length).toBe(2);

      const backendRegionBMove = findMoveForRegion(backendRegionMovesInitial, regionB);
      const sandboxRegionBMove = findMoveForRegion(sandboxRegionMovesInitial, regionB);

      {
        const backendStateBefore = backendEngine.getGameState();
        const outcome = applyProcessTerritoryRegionDecision(
          backendStateBefore,
          backendRegionBMove as Move
        );
        const backendAnyInternal: any = backendEngine;
        backendAnyInternal.gameState = outcome.nextState;
        backendAnyInternal.pendingTerritorySelfElimination = true;
      }

      await sandboxEngine.applyCanonicalMove(sandboxRegionBMove as Move);

      const backendAfterRegionB = backendEngine.getGameState();
      const sandboxAfterRegionB = sandboxEngine.getGameState();

      const backendSnapAfterRegionB = snapshotFromGameState(
        'backend-after-region-B',
        backendAfterRegionB
      );
      const sandboxSnapAfterRegionB = snapshotFromGameState(
        'sandbox-after-region-B',
        sandboxAfterRegionB
      );

      if (!snapshotsEqual(backendSnapAfterRegionB, sandboxSnapAfterRegionB)) {
        console.error('[TerritoryDecisions.SquareTwoRegionThenElim] mismatch after Region B', {
          diff: diffSnapshots(backendSnapAfterRegionB, sandboxSnapAfterRegionB),
        });
      }

      expect(snapshotsEqual(backendSnapAfterRegionB, sandboxSnapAfterRegionB)).toBe(true);

      // --- Step 2: enumerate remaining region decisions and apply Region A ---
      const backendRegionMovesAfterB: Move[] = enumerateProcessTerritoryRegionMoves(
        backendEngine.getGameState(),
        movingPlayer,
        { testOverrideRegions: regions }
      );
      const sandboxRegionMovesAfterB: Move[] = enumerateProcessTerritoryRegionMoves(
        sandboxEngine.getGameState(),
        movingPlayer,
        { testOverrideRegions: regions }
      );

      expect(backendRegionMovesAfterB.length).toBeGreaterThanOrEqual(1);
      expect(sandboxRegionMovesAfterB.length).toBeGreaterThanOrEqual(1);

      const backendRegionAMove = findMoveForRegion(backendRegionMovesAfterB, regionA);
      const sandboxRegionAMove = findMoveForRegion(sandboxRegionMovesAfterB, regionA);

      {
        const backendStateBefore = backendEngine.getGameState();
        const outcome = applyProcessTerritoryRegionDecision(
          backendStateBefore,
          backendRegionAMove as Move
        );
        const backendAnyInternal: any = backendEngine;
        backendAnyInternal.gameState = outcome.nextState;
        backendAnyInternal.pendingTerritorySelfElimination = true;
      }

      await sandboxEngine.applyCanonicalMove(sandboxRegionAMove as Move);

      const backendAfterRegionA = backendEngine.getGameState();
      const sandboxAfterRegionA = sandboxEngine.getGameState();

      const backendSnapAfterRegionA = snapshotFromGameState(
        'backend-after-region-A',
        backendAfterRegionA
      );
      const sandboxSnapAfterRegionA = snapshotFromGameState(
        'sandbox-after-region-A',
        sandboxAfterRegionA
      );

      if (!snapshotsEqual(backendSnapAfterRegionA, sandboxSnapAfterRegionA)) {
        console.error('[TerritoryDecisions.SquareTwoRegionThenElim] mismatch after Region A', {
          diff: diffSnapshots(backendSnapAfterRegionA, sandboxSnapAfterRegionA),
        });
      }

      expect(snapshotsEqual(backendSnapAfterRegionA, sandboxSnapAfterRegionA)).toBe(true);

      // --- Step 3: enumerate elimination decisions and apply a matching target ---
      const backendMovesAfterRegions = backendEngine.getValidMoves(movingPlayer);
      const backendElimMoves = backendMovesAfterRegions.filter(
        (m) => m.type === 'eliminate_rings_from_stack'
      );
      expect(backendElimMoves.length).toBeGreaterThan(0);

      const sandboxElimMoves: Move[] =
        sandboxAny.getValidEliminationDecisionMovesForCurrentPlayer() ?? [];
      expect(sandboxElimMoves.length).toBeGreaterThan(0);

      const backendElimTargets = backendElimMoves
        .map((m) => (m.to ? positionToString(m.to) : ''))
        .filter((k) => k.length > 0)
        .sort();
      const sandboxElimTargets = sandboxElimMoves
        .map((m) => (m.to ? positionToString(m.to) : ''))
        .filter((k) => k.length > 0)
        .sort();

      expect(backendElimTargets).toEqual(sandboxElimTargets);
      expect(backendElimTargets.length).toBeGreaterThan(0);

      const chosenElimKey = backendElimTargets[0];

      const backendElimMove = backendElimMoves.find(
        (m) => m.to && positionToString(m.to) === chosenElimKey
      );
      const sandboxElimMove = sandboxElimMoves.find(
        (m) => m.to && positionToString(m.to) === chosenElimKey
      );

      expect(backendElimMove).toBeDefined();
      expect(sandboxElimMove).toBeDefined();

      {
        const { id, timestamp, moveNumber, ...payload } = backendElimMove as any;
        const result = await backendEngine.makeMove(
          payload as Omit<Move, 'id' | 'timestamp' | 'moveNumber'>
        );
        expect(result.success).toBe(true);
      }

      await sandboxEngine.applyCanonicalMove(sandboxElimMove as Move);

      const backendFinal = backendEngine.getGameState();
      const sandboxFinal = sandboxEngine.getGameState();

      const backendAnyFinal: any = backendEngine;
      const sandboxAnyFinal: any = sandboxEngine;

      expect(backendAnyFinal.pendingTerritorySelfElimination).toBe(false);
      expect(sandboxAnyFinal._pendingTerritorySelfElimination).toBe(false);

      expect(backendFinal.currentPhase).not.toBe('territory_processing');
      expect(sandboxFinal.currentPhase).not.toBe('territory_processing');
      expect(backendFinal.currentPlayer).toBe(sandboxFinal.currentPlayer);

      const backendFinalSnap = snapshotFromGameState('backend-final', backendFinal);
      const sandboxFinalSnap = snapshotFromGameState('sandbox-final', sandboxFinal);

      if (!snapshotsEqual(backendFinalSnap, sandboxFinalSnap)) {
        console.error('[TerritoryDecisions.SquareTwoRegionThenElim] final snapshot mismatch', {
          diff: diffSnapshots(backendFinalSnap, sandboxFinalSnap),
        });
      }

      expect(snapshotsEqual(backendFinalSnap, sandboxFinalSnap)).toBe(true);
    });
  }
);
