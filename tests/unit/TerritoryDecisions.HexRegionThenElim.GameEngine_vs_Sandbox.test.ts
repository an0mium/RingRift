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
import { createHexTerritoryRegionScenario } from '../helpers/hexTerritoryScenario';
import { snapshotFromGameState, snapshotsEqual, diffSnapshots } from '../utils/stateSnapshots';

// Skip by default when the orchestrator adapter is enabled, unless an explicit
// override is provided. This keeps these suites as non-blocking diagnostics for
// orchestrator parity work while still allowing targeted runs under
// TERRITORY_PARITY_ALLOW_ORCHESTRATOR=true.
const skipWithOrchestrator =
  process.env.ORCHESTRATOR_ADAPTER_ENABLED === 'true' &&
  process.env.TERRITORY_PARITY_ALLOW_ORCHESTRATOR !== 'true';

(skipWithOrchestrator ? describe.skip : describe)(
  'Hex territory region + elimination parity (GameEngine vs ClientSandboxEngine)',
  () => {
    function clonePlayers(players: Player[]): Player[] {
      return players.map((p) => ({ ...p }));
    }

    test('backend and sandbox stay in parity through hex region processing and self-elimination', async () => {
      const { initialState, region, outsideStackPosition } = createHexTerritoryRegionScenario(
        'hex-territory-region-then-elim'
      );

      const basePlayers = initialState.players;
      const boardType = initialState.board.type;
      const movingPlayer = initialState.currentPlayer;

      // --- Backend setup ---
      const backendPlayers = clonePlayers(basePlayers);
      const backendTimeControl = initialState.timeControl;

      const backendEngine = new GameEngine(
        'hex-territory-region-then-elim-backend',
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
        'backend-hex-initial',
        backendEngine.getGameState()
      );
      const sandboxInitialSnap = snapshotFromGameState(
        'sandbox-hex-initial',
        sandboxEngine.getGameState()
      );
      expect(snapshotsEqual(backendInitialSnap, sandboxInitialSnap)).toBe(true);

      // --- Step 1: enumerate hex region decisions using shared helper + override region ---
      const overrideRegions: Territory[] = [region];

      const backendRegionMoves: Move[] = enumerateProcessTerritoryRegionMoves(
        backendEngine.getGameState(),
        movingPlayer,
        { testOverrideRegions: overrideRegions }
      );
      const sandboxRegionMoves: Move[] = enumerateProcessTerritoryRegionMoves(
        sandboxEngine.getGameState(),
        movingPlayer,
        { testOverrideRegions: overrideRegions }
      );

      expect(backendRegionMoves.length).toBe(1);
      expect(sandboxRegionMoves.length).toBe(1);

      const backendRegionMove = backendRegionMoves[0];
      const sandboxRegionMove = sandboxRegionMoves[0];

      const regionKey = (spaces: Position[]): string =>
        spaces
          .map((p) => positionToString(p))
          .sort()
          .join('|');

      const backendRegionSpaces =
        (backendRegionMove.disconnectedRegions &&
          backendRegionMove.disconnectedRegions[0]?.spaces) ||
        [];
      const sandboxRegionSpaces =
        (sandboxRegionMove.disconnectedRegions &&
          sandboxRegionMove.disconnectedRegions[0]?.spaces) ||
        [];

      expect(regionKey(backendRegionSpaces)).toBe(regionKey(region.spaces));
      expect(regionKey(sandboxRegionSpaces)).toBe(regionKey(region.spaces));

      // --- Step 2: apply region decision in both hosts ---
      {
        const backendStateBefore = backendEngine.getGameState();
        const outcome = applyProcessTerritoryRegionDecision(backendStateBefore, backendRegionMove);
        backendAny.gameState = outcome.nextState;

        // Mirror GameEngine's flag lifecycle for pending self-elimination.
        backendAny.pendingTerritorySelfElimination = outcome.pendingSelfElimination;
      }

      await sandboxEngine.applyCanonicalMove(sandboxRegionMove as Move);

      const backendAfterRegion = backendEngine.getGameState();
      const sandboxAfterRegion = sandboxEngine.getGameState();

      const backendAfterRegionSnap = snapshotFromGameState(
        'backend-hex-after-region',
        backendAfterRegion
      );
      const sandboxAfterRegionSnap = snapshotFromGameState(
        'sandbox-hex-after-region',
        sandboxAfterRegion
      );

      if (!snapshotsEqual(backendAfterRegionSnap, sandboxAfterRegionSnap)) {
        console.error(
          '[HexTerritoryParity] mismatch after choose_territory_option',
          diffSnapshots(backendAfterRegionSnap, sandboxAfterRegionSnap)
        );
      }

      expect(snapshotsEqual(backendAfterRegionSnap, sandboxAfterRegionSnap)).toBe(true);

      // --- Step 3: enumerate and apply eliminate_rings_from_stack decisions ---
      const backendMovesAfterRegion = backendEngine.getValidMoves(movingPlayer);
      const backendElimMoves = backendMovesAfterRegion.filter(
        (m) => m.type === 'eliminate_rings_from_stack'
      );
      expect(backendElimMoves.length).toBeGreaterThan(0);

      const sandboxElimMoves: Move[] =
        sandboxAny.getValidEliminationDecisionMovesForCurrentPlayer() ?? [];
      expect(sandboxElimMoves.length).toBeGreaterThan(0);

      const keyFromPos = (pos: Position | undefined): string => (pos ? positionToString(pos) : '');

      const backendTargets = backendElimMoves
        .map((m) => keyFromPos(m.to))
        .filter((k) => k.length > 0)
        .sort();
      const sandboxTargets = sandboxElimMoves
        .map((m) => keyFromPos(m.to))
        .filter((k) => k.length > 0)
        .sort();

      expect(backendTargets).toEqual(sandboxTargets);
      expect(backendTargets).toContain(positionToString(outsideStackPosition));

      const chosenKey = positionToString(outsideStackPosition);

      const backendElimMove = backendElimMoves.find(
        (m) => m.to && positionToString(m.to) === chosenKey
      );
      const sandboxElimMove = sandboxElimMoves.find(
        (m) => m.to && positionToString(m.to) === chosenKey
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

      const backendFinalSnap = snapshotFromGameState('backend-hex-final', backendFinal);
      const sandboxFinalSnap = snapshotFromGameState('sandbox-hex-final', sandboxFinal);

      if (!snapshotsEqual(backendFinalSnap, sandboxFinalSnap)) {
        console.error(
          '[HexTerritoryParity] mismatch after eliminate_rings_from_stack',
          diffSnapshots(backendFinalSnap, sandboxFinalSnap)
        );
      }

      expect(snapshotsEqual(backendFinalSnap, sandboxFinalSnap)).toBe(true);
    });
  }
);
