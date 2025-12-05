import {
  ClientSandboxEngine,
  type SandboxConfig,
  type SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type {
  GameState,
  Move,
  PlayerChoice,
  PlayerChoiceResponseFor,
} from '../../src/shared/types/game';
import { createSquareTwoRegionTerritoryScenario } from '../helpers/squareTerritoryScenario';
import * as TerritoryAggregate from '../../src/shared/engine/aggregates/TerritoryAggregate';

describe('ClientSandboxEngine – territory processing skip flow (orchestrator)', () => {
  /**
   * Minimal interaction handler that always selects the first option.
   * For this test we drive canonical territory moves directly via
   * getValidMoves + adapter.processMove, so choices are not expected
   * to surface, but the handler keeps the engine wiring complete.
   */
  const interactionHandler: SandboxInteractionHandler = {
    async requestChoice<TChoice extends PlayerChoice>(
      choice: TChoice
    ): Promise<PlayerChoiceResponseFor<TChoice>> {
      const anyChoice = choice as any;
      const options = (anyChoice.options as any[]) ?? [];
      const selectedOption = options.length > 0 ? options[0] : undefined;
      return {
        choiceId: anyChoice.id,
        playerNumber: anyChoice.playerNumber,
        choiceType: anyChoice.type,
        selectedOption,
      } as PlayerChoiceResponseFor<TChoice>;
    },
  };

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('surfaces region + skip territory moves and applying skip advances phase without changing board', async () => {
    const { initialState, regionA, regionB } = createSquareTwoRegionTerritoryScenario(
      'sandbox-orchestrator-territory-skip-flow'
    );

    const state: GameState = initialState as GameState;
    expect(state.currentPhase).toBe('territory_processing');
    const currentPlayer = state.currentPlayer;

    const config: SandboxConfig = {
      boardType: state.board.type,
      numPlayers: state.players.length,
      playerKinds: state.players.map(() => 'human'),
    };

    // Prepare synthetic territory decision moves for the shared orchestrator.
    // We stub the aggregate enumerators so that getValidMoves(...) surfaces
    // one or more process_territory_region moves for the current player and
    // no explicit elimination options, ensuring skip_territory_processing is
    // also present in the canonical Move surface.
    const baseMoveNumber = state.moveHistory.length + 1;
    const regionMoveA: Move = {
      id: 'process-region-a',
      type: 'process_territory_region',
      player: currentPlayer,
      to: regionA.spaces[0],
      disconnectedRegions: [regionA],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: baseMoveNumber,
    } as Move;
    const regionMoveB: Move = {
      id: 'process-region-b',
      type: 'process_territory_region',
      player: currentPlayer,
      to: regionB.spaces[0],
      disconnectedRegions: [regionB],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: baseMoveNumber,
    } as Move;

    jest
      .spyOn(TerritoryAggregate, 'enumerateProcessTerritoryRegionMoves')
      .mockImplementation(() => [regionMoveA, regionMoveB]);

    jest
      .spyOn(TerritoryAggregate, 'enumerateTerritoryEliminationMoves')
      .mockImplementation(() => []);

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler,
      traceMode: false,
    });

    // Seed the sandbox engine with the prepared territory-processing state.
    const engineAny: any = engine;
    engineAny.gameState = state;
    engineAny.orchestratorAdapter = null;

    const before = engine.getGameState();
    const beforeP1 = before.players.find((p) => p.playerNumber === currentPlayer)!;
    const beforeTerritory = beforeP1.territorySpaces;

    // 1) From the orchestrator-backed sandbox surface, enumerate valid moves
    // for the current player in territory_processing. We expect at least one
    // process_territory_region move and a skip_territory_processing move.
    const initialMoves: Move[] = engine.getValidMoves(currentPlayer);
    const initialRegionMoves = initialMoves.filter((m) => m.type === 'process_territory_region');
    const initialSkipMoves = initialMoves.filter((m) => m.type === 'skip_territory_processing');

    expect(initialRegionMoves.length).toBeGreaterThan(0);
    expect(initialSkipMoves.length).toBeGreaterThan(0);

    // Snapshot the board before applying skip so we can verify that
    // skip_territory_processing leaves geometry unchanged.
    const stacksBeforeSkip = new Map(before.board.stacks);
    const collapsedBeforeSkip = new Map(before.board.collapsedSpaces);
    const eliminatedBeforeSkip = { ...before.board.eliminatedRings };

    // 2) Apply the skip_territory_processing move via the sandbox
    // orchestrator adapter. This should advance phase/turn but not change
    // the board geometry.
    const skipMove = initialSkipMoves[0];
    const adapter = engineAny.getOrchestratorAdapter() as {
      processMove: (move: Move) => Promise<{ success: boolean }>;
    };
    const result = await adapter.processMove(skipMove as Move);
    expect(result.success).toBe(true);

    const afterSkip = engine.getGameState();

    // Board geometry and aggregate elimination totals should be unchanged
    // by skip_territory_processing.
    expect(afterSkip.board.stacks.size).toBe(stacksBeforeSkip.size);
    expect(afterSkip.board.collapsedSpaces.size).toBe(collapsedBeforeSkip.size);
    expect(afterSkip.board.eliminatedRings).toEqual(eliminatedBeforeSkip);
    expect(afterSkip.players.find((p) => p.playerNumber === currentPlayer)?.territorySpaces).toBe(
      beforeTerritory
    );

    // Phase should have advanced out of territory_processing and the turn
    // should no longer be on the same player.
    expect(afterSkip.currentPhase).not.toBe('territory_processing');
    expect(afterSkip.currentPlayer).not.toBe(before.currentPlayer);
  });

  it('supports skipping further territory processing from a post-region state without changing the post-region board', async () => {
    const { initialState, regionA, regionB } = createSquareTwoRegionTerritoryScenario(
      'sandbox-orchestrator-territory-skip-after-region'
    );

    const baseState: GameState = initialState as GameState;
    expect(baseState.currentPhase).toBe('territory_processing');
    const currentPlayer = baseState.currentPlayer;

    const config: SandboxConfig = {
      boardType: baseState.board.type,
      numPlayers: baseState.players.length,
      playerKinds: baseState.players.map(() => 'human'),
    };

    const baseMoveNumber = baseState.moveHistory.length + 1;
    const regionMoveA: Move = {
      id: 'process-region-a',
      type: 'process_territory_region',
      player: currentPlayer,
      to: regionA.spaces[0],
      disconnectedRegions: [regionA],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: baseMoveNumber,
    } as Move;
    const regionMoveB: Move = {
      id: 'process-region-b',
      type: 'process_territory_region',
      player: currentPlayer,
      to: regionB.spaces[0],
      disconnectedRegions: [regionB],
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: baseMoveNumber + 1,
    } as Move;

    // 1) Apply one region-processing move using the shared TerritoryAggregate
    // helper on a fresh copy of the fixture state to obtain a realistic
    // post-region board.
    const helperState: GameState = baseState as GameState;
    const helperOutcome = TerritoryAggregate.applyProcessTerritoryRegionDecision(
      helperState,
      regionMoveA
    );
    const postRegionState: GameState = {
      ...helperOutcome.nextState,
      currentPhase: 'territory_processing',
    };

    for (const pos of regionA.spaces) {
      const key = `${pos.x},${pos.y}`;
      expect(postRegionState.board.collapsedSpaces.get(key)).toBe(currentPlayer);
      expect(postRegionState.board.stacks.has(key)).toBe(false);
    }

    const beforeTerritorySpaces =
      baseState.players.find((p) => p.playerNumber === currentPlayer)?.territorySpaces ?? 0;
    const afterRegionTerritorySpaces =
      postRegionState.players.find((p) => p.playerNumber === currentPlayer)?.territorySpaces ?? 0;
    expect(afterRegionTerritorySpaces).toBeGreaterThanOrEqual(beforeTerritorySpaces);

    // 2) Seed a sandbox engine with this post-region state and stub
    // territory enumerators so that getValidMoves surfaces both another
    // process_territory_region and a skip_territory_processing move.
    jest
      .spyOn(TerritoryAggregate, 'enumerateProcessTerritoryRegionMoves')
      .mockImplementation(() => [regionMoveB]);

    jest
      .spyOn(TerritoryAggregate, 'enumerateTerritoryEliminationMoves')
      .mockImplementation(() => []);

    const engine = new ClientSandboxEngine({
      config,
      interactionHandler,
      traceMode: false,
    });

    const engineAny: any = engine;
    engineAny.gameState = postRegionState;
    engineAny.orchestratorAdapter = null;

    const before = engine.getGameState();
    const stacksBefore = new Map(before.board.stacks);
    const collapsedBefore = new Map(before.board.collapsedSpaces);
    const eliminatedBefore = { ...before.board.eliminatedRings };

    const movesAfterRegion: Move[] = engine.getValidMoves(currentPlayer);
    const regionMovesAfter = movesAfterRegion.filter((m) => m.type === 'process_territory_region');
    const skipMovesAfter = movesAfterRegion.filter((m) => m.type === 'skip_territory_processing');

    expect(regionMovesAfter.length).toBeGreaterThan(0);
    expect(skipMovesAfter.length).toBeGreaterThan(0);

    const adapter = engineAny.getOrchestratorAdapter() as {
      processMove: (move: Move) => Promise<{ success: boolean }>;
    };

    // 3) Apply skip_territory_processing and confirm the board is unchanged
    // relative to the post-region state while phase/turn advance.
    const skipMove = skipMovesAfter[0];
    const skipResult = await adapter.processMove(skipMove);
    expect(skipResult.success).toBe(true);

    const afterSkip = engine.getGameState();

    expect(afterSkip.board.stacks.size).toBe(stacksBefore.size);
    expect(afterSkip.board.collapsedSpaces.size).toBe(collapsedBefore.size);
    expect(afterSkip.board.eliminatedRings).toEqual(eliminatedBefore);
    expect(afterSkip.currentPhase).not.toBe('territory_processing');
    expect(afterSkip.currentPlayer).not.toBe(currentPlayer);
  });
});
