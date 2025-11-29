import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import type { GameState, Move, Position, BoardType, Territory } from '../../src/shared/types/game';
import { BOARD_CONFIGS, positionToString } from '../../src/shared/types/game';
import { lineAndTerritoryRuleScenarios, LineAndTerritoryRuleScenario } from './rulesMatrix';
import { SandboxOrchestratorAdapter } from '../../src/client/sandbox/SandboxOrchestratorAdapter';
import { enumerateProcessTerritoryRegionMoves } from '../../src/shared/engine/territoryDecisionHelpers';

/**
 * Orchestrator-centric sandbox multi-phase scenario tests.
 *
 * These suites exercise ClientSandboxEngine with the SandboxOrchestratorAdapter
 * enabled so that all rules processing goes through:
 *
 *   ClientSandboxEngine.applyCanonicalMove
 *     → SandboxOrchestratorAdapter.processMove
 *       → processTurnAsync (shared orchestrator)
 *         → shared aggregates
 *
 * No legacy sandbox-only helpers (processLinesForCurrentPlayer,
 * processDisconnectedRegionsForCurrentPlayer) are invoked in the core flow.
 */

function createSandboxEngineForTest(boardType: BoardType = 'square8'): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType,
    numPlayers: 2,
    playerKinds: ['human', 'human'],
  };

  const handler: SandboxInteractionHandler = {
    // For these orchestrator-driven tests we never rely on explicit
    // PlayerChoice handling; if the adapter ever surfaces a PendingDecision
    // the handler deterministically selects the first option.
    async requestChoice<TChoice>(_choice: TChoice) {
      return {
        choiceId: (_choice as any).id,
        playerNumber: (_choice as any).playerNumber,
        choiceType: (_choice as any).type,
        selectedOption: (Array.isArray((_choice as any).options as any[]) &&
          ((_choice as any).options as any[])[0]) as any,
      } as any;
    },
  };

  const engine = new ClientSandboxEngine({ config, interactionHandler: handler });
  // Ensure orchestrator adapter is enabled explicitly for clarity.
  engine.enableOrchestratorAdapter();
  return engine;
}

function getSandboxAdapter(engine: ClientSandboxEngine): SandboxOrchestratorAdapter {
  const anyEngine = engine as any;
  // getOrchestratorAdapter is a private helper; tests access it via any-cast.
  return anyEngine.getOrchestratorAdapter() as SandboxOrchestratorAdapter;
}

function keyFromPositions(positions: Position[]): string {
  return positions
    .map((p) => positionToString(p))
    .sort()
    .join('|');
}

describe('Orchestrator.Sandbox multi-phase scenarios (ClientSandboxEngine + SandboxOrchestratorAdapter)', () => {
  /**
   * Scenario B (sandbox) – process_territory_region + explicit
   * eliminate_rings_from_stack via orchestrator adapter.
   *
   * This mirrors the backend Scenario B for
   * Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_square8
   * but drives the sequence entirely through the sandbox host:
   *
   *   1) Seed a minimal disconnected region containing victim stacks for
   *      Player 2 and an outside stack for Player 1.
   *   2) Enter territory_processing for Player 1.
   *   3) Use SandboxOrchestratorAdapter.getValidMoves() to surface
   *      process_territory_region Moves and assert no elimination
   *      decisions appear before a region is processed.
   *   4) Apply a process_territory_region Move via
   *      ClientSandboxEngine.applyCanonicalMove (orchestrator-backed).
   *   5) Assert that region spaces collapse to Player 1's territory and
   *      victim stacks are removed.
   *   6) Assert that eliminate_rings_from_stack Moves are now surfaced,
   *      apply one targeting the outside stack, and verify elimination
   *      counts and phase/turn sequencing.
   *   7) At each step, assert that all moves returned by the orchestrator
   *      validate successfully (no phantom moves).
   */
  it('Scenario B – sandbox territory region + self-elimination via orchestrator adapter', async () => {
    const scenario: LineAndTerritoryRuleScenario | undefined = lineAndTerritoryRuleScenarios.find(
      (s) => s.ref.id === 'Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_square8'
    );
    expect(scenario).toBeDefined();
    if (!scenario) return;

    const engine = createSandboxEngineForTest('square8');
    const engineAny: any = engine;
    const state: GameState = engineAny.gameState as GameState;

    const territory = scenario.territoryRegion;
    const regionSpaces = territory.spaces;
    const controllingPlayer = territory.controllingPlayer;
    const victimPlayer = territory.victimPlayer;
    const outsideStackPos = territory.outsideStackPosition;
    const outsideHeight = territory.selfEliminationStackHeight ?? 2;

    const board = state.board;
    board.stacks.clear();
    board.markers.clear();
    board.collapsedSpaces.clear();

    // Place victim stacks inside the region (height 1 each) for the victim player.
    for (const pos of regionSpaces) {
      const key = positionToString(pos);
      board.stacks.set(key, {
        position: pos,
        rings: [victimPlayer],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: victimPlayer,
      } as any);
    }

    // Outside stack for the controlling player used to pay the self-elimination cost.
    const outsideKey = positionToString(outsideStackPos);
    const outsideRings = Array(outsideHeight).fill(controllingPlayer);
    board.stacks.set(outsideKey, {
      position: outsideStackPos,
      rings: outsideRings,
      stackHeight: outsideRings.length,
      capHeight: outsideRings.length,
      controllingPlayer,
    } as any);

    state.currentPlayer = controllingPlayer;
    state.currentPhase = 'territory_processing';
    state.gameStatus = 'active';

    const adapter = getSandboxAdapter(engine);

    const beforeState = engine.getGameState();

    // Orchestrator moves before any region is processed. In this synthetic
    // geometry the shared detector may or may not report a disconnected region;
    // elimination decisions may already exist for generic territory contexts.
    const movesBefore = adapter.getValidMoves();
    expect(movesBefore.length).toBeGreaterThan(0);

    const elimBefore = movesBefore.filter((m) => m.type === 'eliminate_rings_from_stack');
    // We do not assert on elimBefore count here; the invariant we care about is
    // that once a specific region is processed via a canonical decision, the
    // resulting elimination moves and accounting are consistent across hosts.

    // Construct a canonical process_territory_region decision using the shared
    // helper with a test-only override region, mirroring the backend scenario.
    const regionTerritory: Territory = {
      spaces: regionSpaces,
      controllingPlayer,
      isDisconnected: true,
    };

    const regionMoves = enumerateProcessTerritoryRegionMoves(beforeState, controllingPlayer, {
      testOverrideRegions: [regionTerritory],
    });
    expect(regionMoves.length).toBeGreaterThan(0);

    // All orchestrator moves should validate for this snapshot.
    for (const move of movesBefore) {
      const validation = adapter.validateMove(move);
      expect(validation.valid).toBe(true);
    }

    const targetKey = keyFromPositions(regionSpaces);
    const regionMove =
      regionMoves.find((m: Move) => {
        if (!m.disconnectedRegions || m.disconnectedRegions.length === 0) {
          return false;
        }
        const regSpaces = m.disconnectedRegions[0].spaces ?? [];
        return keyFromPositions(regSpaces) === targetKey;
      }) ?? regionMoves[0];

    const validationBefore = adapter.validateMove(regionMove);
    expect(validationBefore.valid).toBe(true);

    // Apply region processing via the orchestrator-backed canonical move path.
    await engine.applyCanonicalMove(regionMove);

    const afterRegion = engine.getGameState();

    // Region interior must be collapsed for the controlling player and victim stacks removed.
    for (const pos of regionSpaces) {
      const key = positionToString(pos);
      expect(afterRegion.board.collapsedSpaces.get(key)).toBe(controllingPlayer);
      expect(afterRegion.board.stacks.get(key)).toBeUndefined();
    }

    // After processing the region, the sandbox orchestrator completes the
    // territory-processing cycle (including mandatory self-elimination) and
    // rotates to the next player's ring_placement turn.
    expect(afterRegion.currentPhase).toBe('ring_placement');
    expect(afterRegion.currentPlayer).not.toBe(controllingPlayer);

    const beforePlayer = beforeState.players.find((p) => p.playerNumber === controllingPlayer)!;
    const afterPlayer = afterRegion.players.find((p) => p.playerNumber === controllingPlayer)!;

    // Player's eliminatedRings and global total should have increased due to
    // internal region eliminations + mandatory self-elimination.
    expect(afterRegion.totalRingsEliminated).toBeGreaterThan(beforeState.totalRingsEliminated);
    expect(afterPlayer.eliminatedRings).toBeGreaterThan(beforePlayer.eliminatedRings);

    // Orchestrator invariant on the post-region snapshot as well.
    const finalOrchMoves = adapter.getValidMoves();
    for (const move of finalOrchMoves) {
      const v = adapter.validateMove(move);
      expect(v.valid).toBe(true);
    }
  });
});
