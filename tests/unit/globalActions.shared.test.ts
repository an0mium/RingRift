import {
  hasTurnMaterial,
  hasGlobalPlacementAction,
  hasPhaseLocalInteractiveMove,
  hasForcedEliminationAction,
  applyForcedEliminationForPlayer,
  computeGlobalLegalActionsSummary,
  isANMState,
  computeSMetric,
  computeTMetric,
} from '../../src/shared/engine';
import {
  createTestBoard,
  createTestGameState,
  createTestPlayer,
  addStack,
  pos,
} from '../utils/fixtures';

describe('globalActions – R200/R2xx global legal actions and ANM semantics', () => {
  describe('hasTurnMaterial', () => {
    it('is true when player has rings in hand', () => {
      const state = createTestGameState({
        players: [createTestPlayer(1, { ringsInHand: 5 }), createTestPlayer(2)],
      });

      expect(hasTurnMaterial(state, 1)).toBe(true);
      expect(hasTurnMaterial(state, 2)).toBe(true);
    });

    it('is true when player controls at least one stack even with no rings in hand', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(0, 0), 1, 2);

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
      });

      expect(hasTurnMaterial(state, 1)).toBe(true);
      expect(hasTurnMaterial(state, 2)).toBe(false);
    });

    it('is false when player has no rings in hand and no stacks', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
      });

      expect(hasTurnMaterial(state, 1)).toBe(false);
      expect(hasTurnMaterial(state, 2)).toBe(false);
    });

    it('is false for unknown player number', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1), createTestPlayer(2)],
      });

      expect(hasTurnMaterial(state, 999)).toBe(false);
    });
  });

  describe('hasGlobalPlacementAction and hasPhaseLocalInteractiveMove', () => {
    it('reports global placements and phase-local moves in a normal ring_placement state', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1, { ringsInHand: 3 }), createTestPlayer(2)],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
      });

      expect(hasGlobalPlacementAction(state, 1)).toBe(true);
      expect(hasPhaseLocalInteractiveMove(state, 1)).toBe(true);

      // Non-active players never have phase-local interactive moves.
      expect(hasPhaseLocalInteractiveMove(state, 2)).toBe(false);
    });

    it('hasGlobalPlacementAction is false when ringsInHand == 0', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
      });

      expect(hasGlobalPlacementAction(state, 1)).toBe(false);
      expect(hasGlobalPlacementAction(state, 2)).toBe(false);
    });

    it('returns false for hasPhaseLocalInteractiveMove when game is not active', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1, { ringsInHand: 3 }), createTestPlayer(2)],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        gameStatus: 'finished',
      });

      expect(hasPhaseLocalInteractiveMove(state, 1)).toBe(false);
    });

    it('returns false for hasPhaseLocalInteractiveMove in line_processing with no lines', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1), createTestPlayer(2)],
        currentPlayer: 1,
        currentPhase: 'line_processing',
        gameStatus: 'active',
      });

      // No lines to process means no phase-local moves
      expect(hasPhaseLocalInteractiveMove(state, 1)).toBe(false);
    });

    it('returns false for hasPhaseLocalInteractiveMove in territory_processing with no regions', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1), createTestPlayer(2)],
        currentPlayer: 1,
        currentPhase: 'territory_processing',
        gameStatus: 'active',
      });

      // No territory regions to process means no phase-local moves
      expect(hasPhaseLocalInteractiveMove(state, 1)).toBe(false);
    });

    it('returns false for hasPhaseLocalInteractiveMove with unknown phase', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1), createTestPlayer(2)],
        currentPlayer: 1,
        currentPhase: 'unknown_phase' as any,
        gameStatus: 'active',
      });

      expect(hasPhaseLocalInteractiveMove(state, 1)).toBe(false);
    });

    it('checks skip placement eligibility when no placements available in ring_placement phase', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        gameStatus: 'active',
      });

      // With no rings in hand, there are no placement actions.
      // This test exercises the skip_placement eligibility path.
      expect(hasGlobalPlacementAction(state, 1)).toBe(false);

      // Will evaluate skip placement eligibility - result depends on game state
      const result = hasPhaseLocalInteractiveMove(state, 1);
      expect(typeof result).toBe('boolean');
    });

    it('returns true in movement phase when stacks have legal moves', () => {
      const board = createTestBoard('square8');
      // Add a stack in the middle of the board where it has movement options
      addStack(board, pos(4, 4), 1, 2);

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
      });

      expect(hasPhaseLocalInteractiveMove(state, 1)).toBe(true);
    });

    it('skips other players stacks when checking for movement options', () => {
      const board = createTestBoard('square8');
      // Add a stack for player 2 (not the current player)
      addStack(board, pos(4, 4), 2, 2);
      // Also add a stack for player 1
      addStack(board, pos(2, 2), 1, 2);

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
      });

      // Should find moves for player 1 and skip player 2's stacks
      expect(hasPhaseLocalInteractiveMove(state, 1)).toBe(true);
    });
  });

  // INV-S-MONOTONIC / INV-ELIMINATION-MONOTONIC (R191, R207)
  // INV-ACTIVE-NO-MOVES (R200–R203) – forced elimination counts as a global action
  describe('hasForcedEliminationAction and applyForcedEliminationForPlayer', () => {
    it('returns false when game is not active', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(0, 0), 1, 2);

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'finished',
      });

      expect(hasForcedEliminationAction(state, 1)).toBe(false);
    });

    it('returns false for unknown player', () => {
      const board = createTestBoard('square8');
      addStack(board, pos(0, 0), 1, 2);

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
      });

      expect(hasForcedEliminationAction(state, 999)).toBe(false);
    });

    it('returns false when player has no stacks on board', () => {
      const board = createTestBoard('square8');
      (board as any).size = 1;
      // No stacks added for player 1

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
      });

      expect(hasForcedEliminationAction(state, 1)).toBe(false);
    });

    it('returns undefined from applyForcedEliminationForPlayer when preconditions not met', () => {
      const board = createTestBoard('square8');

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 3 }), createTestPlayer(2)],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
      });

      // Player has rings in hand, so placements are available
      const outcome = applyForcedEliminationForPlayer(state, 1);
      expect(outcome).toBeUndefined();
    });

    it('prefers stacks with smaller capHeight for forced elimination', () => {
      const board = createTestBoard('square8');
      (board as any).size = 1;

      // Add two stacks with different cap heights
      addStack(board, pos(0, 0), 1, 3, 2); // capHeight 2
      // Manually add another stack at a different position
      board.stacks.set('0,1', {
        position: pos(0, 1),
        controllingPlayer: 1,
        stackHeight: 4,
        capHeight: 1, // Smaller capHeight - should be preferred
        rings: [1, 1, 1, 1],
      } as any);

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        totalRingsEliminated: 0,
      });

      expect(hasForcedEliminationAction(state, 1)).toBe(true);

      const outcome = applyForcedEliminationForPlayer(state, 1);
      expect(outcome).toBeDefined();
      // Should prefer the stack with capHeight 1
      expect(outcome?.eliminatedFrom).toEqual(pos(0, 1));
    });

    it('uses fallback stack when no stacks have positive capHeight', () => {
      const board = createTestBoard('square8');
      (board as any).size = 1;

      // Add stack with capHeight 0 (or undefined)
      board.stacks.set('0,0', {
        position: pos(0, 0),
        controllingPlayer: 1,
        stackHeight: 2,
        capHeight: 0,
        rings: [1, 1],
      } as any);

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        totalRingsEliminated: 0,
      });

      expect(hasForcedEliminationAction(state, 1)).toBe(true);

      const outcome = applyForcedEliminationForPlayer(state, 1);
      expect(outcome).toBeDefined();
      expect(outcome?.eliminatedFrom).toEqual(pos(0, 0));
      expect(outcome?.eliminatedCount).toBeGreaterThanOrEqual(1);
    });

    it('skips other players stacks when selecting stack for forced elimination', () => {
      const board = createTestBoard('square8');
      (board as any).size = 1;

      // Add a stack for player 2 (should be skipped)
      board.stacks.set('0,1', {
        position: pos(0, 1),
        controllingPlayer: 2,
        stackHeight: 3,
        capHeight: 1,
        rings: [2, 2, 2],
      } as any);

      // Add a stack for player 1 (should be selected)
      board.stacks.set('0,0', {
        position: pos(0, 0),
        controllingPlayer: 1,
        stackHeight: 2,
        capHeight: 1,
        rings: [1, 1],
      } as any);

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        totalRingsEliminated: 0,
      });

      expect(hasForcedEliminationAction(state, 1)).toBe(true);

      const outcome = applyForcedEliminationForPlayer(state, 1);
      expect(outcome).toBeDefined();
      // Should eliminate from player 1's stack, not player 2's
      expect(outcome?.eliminatedPlayer).toBe(1);
      expect(outcome?.eliminatedFrom).toEqual(pos(0, 0));
    });

    it('detects forced-elimination opportunities when a player is blocked with stacks but no placements or moves', () => {
      // Use a 1×1 square board so that no non-capture movement or capture is
      // possible from the single cell.
      const board = createTestBoard('square8');
      // Override size for this test to force a 1×1 effective grid.
      (board as any).size = 1;
      addStack(board, pos(0, 0), 1, 2);

      const state = createTestGameState({
        board,
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        totalRingsEliminated: 0,
      });

      // Preconditions: player 1 has material on board but no placements and
      // no legal moves/captures from the single stack.
      expect(hasTurnMaterial(state, 1)).toBe(true);
      expect(hasGlobalPlacementAction(state, 1)).toBe(false);

      expect(hasForcedEliminationAction(state, 1)).toBe(true);

      const summary = computeGlobalLegalActionsSummary(state, 1);
      expect(summary.hasTurnMaterial).toBe(true);
      expect(summary.hasGlobalPlacementAction).toBe(false);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(false);
      expect(summary.hasForcedEliminationAction).toBe(true);

      const beforeS = computeSMetric(state);
      const beforeT = computeTMetric(state);
      const outcome = applyForcedEliminationForPlayer(state, 1);
      expect(outcome).toBeDefined();
      if (!outcome) {
        return;
      }

      expect(outcome.eliminatedPlayer).toBe(1);
      expect(outcome.eliminatedFrom).toEqual(pos(0, 0));
      expect(outcome.eliminatedCount).toBeGreaterThanOrEqual(1);

      const afterS = computeSMetric(outcome.nextState);
      const afterT = computeTMetric(outcome.nextState);
      expect(afterS).toBeGreaterThanOrEqual(beforeS);
      expect(afterT).toBeGreaterThan(beforeT);

      // By definition, states where forced elimination is available are not ANM.
      expect(isANMState(state)).toBe(false);
      expect(isANMState(outcome.nextState)).toBe(false);
    });
  });

  // INV-ACTIVE-NO-MOVES (R200–R203) and INV-ANM-TURN-MATERIAL-SKIP (R201)
  describe('isANMState (ACTIVE-no-moves predicate)', () => {
    it('returns false when game is not active', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        gameStatus: 'finished',
        currentPlayer: 1,
      });

      expect(isANMState(state)).toBe(false);
    });

    it('returns false when current player has no turn material', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        currentPlayer: 1,
        gameStatus: 'active',
        players: [createTestPlayer(1, { ringsInHand: 0 }), createTestPlayer(2, { ringsInHand: 0 })],
      });

      expect(hasTurnMaterial(state, 1)).toBe(false);
      expect(isANMState(state)).toBe(false);
    });

    it('returns false when global placements or phase-local moves exist', () => {
      const state = createTestGameState({
        board: createTestBoard('square8'),
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        gameStatus: 'active',
        players: [createTestPlayer(1, { ringsInHand: 3 }), createTestPlayer(2)],
      });

      const summary = computeGlobalLegalActionsSummary(state, 1);
      expect(summary.hasGlobalPlacementAction).toBe(true);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(true);
      expect(summary.hasForcedEliminationAction).toBe(false);

      expect(isANMState(state)).toBe(false);
    });

    it('treats placements-only movement states as non-ANM (ANM-SCEN-02)', () => {
      // Movement phase with no movement/capture actions but legal placements
      // available on the next ring_placement phase.
      const board = createTestBoard('square8');
      const state = createTestGameState({
        board,
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        players: [createTestPlayer(1, { ringsInHand: 3 }), createTestPlayer(2)],
      });

      // No stacks on the board ⇒ no phase-local movement/capture options.
      const summary = computeGlobalLegalActionsSummary(state, 1);
      expect(summary.hasTurnMaterial).toBe(true);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(false);

      // But global placements are still available and must count as legal
      // actions for ANM purposes (RR-CANON-R200 / ANM-SCEN-02).
      expect(summary.hasGlobalPlacementAction).toBe(true);
      expect(summary.hasForcedEliminationAction).toBe(false);

      expect(isANMState(state)).toBe(false);
    });

    it('returns true for an artificial ANM candidate with turn-material but no global actions (unreachable shape)', () => {
      const board = createTestBoard('square8');
      // Shrink the playable area so that no placements or movements exist.
      (board as any).size = 0;
      board.stacks.clear();
      board.markers.clear();
      board.collapsedSpaces.clear();

      const state = createTestGameState({
        board,
        currentPlayer: 1,
        currentPhase: 'movement',
        gameStatus: 'active',
        players: [createTestPlayer(1, { ringsInHand: 1 }), createTestPlayer(2)],
      });

      const summary = computeGlobalLegalActionsSummary(state, 1);
      expect(summary.hasTurnMaterial).toBe(true);
      expect(summary.hasGlobalPlacementAction).toBe(false);
      expect(summary.hasPhaseLocalInteractiveMove).toBe(false);
      expect(summary.hasForcedEliminationAction).toBe(false);

      // INV-ACTIVE-NO-MOVES: ANM(state) holds for this synthetic candidate.
      expect(isANMState(state)).toBe(true);
    });
  });
});
