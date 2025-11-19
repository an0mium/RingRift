import { BoardManager } from '../../src/server/game/BoardManager';
import { RuleEngine } from '../../src/server/game/RuleEngine';
import { GameState, Move } from '../../src/shared/types/game';
import { createTestGameState, pos } from '../utils/fixtures';

/**
 * Scenario Tests: RuleEngine movement scenarios
 *
 * These tests encode small, FAQ‑style examples for non‑capture movement
 * and blocking, keyed to the compact rules §3.1–3.2 and
 * `ringrift_complete_rules.md` §8.2–8.3 (FAQ Q2–Q3).
 */

describe('RuleEngine movement scenarios (Section 8.2–8.3; FAQ 2–3)', () => {
  function createState(): {
    gameState: GameState;
    boardManager: BoardManager;
    ruleEngine: RuleEngine;
  } {
    const gameState = createTestGameState({ boardType: 'square8' });

    // Replace the board with one created by BoardManager so that
    // RuleEngine and BoardManager share the same BoardState instance.
    const boardManager = new BoardManager('square8');
    gameState.board = boardManager.createBoard();

    const ruleEngine = new RuleEngine(boardManager, 'square8');

    gameState.currentPlayer = 1;
    gameState.currentPhase = 'movement';

    return { gameState, boardManager, ruleEngine };
  }

  it('FAQ_2_1_minimum_distance_at_least_stack_height_square8', () => {
    // Rules reference:
    // - Compact rules §3.1; complete rules §8.2; FAQ Q2.
    // - A stack of height H must move at least distance H along a
    //   straight line; shorter landings are illegal.

    const { gameState, boardManager, ruleEngine } = createState();

    // Single Player 1 stack of height 2 at (3,3).
    const origin = pos(3, 3);
    const rings = [1, 1];
    boardManager.setStack(
      origin,
      {
        position: origin,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: 1,
      },
      gameState.board
    );

    const moves = ruleEngine.getValidMoves(gameState);
    const movementMoves = moves.filter(
      (m) => m.type === 'move_stack' || m.type === 'move_ring'
    ) as Move[];

    // There must be at least one legal move of distance >= 2.
    expect(movementMoves.length).toBeGreaterThan(0);

    // No move is allowed that lands at Chebyshev distance 1 from origin.
    const hasTooShortMove = movementMoves.some((m) => {
      if (!m.to || !m.from) return false;
      const dx = Math.abs(m.to.x - m.from.x);
      const dy = Math.abs(m.to.y - m.from.y);
      const dist = Math.max(dx, dy);
      return dist < rings.length; // distance < stackHeight
    });

    expect(hasTooShortMove).toBe(false);
  });

  it('FAQ_2_2_blocked_by_collapsed_spaces_and_stacks_square8', () => {
    // Rules reference:
    // - Compact rules §3.1–3.2; complete rules §8.2–8.3; FAQ Q2–Q3.
    // - Movement rays are blocked by collapsed spaces and stacks; you
    //   cannot move "through" them, even if the landing cell would be
    //   far enough away.

    const { gameState, boardManager, ruleEngine } = createState();

    const origin = pos(3, 3);
    const rings = [1, 1];
    boardManager.setStack(
      origin,
      {
        position: origin,
        rings,
        stackHeight: rings.length,
        capHeight: rings.length,
        controllingPlayer: 1,
      },
      gameState.board
    );

    // Place a blocking stack in the positive X direction and a
    // collapsed space in the positive Y direction.
    const stackBlock = pos(5, 3); // same row to the right
    const blockerRings = [2];
    boardManager.setStack(
      stackBlock,
      {
        position: stackBlock,
        rings: blockerRings,
        stackHeight: blockerRings.length,
        capHeight: blockerRings.length,
        controllingPlayer: 2,
      },
      gameState.board
    );

    const collapsedBlock = pos(3, 5); // same column upward
    gameState.board.collapsedSpaces.set(`${collapsedBlock.x},${collapsedBlock.y}`, 0);

    const moves = ruleEngine.getValidMoves(gameState);
    const movementMoves = moves.filter(
      (m) => m.type === 'move_stack' || m.type === 'move_ring'
    ) as Move[];

    // No legal move may "jump over" the stack at (5,3) or the
    // collapsed space at (3,5) along their respective rays.
    const illegalThroughStack = movementMoves.some((m) => {
      if (!m.to || !m.from) return false;
      // Any landing strictly to the right of the blocking stack must
      // have passed through it along the horizontal ray.
      return (
        m.from.x === origin.x &&
        m.from.y === origin.y &&
        m.to.y === origin.y &&
        m.to.x > stackBlock.x
      );
    });

    const illegalThroughCollapsed = movementMoves.some((m) => {
      if (!m.to || !m.from) return false;
      // Any landing strictly above the collapsedBlock on the same
      // column would require crossing it.
      return (
        m.from.x === origin.x &&
        m.from.y === origin.y &&
        m.to.x === origin.x &&
        m.to.y > collapsedBlock.y
      );
    });

    expect(illegalThroughStack).toBe(false);
    expect(illegalThroughCollapsed).toBe(false);
  });
});
