/**
 * Movement / no_movement_action ANM parity tests
 *
 * These tests lock in the canonical semantics for no_movement_action in the
 * movement phase, aligned with RR-CANON ANM rules and Python's phase machine:
 *
 * - When the current player has at least one legal movement/capture anywhere
 *   on the board, NO_MOVEMENT_ACTION must be rejected as an illegal skip.
 * - When the current player has no legal movement/capture at all, then
 *   NO_MOVEMENT_ACTION is the only canonical movement bookkeeping move and
 *   must be accepted by the FSM and orchestrator.
 */

import {
  validateMoveWithFSM,
} from '../../src/shared/engine/fsm/FSMAdapter';
import {
  processTurn,
  getValidMoves,
} from '../../src/shared/engine/orchestration/turnOrchestrator';
import type {
  GameState,
  Move,
  Player,
  BoardState,
  Position,
  GamePhase,
} from '../../src/shared/types/game';

describe('Movement no_movement_action ANM semantics', () => {
  const createPlayer = (playerNumber: number, ringsInHand: number): Player => ({
    id: `player-${playerNumber}`,
    username: `Player ${playerNumber}`,
    playerNumber,
    type: 'human',
    isReady: true,
    timeRemaining: 600000,
    ringsInHand,
    eliminatedRings: 0,
    territorySpaces: 0,
  });

  const createBoard = (overrides: Partial<BoardState> = {}): BoardState => ({
    type: 'square8',
    size: 8,
    stacks: new Map(),
    markers: new Map(),
    territories: new Map(),
    formedLines: [],
    collapsedSpaces: new Map(),
    eliminatedRings: {},
    ...overrides,
  });

  const createGameState = (overrides: Partial<GameState> = {}): GameState => ({
    id: 'anm-test-game',
    boardType: 'square8',
    board: createBoard(),
    players: [createPlayer(1, 0), createPlayer(2, 0)],
    currentPlayer: 1,
    currentPhase: 'movement' as GamePhase,
    gameStatus: 'active',
    moveHistory: [],
    history: [],
    timeControl: { type: 'rapid', initialTime: 600000, increment: 0 },
    rulesOptions: undefined,
    spectators: [],
    winner: undefined,
    createdAt: new Date(),
    lastMoveAt: new Date(),
    isRated: false,
    maxPlayers: 2,
    totalRingsInPlay: 0,
    totalRingsEliminated: 0,
    victoryThreshold: 19,
    territoryVictoryThreshold: 33,
    ...overrides,
  });

  const createMove = (
    type: Move['type'],
    player: number,
    extras: Partial<Move> = {},
  ): Move => ({
    id: `anm-test-move-${Date.now()}`,
    type,
    player,
    to: { x: 0, y: 0 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
    ...extras,
  });

  describe('forced ANM scenario (no legal movements/captures anywhere)', () => {
    /**
     * Shape:
     * - currentPhase == movement
     * - currentPlayer == 1
     * - Single-cell board (1x1) with a single stack for player 1
     * - No rings in hand for player 1 (only stack material)
     * - No legal non-capturing moves or overtaking captures exist
     *
     * Expectation:
     * - getValidMoves returns no interactive movement/capture options.
     * - FSM derives MovementState.canMove === false.
     * - validateMoveWithFSM accepts no_movement_action.
     * - processTurn(no_movement_action) transitions to line_processing and
     *   surfaces no_line_action_required when no lines exist.
     */
    it('accepts no_movement_action when no movement exists (forced ANM)', () => {
      const singleCellBoard: BoardState = createBoard({
        size: 1,
        stacks: new Map([
          ['0,0', {
            position: { x: 0, y: 0 },
            stackHeight: 1,
            controllingPlayer: 1,
            composition: [{ player: 1, count: 1 }],
            rings: [1],
            capHeight: 1,
          }],
        ]),
      });

      const state: GameState = createGameState({
        board: singleCellBoard,
        currentPhase: 'movement',
        currentPlayer: 1,
        players: [createPlayer(1, 0), createPlayer(2, 0)],
      });

      const interactiveMoves = getValidMoves(state);
      const hasMovementLikeMove = interactiveMoves.some(
        (m) =>
          m.type === 'move_stack' ||
          m.type === 'move_ring' ||
          m.type === 'overtaking_capture' ||
          m.type === 'continue_capture_segment',
      );
      expect(hasMovementLikeMove).toBe(false);

      const noMovementMove: Move = createMove('no_movement_action', 1);

      const fsmResult = validateMoveWithFSM(state, noMovementMove);
      expect(fsmResult.valid).toBe(true);
      expect(fsmResult.currentPhase).toBe('movement');

      const turnResult = processTurn(state, noMovementMove);

      expect(turnResult.nextState.currentPhase).toBe('line_processing');
      expect(['complete', 'awaiting_decision']).toContain(turnResult.status);

      if (turnResult.status === 'awaiting_decision') {
        expect(turnResult.pendingDecision?.type).toBe('no_line_action_required');
      }
    });
  });

  describe('non-ANM scenario (movement/capture exists)', () => {
    /**
     * Shape:
     * - currentPhase == movement
     * - currentPlayer == 1
     * - Board has at least one stack for player 1 with legal movement
     *
     * Expectation:
     * - Some interactive movement/capture options exist.
     * - FSM derives MovementState.canMove === true.
     * - validateMoveWithFSM rejects no_movement_action with GUARD_FAILED and
     *   the canonical "Cannot skip movement when valid moves exist" reason.
     */
    it('rejects no_movement_action when legal movement exists', () => {
      const boardWithMovement: BoardState = createBoard({
        size: 8,
        stacks: new Map([
          ['3,3', {
            position: { x: 3, y: 3 },
            stackHeight: 1,
            controllingPlayer: 1,
            composition: [{ player: 1, count: 1 }],
            rings: [1],
            capHeight: 1,
          }],
        ]),
      });

      const state: GameState = createGameState({
        board: boardWithMovement,
        currentPhase: 'movement',
        currentPlayer: 1,
        players: [createPlayer(1, 0), createPlayer(2, 0)],
      });

      const interactiveMoves = getValidMoves(state);
      const hasMovementLikeMove = interactiveMoves.some(
        (m) =>
          m.type === 'move_stack' ||
          m.type === 'move_ring' ||
          m.type === 'overtaking_capture' ||
          m.type === 'continue_capture_segment',
      );
      expect(hasMovementLikeMove).toBe(true);

      const noMovementMove: Move = createMove('no_movement_action', 1);

      const fsmResult = validateMoveWithFSM(state, noMovementMove);
      expect(fsmResult.valid).toBe(false);
      expect(fsmResult.errorCode).toBe('GUARD_FAILED');
      expect(fsmResult.reason).toMatch(/Cannot skip movement when valid moves exist/);
    });
  });
});