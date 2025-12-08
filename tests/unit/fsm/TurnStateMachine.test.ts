/**
 * TurnStateMachine unit tests
 */

import {
  TurnStateMachine,
  transition,
  type TurnState,
  type TurnEvent,
  type GameContext,
  type RingPlacementState,
  type MovementState,
  type ChainCaptureState,
} from '../../../src/shared/engine/fsm';

describe('TurnStateMachine', () => {
  const context: GameContext = {
    boardType: 'square8',
    numPlayers: 2,
    ringsPerPlayer: 18,
    lineLength: 3,
  };

  describe('transition function', () => {
    it('should transition from ring_placement to movement on PLACE_RING', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        canPlace: true,
        validPositions: [{ x: 3, y: 3 }],
      };

      const event: TurnEvent = { type: 'PLACE_RING', to: { x: 3, y: 3 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('movement');
        expect(result.actions).toContainEqual({
          type: 'PLACE_RING',
          position: { x: 3, y: 3 },
          player: 1,
        });
        expect(result.actions).toContainEqual({
          type: 'LEAVE_MARKER',
          position: { x: 3, y: 3 },
          player: 1,
        });
      }
    });

    it('should reject PLACE_RING when canPlace is false', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        canPlace: false,
        validPositions: [],
      };

      const event: TurnEvent = { type: 'PLACE_RING', to: { x: 3, y: 3 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should reject invalid event for phase', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        canPlace: true,
        validPositions: [{ x: 3, y: 3 }],
      };

      // MOVE_STACK is not valid in ring_placement phase
      const event: TurnEvent = { type: 'MOVE_STACK', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('INVALID_EVENT');
        expect(result.error.currentPhase).toBe('ring_placement');
        expect(result.error.eventType).toBe('MOVE_STACK');
      }
    });

    it('should transition from movement to line_processing on MOVE_STACK', () => {
      const state: MovementState = {
        phase: 'movement',
        player: 1,
        canMove: true,
        placedRingAt: { x: 3, y: 3 },
      };

      const event: TurnEvent = { type: 'MOVE_STACK', from: { x: 3, y: 3 }, to: { x: 4, y: 4 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('line_processing');
        expect(result.actions).toContainEqual({
          type: 'MOVE_STACK',
          from: { x: 3, y: 3 },
          to: { x: 4, y: 4 },
        });
      }
    });

    it('should transition to game_over on RESIGN', () => {
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        canPlace: true,
        validPositions: [{ x: 3, y: 3 }],
      };

      const event: TurnEvent = { type: 'RESIGN', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('resignation');
        }
      }
    });
  });

  describe('TurnStateMachine class', () => {
    it('should track state through transitions', () => {
      const initialState = TurnStateMachine.createInitialState(1);
      const fsm = new TurnStateMachine(
        { ...initialState, validPositions: [{ x: 3, y: 3 }] },
        context
      );

      expect(fsm.phase).toBe('ring_placement');
      expect(fsm.currentPlayer).toBe(1);

      const actions = fsm.send({ type: 'PLACE_RING', to: { x: 3, y: 3 } });
      expect(fsm.phase).toBe('movement');
      expect(actions.length).toBeGreaterThan(0);
    });

    it('should throw on invalid transitions', () => {
      const initialState = TurnStateMachine.createInitialState(1);
      const fsm = new TurnStateMachine(
        { ...initialState, validPositions: [{ x: 3, y: 3 }] },
        context
      );

      // MOVE_STACK is not valid in ring_placement phase
      expect(() => {
        fsm.send({ type: 'MOVE_STACK', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } });
      }).toThrow('[FSM]');
    });

    it('should track history', () => {
      const initialState = TurnStateMachine.createInitialState(1);
      const fsm = new TurnStateMachine(
        { ...initialState, validPositions: [{ x: 3, y: 3 }] },
        context
      );

      fsm.send({ type: 'PLACE_RING', to: { x: 3, y: 3 } });

      const history = fsm.getHistory();
      expect(history.length).toBe(1);
      expect(history[0].event.type).toBe('PLACE_RING');
    });

    it('canSend should return true for valid events', () => {
      const initialState = TurnStateMachine.createInitialState(1);
      const fsm = new TurnStateMachine(
        { ...initialState, validPositions: [{ x: 3, y: 3 }] },
        context
      );

      expect(fsm.canSend({ type: 'PLACE_RING', to: { x: 3, y: 3 } })).toBe(true);
      expect(fsm.canSend({ type: 'MOVE_STACK', from: { x: 0, y: 0 }, to: { x: 1, y: 1 } })).toBe(false);
    });
  });

  describe('chain capture transitions', () => {
    it('should allow CONTINUE_CHAIN when valid target exists', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [
          { target: { x: 5, y: 5 }, capturingPlayer: 1, isChainCapture: true },
        ],
        segmentCount: 1,
        isFirstSegment: false,
      };

      const event: TurnEvent = { type: 'CONTINUE_CHAIN', target: { x: 5, y: 5 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('chain_capture');
        expect(result.actions).toContainEqual({
          type: 'EXECUTE_CAPTURE',
          target: { x: 5, y: 5 },
          capturer: 1,
        });
        if (result.state.phase === 'chain_capture') {
          expect(result.state.capturedTargets).toContainEqual({ x: 5, y: 5 });
          expect(result.state.segmentCount).toBe(2);
        }
      }
    });

    it('should reject CONTINUE_CHAIN with invalid target', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [
          { target: { x: 5, y: 5 }, capturingPlayer: 1, isChainCapture: true },
        ],
        segmentCount: 1,
        isFirstSegment: false,
      };

      // Invalid target - not in availableContinuations
      const event: TurnEvent = { type: 'CONTINUE_CHAIN', target: { x: 6, y: 6 } };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should allow END_CHAIN when no mandatory captures remain', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [], // No more targets
        segmentCount: 1,
        isFirstSegment: false,
      };

      const event: TurnEvent = { type: 'END_CHAIN' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('line_processing');
      }
    });

    it('should reject END_CHAIN when mandatory captures remain', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [
          { target: { x: 5, y: 5 }, capturingPlayer: 1, isChainCapture: true },
        ],
        segmentCount: 1,
        isFirstSegment: false,
      };

      const event: TurnEvent = { type: 'END_CHAIN' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
      }
    });

    it('should transition to game_over on RESIGN from chain_capture', () => {
      const state: ChainCaptureState = {
        phase: 'chain_capture',
        player: 1,
        attackerPosition: { x: 4, y: 4 },
        capturedTargets: [{ x: 3, y: 3 }],
        availableContinuations: [],
        segmentCount: 1,
        isFirstSegment: false,
      };

      const event: TurnEvent = { type: 'RESIGN', player: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('game_over');
        if (result.state.phase === 'game_over') {
          expect(result.state.reason).toBe('resignation');
        }
      }
    });
  });
});
