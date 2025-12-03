/**
 * Chain Capture Tracking Module Unit Tests
 *
 * Tests for the chain capture state management including:
 * - Creating empty/minimal/full chain capture state
 * - Updating chain capture position and state
 * - Evaluating chain capture continuation
 * - Validating continuation moves
 * - Getting chain capture moves
 */

import {
  createEmptyChainCaptureState,
  createMinimalChainCaptureState,
  createFullChainCaptureState,
  updateChainCapturePosition,
  updateFullChainCaptureState,
  isChainCapturePhase,
  isChainCaptureActive,
  getChainCapturePosition,
  getChainCapturePlayer,
  clearChainCaptureState,
  validateChainCaptureContinuation,
  getChainCaptureMoves,
  evaluateChainCaptureContinuation,
  processChainCaptureResult,
  MinimalChainCaptureState,
  ChainCaptureState,
} from '../../src/shared/engine/chainCaptureTracking';
import * as CaptureAggregate from '../../src/shared/engine/aggregates/CaptureAggregate';
import type { Move, Position, GameState, GamePhase } from '../../src/shared/types/game';

// Mock the CaptureAggregate module
jest.mock('../../src/shared/engine/aggregates/CaptureAggregate', () => ({
  getChainCaptureContinuationInfo: jest.fn(),
  updateChainCaptureStateAfterCapture: jest.fn(),
}));

describe('chainCaptureTracking module', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('createEmptyChainCaptureState', () => {
    it('should return null', () => {
      const result = createEmptyChainCaptureState();
      expect(result).toBeNull();
    });
  });

  describe('createMinimalChainCaptureState', () => {
    it('should create state with correct playerNumber', () => {
      const state = createMinimalChainCaptureState(2, { x: 3, y: 4 });
      expect(state.playerNumber).toBe(2);
    });

    it('should create state with correct currentPosition', () => {
      const position: Position = { x: 5, y: 6 };
      const state = createMinimalChainCaptureState(1, position);
      expect(state.currentPosition).toEqual(position);
    });

    it('should create state with isActive=true', () => {
      const state = createMinimalChainCaptureState(1, { x: 0, y: 0 });
      expect(state.isActive).toBe(true);
    });
  });

  describe('createFullChainCaptureState', () => {
    it('should delegate to updateChainCaptureStateAfterCapture with undefined', () => {
      const mockMove: Move = {
        type: 'capture',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 2, y: 2 },
      } as Move;

      const mockResult = {
        playerNumber: 1,
        currentPosition: { x: 2, y: 2 },
        segments: [],
        visitedPositions: new Set<string>(),
        availableMoves: [],
      };

      (CaptureAggregate.updateChainCaptureStateAfterCapture as jest.Mock).mockReturnValue(
        mockResult
      );

      const result = createFullChainCaptureState(mockMove, 3);

      expect(CaptureAggregate.updateChainCaptureStateAfterCapture).toHaveBeenCalledWith(
        undefined,
        mockMove,
        3
      );
      expect(result).toBe(mockResult);
    });
  });

  describe('updateChainCapturePosition', () => {
    it('should update currentPosition immutably', () => {
      const state: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };
      const newPosition: Position = { x: 5, y: 5 };

      const result = updateChainCapturePosition(state, newPosition);

      expect(result.currentPosition).toEqual(newPosition);
      expect(result).not.toBe(state);
    });

    it('should preserve playerNumber', () => {
      const state: MinimalChainCaptureState = {
        playerNumber: 3,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };

      const result = updateChainCapturePosition(state, { x: 1, y: 1 });

      expect(result.playerNumber).toBe(3);
    });

    it('should preserve isActive', () => {
      const state: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };

      const result = updateChainCapturePosition(state, { x: 1, y: 1 });

      expect(result.isActive).toBe(true);
    });
  });

  describe('updateFullChainCaptureState', () => {
    it('should delegate to updateChainCaptureStateAfterCapture', () => {
      const existingState = {
        playerNumber: 1,
        currentPosition: { x: 2, y: 2 },
        segments: [],
        visitedPositions: new Set<string>(),
        availableMoves: [],
      };
      const mockMove: Move = {
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 2, y: 2 },
        to: { x: 4, y: 4 },
      } as Move;

      (CaptureAggregate.updateChainCaptureStateAfterCapture as jest.Mock).mockReturnValue({
        ...existingState,
        currentPosition: { x: 4, y: 4 },
      });

      updateFullChainCaptureState(existingState, mockMove, 2);

      expect(CaptureAggregate.updateChainCaptureStateAfterCapture).toHaveBeenCalledWith(
        existingState,
        mockMove,
        2
      );
    });
  });

  describe('isChainCapturePhase', () => {
    it('should return true for chain_capture phase', () => {
      expect(isChainCapturePhase('chain_capture')).toBe(true);
    });

    it('should return false for capture phase', () => {
      expect(isChainCapturePhase('capture')).toBe(false);
    });

    it('should return false for movement phase', () => {
      expect(isChainCapturePhase('movement')).toBe(false);
    });

    it('should return false for ring_placement phase', () => {
      expect(isChainCapturePhase('ring_placement')).toBe(false);
    });

    it('should return false for line_formed phase', () => {
      expect(isChainCapturePhase('line_formed')).toBe(false);
    });
  });

  describe('isChainCaptureActive', () => {
    it('should return false for null state', () => {
      expect(isChainCaptureActive(null)).toBe(false);
    });

    it('should return false for undefined state', () => {
      expect(isChainCaptureActive(undefined)).toBe(false);
    });

    it('should return isActive value for MinimalChainCaptureState', () => {
      const activeState: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };
      const inactiveState: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        isActive: false,
      };

      expect(isChainCaptureActive(activeState)).toBe(true);
      expect(isChainCaptureActive(inactiveState)).toBe(false);
    });

    it('should return true for full ChainCaptureState (always active if exists)', () => {
      const fullState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        segments: [],
        visitedPositions: new Set<string>(),
        availableMoves: [],
      };

      expect(isChainCaptureActive(fullState)).toBe(true);
    });
  });

  describe('getChainCapturePosition', () => {
    it('should return null for null state', () => {
      expect(getChainCapturePosition(null)).toBeNull();
    });

    it('should return null for undefined state', () => {
      expect(getChainCapturePosition(undefined)).toBeNull();
    });

    it('should return currentPosition from state', () => {
      const state: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 7, y: 8 },
        isActive: true,
      };

      expect(getChainCapturePosition(state)).toEqual({ x: 7, y: 8 });
    });
  });

  describe('getChainCapturePlayer', () => {
    it('should return null for null state', () => {
      expect(getChainCapturePlayer(null)).toBeNull();
    });

    it('should return null for undefined state', () => {
      expect(getChainCapturePlayer(undefined)).toBeNull();
    });

    it('should return playerNumber from state', () => {
      const state: MinimalChainCaptureState = {
        playerNumber: 4,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };

      expect(getChainCapturePlayer(state)).toBe(4);
    });
  });

  describe('clearChainCaptureState', () => {
    it('should return null', () => {
      expect(clearChainCaptureState()).toBeNull();
    });
  });

  describe('validateChainCaptureContinuation', () => {
    it('should return false for null chainState', () => {
      const move: Move = {
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 2, y: 2 },
      } as Move;

      expect(validateChainCaptureContinuation(move, null)).toBe(false);
    });

    it('should return false for undefined chainState', () => {
      const move: Move = {
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 2, y: 2 },
      } as Move;

      expect(validateChainCaptureContinuation(move, undefined)).toBe(false);
    });

    it('should return false for wrong move type', () => {
      const move: Move = {
        type: 'capture',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 2, y: 2 },
      } as Move;
      const state: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };

      expect(validateChainCaptureContinuation(move, state)).toBe(false);
    });

    it('should return false for wrong player', () => {
      const move: Move = {
        type: 'continue_capture_segment',
        player: 2,
        from: { x: 0, y: 0 },
        to: { x: 2, y: 2 },
      } as Move;
      const state: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };

      expect(validateChainCaptureContinuation(move, state)).toBe(false);
    });

    it('should return false for missing from position', () => {
      const move: Move = {
        type: 'continue_capture_segment',
        player: 1,
        to: { x: 2, y: 2 },
      } as Move;
      const state: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };

      expect(validateChainCaptureContinuation(move, state)).toBe(false);
    });

    it('should return false when from position does not match currentPosition', () => {
      const move: Move = {
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 3, y: 3 },
        to: { x: 5, y: 5 },
      } as Move;
      const state: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };

      expect(validateChainCaptureContinuation(move, state)).toBe(false);
    });

    it('should return true for valid continuation', () => {
      const move: Move = {
        type: 'continue_capture_segment',
        player: 1,
        from: { x: 2, y: 3 },
        to: { x: 4, y: 5 },
      } as Move;
      const state: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 2, y: 3 },
        isActive: true,
      };

      expect(validateChainCaptureContinuation(move, state)).toBe(true);
    });
  });

  describe('evaluateChainCaptureContinuation', () => {
    it('should call getChainCaptureContinuationInfo and return formatted result', () => {
      const mockGameState = {} as GameState;
      (CaptureAggregate.getChainCaptureContinuationInfo as jest.Mock).mockReturnValue({
        mustContinue: true,
        availableContinuations: [{ type: 'continue_capture_segment' } as Move],
      });

      const result = evaluateChainCaptureContinuation(mockGameState, 1, { x: 2, y: 2 });

      expect(CaptureAggregate.getChainCaptureContinuationInfo).toHaveBeenCalledWith(
        mockGameState,
        1,
        { x: 2, y: 2 }
      );
      expect(result.mustContinue).toBe(true);
      expect(result.availableMoves).toHaveLength(1);
      expect(result.recommendedPhase).toBe('chain_capture');
    });

    it('should recommend line_processing when mustContinue is false', () => {
      const mockGameState = {} as GameState;
      (CaptureAggregate.getChainCaptureContinuationInfo as jest.Mock).mockReturnValue({
        mustContinue: false,
        availableContinuations: [],
      });

      const result = evaluateChainCaptureContinuation(mockGameState, 1, { x: 2, y: 2 });

      expect(result.mustContinue).toBe(false);
      expect(result.recommendedPhase).toBe('line_processing');
    });
  });

  describe('getChainCaptureMoves', () => {
    it('should return empty array for null state', () => {
      const mockGameState = {} as GameState;
      expect(getChainCaptureMoves(mockGameState, null)).toEqual([]);
    });

    it('should return empty array for undefined state', () => {
      const mockGameState = {} as GameState;
      expect(getChainCaptureMoves(mockGameState, undefined)).toEqual([]);
    });

    it('should return cached availableMoves from full state', () => {
      const mockGameState = {} as GameState;
      const cachedMoves = [{ type: 'continue_capture_segment' } as Move];
      const state = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        segments: [],
        visitedPositions: new Set<string>(),
        availableMoves: cachedMoves,
      };

      const result = getChainCaptureMoves(mockGameState, state);

      expect(result).toBe(cachedMoves);
      expect(CaptureAggregate.getChainCaptureContinuationInfo).not.toHaveBeenCalled();
    });

    it('should enumerate moves when no cached moves', () => {
      const mockGameState = {} as GameState;
      const state: MinimalChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        isActive: true,
      };
      const enumeratedMoves = [{ type: 'continue_capture_segment' } as Move];

      (CaptureAggregate.getChainCaptureContinuationInfo as jest.Mock).mockReturnValue({
        mustContinue: true,
        availableContinuations: enumeratedMoves,
      });

      const result = getChainCaptureMoves(mockGameState, state);

      expect(result).toEqual(enumeratedMoves);
    });

    it('should enumerate moves when cached moves is empty array', () => {
      const mockGameState = {} as GameState;
      const state = {
        playerNumber: 1,
        currentPosition: { x: 0, y: 0 },
        segments: [],
        visitedPositions: new Set<string>(),
        availableMoves: [],
      };
      const enumeratedMoves = [{ type: 'continue_capture_segment' } as Move];

      (CaptureAggregate.getChainCaptureContinuationInfo as jest.Mock).mockReturnValue({
        mustContinue: true,
        availableContinuations: enumeratedMoves,
      });

      const result = getChainCaptureMoves(mockGameState, state);

      expect(result).toEqual(enumeratedMoves);
    });
  });

  describe('processChainCaptureResult', () => {
    it('should update chain state and evaluate continuation', () => {
      const mockGameState = {} as GameState;
      const mockMove: Move = {
        type: 'capture',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 2, y: 2 },
      } as Move;
      const existingState = undefined;

      const updatedState = {
        playerNumber: 1,
        currentPosition: { x: 2, y: 2 },
        segments: [],
        visitedPositions: new Set<string>(),
        availableMoves: [],
      };

      (CaptureAggregate.updateChainCaptureStateAfterCapture as jest.Mock).mockReturnValue(
        updatedState
      );
      (CaptureAggregate.getChainCaptureContinuationInfo as jest.Mock).mockReturnValue({
        mustContinue: true,
        availableContinuations: [{ type: 'continue_capture_segment' } as Move],
      });

      const result = processChainCaptureResult(mockGameState, mockMove, 3, existingState);

      expect(result.chainState).toBe(updatedState);
      expect(result.evaluation.mustContinue).toBe(true);
      expect(result.shouldTransitionToChainPhase).toBe(true);
    });

    it('should clear chain state when continuation not needed', () => {
      const mockGameState = {} as GameState;
      const mockMove: Move = {
        type: 'capture',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 2, y: 2 },
      } as Move;

      (CaptureAggregate.updateChainCaptureStateAfterCapture as jest.Mock).mockReturnValue({
        playerNumber: 1,
        currentPosition: { x: 2, y: 2 },
        segments: [],
        visitedPositions: new Set<string>(),
        availableMoves: [],
      });
      (CaptureAggregate.getChainCaptureContinuationInfo as jest.Mock).mockReturnValue({
        mustContinue: false,
        availableContinuations: [],
      });

      const result = processChainCaptureResult(mockGameState, mockMove, 3, undefined);

      expect(result.chainState).toBeUndefined();
      expect(result.shouldTransitionToChainPhase).toBe(false);
    });

    it('should cache available moves in chain state when mustContinue', () => {
      const mockGameState = {} as GameState;
      const mockMove: Move = {
        type: 'capture',
        player: 1,
        from: { x: 0, y: 0 },
        to: { x: 2, y: 2 },
      } as Move;

      const updatedState: ChainCaptureState = {
        playerNumber: 1,
        currentPosition: { x: 2, y: 2 },
        segments: [],
        visitedPositions: new Set<string>(),
        availableMoves: [],
      };

      const continuationMoves = [{ type: 'continue_capture_segment' } as Move];

      (CaptureAggregate.updateChainCaptureStateAfterCapture as jest.Mock).mockReturnValue(
        updatedState
      );
      (CaptureAggregate.getChainCaptureContinuationInfo as jest.Mock).mockReturnValue({
        mustContinue: true,
        availableContinuations: continuationMoves,
      });

      processChainCaptureResult(mockGameState, mockMove, 3, undefined);

      expect(updatedState.availableMoves).toBe(continuationMoves);
    });
  });
});
