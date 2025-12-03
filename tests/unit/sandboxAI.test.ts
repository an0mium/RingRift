/**
 * Unit tests for sandboxAI.ts
 *
 * Tests for sandbox AI decision-making functions: buildSandboxMovementCandidates,
 * selectSandboxMovementMove, and maybeRunAITurnSandbox.
 */

import {
  buildSandboxMovementCandidates,
  selectSandboxMovementMove,
  maybeRunAITurnSandbox,
  SandboxAIHooks,
} from '../../src/client/sandbox/sandboxAI';
import {
  createTestGameState,
  createTestPlayer,
  createTestBoard,
  pos,
  posStr,
} from '../utils/fixtures';
import type {
  GameState,
  Move,
  Position,
  BoardState,
  LocalAIRng,
} from '../../src/shared/types/game';

describe('sandboxAI', () => {
  // Create a mock RNG for deterministic tests
  // LocalAIRng is just () => number
  function createMockRng(values: number[] = [0.5]): LocalAIRng {
    let index = 0;
    return () => {
      const value = values[index % values.length];
      index++;
      return value;
    };
  }

  // Helper to create AI player
  function createAIPlayer(playerNumber: number, overrides: Partial<any> = {}) {
    return createTestPlayer(playerNumber, { type: 'ai', ...overrides });
  }

  // Helper to create human player
  function createHumanPlayer(playerNumber: number, overrides: Partial<any> = {}) {
    return createTestPlayer(playerNumber, { type: 'human', ...overrides });
  }

  // Create mock hooks with sensible defaults
  function createMockHooks(overrides: Partial<SandboxAIHooks> = {}): SandboxAIHooks {
    return {
      getPlayerStacks: jest.fn().mockReturnValue([]),
      hasAnyLegalMoveOrCaptureFrom: jest.fn().mockReturnValue(true),
      enumerateLegalRingPlacements: jest.fn().mockReturnValue([]),
      getValidMovesForCurrentPlayer: jest.fn().mockReturnValue([]),
      createHypotheticalBoardWithPlacement: jest.fn().mockImplementation((board) => board),
      tryPlaceRings: jest.fn().mockResolvedValue(true),
      enumerateCaptureSegmentsFrom: jest.fn().mockReturnValue([]),
      enumerateSimpleMovementLandings: jest.fn().mockReturnValue([]),
      maybeProcessForcedEliminationForCurrentPlayer: jest.fn().mockReturnValue(false),
      handleMovementClick: jest.fn().mockResolvedValue(undefined),
      appendHistoryEntry: jest.fn(),
      getGameState: jest.fn().mockReturnValue(
        createTestGameState({
          currentPlayer: 1,
          currentPhase: 'ring_placement',
          gameStatus: 'active',
          players: [createAIPlayer(1), createAIPlayer(2)],
        })
      ),
      setGameState: jest.fn(),
      setLastAIMove: jest.fn(),
      setSelectedStackKey: jest.fn(),
      getMustMoveFromStackKey: jest.fn().mockReturnValue(undefined),
      applyCanonicalMove: jest.fn().mockResolvedValue(undefined),
      hasPendingTerritorySelfElimination: jest.fn().mockReturnValue(false),
      hasPendingLineRewardElimination: jest.fn().mockReturnValue(false),
      canCurrentPlayerSwapSides: jest.fn().mockReturnValue(false),
      applySwapSidesForCurrentPlayer: jest.fn().mockReturnValue(false),
      ...overrides,
    };
  }

  describe('buildSandboxMovementCandidates', () => {
    it('returns empty candidates when no stacks available', () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'movement',
        players: [createAIPlayer(1), createAIPlayer(2)],
      });
      const hooks = createMockHooks({
        getPlayerStacks: jest.fn().mockReturnValue([]),
        enumerateCaptureSegmentsFrom: jest.fn().mockReturnValue([]),
        enumerateSimpleMovementLandings: jest.fn().mockReturnValue([]),
      });
      const rng = createMockRng();

      const result = buildSandboxMovementCandidates(gameState, hooks, rng);

      expect(result.candidates).toHaveLength(0);
      expect(result.debug.captureCount).toBe(0);
      expect(result.debug.simpleMoveCount).toBe(0);
    });

    it('builds capture candidates from player stacks', () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'movement',
        players: [createAIPlayer(1), createAIPlayer(2)],
      });

      const stack = {
        position: pos(2, 2),
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      };

      const captureSegment = {
        from: pos(2, 2),
        target: pos(3, 2),
        landing: pos(4, 2),
      };

      const hooks = createMockHooks({
        getPlayerStacks: jest.fn().mockReturnValue([stack]),
        enumerateCaptureSegmentsFrom: jest.fn().mockReturnValue([captureSegment]),
        enumerateSimpleMovementLandings: jest.fn().mockReturnValue([]),
      });
      const rng = createMockRng();

      const result = buildSandboxMovementCandidates(gameState, hooks, rng);

      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].type).toBe('overtaking_capture');
      expect(result.debug.captureCount).toBe(1);
    });

    it('builds simple movement candidates', () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'movement',
        players: [createAIPlayer(1), createAIPlayer(2)],
      });

      const hooks = createMockHooks({
        getPlayerStacks: jest.fn().mockReturnValue([]),
        enumerateCaptureSegmentsFrom: jest.fn().mockReturnValue([]),
        enumerateSimpleMovementLandings: jest.fn().mockReturnValue([
          { fromKey: '2,2', to: pos(2, 4) },
          { fromKey: '2,2', to: pos(4, 2) },
        ]),
      });
      const rng = createMockRng();

      const result = buildSandboxMovementCandidates(gameState, hooks, rng);

      expect(result.candidates).toHaveLength(2);
      expect(result.candidates[0].type).toBe('move_stack');
      expect(result.debug.simpleMoveCount).toBe(2);
    });

    it('filters candidates by mustMoveFromStackKey', () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'movement',
        players: [createAIPlayer(1), createAIPlayer(2)],
      });

      const stack1 = {
        position: pos(2, 2),
        rings: [1, 1],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 1,
      };
      const stack2 = {
        position: pos(4, 4),
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      };

      const hooks = createMockHooks({
        getPlayerStacks: jest.fn().mockReturnValue([stack1, stack2]),
        getMustMoveFromStackKey: jest.fn().mockReturnValue('2,2'),
        enumerateCaptureSegmentsFrom: jest.fn().mockImplementation((position) => {
          // Only return captures from the must-move stack
          if (position.x === 2 && position.y === 2) {
            return [{ from: pos(2, 2), target: pos(3, 2), landing: pos(4, 2) }];
          }
          return [];
        }),
        enumerateSimpleMovementLandings: jest.fn().mockReturnValue([
          { fromKey: '2,2', to: pos(2, 4) },
          { fromKey: '4,4', to: pos(4, 6) }, // This should be filtered out
        ]),
      });
      const rng = createMockRng();

      const result = buildSandboxMovementCandidates(gameState, hooks, rng);

      // Only the stack at 2,2 should have its moves included
      expect(result.candidates.length).toBe(2); // 1 capture + 1 simple move from 2,2
      expect(result.candidates.every((c) => c.from?.x === 2 && c.from?.y === 2)).toBe(true);
    });

    it('handles hex positions with z coordinate in fromKey', () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'movement',
        boardType: 'hexagonal',
        players: [createAIPlayer(1), createAIPlayer(2)],
      });

      const hooks = createMockHooks({
        getPlayerStacks: jest.fn().mockReturnValue([]),
        enumerateCaptureSegmentsFrom: jest.fn().mockReturnValue([]),
        enumerateSimpleMovementLandings: jest
          .fn()
          .mockReturnValue([{ fromKey: '1,0,-1', to: { x: 2, y: 0, z: -2 } }]),
      });
      const rng = createMockRng();

      const result = buildSandboxMovementCandidates(gameState, hooks, rng);

      expect(result.candidates).toHaveLength(1);
      expect(result.candidates[0].from).toEqual({ x: 1, y: 0, z: -1 });
    });

    it('handles invalid fromKey format with fallback', () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'movement',
        players: [createAIPlayer(1), createAIPlayer(2)],
      });

      const hooks = createMockHooks({
        getPlayerStacks: jest.fn().mockReturnValue([]),
        enumerateCaptureSegmentsFrom: jest.fn().mockReturnValue([]),
        enumerateSimpleMovementLandings: jest
          .fn()
          .mockReturnValue([{ fromKey: 'invalid', to: pos(2, 4) }]),
      });
      const rng = createMockRng();

      const result = buildSandboxMovementCandidates(gameState, hooks, rng);

      expect(result.candidates).toHaveLength(1);
      // Invalid key should fallback to {0, 0}
      expect(result.candidates[0].from).toEqual({ x: 0, y: 0 });
    });
  });

  describe('selectSandboxMovementMove', () => {
    it('returns null when candidates are empty', () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        players: [createAIPlayer(1), createAIPlayer(2)],
      });
      const rng = createMockRng();

      const result = selectSandboxMovementMove(gameState, [], rng, false);

      expect(result).toBeNull();
    });

    it('selects a move in parity mode', () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        players: [createAIPlayer(1), createAIPlayer(2)],
      });
      const candidates: Move[] = [
        {
          type: 'move_stack',
          player: 1,
          from: pos(2, 2),
          to: pos(2, 4),
          id: '',
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ];
      const rng = createMockRng([0.1]);

      const result = selectSandboxMovementMove(gameState, candidates, rng, true);

      expect(result).not.toBeNull();
      expect(result?.type).toBe('move_stack');
    });

    it('selects a move in default (non-parity) mode', () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        players: [createAIPlayer(1), createAIPlayer(2)],
      });
      const candidates: Move[] = [
        {
          type: 'overtaking_capture',
          player: 1,
          from: pos(2, 2),
          captureTarget: pos(3, 2),
          to: pos(4, 2),
          id: '',
          timestamp: new Date(),
          thinkTime: 0,
          moveNumber: 1,
        },
      ];
      const rng = createMockRng([0.5]);

      const result = selectSandboxMovementMove(gameState, candidates, rng, false);

      expect(result).not.toBeNull();
      expect(result?.type).toBe('overtaking_capture');
    });
  });

  describe('maybeRunAITurnSandbox', () => {
    it('does nothing when current player is not AI', async () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        gameStatus: 'active',
        players: [createHumanPlayer(1), createAIPlayer(2)],
      });
      const hooks = createMockHooks({
        getGameState: jest.fn().mockReturnValue(gameState),
      });
      const rng = createMockRng();

      await maybeRunAITurnSandbox(hooks, rng);

      expect(hooks.applyCanonicalMove).not.toHaveBeenCalled();
    });

    it('does nothing when game is not active', async () => {
      const gameState = createTestGameState({
        currentPlayer: 1,
        currentPhase: 'ring_placement',
        gameStatus: 'completed',
        players: [createAIPlayer(1), createAIPlayer(2)],
      });
      const hooks = createMockHooks({
        getGameState: jest.fn().mockReturnValue(gameState),
      });
      const rng = createMockRng();

      await maybeRunAITurnSandbox(hooks, rng);

      expect(hooks.applyCanonicalMove).not.toHaveBeenCalled();
    });

    describe('ring_placement phase', () => {
      it('applies placement move when candidates are available', async () => {
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'ring_placement',
          gameStatus: 'active',
          players: [createAIPlayer(1, { ringsInHand: 18 }), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          enumerateLegalRingPlacements: jest.fn().mockReturnValue([pos(3, 3), pos(4, 4)]),
          getPlayerStacks: jest.fn().mockReturnValue([]),
          hasAnyLegalMoveOrCaptureFrom: jest.fn().mockReturnValue(true),
          createHypotheticalBoardWithPlacement: jest.fn().mockImplementation((board) => board),
        });
        const rng = createMockRng([0.1]);

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.applyCanonicalMove).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'place_ring',
            player: 1,
          })
        );
      });

      it('triggers forced elimination when no placements and cannot skip', async () => {
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'ring_placement',
          gameStatus: 'active',
          players: [createAIPlayer(1, { ringsInHand: 5 }), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          enumerateLegalRingPlacements: jest.fn().mockReturnValue([]),
          getPlayerStacks: jest.fn().mockReturnValue([]),
          maybeProcessForcedEliminationForCurrentPlayer: jest.fn().mockReturnValue(true),
        });
        const rng = createMockRng();

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.maybeProcessForcedEliminationForCurrentPlayer).toHaveBeenCalled();
      });

      it('handles zero rings in hand by attempting forced elimination', async () => {
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'ring_placement',
          gameStatus: 'active',
          players: [createAIPlayer(1, { ringsInHand: 0 }), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          getPlayerStacks: jest.fn().mockReturnValue([]),
          maybeProcessForcedEliminationForCurrentPlayer: jest.fn().mockReturnValue(false),
        });
        const rng = createMockRng();

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.maybeProcessForcedEliminationForCurrentPlayer).toHaveBeenCalled();
      });

      it('applies skip_placement when no placements but can skip', async () => {
        const board = createTestBoard('square8');
        board.stacks.set(posStr(3, 3), {
          position: pos(3, 3),
          rings: [1, 1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 1,
        });

        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'ring_placement',
          gameStatus: 'active',
          board,
          players: [createAIPlayer(1, { ringsInHand: 5 }), createAIPlayer(2)],
        });

        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          enumerateLegalRingPlacements: jest.fn().mockReturnValue([]), // No placements
          getPlayerStacks: jest
            .fn()
            .mockReturnValue([
              {
                position: pos(3, 3),
                rings: [1, 1],
                stackHeight: 2,
                capHeight: 2,
                controllingPlayer: 1,
              },
            ]),
          hasAnyLegalMoveOrCaptureFrom: jest.fn().mockReturnValue(true), // Can move from stack
        });
        const rng = createMockRng();

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.applyCanonicalMove).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'skip_placement',
          })
        );
      });
    });

    describe('movement phase', () => {
      it('applies move_stack when only simple moves available', async () => {
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'movement',
          gameStatus: 'active',
          players: [createAIPlayer(1), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          getValidMovesForCurrentPlayer: jest.fn().mockReturnValue([
            {
              type: 'move_stack',
              player: 1,
              from: pos(2, 2),
              to: pos(2, 4),
              id: '',
              timestamp: new Date(),
              thinkTime: 0,
              moveNumber: 1,
            },
          ]),
        });
        const rng = createMockRng([0.1]);

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.applyCanonicalMove).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'move_stack',
            player: 1,
          })
        );
      });

      it('applies overtaking_capture when available', async () => {
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'movement',
          gameStatus: 'active',
          players: [createAIPlayer(1), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          getValidMovesForCurrentPlayer: jest.fn().mockReturnValue([
            {
              type: 'overtaking_capture',
              player: 1,
              from: pos(2, 2),
              captureTarget: pos(3, 2),
              to: pos(4, 2),
              id: '',
              timestamp: new Date(),
              thinkTime: 0,
              moveNumber: 1,
            },
          ]),
          enumerateCaptureSegmentsFrom: jest.fn().mockReturnValue([]), // No chain captures
        });
        const rng = createMockRng([0.1]);

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.applyCanonicalMove).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'overtaking_capture',
          })
        );
      });

      it('triggers forced elimination when no moves available', async () => {
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'movement',
          gameStatus: 'active',
          players: [createAIPlayer(1), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          getValidMovesForCurrentPlayer: jest.fn().mockReturnValue([]),
          getPlayerStacks: jest.fn().mockReturnValue([]),
          enumerateCaptureSegmentsFrom: jest.fn().mockReturnValue([]),
          enumerateSimpleMovementLandings: jest.fn().mockReturnValue([]),
          maybeProcessForcedEliminationForCurrentPlayer: jest.fn().mockReturnValue(true),
        });
        const rng = createMockRng();

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.maybeProcessForcedEliminationForCurrentPlayer).toHaveBeenCalled();
      });

      it('uses fallback when getValidMoves returns empty but helpers have moves', async () => {
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'movement',
          gameStatus: 'active',
          players: [createAIPlayer(1), createAIPlayer(2)],
        });

        const stack = {
          position: pos(3, 3),
          rings: [1, 1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 1,
        };

        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          // getValidMovesForCurrentPlayer returns empty to trigger fallback
          getValidMovesForCurrentPlayer: jest.fn().mockReturnValue([]),
          getPlayerStacks: jest.fn().mockReturnValue([stack]),
          enumerateCaptureSegmentsFrom: jest.fn().mockReturnValue([]),
          enumerateSimpleMovementLandings: jest
            .fn()
            .mockReturnValue([{ fromKey: '3,3', to: pos(3, 5) }]),
        });
        const rng = createMockRng([0.1]);

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.applyCanonicalMove).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'move_stack',
          })
        );
      });
    });

    describe('line_processing phase', () => {
      it('applies process_line move when available', async () => {
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'line_processing',
          gameStatus: 'active',
          players: [createAIPlayer(1), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          getValidMovesForCurrentPlayer: jest.fn().mockReturnValue([
            {
              type: 'process_line',
              player: 1,
              id: '',
              timestamp: new Date(),
              thinkTime: 0,
              moveNumber: 1,
            },
          ]),
          hasPendingLineRewardElimination: jest.fn().mockReturnValue(false),
        });
        const rng = createMockRng([0.1]);

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.applyCanonicalMove).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'process_line',
          })
        );
      });

      it('applies eliminate_rings_from_stack when pending line reward elimination', async () => {
        const board = createTestBoard('square8');
        board.stacks.set(posStr(3, 3), {
          position: pos(3, 3),
          rings: [1, 1],
          stackHeight: 2,
          capHeight: 2,
          controllingPlayer: 1,
        });

        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'line_processing',
          gameStatus: 'active',
          board,
          players: [createAIPlayer(1), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          hasPendingLineRewardElimination: jest.fn().mockReturnValue(true),
          getPlayerStacks: jest
            .fn()
            .mockReturnValue([
              {
                position: pos(3, 3),
                rings: [1, 1],
                stackHeight: 2,
                capHeight: 2,
                controllingPlayer: 1,
              },
            ]),
        });
        const rng = createMockRng([0.1]);

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.applyCanonicalMove).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'eliminate_rings_from_stack',
          })
        );
      });

      it('does nothing when no decision candidates available', async () => {
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'line_processing',
          gameStatus: 'active',
          players: [createAIPlayer(1), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
          getValidMovesForCurrentPlayer: jest.fn().mockReturnValue([]),
          hasPendingLineRewardElimination: jest.fn().mockReturnValue(false),
        });
        const rng = createMockRng();

        await maybeRunAITurnSandbox(hooks, rng);

        expect(hooks.applyCanonicalMove).not.toHaveBeenCalled();
      });
    });

    // Note: territory_processing phase tests are covered by integration tests
    // (SandboxAI.ringPlacementNoopRegression.test.ts) due to module-level state
    // making isolated unit tests unreliable.

    describe('other phases', () => {
      it('does nothing when in chain_capture phase initially', async () => {
        // chain_capture is only handled as continuation after overtaking_capture
        const gameState = createTestGameState({
          currentPlayer: 1,
          currentPhase: 'chain_capture',
          gameStatus: 'active',
          players: [createAIPlayer(1), createAIPlayer(2)],
        });
        const hooks = createMockHooks({
          getGameState: jest.fn().mockReturnValue(gameState),
        });
        const rng = createMockRng();

        await maybeRunAITurnSandbox(hooks, rng);

        // chain_capture is not handled as an initial phase
        expect(hooks.applyCanonicalMove).not.toHaveBeenCalled();
      });
    });
  });
});
