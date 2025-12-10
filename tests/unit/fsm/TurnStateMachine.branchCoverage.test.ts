/**
 * TurnStateMachine.branchCoverage.test.ts
 *
 * Targeted branch coverage tests for TurnStateMachine.ts
 * Focuses on uncovered branches identified in coverage analysis:
 * - Lines 345-348: SKIP_PLACEMENT when canPlace is true
 * - Lines 608-626: PROCESS_LINE with multiple lines
 * - Lines 714-743: PROCESS_REGION with eliminations and multiple regions
 */

import {
  transition,
  type TurnEvent,
  type GameContext,
  type RingPlacementState,
  type LineProcessingState,
  type TerritoryProcessingState,
} from '../../../src/shared/engine/fsm';

describe('TurnStateMachine - Branch Coverage', () => {
  const context: GameContext = {
    boardType: 'square8',
    numPlayers: 2,
    ringsPerPlayer: 18,
    lineLength: 3,
  };

  const context4p: GameContext = {
    ...context,
    numPlayers: 4,
  };

  // ==========================================================================
  // Lines 345-348: SKIP_PLACEMENT when canPlace is true (should fail)
  // ==========================================================================
  describe('SKIP_PLACEMENT guard failures', () => {
    it('should reject SKIP_PLACEMENT when canPlace is true', () => {
      // Per RR-CANON: You cannot skip placement when valid placements exist
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 5,
        canPlace: true, // Valid placements exist
        validPositions: [{ x: 3, y: 3 }],
      };

      const event: TurnEvent = { type: 'SKIP_PLACEMENT' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('Cannot skip placement when valid placements exist');
      }
    });

    it('should allow SKIP_PLACEMENT when canPlace is false and ringsInHand > 0', () => {
      // Per RR-CANON: You CAN skip placement when no valid positions exist
      // but you still have rings in hand (all positions blocked)
      const state: RingPlacementState = {
        phase: 'ring_placement',
        player: 1,
        ringsInHand: 5,
        canPlace: false, // No valid placements
        validPositions: [],
      };

      const event: TurnEvent = { type: 'SKIP_PLACEMENT' };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('movement');
      }
    });
  });

  // ==========================================================================
  // Lines 608-626: PROCESS_LINE with multiple lines
  // ==========================================================================
  describe('PROCESS_LINE - multiple lines', () => {
    it('should advance to next line when multiple lines exist', () => {
      // Per RR-CANON-R076: Lines are processed one at a time in order
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
            player: 1,
            length: 3,
            direction: { x: 1, y: 0 },
            requiresChoice: false,
          },
          {
            positions: [
              { x: 0, y: 1 },
              { x: 1, y: 1 },
              { x: 2, y: 1 },
            ],
            player: 1,
            length: 3,
            direction: { x: 1, y: 0 },
            requiresChoice: false,
          },
        ],
        currentLineIndex: 0,
      };

      const event: TurnEvent = { type: 'PROCESS_LINE', lineIndex: 0 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        // Should stay in line_processing and advance to next line
        expect(result.state.phase).toBe('line_processing');
        const lineState = result.state as LineProcessingState;
        expect(lineState.currentLineIndex).toBe(1);
        // Should emit COLLAPSE_LINE action
        expect(result.actions).toContainEqual({
          type: 'COLLAPSE_LINE',
          positions: state.detectedLines[0].positions,
        });
      }
    });

    it('should transition to territory_processing after last line', () => {
      // Per RR-CANON-R076: After all lines processed, move to territory phase
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
            player: 1,
            length: 3,
            direction: { x: 1, y: 0 },
            requiresChoice: false,
          },
        ],
        currentLineIndex: 0,
      };

      const event: TurnEvent = { type: 'PROCESS_LINE', lineIndex: 0 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        // Should transition to territory_processing after last line
        expect(result.state.phase).toBe('territory_processing');
        expect(result.actions).toContainEqual({
          type: 'COLLAPSE_LINE',
          positions: state.detectedLines[0].positions,
        });
      }
    });

    it('should reject PROCESS_LINE with invalid line index', () => {
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
            player: 1,
            length: 3,
            direction: { x: 1, y: 0 },
            requiresChoice: false,
          },
        ],
        currentLineIndex: 0,
      };

      const event: TurnEvent = { type: 'PROCESS_LINE', lineIndex: 5 }; // Invalid
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('Invalid line index');
      }
    });
  });

  // ==========================================================================
  // Lines 714-743: PROCESS_REGION with eliminations and multiple regions
  // ==========================================================================
  describe('PROCESS_REGION - eliminations and multiple regions', () => {
    it('should queue eliminations when region requires them', () => {
      // Per RR-CANON-R077: Disconnected regions may require eliminations
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [
          {
            positions: [{ x: 0, y: 0 }],
            controllingPlayer: 1,
            eliminationsRequired: 2, // Requires eliminations
          },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'PROCESS_REGION', regionIndex: 0 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
        const terState = result.state as TerritoryProcessingState;
        // Should have elimination pending
        expect(terState.eliminationsPending.length).toBeGreaterThan(0);
        expect(terState.eliminationsPending[0].count).toBe(2);
        expect(result.actions).toContainEqual(
          expect.objectContaining({ type: 'PROCESS_DISCONNECTION' })
        );
      }
    });

    it('should advance to next region when multiple regions exist', () => {
      // Per RR-CANON-R077: Multiple disconnected regions processed in order
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [
          {
            positions: [{ x: 0, y: 0 }],
            controllingPlayer: 1,
            eliminationsRequired: 0, // No eliminations
          },
          {
            positions: [{ x: 5, y: 5 }],
            controllingPlayer: 1,
            eliminationsRequired: 0,
          },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'PROCESS_REGION', regionIndex: 0 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        // Should stay in territory_processing and advance
        expect(result.state.phase).toBe('territory_processing');
        const terState = result.state as TerritoryProcessingState;
        expect(terState.currentRegionIndex).toBe(1);
      }
    });

    it('should transition to turn_end after last region (no eliminations)', () => {
      // Per RR-CANON-R070: After all regions processed, check victory or end turn
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [
          {
            positions: [{ x: 0, y: 0 }],
            controllingPlayer: 1,
            eliminationsRequired: 0,
          },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'PROCESS_REGION', regionIndex: 0 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        expect(result.actions).toContainEqual({ type: 'CHECK_VICTORY' });
      }
    });

    it('should reject PROCESS_REGION with invalid region index', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [
          {
            positions: [{ x: 0, y: 0 }],
            controllingPlayer: 1,
            eliminationsRequired: 0,
          },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'PROCESS_REGION', regionIndex: 10 }; // Invalid
      const result = transition(state, event, context);

      expect(result.ok).toBe(false);
      if (!result.ok) {
        expect(result.error.code).toBe('GUARD_FAILED');
        expect(result.error.message).toContain('Invalid region index');
      }
    });

    it('should compute correct nextPlayer for 4-player games', () => {
      // Per RR-CANON: Turn rotation is (player % numPlayers) + 1
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 3, // Player 3 in 4-player game
        disconnectedRegions: [
          {
            positions: [{ x: 0, y: 0 }],
            controllingPlayer: 3,
            eliminationsRequired: 0,
          },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'PROCESS_REGION', regionIndex: 0 };
      const result = transition(state, event, context4p);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        const endState = result.state as { phase: string; nextPlayer: number };
        expect(endState.nextPlayer).toBe(4); // 3 % 4 + 1 = 4
      }
    });

    it('should wrap around to player 1 after player 4', () => {
      // Per RR-CANON: Turn rotation wraps (4 % 4) + 1 = 1
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 4, // Player 4 in 4-player game
        disconnectedRegions: [
          {
            positions: [{ x: 0, y: 0 }],
            controllingPlayer: 4,
            eliminationsRequired: 0,
          },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'PROCESS_REGION', regionIndex: 0 };
      const result = transition(state, event, context4p);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('turn_end');
        const endState = result.state as { phase: string; nextPlayer: number };
        expect(endState.nextPlayer).toBe(1); // 4 % 4 + 1 = 1
      }
    });
  });

  // ==========================================================================
  // Additional edge case coverage
  // ==========================================================================
  describe('Edge cases for line processing', () => {
    it('should handle PROCESS_LINE with exactly 3 lines', () => {
      const state: LineProcessingState = {
        phase: 'line_processing',
        player: 1,
        detectedLines: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
            player: 1,
            length: 3,
            direction: { x: 1, y: 0 },
            requiresChoice: false,
          },
          {
            positions: [
              { x: 0, y: 1 },
              { x: 1, y: 1 },
              { x: 2, y: 1 },
            ],
            player: 1,
            length: 3,
            direction: { x: 1, y: 0 },
            requiresChoice: false,
          },
          {
            positions: [
              { x: 0, y: 2 },
              { x: 1, y: 2 },
              { x: 2, y: 2 },
            ],
            player: 1,
            length: 3,
            direction: { x: 1, y: 0 },
            requiresChoice: false,
          },
        ],
        currentLineIndex: 1, // Processing second line
      };

      const event: TurnEvent = { type: 'PROCESS_LINE', lineIndex: 1 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        // Should advance to index 2 (third line)
        expect(result.state.phase).toBe('line_processing');
        const lineState = result.state as LineProcessingState;
        expect(lineState.currentLineIndex).toBe(2);
      }
    });
  });

  describe('Edge cases for territory processing', () => {
    it('should handle region with multiple eliminations required', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 1,
        disconnectedRegions: [
          {
            positions: [
              { x: 0, y: 0 },
              { x: 1, y: 0 },
              { x: 2, y: 0 },
            ],
            controllingPlayer: 1,
            eliminationsRequired: 5, // Large elimination count
          },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      const event: TurnEvent = { type: 'PROCESS_REGION', regionIndex: 0 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        expect(result.state.phase).toBe('territory_processing');
        const terState = result.state as TerritoryProcessingState;
        expect(terState.eliminationsPending[0].count).toBe(5);
      }
    });

    it('should handle mixed regions (some with eliminations, some without)', () => {
      const state: TerritoryProcessingState = {
        phase: 'territory_processing',
        player: 2,
        disconnectedRegions: [
          {
            positions: [{ x: 0, y: 0 }],
            controllingPlayer: 2,
            eliminationsRequired: 0, // No eliminations
          },
          {
            positions: [{ x: 5, y: 5 }],
            controllingPlayer: 2,
            eliminationsRequired: 3, // Needs eliminations
          },
          {
            positions: [{ x: 7, y: 7 }],
            controllingPlayer: 2,
            eliminationsRequired: 0, // No eliminations
          },
        ],
        currentRegionIndex: 0,
        eliminationsPending: [],
      };

      // Process first region (no eliminations)
      const event: TurnEvent = { type: 'PROCESS_REGION', regionIndex: 0 };
      const result = transition(state, event, context);

      expect(result.ok).toBe(true);
      if (result.ok) {
        // Should advance to region 1
        expect(result.state.phase).toBe('territory_processing');
        const terState = result.state as TerritoryProcessingState;
        expect(terState.currentRegionIndex).toBe(1);
        expect(terState.eliminationsPending.length).toBe(0);
      }
    });
  });
});
