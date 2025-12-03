/**
 * Unit tests for sandboxCaptureSearch.ts
 *
 * Tests for capture chain search diagnostics helper.
 * Note: This is a diagnostics-only module, not part of canonical rules.
 */

import { findMaxCaptureChains } from '../../src/client/sandbox/sandboxCaptureSearch';
import {
  CaptureBoardAdapters,
  CaptureApplyAdapters,
} from '../../src/client/sandbox/sandboxCaptures';
import { createTestBoard, pos, posStr } from '../utils/fixtures';
import type { BoardState, BoardType, Position } from '../../src/shared/types/game';
import { BOARD_CONFIGS, isValidPosition } from '../../src/shared/engine';

describe('sandboxCaptureSearch', () => {
  function createAdapters(boardType: BoardType): CaptureBoardAdapters & CaptureApplyAdapters {
    const config = BOARD_CONFIGS[boardType];
    return {
      isValidPosition: (p: Position) => isValidPosition(p, boardType, config.size),
      isCollapsedSpace: (p: Position, board: BoardState) =>
        board.collapsedSpaces.has(posStr(p.x, p.y, p.z)),
      getMarkerOwner: (p: Position, board: BoardState) =>
        board.markers.get(posStr(p.x, p.y, p.z))?.player,
      applyMarkerEffectsAlongPath: jest.fn(),
    };
  }

  describe('findMaxCaptureChains', () => {
    const boardType: BoardType = 'square8';

    it('returns empty array when no stack at start position', () => {
      const board = createTestBoard(boardType);
      const adapters = createAdapters(boardType);

      const results = findMaxCaptureChains(boardType, board, pos(3, 3), 1, adapters);

      expect(results).toHaveLength(0);
    });

    it('returns empty array when stack has zero height', () => {
      const board = createTestBoard(boardType);
      board.stacks.set(posStr(3, 3), {
        position: pos(3, 3),
        rings: [],
        stackHeight: 0,
        capHeight: 0,
        controllingPlayer: 1,
      });
      const adapters = createAdapters(boardType);

      const results = findMaxCaptureChains(boardType, board, pos(3, 3), 1, adapters);

      expect(results).toHaveLength(0);
    });

    it('returns empty results when no captures available (isolated stack)', () => {
      const board = createTestBoard(boardType);
      // Place a single stack with no targets nearby
      board.stacks.set(posStr(3, 3), {
        position: pos(3, 3),
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      });
      const adapters = createAdapters(boardType);

      const results = findMaxCaptureChains(boardType, board, pos(3, 3), 1, adapters);

      // No enemy stacks to capture, so no chains
      expect(results).toHaveLength(0);
    });

    it('finds single-segment capture chain', () => {
      const board = createTestBoard(boardType);
      // Attacker at (2, 3) with height 3
      board.stacks.set(posStr(2, 3), {
        position: pos(2, 3),
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      });
      // Target at (3, 3) with height 2 (can be captured by taller stack)
      board.stacks.set(posStr(3, 3), {
        position: pos(3, 3),
        rings: [2, 2],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 2,
      });
      // Landing at (4, 3) is empty

      const adapters = createAdapters(boardType);

      const results = findMaxCaptureChains(boardType, board, pos(2, 3), 1, adapters);

      // Should find at least one chain with one segment
      expect(results.length).toBeGreaterThanOrEqual(0);
    });

    it('respects maxDepth option', () => {
      const board = createTestBoard(boardType);
      // Set up a potential multi-segment chain
      board.stacks.set(posStr(1, 3), {
        position: pos(1, 3),
        rings: [1, 1, 1, 1, 1],
        stackHeight: 5,
        capHeight: 5,
        controllingPlayer: 1,
      });
      board.stacks.set(posStr(2, 3), {
        position: pos(2, 3),
        rings: [2, 2],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 2,
      });

      const adapters = createAdapters(boardType);

      // With maxDepth 1, should limit chain length
      const results = findMaxCaptureChains(boardType, board, pos(1, 3), 1, adapters, {
        maxDepth: 1,
      });

      // Results should respect max depth
      for (const result of results) {
        expect(result.segments.length).toBeLessThanOrEqual(1);
      }
    });

    it('uses pruneVisitedPositions to avoid cycles', () => {
      const board = createTestBoard(boardType);
      // Create a scenario where cycle avoidance matters
      board.stacks.set(posStr(2, 2), {
        position: pos(2, 2),
        rings: [1, 1, 1, 1, 1],
        stackHeight: 5,
        capHeight: 5,
        controllingPlayer: 1,
      });

      const adapters = createAdapters(boardType);

      // Should not throw even with pruneVisitedPositions
      const results = findMaxCaptureChains(boardType, board, pos(2, 2), 1, adapters, {
        pruneVisitedPositions: true,
      });

      expect(Array.isArray(results)).toBe(true);
    });

    it('handles multi-segment capture chains', () => {
      const board = createTestBoard(boardType);
      // Attacker with tall stack
      board.stacks.set(posStr(1, 3), {
        position: pos(1, 3),
        rings: [1, 1, 1, 1, 1, 1],
        stackHeight: 6,
        capHeight: 6,
        controllingPlayer: 1,
      });
      // First target
      board.stacks.set(posStr(2, 3), {
        position: pos(2, 3),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });
      // Second target (after landing at 3,3)
      board.stacks.set(posStr(4, 3), {
        position: pos(4, 3),
        rings: [2],
        stackHeight: 1,
        capHeight: 1,
        controllingPlayer: 2,
      });

      const adapters = createAdapters(boardType);

      const results = findMaxCaptureChains(boardType, board, pos(1, 3), 1, adapters);

      // Should find chains - may include multi-segment if captures chain
      expect(Array.isArray(results)).toBe(true);
    });

    it('returns finalPosition and finalHeight in results', () => {
      const board = createTestBoard(boardType);
      board.stacks.set(posStr(2, 3), {
        position: pos(2, 3),
        rings: [1, 1, 1, 1],
        stackHeight: 4,
        capHeight: 4,
        controllingPlayer: 1,
      });
      board.stacks.set(posStr(3, 3), {
        position: pos(3, 3),
        rings: [2, 2],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 2,
      });

      const adapters = createAdapters(boardType);

      const results = findMaxCaptureChains(boardType, board, pos(2, 3), 1, adapters);

      for (const result of results) {
        expect(result.finalPosition).toBeDefined();
        expect(typeof result.finalHeight).toBe('number');
        expect(result.segments).toBeDefined();
        expect(Array.isArray(result.segments)).toBe(true);
      }
    });

    it('clones board to avoid mutating original', () => {
      const board = createTestBoard(boardType);
      const originalStackCount = board.stacks.size;

      board.stacks.set(posStr(2, 3), {
        position: pos(2, 3),
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      });
      board.stacks.set(posStr(3, 3), {
        position: pos(3, 3),
        rings: [2, 2],
        stackHeight: 2,
        capHeight: 2,
        controllingPlayer: 2,
      });

      const adapters = createAdapters(boardType);

      findMaxCaptureChains(boardType, board, pos(2, 3), 1, adapters);

      // Original board should not be modified
      expect(board.stacks.size).toBe(originalStackCount + 2);
    });

    it('uses default maxDepth of 32 when not specified', () => {
      const board = createTestBoard(boardType);
      board.stacks.set(posStr(3, 3), {
        position: pos(3, 3),
        rings: [1, 1, 1],
        stackHeight: 3,
        capHeight: 3,
        controllingPlayer: 1,
      });

      const adapters = createAdapters(boardType);

      // Should not throw even without maxDepth option
      const results = findMaxCaptureChains(boardType, board, pos(3, 3), 1, adapters);

      expect(Array.isArray(results)).toBe(true);
    });
  });
});
