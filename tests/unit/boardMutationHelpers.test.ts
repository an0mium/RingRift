/**
 * Board Mutation Helpers Unit Tests
 *
 * Tests for in-place board state mutation utilities.
 * These helpers preserve object references while replacing contents.
 */

import {
  replaceMapContents,
  replaceArrayContents,
  replaceObjectContents,
  copyBoardStateInPlace,
} from '../../src/shared/engine/boardMutationHelpers';
import type { BoardState, RingStack, FormedLine } from '../../src/shared/types/game';

describe('boardMutationHelpers', () => {
  describe('replaceMapContents', () => {
    it('should clear target and copy all entries from source', () => {
      const target = new Map<string, number>([
        ['a', 1],
        ['b', 2],
      ]);
      const source = new Map<string, number>([
        ['x', 10],
        ['y', 20],
        ['z', 30],
      ]);

      replaceMapContents(target, source);

      expect(target.size).toBe(3);
      expect(target.get('x')).toBe(10);
      expect(target.get('y')).toBe(20);
      expect(target.get('z')).toBe(30);
      expect(target.has('a')).toBe(false);
      expect(target.has('b')).toBe(false);
    });

    it('should preserve target Map reference', () => {
      const target = new Map<string, number>();
      const originalRef = target;
      const source = new Map([['key', 42]]);

      replaceMapContents(target, source);

      expect(target).toBe(originalRef);
    });

    it('should handle empty source', () => {
      const target = new Map<string, number>([['existing', 1]]);
      const source = new Map<string, number>();

      replaceMapContents(target, source);

      expect(target.size).toBe(0);
    });

    it('should handle empty target', () => {
      const target = new Map<string, number>();
      const source = new Map([['new', 99]]);

      replaceMapContents(target, source);

      expect(target.size).toBe(1);
      expect(target.get('new')).toBe(99);
    });

    it('should work with complex value types', () => {
      const target = new Map<string, { data: number[] }>();
      const source = new Map<string, { data: number[] }>([
        ['obj1', { data: [1, 2, 3] }],
        ['obj2', { data: [4, 5] }],
      ]);

      replaceMapContents(target, source);

      expect(target.get('obj1')?.data).toEqual([1, 2, 3]);
      expect(target.get('obj2')?.data).toEqual([4, 5]);
    });
  });

  describe('replaceArrayContents', () => {
    it('should clear target and copy all elements from source', () => {
      const target = [1, 2, 3];
      const source = [10, 20, 30, 40];

      replaceArrayContents(target, source);

      expect(target).toEqual([10, 20, 30, 40]);
    });

    it('should preserve target array reference', () => {
      const target = [1, 2];
      const originalRef = target;
      const source = [5, 6, 7];

      replaceArrayContents(target, source);

      expect(target).toBe(originalRef);
    });

    it('should handle empty source', () => {
      const target = [1, 2, 3];
      const source: number[] = [];

      replaceArrayContents(target, source);

      expect(target).toEqual([]);
      expect(target.length).toBe(0);
    });

    it('should handle empty target', () => {
      const target: number[] = [];
      const source = [1, 2, 3];

      replaceArrayContents(target, source);

      expect(target).toEqual([1, 2, 3]);
    });

    it('should work with object elements', () => {
      const target: { id: number }[] = [{ id: 1 }];
      const source = [{ id: 10 }, { id: 20 }];

      replaceArrayContents(target, source);

      expect(target).toEqual([{ id: 10 }, { id: 20 }]);
    });
  });

  describe('replaceObjectContents', () => {
    it('should clear target and copy all properties from source', () => {
      const target = { a: 1, b: 2 } as Record<string, number>;
      const source = { x: 10, y: 20, z: 30 } as Record<string, number>;

      replaceObjectContents(target, source);

      expect(target).toEqual({ x: 10, y: 20, z: 30 });
      expect(target.a).toBeUndefined();
      expect(target.b).toBeUndefined();
    });

    it('should preserve target object reference', () => {
      const target = { key: 1 };
      const originalRef = target;
      const source = { newKey: 99 };

      replaceObjectContents(target, source);

      expect(target).toBe(originalRef);
    });

    it('should handle empty source', () => {
      const target = { existing: 1 } as Record<string, number>;
      const source = {} as Record<string, number>;

      replaceObjectContents(target, source);

      expect(Object.keys(target)).toEqual([]);
    });

    it('should handle empty target', () => {
      const target = {} as Record<string, number>;
      const source = { new: 42 };

      replaceObjectContents(target, source);

      expect(target).toEqual({ new: 42 });
    });

    it('should work with nested objects', () => {
      const target = { old: { nested: 1 } } as Record<string, unknown>;
      const source = { data: { value: [1, 2, 3] } };

      replaceObjectContents(target, source);

      expect(target).toEqual({ data: { value: [1, 2, 3] } });
    });

    it('should work with numeric keys', () => {
      const target = { 1: 'a', 2: 'b' } as Record<number, string>;
      const source = { 3: 'c', 4: 'd' } as Record<number, string>;

      replaceObjectContents(target, source);

      expect(target).toEqual({ 3: 'c', 4: 'd' });
    });
  });

  describe('copyBoardStateInPlace', () => {
    const createEmptyBoardState = (): BoardState =>
      ({
        type: 'square8',
        size: 8,
        stacks: new Map(),
        markers: new Map(),
        collapsedSpaces: new Map(),
        territories: new Map(),
        formedLines: [],
        eliminatedRings: {},
      }) as BoardState;

    const createPopulatedBoardState = (): BoardState =>
      ({
        type: 'square8',
        size: 10,
        stacks: new Map([
          ['0,0', { controllingPlayer: 1, stackHeight: 3 } as RingStack],
          ['1,1', { controllingPlayer: 2, stackHeight: 2 } as RingStack],
        ]),
        markers: new Map([
          ['2,2', { player: 1, placed: true }],
          ['3,3', { player: 2, placed: true }],
        ]),
        collapsedSpaces: new Map([['4,4', { collapsedAt: 5 }]]),
        territories: new Map([['territory-1', { id: 'territory-1', spaces: [] }]]),
        formedLines: [
          { player: 1, length: 4, positions: [], direction: { x: 1, y: 0 } } as FormedLine,
        ],
        eliminatedRings: { 1: 2, 2: 3 },
      }) as BoardState;

    it('should copy all Map contents from source to target', () => {
      const target = createEmptyBoardState();
      const source = createPopulatedBoardState();

      copyBoardStateInPlace(target, source);

      expect(target.stacks.size).toBe(2);
      expect(target.markers.size).toBe(2);
      expect(target.collapsedSpaces.size).toBe(1);
      expect(target.territories.size).toBe(1);
    });

    it('should copy array contents from source to target', () => {
      const target = createEmptyBoardState();
      const source = createPopulatedBoardState();

      copyBoardStateInPlace(target, source);

      expect(target.formedLines).toHaveLength(1);
      expect(target.formedLines[0].player).toBe(1);
    });

    it('should copy eliminatedRings object', () => {
      const target = createEmptyBoardState();
      const source = createPopulatedBoardState();

      copyBoardStateInPlace(target, source);

      expect(target.eliminatedRings).toEqual({ 1: 2, 2: 3 });
    });

    it('should copy primitive values like size', () => {
      const target = createEmptyBoardState();
      const source = createPopulatedBoardState();

      copyBoardStateInPlace(target, source);

      expect(target.size).toBe(10);
    });

    it('should preserve target object references for Maps', () => {
      const target = createEmptyBoardState();
      const originalStacksRef = target.stacks;
      const originalMarkersRef = target.markers;
      const source = createPopulatedBoardState();

      copyBoardStateInPlace(target, source);

      expect(target.stacks).toBe(originalStacksRef);
      expect(target.markers).toBe(originalMarkersRef);
    });

    it('should preserve target object reference for formedLines array', () => {
      const target = createEmptyBoardState();
      const originalArrayRef = target.formedLines;
      const source = createPopulatedBoardState();

      copyBoardStateInPlace(target, source);

      expect(target.formedLines).toBe(originalArrayRef);
    });

    it('should preserve target object reference for eliminatedRings', () => {
      const target = createEmptyBoardState();
      const originalObjRef = target.eliminatedRings;
      const source = createPopulatedBoardState();

      copyBoardStateInPlace(target, source);

      expect(target.eliminatedRings).toBe(originalObjRef);
    });

    it('should handle copying to a populated target', () => {
      const target = createPopulatedBoardState();
      const source = createEmptyBoardState();

      copyBoardStateInPlace(target, source);

      expect(target.stacks.size).toBe(0);
      expect(target.markers.size).toBe(0);
      expect(target.formedLines).toHaveLength(0);
      expect(target.size).toBe(8);
    });

    it('should correctly copy stack data', () => {
      const target = createEmptyBoardState();
      const source = createPopulatedBoardState();

      copyBoardStateInPlace(target, source);

      const stack = target.stacks.get('0,0') as RingStack;
      expect(stack.controllingPlayer).toBe(1);
      expect(stack.stackHeight).toBe(3);
    });
  });
});
