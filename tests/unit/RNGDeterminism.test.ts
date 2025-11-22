/**
 * RNG Determinism Tests
 *
 * Verifies that the SeededRNG implementation produces consistent,
 * deterministic sequences across multiple instantiations with the
 * same seed and different sequences with different seeds.
 */

import { SeededRNG, generateGameSeed } from '../../src/shared/utils/rng';

describe('SeededRNG Determinism', () => {
  describe('Basic Determinism', () => {
    it('should produce identical sequences with same seed', () => {
      const seed = 42;
      const rng1 = new SeededRNG(seed);
      const rng2 = new SeededRNG(seed);

      for (let i = 0; i < 100; i++) {
        expect(rng1.next()).toBe(rng2.next());
      }
    });

    it('should produce different sequences with different seeds', () => {
      const rng1 = new SeededRNG(42);
      const rng2 = new SeededRNG(43);

      let different = false;
      for (let i = 0; i < 100; i++) {
        if (rng1.next() !== rng2.next()) {
          different = true;
          break;
        }
      }
      expect(different).toBe(true);
    });

    it('should produce values in [0, 1) range', () => {
      const rng = new SeededRNG(12345);

      for (let i = 0; i < 1000; i++) {
        const val = rng.next();
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(1);
      }
    });
  });

  describe('nextInt', () => {
    it('should produce integers in specified range', () => {
      const rng = new SeededRNG(42);
      const min = 10;
      const max = 20;

      for (let i = 0; i < 100; i++) {
        const val = rng.nextInt(min, max);
        expect(val).toBeGreaterThanOrEqual(min);
        expect(val).toBeLessThan(max);
        expect(Number.isInteger(val)).toBe(true);
      }
    });

    it('should produce deterministic integer sequences', () => {
      const rng1 = new SeededRNG(777);
      const rng2 = new SeededRNG(777);

      for (let i = 0; i < 50; i++) {
        expect(rng1.nextInt(0, 100)).toBe(rng2.nextInt(0, 100));
      }
    });
  });

  describe('shuffle', () => {
    it('should produce deterministic shuffles', () => {
      const arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const arr2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      const rng1 = new SeededRNG(999);
      const rng2 = new SeededRNG(999);

      rng1.shuffle(arr1);
      rng2.shuffle(arr2);

      expect(arr1).toEqual(arr2);
    });

    it('should produce different shuffles with different seeds', () => {
      const arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const arr2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      const rng1 = new SeededRNG(100);
      const rng2 = new SeededRNG(200);

      rng1.shuffle(arr1);
      rng2.shuffle(arr2);

      expect(arr1).not.toEqual(arr2);
    });

    it('should preserve array elements', () => {
      const original = [1, 2, 3, 4, 5];
      const arr = [...original];

      const rng = new SeededRNG(42);
      rng.shuffle(arr);

      expect(arr.sort()).toEqual(original.sort());
    });
  });

  describe('choice', () => {
    it('should select deterministic elements', () => {
      const array = ['a', 'b', 'c', 'd', 'e'];
      const rng1 = new SeededRNG(555);
      const rng2 = new SeededRNG(555);

      for (let i = 0; i < 20; i++) {
        expect(rng1.choice(array)).toBe(rng2.choice(array));
      }
    });

    it('should throw on empty array', () => {
      const rng = new SeededRNG(42);
      expect(() => rng.choice([])).toThrow('Cannot select from empty array');
    });

    it('should always return the single element for 1-element array', () => {
      const rng = new SeededRNG(42);
      const array = ['only'];

      for (let i = 0; i < 10; i++) {
        expect(rng.choice(array)).toBe('only');
      }
    });
  });

  describe('generateGameSeed', () => {
    it('should generate valid seeds in range', () => {
      for (let i = 0; i < 100; i++) {
        const seed = generateGameSeed();
        expect(seed).toBeGreaterThanOrEqual(0);
        expect(seed).toBeLessThanOrEqual(0x7fffffff);
        expect(Number.isInteger(seed)).toBe(true);
      }
    });

    it('should generate different seeds on consecutive calls', () => {
      const seeds = new Set<number>();
      for (let i = 0; i < 100; i++) {
        seeds.add(generateGameSeed());
      }
      // Very unlikely to have collisions in 100 calls
      expect(seeds.size).toBeGreaterThan(90);
    });
  });

  describe('Long Sequence Stability', () => {
    it('should maintain determinism over long sequences', () => {
      const seed = 0xabcdef12;
      const rng1 = new SeededRNG(seed);
      const rng2 = new SeededRNG(seed);

      // Generate 10,000 values
      for (let i = 0; i < 10000; i++) {
        expect(rng1.next()).toBe(rng2.next());
      }
    });

    it('should produce distributed values over long sequences', () => {
      const rng = new SeededRNG(42);
      const buckets = Array(10).fill(0);

      for (let i = 0; i < 10000; i++) {
        const val = rng.next();
        const bucket = Math.floor(val * 10);
        buckets[bucket]++;
      }

      // Each bucket should have roughly 1000 values (±200 is reasonable)
      for (const count of buckets) {
        expect(count).toBeGreaterThan(800);
        expect(count).toBeLessThan(1200);
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle seed = 0', () => {
      const rng = new SeededRNG(0);
      const val = rng.next();
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThan(1);
    });

    it('should handle maximum seed value', () => {
      const rng = new SeededRNG(0x7fffffff);
      const val = rng.next();
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThan(1);
    });

    it('should handle negative seeds by wrapping', () => {
      const rng = new SeededRNG(-1);
      const val = rng.next();
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThan(1);
    });
  });
});
