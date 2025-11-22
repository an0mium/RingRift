/**
 * Seedable pseudo-random number generator using xorshift128+
 * Provides deterministic random sequences for game replay and testing
 */
export class SeededRNG {
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;

  constructor(seed: number) {
    // Initialize state from seed using splitmix32
    this.s0 = this.splitmix32(seed);
    this.s1 = this.splitmix32(this.s0);
    this.s2 = this.splitmix32(this.s1);
    this.s3 = this.splitmix32(this.s2);
  }

  private splitmix32(a: number): number {
    a |= 0;
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    return (t ^ (t >>> 15)) >>> 0;
  }

  /**
   * Returns next random number in [0, 1)
   */
  next(): number {
    // xorshift128+ algorithm
    const t = this.s1 ^ (this.s1 << 23);
    this.s1 = this.s0;
    this.s0 = this.s3;
    this.s3 = this.s2;
    this.s2 = t ^ this.s0 ^ (this.s1 >>> 26) ^ (this.s0 >>> 5);
    return ((this.s0 + this.s2) >>> 0) / 0x100000000;
  }

  /**
   * Returns random integer in [min, max)
   */
  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min)) + min;
  }

  /**
   * Shuffles array in-place using Fisher-Yates algorithm
   */
  shuffle<T>(array: T[]): T[] {
    for (let i = array.length - 1; i > 0; i--) {
      const j = this.nextInt(0, i + 1);
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  /**
   * Selects a random element from an array
   */
  choice<T>(array: T[]): T {
    if (array.length === 0) {
      throw new Error('Cannot select from empty array');
    }
    return array[this.nextInt(0, array.length)];
  }
}

/**
 * Generates a random seed for new games
 * Uses current timestamp and random value for uniqueness
 */
export function generateGameSeed(): number {
  return Math.floor(Math.random() * 0x7fffffff);
}
