/**
 * Board Mutation Helpers
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Utility functions for in-place mutation of board state objects.
 *
 * These helpers preserve object references while replacing contents,
 * which is important for callers that cache references to board.stacks,
 * board.markers, etc.
 *
 * @module boardMutationHelpers
 */

import type { BoardState } from '../types/game';

/**
 * Replace all entries in a Map with entries from another Map.
 *
 * Preserves the target Map's object reference while replacing its contents.
 * This is useful when you need to update a Map in-place without breaking
 * references held by other parts of the code.
 *
 * @param target - The Map to update (will be cleared and repopulated)
 * @param source - The Map to copy entries from
 */
export function replaceMapContents<K, V>(target: Map<K, V>, source: Map<K, V>): void {
  target.clear();
  for (const [key, value] of source) {
    target.set(key, value);
  }
}

/**
 * Replace all elements in an array with elements from another array.
 *
 * Preserves the target array's object reference while replacing its contents.
 *
 * @param target - The array to update (will be emptied and repopulated)
 * @param source - The array to copy elements from
 */
export function replaceArrayContents<T>(target: T[], source: T[]): void {
  target.length = 0;
  target.push(...source);
}

/**
 * Replace all properties in a plain object with properties from another object.
 *
 * Preserves the target object's reference while replacing its contents.
 * Deletes all existing properties before copying from source.
 *
 * @param target - The object to update (will be cleared and repopulated)
 * @param source - The object to copy properties from
 */
export function replaceObjectContents<T extends Record<string | number, unknown>>(
  target: T,
  source: T
): void {
  for (const key of Object.keys(target)) {
    delete target[key as keyof T];
  }
  for (const [key, value] of Object.entries(source)) {
    (target as Record<string, unknown>)[key] = value;
  }
}

/**
 * Copy all board state from source to target, preserving target's object references.
 *
 * This is the main helper for updating a BoardState in-place. It preserves
 * the target BoardState's Maps, arrays, and nested objects while replacing
 * their contents with values from the source.
 *
 * Use this when you need to apply a new board state but want to preserve
 * references that other code may be holding to board.stacks, board.markers, etc.
 *
 * @param target - The BoardState to update in-place
 * @param source - The BoardState to copy values from
 */
export function copyBoardStateInPlace(target: BoardState, source: BoardState): void {
  // Copy Map contents
  replaceMapContents(target.stacks, source.stacks);
  replaceMapContents(target.markers, source.markers);
  replaceMapContents(target.collapsedSpaces, source.collapsedSpaces);
  replaceMapContents(target.territories, source.territories);

  // Copy array contents
  replaceArrayContents(target.formedLines, source.formedLines);

  // Copy plain object contents
  replaceObjectContents(target.eliminatedRings, source.eliminatedRings);

  // Copy primitive values
  target.size = source.size;
}
