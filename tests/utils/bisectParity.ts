import { BoardType, GameState, Move } from '../../src/shared/types/game';
import { runSandboxAITrace } from './traces';
import {
  backendAdapter,
  sandboxAdapter,
  findFirstMismatchIndex,
  compareEnginesAtPrefix,
} from './traceReplayer';
import { snapshotFromGameState, ComparableSnapshot } from './stateSnapshots';

/**
 * Shared helper for backend vs sandbox snapshot-parity bisect harnesses.
 *
 * Responsibilities:
 *   1) Generate a sandbox AI trace for a given boardType / numPlayers / seed.
 *   2) Binary-search (via traceReplayer) for the smallest prefix length at
 *      which backend and sandbox snapshots diverge when replaying the
 *      canonical move list from the common initial state.
 *   3) When a mismatch is found, compute backend vs sandbox snapshots at the
 *      earliest mismatching prefix so callers can log focused diagnostics.
 *
 * This keeps the core bisect mechanics in one place so specialised tests
 * (seed-specific diagnostics, board-type variations, etc.) can focus on
 * expectations and logging rather than duplicating trace + replay logic.
 */

export interface BisectConfig {
  boardType: BoardType;
  numPlayers: number;
  seed: number;
  maxSteps: number;
}

export interface BisectOutcome {
  /** Canonical move list extracted from the sandbox trace. */
  moves: Move[];
  /** Whether backend and sandbox snapshots were equal for the full move list. */
  allEqual: boolean;
  /** Index of first mismatching prefix in [0, moves.length], or moves.length when none. */
  firstMismatchIndex: number;
  /**
   * Backend snapshot at the earliest mismatching prefix, when a mismatch is
   * found. Undefined when allEqual === true.
   */
  backendSnapAtMismatch?: ComparableSnapshot;
  /**
   * Sandbox snapshot at the earliest mismatching prefix, when a mismatch is
   * found. Undefined when allEqual === true.
   */
  sandboxSnapAtMismatch?: ComparableSnapshot;
  /** Initial GameState from which both engines are constructed. */
  initialState: GameState;
}

export async function runBackendVsSandboxBisect(config: BisectConfig): Promise<BisectOutcome> {
  const { boardType, numPlayers, seed, maxSteps } = config;

  const trace = await runSandboxAITrace(boardType, numPlayers, seed, maxSteps);
  const moves: Move[] = trace.entries.map((e) => e.action as Move);

  const { allEqual, firstMismatchIndex } = await findFirstMismatchIndex(
    backendAdapter,
    sandboxAdapter,
    trace.initialState,
    moves
  );

  let backendSnapAtMismatch: ComparableSnapshot | undefined;
  let sandboxSnapAtMismatch: ComparableSnapshot | undefined;

  if (!allEqual) {
    const prefix = firstMismatchIndex;

    // For prefix 0 we simply compare the initial state; for k > 0, reuse the
    // shared compareEnginesAtPrefix helper to derive focused snapshots.
    if (prefix === 0) {
      const snap = snapshotFromGameState('initial-state', trace.initialState);
      backendSnapAtMismatch = snap;
      sandboxSnapAtMismatch = snap;
    } else {
      const { backendSnap, sandboxSnap } = await compareEnginesAtPrefix(
        backendAdapter,
        sandboxAdapter,
        trace.initialState,
        moves,
        prefix
      );
      backendSnapAtMismatch = backendSnap;
      sandboxSnapAtMismatch = sandboxSnap;
    }
  }

  return {
    moves,
    allEqual,
    firstMismatchIndex,
    backendSnapAtMismatch,
    sandboxSnapAtMismatch,
    initialState: trace.initialState,
  };
}
