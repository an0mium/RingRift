import {
  ClientSandboxEngine,
  SandboxConfig,
  SandboxInteractionHandler,
} from '../../src/client/sandbox/ClientSandboxEngine';
import {
  BoardType,
  GameState,
  PlayerChoice,
  PlayerChoiceResponseFor,
  CaptureDirectionChoice,
} from '../../src/shared/types/game';
import { hashGameState, computeProgressSnapshot } from '../../src/shared/engine/core';
import { ComparableSnapshot, snapshotFromGameState } from './stateSnapshots';

/**
 * Helpers for reproducing specific AI fuzz-harness seeds and extracting
 * mid-game snapshots in a reusable, deterministic way.
 *
 * These utilities intentionally mirror the logic in
 * `ClientSandboxEngine.aiSimulation.test.ts`, but are packaged as
 * lightweight, targeted repro functions.
 */

function makePrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    // LCG parameters from Numerical Recipes (same as fuzz harness).
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/**
 * Create a sandbox engine wired exactly like the fuzz harness: all players
 * are AI with the same deterministic capture-direction tie-breaker.
 */
export function createSeedHarnessEngine(
  boardType: BoardType,
  numPlayers: number
): ClientSandboxEngine {
  const config: SandboxConfig = {
    boardType,
    numPlayers,
    playerKinds: Array.from({ length: numPlayers }, () => 'ai'),
  };

  const handler: SandboxInteractionHandler = {
    async requestChoice<TChoice extends PlayerChoice>(
      choice: TChoice
    ): Promise<PlayerChoiceResponseFor<TChoice>> {
      const anyChoice = choice as any;

      if (anyChoice.type === 'capture_direction') {
        const cd = anyChoice as CaptureDirectionChoice;
        const options = cd.options || [];
        if (options.length === 0) {
          throw new Error('Test SandboxInteractionHandler: no options for capture_direction');
        }

        // Deterministically pick the option with the smallest landing x,y
        // to keep simulations reproducible given a fixed Math.random.
        let selected = options[0];
        for (const opt of options) {
          if (
            opt.landingPosition.x < selected.landingPosition.x ||
            (opt.landingPosition.x === selected.landingPosition.x &&
              opt.landingPosition.y < selected.landingPosition.y)
          ) {
            selected = opt;
          }
        }

        return {
          choiceId: cd.id,
          playerNumber: cd.playerNumber,
          choiceType: cd.type,
          selectedOption: selected,
        } as PlayerChoiceResponseFor<TChoice>;
      }

      const selectedOption = anyChoice.options ? anyChoice.options[0] : undefined;
      return {
        choiceId: anyChoice.id,
        playerNumber: anyChoice.playerNumber,
        choiceType: anyChoice.type,
        selectedOption,
      } as PlayerChoiceResponseFor<TChoice>;
    },
  };

  return new ClientSandboxEngine({ config, interactionHandler: handler });
}

export interface SeedReproResult {
  /** Full sandbox GameState at the chosen checkpoint. */
  state: GameState;
  /** Order-stable snapshot derived from the GameState for parity-style asserts. */
  snapshot: ComparableSnapshot;
  /** Number of AI actions actually taken before the checkpoint (may be < targetActionIndex). */
  actionsTaken: number;
  /** Live engine instance positioned at the checkpoint, for further targeted probes. */
  engine: ClientSandboxEngine;
}

/**
 * Generic helper: reproduce the sandbox AI fuzz-harness configuration
 *   - boardType: 'square8'
 *   - numPlayers: 2 (both AI)
 *   - seed: caller-provided
 *
 * and advance the game using maybeRunAITurn until we either:
 *   - reach the specified action index, or
 *   - hit an early stall/termination according to the stagnation heuristic.
 */
export async function reproduceSquare8TwoAiAtSeed(
  seed: number,
  targetActionIndex: number,
  labelOverride?: string
): Promise<SeedReproResult> {
  const boardType: BoardType = 'square8';
  const numPlayers = 2;

  const rng = makePrng(seed);
  const originalRandom = Math.random;
  (Math as any).random = rng;

  try {
    const engine = createSeedHarnessEngine(boardType, numPlayers);

    let actionsTaken = 0;
    let lastHash = hashGameState(engine.getGameState());
    let stagnantSteps = 0;

    while (actionsTaken < targetActionIndex) {
      const before = engine.getGameState();
      const beforeProgress = computeProgressSnapshot(before);

      if (before.gameStatus !== 'active') {
        break;
      }

      const currentPlayer = before.players.find(
        (p) => p.playerNumber === before.currentPlayer
      );
      if (!currentPlayer || currentPlayer.type !== 'ai') {
        // For this configuration all players are AI, but keep the
        // guard for completeness.
        break;
      }

      await engine.maybeRunAITurn();
      actionsTaken += 1;

      const after = engine.getGameState();
      const afterHash = hashGameState(after);
      const afterProgress = computeProgressSnapshot(after);

      // Maintain a light-touch invariant check similar to the harness.
      if (!(afterProgress.S >= beforeProgress.S)) {
        throw new Error(
          `Repro harness invariant violation (seed=${seed}): S decreased at action=${actionsTaken} (before=${beforeProgress.S}, after=${afterProgress.S})`
        );
      }

      if (afterHash === lastHash && after.gameStatus === 'active') {
        stagnantSteps += 1;
      } else {
        stagnantSteps = 0;
      }
      lastHash = afterHash;

      // If we encounter an obvious stall while trying to reach the
      // requested index, stop early and treat that stalled state as the
      // snapshot—we are mainly interested in mid-game behaviour near a
      // stall plateau.
      if (stagnantSteps >= 8) {
        break;
      }
    }

    const state = engine.getGameState();
    const label =
      labelOverride ?? `square8-2p-seed${seed}-action-${targetActionIndex}`;
    const snapshot = snapshotFromGameState(label, state);
    return { state, snapshot, actionsTaken, engine };
  } finally {
    (Math as any).random = originalRandom;
  }
}

/**
 * Backwards-compat wrapper for the historical stall seed (seed=1).
 */
export async function reproduceSquare8TwoAiSeed1AtAction(
  targetActionIndex: number,
  label = `square8-2p-seed1-action-${targetActionIndex}`
): Promise<SeedReproResult> {
  return reproduceSquare8TwoAiAtSeed(1, targetActionIndex, label);
}

/**
 * Convenience helper for the single-seed debug harness (seed=18). This
 * mirrors the configuration used in ClientSandboxEngine.aiSingleSeedDebug
 * and allows other tests (including parity/scenario suites) to obtain a
 * comparable mid-game snapshot without duplicating seed logic.
 */
export async function reproduceSquare8TwoAiSeed18AtAction(
  targetActionIndex: number,
  label = `square8-2p-seed18-action-${targetActionIndex}`
): Promise<SeedReproResult> {
  return reproduceSquare8TwoAiAtSeed(18, targetActionIndex, label);
}
