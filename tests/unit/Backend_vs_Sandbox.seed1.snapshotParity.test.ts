import { GameEngine } from '../../src/server/game/GameEngine';
import { snapshotFromGameState, snapshotsEqual } from '../utils/stateSnapshots';
import { reproduceSquare8TwoAiSeed1AtAction } from '../utils/aiSeedSnapshots';

/**
 * Backend vs Sandbox parity snapshot for the square8 / 2 AI / seed=1
 * configuration corresponding to the historical fuzz-harness stall
 * plateau.
 *
 * As with the seed18 parity test, this suite:
 *   1. Uses the shared seed harness to obtain a canonical mid-game
 *      GameState + ComparableSnapshot from the sandbox.
 *   2. Constructs a backend GameEngine with an equivalent configuration
 *      and replaces its internal gameState with the sandbox plateau.
 *   3. Takes a backend snapshot via snapshotFromGameState and asserts
 *      snapshot equality vs the sandbox snapshot.
 */

test('Backend_vs_Sandbox: square8 / 2 AI / seed=1 plateau snapshot parity', async () => {
  const targetActionIndex = 58;
  const { state: sandboxState, snapshot: sandboxSnapshot } =
    await reproduceSquare8TwoAiSeed1AtAction(targetActionIndex);

  // Sanity checks on sandbox plateau.
  expect(sandboxState.boardType).toBe('square8');
  expect(sandboxState.players.length).toBe(2);

  const backend = new GameEngine(
    'backend-seed1-plateau',
    sandboxState.boardType,
    sandboxState.players,
    sandboxState.timeControl,
    false
  );

  (backend as any).gameState = {
    ...sandboxState,
    id: 'backend-seed1-plateau',
  };

  const backendState = backend.getGameState();
  const backendSnapshot = snapshotFromGameState('backend-seed1-plateau', backendState);

  expect(snapshotsEqual(sandboxSnapshot, backendSnapshot)).toBe(true);
});
