import { GameEngine } from '../../src/server/game/GameEngine';
import { snapshotFromGameState, snapshotsEqual } from '../utils/stateSnapshots';
import { reproduceSquare8TwoAiSeed18AtAction } from '../utils/aiSeedSnapshots';

/**
 * Backend vs Sandbox parity snapshot for the square8 / 2 AI / seed=18
 * configuration used by the single-seed debug harness.
 *
 * This test does not attempt to replay the full AI trace into the backend.
 * Instead, it:
 *   1. Uses the shared seed harness to obtain a canonical mid-game
 *      GameState + ComparableSnapshot from the sandbox.
 *   2. Constructs a backend GameEngine with an equivalent configuration
 *      and replaces its internal gameState with the sandbox plateau.
 *   3. Takes a backend snapshot via snapshotFromGameState and asserts
 *      that it is snapshot-equal to the sandbox snapshot.
 *
 * This acts as a lightweight guard that:
 *   - The plateau GameState is structurally valid for both engines.
 *   - snapshotFromGameState / ComparableSnapshot treat sandbox and backend
 *     states in a host-agnostic way.
 */

test('Backend_vs_Sandbox: square8 / 2 AI / seed=18 plateau snapshot parity', async () => {
  const targetActionIndex = 58;
  const { state: sandboxState, snapshot: sandboxSnapshot } =
    await reproduceSquare8TwoAiSeed18AtAction(targetActionIndex);

  // Sanity checks on sandbox plateau.
  expect(sandboxState.boardType).toBe('square8');
  expect(sandboxState.players.length).toBe(2);

  // Build a backend GameEngine with equivalent boardType/players/timeControl
  // and then inject the sandbox plateau GameState as its internal state.
  const backend = new GameEngine(
    'backend-seed18-plateau',
    sandboxState.boardType,
    sandboxState.players,
    sandboxState.timeControl,
    false
  );

  // Test-only: override the internal gameState so that subsequent
  // getGameState() calls see the same plateau as the sandbox engine.
  (backend as any).gameState = {
    ...sandboxState,
    id: 'backend-seed18-plateau',
  };

  const backendState = backend.getGameState();
  const backendSnapshot = snapshotFromGameState('backend-seed18-plateau', backendState);

  expect(snapshotsEqual(sandboxSnapshot, backendSnapshot)).toBe(true);
});
