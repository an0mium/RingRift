import { GameEngine } from '../../../src/server/game/GameEngine';
import { snapshotFromGameState, snapshotsEqual } from '../../../tests/utils/stateSnapshots';
import { reproduceSquare8TwoAiSeed1AtAction } from '../../../tests/utils/aiSeedSnapshots';

/**
 * Archived Backend vs Sandbox parity snapshot for the square8 / 2 AI / seed=1
 * plateau configuration.
 *
 * This diagnostic harness was historically used to compare a mid‑game
 * "plateau" snapshot between the backend GameEngine and ClientSandboxEngine
 * for seed 1. It has been superseded for semantic parity purposes by:
 *
 *   - Shared-engine rules and invariant suites (movement/capture/territory).
 *   - Contract vectors under tests/fixtures/contract-vectors/v2.
 *   - Territory and chain‑capture scenario tests and parity suites.
 *
 * The file is kept under archive/tests/** for historical debugging only and
 * is not part of CI-gated or default Jest runs.
 */

test('Backend_vs_Sandbox: square8 / 2 AI / seed=1 plateau snapshot parity (archived)', async () => {
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
