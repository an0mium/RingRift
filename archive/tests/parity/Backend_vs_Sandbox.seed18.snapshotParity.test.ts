import { GameEngine } from '../../../src/server/game/GameEngine';
import { snapshotFromGameState, snapshotsEqual } from '../../../tests/utils/stateSnapshots';
import { reproduceSquare8TwoAiSeed18AtAction } from '../../../tests/utils/aiSeedSnapshots';

/**
 * Archived Backend vs Sandbox parity snapshot for the square8 / 2 AI / seed=18
 * plateau configuration.
 *
 * This diagnostic harness was historically used to validate that a canonical
 * mid‑game "plateau" snapshot for seed 18 produced identical snapshots in the
 * backend GameEngine and ClientSandboxEngine. It has since been superseded by
 * shared‑engine rules/invariant suites and the v2 contract vectors.
 *
 * The file is kept under archive/tests/** for historical debugging only and
 * is not part of CI-gated or default Jest runs.
 */

test('Backend_vs_Sandbox: square8 / 2 AI / seed=18 plateau snapshot parity (archived)', async () => {
  const targetActionIndex = 58;
  const { state: sandboxState, snapshot: sandboxSnapshot } =
    await reproduceSquare8TwoAiSeed18AtAction(targetActionIndex);

  // Sanity checks on sandbox plateau.
  expect(sandboxState.boardType).toBe('square8');
  expect(sandboxState.players.length).toBe(2);

  const backend = new GameEngine(
    'backend-seed18-plateau',
    sandboxState.boardType,
    sandboxState.players,
    sandboxState.timeControl,
    false
  );

  (backend as any).gameState = {
    ...sandboxState,
    id: 'backend-seed18-plateau',
  };

  const backendState = backend.getGameState();
  const backendSnapshot = snapshotFromGameState('backend-seed18-plateau', backendState);

  expect(snapshotsEqual(sandboxSnapshot, backendSnapshot)).toBe(true);
});
