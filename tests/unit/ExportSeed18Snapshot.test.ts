import fs from 'fs';
import path from 'path';
import { reproduceSquare8TwoAiSeed18AtAction } from '../utils/aiSeedSnapshots';

/**
 * Utility test to export a canonical ComparableSnapshot for the
 * square8 / 2 AI / seed=18 plateau into a JSON file that can be
 * consumed by the Python ai-service parity tests.
 *
 * This test is gated by RINGRIFT_EXPORT_PARITY_SNAPSHOTS. Under normal
 * Jest runs it is skipped and has no side effects. To (re)generate the
 * fixture, run:
 *
 *   RINGRIFT_EXPORT_PARITY_SNAPSHOTS=1 \
 *   npx jest tests/unit/ExportSeed18Snapshot.test.ts --runInBand
 *
 * The snapshot will be written to:
 *   ai-service/tests/parity/square8_2p_seed18_plateau.snapshot.json
 */

const EXPORT_ENABLED =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_EXPORT_PARITY_SNAPSHOTS ?? '');

const maybeTest = EXPORT_ENABLED ? test : test.skip;

maybeTest('export square8 / 2 AI / seed=18 plateau snapshot for Python parity tests', async () => {
  const targetActionIndex = 58;
  const { snapshot } = await reproduceSquare8TwoAiSeed18AtAction(targetActionIndex);

  const outDir = path.join(process.cwd(), 'ai-service', 'tests', 'parity');
  fs.mkdirSync(outDir, { recursive: true });

  const outPath = path.join(outDir, 'square8_2p_seed18_plateau.snapshot.json');
  fs.writeFileSync(outPath, JSON.stringify(snapshot, null, 2), 'utf8');
});
