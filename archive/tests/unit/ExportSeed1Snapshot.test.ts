import fs from 'fs';
import path from 'path';
import { reproduceSquare8TwoAiSeed1AtAction } from '../../../tests/utils/aiSeedSnapshots';

/**
 * Archived utility test to export a canonical ComparableSnapshot for the
 * square8 / 2 AI / seed=1 plateau into a JSON file for Python ai-service
 * parity tests.
 *
 * This suite has been moved under archive/tests/unit and is no longer part
 * of normal CI or diagnostics. To (re)generate the snapshot manually:
 *
 *   RINGRIFT_EXPORT_PARITY_SNAPSHOTS=1 \\
 *   npx jest archive/tests/unit/ExportSeed1Snapshot.test.ts --runInBand
 *
 * The snapshot will be written to:
 *   ai-service/tests/parity/square8_2p_seed1_plateau.snapshot.json
 */

const EXPORT_ENABLED =
  typeof process !== 'undefined' &&
  !!(process as any).env &&
  ['1', 'true', 'TRUE'].includes((process as any).env.RINGRIFT_EXPORT_PARITY_SNAPSHOTS ?? '');

const maybeTest = EXPORT_ENABLED ? test : test.skip;

maybeTest('export square8 / 2 AI / seed=1 plateau snapshot for Python parity tests', async () => {
  const targetActionIndex = 58;
  const { snapshot } = await reproduceSquare8TwoAiSeed1AtAction(targetActionIndex);

  const outDir = path.join(process.cwd(), 'ai-service', 'tests', 'parity');
  fs.mkdirSync(outDir, { recursive: true });

  const outPath = path.join(outDir, 'square8_2p_seed1_plateau.snapshot.json');
  fs.writeFileSync(outPath, JSON.stringify(snapshot, null, 2), 'utf8');
});
