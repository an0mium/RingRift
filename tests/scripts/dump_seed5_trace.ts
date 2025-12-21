import { runSandboxAITrace } from '../utils/traces';
import { formatMove } from '../../src/shared/engine/notation';

async function dump() {
  const trace = await runSandboxAITrace('square8', 2, 5, 60);
  console.log('--- Seed 5 Trace Moves 40-50 ---');
  trace.entries.slice(40, 55).forEach((e) => {
    console.log(
      `${e.moveNumber}: ${e.action.type} (${e.actor}) - ${formatMove(e.action, { boardType: 'square8' })}`
    );
    if (e.action.type === 'choose_territory_option') {
      console.log('  Region:', JSON.stringify(e.action.disconnectedRegions?.[0]?.spaces));
    }
  });
}

dump().catch(console.error);
