import fs from 'fs';
import path from 'path';

import { reconstructStateAtMove } from '../../src/shared/engine';
import { jsonlLineToGameRecord, type GameRecord } from '../../src/shared/types/gameRecord';

const FIXTURES_DIR = path.join(__dirname, '..', 'fixtures', 'golden-games');

function loadGoldenGameRecords(): GameRecord[] {
  if (!fs.existsSync(FIXTURES_DIR)) {
    return [];
  }

  const entries = fs.readdirSync(FIXTURES_DIR, { withFileTypes: true });
  const jsonlFiles = entries
    .filter((e) => e.isFile() && e.name.endsWith('.jsonl'))
    .map((e) => path.join(FIXTURES_DIR, e.name));

  const records: GameRecord[] = [];
  for (const file of jsonlFiles) {
    const contents = fs.readFileSync(file, 'utf8');
    const lines = contents
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    for (const line of lines) {
      records.push(jsonlLineToGameRecord(line));
    }
  }

  return records;
}

describe('GoldenGames shared-engine replay (scaffold)', () => {
  const records = loadGoldenGameRecords();

  if (records.length === 0) {
    it('has no golden GameRecords yet (scaffold)', () => {
      expect(records.length).toBe(0);
    });
    return;
  }

  it('reconstructs states for all golden GameRecords without throwing', () => {
    for (const record of records) {
      // Initial state (no moves applied)
      const initial = reconstructStateAtMove(record, 0);
      expect(initial.boardType).toBe(record.boardType);
      expect(initial.players.length).toBe(record.numPlayers);

      // Final state (all moves applied)
      const finalState = reconstructStateAtMove(record, record.totalMoves);
      expect(finalState.boardType).toBe(record.boardType);
      expect(finalState.players.length).toBe(record.numPlayers);
    }
  });
});
