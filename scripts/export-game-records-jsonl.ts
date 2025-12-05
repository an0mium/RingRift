#!/usr/bin/env ts-node
/**
 * Export completed Game records from Postgres as JSONL GameRecord lines.
 *
 * This is the TS/Node companion to the Python-side training exporters:
 * it lets you pull canonical GameRecord JSONL directly from the online
 * games database for use in analysis or training pipelines.
 *
 * Usage:
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-game-records-jsonl.ts \\
 *     --output data/game_records.jsonl
 *
 *   # Filter by board type
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-game-records-jsonl.ts \\
 *     --output data/square8_records.jsonl --board-type square8
 */

import fs from 'fs';
import path from 'path';

import { BoardType } from '@prisma/client';

import {
  gameRecordRepository,
  type GameRecordFilter,
} from '../src/server/services/GameRecordRepository';

interface CliArgs {
  output: string;
  boardType?: BoardType;
}

function printUsage(): void {
  // eslint-disable-next-line no-console
  console.log(
    [
      'Usage: export-game-records-jsonl.ts --output <path> [--board-type square8|square19|hexagonal]',
      '',
      'Example:',
      '  TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/export-game-records-jsonl.ts \\',
      '    --output data/game_records.jsonl --board-type square8',
    ].join('\n')
  );
}

function parseArgs(argv: string[]): CliArgs | null {
  let output: string | undefined;
  let boardType: BoardType | undefined;

  for (let i = 2; i < argv.length; i += 1) {
    const raw = argv[i];
    if (!raw.startsWith('--')) continue;

    const [flag, valueMaybe] = raw.split('=', 2);
    const next = argv[i + 1];
    const value = valueMaybe ?? (next && !next.startsWith('--') ? next : undefined);

    switch (flag) {
      case '--output':
        if (!value) {
          console.error('Missing value for --output');
          return null;
        }
        output = value;
        if (!valueMaybe && next === value) {
          i += 1;
        }
        break;
      case '--board-type':
        if (!value) {
          console.error('Missing value for --board-type');
          return null;
        }
        if (value !== 'square8' && value !== 'square19' && value !== 'hexagonal') {
          console.error(`Invalid --board-type value: ${value}`);
          return null;
        }
        boardType = value as BoardType;
        if (!valueMaybe && next === value) {
          i += 1;
        }
        break;
      default:
        console.warn(`Ignoring unknown flag: ${flag}`);
        if (!valueMaybe && next && !next.startsWith('--')) {
          i += 1;
        }
    }
  }

  if (!output) {
    return null;
  }

  return { output, boardType };
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv);
  if (!args) {
    printUsage();
    process.exitCode = 1;
    return;
  }

  const outputPath = path.isAbsolute(args.output)
    ? args.output
    : path.join(process.cwd(), args.output);

  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });

  const writeStream = fs.createWriteStream(outputPath, { flags: 'w' });

  const filter: GameRecordFilter = {};
  if (args.boardType) {
    filter.boardType = args.boardType;
  }

  let count = 0;

  // eslint-disable-next-line no-console
  console.log(
    `[export-game-records-jsonl] Exporting game records to ${outputPath}${
      args.boardType ? ` (boardType=${args.boardType})` : ''
    }...`
  );

  for await (const line of gameRecordRepository.exportAsJsonl(filter)) {
    writeStream.write(line);
    writeStream.write('\n');
    count += 1;
  }

  await new Promise<void>((resolve, reject) => {
    writeStream.end((err) => {
      if (err) reject(err);
      else resolve();
    });
  });

  // eslint-disable-next-line no-console
  console.log(`[export-game-records-jsonl] Done. Wrote ${count} record(s) to ${outputPath}.`);
}

main().catch((err) => {
  console.error('[export-game-records-jsonl] Fatal error:', err);
  process.exitCode = 1;
});
