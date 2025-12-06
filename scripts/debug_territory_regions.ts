#!/usr/bin/env ts-node
/**
 * Debug helper: inspect TS territory regions for a dumped GameState JSON.
 *
 * Usage (from repo root):
 *
 *   TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/debug_territory_regions.ts \
 *     --state ai-service/ts_py_square19_k200/<file>.ts_state.json \
 *     --player 1
 *
 * This loads the GameState, runs:
 *   - shared territoryDetection.findDisconnectedRegions(board)
 *   - TerritoryAggregate.getProcessableTerritoryRegions(board, { player })
 * and prints region counts and a few sample coordinates.
 */

import * as fs from 'fs';
import * as path from 'path';

import type { GameState, Territory } from '../src/shared/types/game';
import { positionToString } from '../src/shared/types/game';
import { findDisconnectedRegions as findDisconnectedRegionsShared } from '../src/shared/engine/territoryDetection';
import { getProcessableTerritoryRegions } from '../src/shared/engine/aggregates/TerritoryAggregate';

function parseArgs(argv: string[]) {
  let statePath = '';
  let player = 1;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--state' && i + 1 < argv.length) {
      statePath = argv[++i];
    } else if (arg === '--player' && i + 1 < argv.length) {
      player = parseInt(argv[++i], 10);
    }
  }

  if (!statePath) {
    throw new Error('Missing --state <path-to-ts_state.json>');
  }

  return { statePath, player };
}

function stringifyRegionSample(region: Territory, maxSamples = 10): string[] {
  const spaces = region.spaces ?? [];
  return spaces.slice(0, maxSamples).map((p) => positionToString(p));
}

async function main() {
  const { statePath, player } = parseArgs(process.argv.slice(2));
  const absPath = path.resolve(statePath);

  if (!fs.existsSync(absPath)) {
    throw new Error(`State file not found: ${absPath}`);
  }

  const raw = fs.readFileSync(absPath, 'utf8');
  const state = JSON.parse(raw) as GameState;

  console.log('Loaded TS GameState from:', absPath);
  console.log('  boardType:', state.board.type);
  console.log('  size:', state.board.size);
  console.log('  currentPlayer:', state.currentPlayer);
  console.log('  currentPhase:', state.currentPhase);

  // Normalise plain-object board into Map-backed BoardState shape expected by shared helpers.
  const board = {
    ...state.board,
    stacks: new Map(Object.entries((state.board as any).stacks ?? {})),
    markers: new Map(Object.entries((state.board as any).markers ?? {})),
    collapsedSpaces: new Map(Object.entries((state.board as any).collapsedSpaces ?? {})),
  } as GameState['board'];

  const allRegions = findDisconnectedRegionsShared(board);
  console.log('\nAll disconnected regions (shared territoryDetection):', allRegions.length);
  allRegions.slice(0, 3).forEach((r, idx) => {
    const sample = stringifyRegionSample(r);
    console.log(`  region[${idx}] size=${r.spaces.length} sample=${JSON.stringify(sample)}`);
  });

  const processable = getProcessableTerritoryRegions(board, { player });
  console.log(`\nProcessable regions for player ${player}:`, processable.length);
  processable.slice(0, 3).forEach((r, idx) => {
    const sample = stringifyRegionSample(r);
    console.log(`  region[${idx}] size=${r.spaces.length} sample=${JSON.stringify(sample)}`);
  });
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
