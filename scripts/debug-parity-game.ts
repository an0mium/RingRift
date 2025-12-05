#!/usr/bin/env npx ts-node
/**
 * Debug script to trace parity divergence for a specific game.
 * Usage: TS_NODE_PROJECT=tsconfig.server.json npx ts-node scripts/debug-parity-game.ts
 */

import Database from 'better-sqlite3';
import * as path from 'path';
import {
  ClientSandboxEngine,
  type SandboxInteractionHandler,
  type SandboxPlayerKind,
} from '../src/client/sandbox/ClientSandboxEngine';
import { hashGameState } from '../src/shared/engine';
import { deserializeGameState } from '../src/shared/engine/contracts/serialization';
import type { Move, GameState, Position, BoardType } from '../src/shared/engine';
import type { SerializedGameState } from '../src/shared/engine/contracts/serialization';

const GAME_ID = '47f34614-8a2a-420a-8ae2-9d1c51d5196f';
const DB_PATH = path.join(__dirname, '../ai-service/data/games/coverage_selfplay.db');

interface PythonMoveRow {
  move_number: number;
  move_type: string;
  player_number: number;
  from_position: string | null;
  to_position: string | null;
  move_json: string;
}

function loadPythonMoves(db: Database.Database, gameId: string): Move[] {
  const rows = db
    .prepare(
      `
    SELECT move_number, move_type, player_number, from_position, to_position, move_json
    FROM moves
    WHERE game_id = ?
    ORDER BY move_number ASC
  `
    )
    .all(gameId) as PythonMoveRow[];

  return rows.map((row) => {
    const moveData = JSON.parse(row.move_json);
    return {
      id: `py-move-${row.move_number}`,
      type: row.move_type as Move['type'],
      player: row.player_number,
      from: row.from_position ? parsePosition(row.from_position) : undefined,
      to: row.to_position ? parsePosition(row.to_position) : undefined,
      ...moveData,
      timestamp: new Date(),
      thinkTime: 0,
      moveNumber: row.move_number,
    } as Move;
  });
}

function loadPythonStateAtMove(
  db: Database.Database,
  gameId: string,
  moveIndex: number
): SerializedGameState | null {
  // Python uses 0-indexed move_number where get_state_at_move(k) returns state AFTER move k
  const row = db
    .prepare(
      `
    SELECT state_after as state_json
    FROM moves
    WHERE game_id = ? AND move_number = ?
  `
    )
    .get(gameId, moveIndex) as { state_json: string } | undefined;

  if (!row) return null;
  return JSON.parse(row.state_json) as SerializedGameState;
}

function loadInitialState(db: Database.Database, gameId: string): SerializedGameState | null {
  const row = db
    .prepare(
      `
    SELECT initial_state as state_json
    FROM games
    WHERE game_id = ?
  `
    )
    .get(gameId) as { state_json: string } | undefined;

  if (!row) return null;
  return JSON.parse(row.state_json) as SerializedGameState;
}

function parsePosition(posStr: string): Position {
  const [x, y] = posStr.split(',').map(Number);
  return { x, y };
}

function getStacksInfo(state: GameState): string {
  const stacks: string[] = [];
  for (const [key, stack] of state.board.stacks.entries()) {
    stacks.push(`${key}:p${stack.controllingPlayer}h${stack.stackHeight}`);
  }
  return stacks.sort().join(', ');
}

function getPlayerInfo(state: GameState): string {
  return state.players
    .map((p) => `p${p.playerNumber}(elim=${p.eliminatedRings},terr=${p.territorySpaces})`)
    .join(', ');
}

function getCollapsedSpaces(state: GameState): string {
  const spaces: string[] = [];
  for (const key of state.board.collapsedSpaces.keys()) {
    spaces.push(key);
  }
  return spaces.sort().join(', ') || 'none';
}

async function main() {
  console.log(`\n=== Debugging parity for game ${GAME_ID} ===\n`);

  const db = new Database(DB_PATH, { readonly: true });

  // Load initial state and moves from Python DB
  const initialStateSerialized = loadInitialState(db, GAME_ID);
  if (!initialStateSerialized) {
    console.error('Failed to load initial state');
    return;
  }

  const initialState = deserializeGameState(initialStateSerialized);

  const moves = loadPythonMoves(db, GAME_ID);
  console.log(`Loaded ${moves.length} moves from Python DB\n`);

  // Create sandbox config
  const config = {
    boardType: initialState.boardType as BoardType,
    numPlayers: initialState.players.length,
    playerKinds: initialState.players.map(() => 'human' as SandboxPlayerKind),
  };

  const interactionHandler: SandboxInteractionHandler = {
    async requestChoice(choice: any) {
      const options = (choice?.options as any[]) ?? [];
      const selectedOption = options.length > 0 ? options[0] : undefined;
      return {
        choiceId: choice.id,
        playerNumber: choice.playerNumber,
        choiceType: choice.type,
        selectedOption,
      } as any;
    },
  };

  // Create TS sandbox engine in traceMode
  const engine = new ClientSandboxEngine({
    config,
    interactionHandler,
    traceMode: true, // Enable trace mode for parity
  });

  // Initialize with Python's initial state
  engine.initFromSerializedState(initialStateSerialized, config.playerKinds, interactionHandler);

  console.log('Initial state:');
  console.log(`  Phase: ${initialState.currentPhase}, Player: ${initialState.currentPlayer}`);
  console.log(`  Stacks: ${getStacksInfo(initialState)}`);
  console.log();

  // Replay moves and compare states
  for (let k = 0; k < Math.min(moves.length, 20); k++) {
    const move = moves[k];
    const pythonStateAfterSerialized = loadPythonStateAtMove(db, GAME_ID, k);
    const pythonStateAfter = pythonStateAfterSerialized
      ? deserializeGameState(pythonStateAfterSerialized)
      : null;

    console.log(`\n--- Move k=${k}: ${move.type} by p${move.player} ---`);
    if (move.from) console.log(`  From: (${move.from.x},${move.from.y})`);
    if (move.to) console.log(`  To: (${move.to.x},${move.to.y})`);

    // Get TS state BEFORE applying move
    const tsBefore = engine.getGameState();
    console.log(`\n  TS BEFORE:`);
    console.log(
      `    Phase: ${tsBefore.currentPhase}, Player: ${tsBefore.currentPlayer}, Status: ${tsBefore.gameStatus}`
    );
    console.log(`    Players: ${getPlayerInfo(tsBefore)}`);
    console.log(`    Collapsed: ${getCollapsedSpaces(tsBefore)}`);

    // Apply move in TS
    const nextMove = k + 1 < moves.length ? moves[k + 1] : null;
    try {
      await engine.applyCanonicalMoveForReplay(move, nextMove);
    } catch (err) {
      console.error(`  ERROR applying move: ${(err as Error).message}`);
      break;
    }

    // Get TS state AFTER applying move
    const tsAfter = engine.getGameState();
    console.log(`\n  TS AFTER:`);
    console.log(
      `    Phase: ${tsAfter.currentPhase}, Player: ${tsAfter.currentPlayer}, Status: ${tsAfter.gameStatus}`
    );
    console.log(`    Players: ${getPlayerInfo(tsAfter)}`);
    console.log(`    Collapsed: ${getCollapsedSpaces(tsAfter)}`);
    console.log(`    Stacks: ${getStacksInfo(tsAfter)}`);

    // Compare with Python
    if (pythonStateAfter) {
      console.log(`\n  PYTHON state_after:`);
      console.log(
        `    Phase: ${pythonStateAfter.currentPhase}, Player: ${pythonStateAfter.currentPlayer}, Status: ${pythonStateAfter.gameStatus}`
      );
      console.log(`    Players: ${getPlayerInfo(pythonStateAfter)}`);
      console.log(`    Collapsed: ${getCollapsedSpaces(pythonStateAfter)}`);
      console.log(`    Stacks: ${getStacksInfo(pythonStateAfter)}`);

      // Hash comparison
      const tsHash = hashGameState(tsAfter);
      const pyHash = hashGameState(pythonStateAfter);
      if (tsHash !== pyHash) {
        console.log(`\n  *** DIVERGENCE DETECTED ***`);
        console.log(`    TS hash:     ${tsHash}`);
        console.log(`    Python hash: ${pyHash}`);

        // Find specific differences
        if (tsAfter.currentPhase !== pythonStateAfter.currentPhase) {
          console.log(
            `    Phase mismatch: TS=${tsAfter.currentPhase}, Python=${pythonStateAfter.currentPhase}`
          );
        }
        if (tsAfter.currentPlayer !== pythonStateAfter.currentPlayer) {
          console.log(
            `    Player mismatch: TS=${tsAfter.currentPlayer}, Python=${pythonStateAfter.currentPlayer}`
          );
        }

        // Compare stacks
        const tsStacks = new Map<string, { cp: number; h: number }>();
        for (const [key, stack] of tsAfter.board.stacks.entries()) {
          tsStacks.set(key, { cp: stack.controllingPlayer, h: stack.stackHeight });
        }
        const pyStacks = new Map<string, { cp: number; h: number }>();
        for (const [key, stack] of pythonStateAfter.board.stacks.entries()) {
          pyStacks.set(key, { cp: stack.controllingPlayer, h: stack.stackHeight });
        }

        // Find missing/extra stacks
        for (const [key, stack] of pyStacks.entries()) {
          const tsStack = tsStacks.get(key);
          if (!tsStack) {
            console.log(`    Missing in TS: stack at ${key} (p${stack.cp} h${stack.h})`);
          } else if (tsStack.cp !== stack.cp || tsStack.h !== stack.h) {
            console.log(
              `    Differs at ${key}: TS=(p${tsStack.cp} h${tsStack.h}), Py=(p${stack.cp} h${stack.h})`
            );
          }
        }
        for (const [key, stack] of tsStacks.entries()) {
          if (!pyStacks.has(key)) {
            console.log(`    Extra in TS: stack at ${key} (p${stack.cp} h${stack.h})`);
          }
        }
      } else {
        console.log(`\n  ✓ States match (hash: ${tsHash})`);
      }
    }
  }

  db.close();
  console.log('\n=== Done ===\n');
}

main().catch(console.error);
