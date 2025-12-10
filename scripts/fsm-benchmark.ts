/**
 * FSM Performance Benchmark
 *
 * Measures the overhead of FSM validation in a controlled environment.
 */
import { validateMoveWithFSM } from '../src/shared/engine/fsm';
import { createTestGameState, createTestPlayer } from '../tests/utils/fixtures';
import type { Move, MoveType, GamePhase } from '../src/shared/types/game';

function createBenchmarkGameState(phase: GamePhase, currentPlayer: number) {
  return createTestGameState({
    currentPhase: phase,
    currentPlayer: currentPlayer,
    players: [createTestPlayer(1, { ringsInHand: 5 }), createTestPlayer(2, { ringsInHand: 5 })],
    history: [],
    moveHistory: [],
  });
}

function createMockMove(type: MoveType, player: number): Move {
  return {
    id: 'move-bench',
    type,
    player,
    to: { x: 3, y: 3 },
    timestamp: new Date(),
    thinkTime: 0,
    moveNumber: 1,
  };
}

function benchmark(name: string, iterations: number, fn: () => void): void {
  // Warmup
  for (let i = 0; i < 100; i++) {
    fn();
  }

  // Actual benchmark
  const start = process.hrtime.bigint();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = process.hrtime.bigint();

  const totalNs = Number(end - start);
  const avgNs = totalNs / iterations;
  const avgUs = avgNs / 1000;
  const opsPerSec = (1_000_000_000 / avgNs).toFixed(0);

  console.log(`${name}:`);
  console.log(`  Iterations: ${iterations.toLocaleString()}`);
  console.log(`  Total time: ${(totalNs / 1_000_000).toFixed(2)}ms`);
  console.log(`  Avg per call: ${avgUs.toFixed(2)}us`);
  console.log(`  Ops/sec: ${opsPerSec}`);
  console.log();
}

async function main() {
  console.log('FSM Performance Benchmark');
  console.log('='.repeat(50));
  console.log();

  const iterations = 10000;

  // Benchmark 1: PLACE_RING validation (ring_placement phase)
  const placeRingState = createBenchmarkGameState('ring_placement' as GamePhase, 1);
  const placeRingMove = createMockMove('place_ring' as MoveType, 1);

  benchmark('PLACE_RING validation (ring_placement phase)', iterations, () => {
    validateMoveWithFSM(placeRingState, placeRingMove);
  });

  // Benchmark 2: MOVE_STACK validation (movement phase)
  const moveStackState = createBenchmarkGameState('movement' as GamePhase, 1);
  moveStackState.board.stacks.set('2,2', { position: { x: 2, y: 2 }, rings: [{ owner: 1 }] });
  const moveStackMove = createMockMove('move_stack' as MoveType, 1);
  (moveStackMove as any).from = { x: 2, y: 2 };
  moveStackMove.to = { x: 4, y: 4 };

  benchmark('MOVE_STACK validation (movement phase)', iterations, () => {
    validateMoveWithFSM(moveStackState, moveStackMove);
  });

  // Benchmark 3: Invalid phase (PLACE_RING in movement phase - should fail fast)
  benchmark('PLACE_RING in wrong phase (fast failure)', iterations, () => {
    validateMoveWithFSM(moveStackState, placeRingMove);
  });

  // Benchmark 4: Wrong player (should fail very fast)
  const wrongPlayerMove = createMockMove('move_stack' as MoveType, 2); // Player 2 but state has player 1
  benchmark('Wrong player validation (fast failure)', iterations, () => {
    validateMoveWithFSM(moveStackState, wrongPlayerMove);
  });

  // Benchmark 5: With debug context (slower path)
  benchmark('PLACE_RING with debug context', iterations / 10, () => {
    validateMoveWithFSM(placeRingState, placeRingMove, true);
  });

  // Benchmark 6: NO_LINE_ACTION in line_processing (bookkeeping)
  const lineState = createBenchmarkGameState('line_processing' as GamePhase, 1);
  const noLineMove = createMockMove('no_line_action' as MoveType, 1);

  benchmark('NO_LINE_ACTION (bookkeeping move)', iterations, () => {
    validateMoveWithFSM(lineState, noLineMove);
  });

  // Benchmark 7: SKIP_PLACEMENT (fast path - moveHint skips getValidMoves)
  const skipPlacementMove = createMockMove('skip_placement' as MoveType, 1);

  benchmark('SKIP_PLACEMENT (fast path)', iterations, () => {
    validateMoveWithFSM(placeRingState, skipPlacementMove);
  });

  // Summary
  console.log('='.repeat(50));
  console.log('Summary:');
  console.log('  - PLACE_RING is slower (~24us) because it computes valid positions');
  console.log('  - MOVE_STACK/captures are fast (~0.2us) - quick boolean check');
  console.log('  - Wrong player/phase failures are very fast (~0.13-0.18us)');
  console.log('  - Bookkeeping moves (no_*) are fast due to moveHint optimization');
  console.log();
  console.log('Benchmark complete.');
}

main().catch(console.error);
