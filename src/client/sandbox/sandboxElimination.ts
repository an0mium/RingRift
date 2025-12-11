import type { BoardState, Player, RingStack } from '../../shared/engine';
import { positionToString, calculateCapHeight } from '../../shared/engine';
import { flagEnabled, isTestEnvironment } from '../../shared/utils/envFlags';

const TERRITORY_TRACE_DEBUG = flagEnabled('RINGRIFT_TRACE_DEBUG');

export interface ForcedEliminationResult {
  board: BoardState;
  players: Player[];
  totalRingsEliminatedDelta: number;
}

function assertForcedEliminationConsistency(
  context: string,
  before: { board: BoardState; players: Player[] },
  after: { board: BoardState; players: Player[]; delta: number },
  playerNumber: number
): void {
  const isTestEnv = isTestEnvironment();

  const sumEliminated = (players: Player[]): number =>
    players.reduce((acc, p) => acc + p.eliminatedRings, 0);

  const sumBoardEliminated = (board: BoardState): number =>
    Object.values(board.eliminatedRings ?? {}).reduce((acc, v) => acc + v, 0);

  const beforePlayerTotal = sumEliminated(before.players);
  const beforeBoardTotal = sumBoardEliminated(before.board);
  const afterPlayerTotal = sumEliminated(after.players);
  const afterBoardTotal = sumBoardEliminated(after.board);

  const deltaPlayers = afterPlayerTotal - beforePlayerTotal;
  const deltaBoard = afterBoardTotal - beforeBoardTotal;

  const errors: string[] = [];

  if (deltaPlayers !== after.delta) {
    errors.push(
      `forced elimination (${context}) player delta mismatch: expected ${after.delta}, actual ${deltaPlayers}`
    );
  }

  if (deltaBoard !== after.delta) {
    errors.push(
      `forced elimination (${context}) board delta mismatch: expected ${after.delta}, actual ${deltaBoard}`
    );
  }

  if (after.delta < 0) {
    errors.push(
      `forced elimination (${context}) produced negative delta=${after.delta} for player ${playerNumber}`
    );
  }

  if (errors.length === 0) {
    return;
  }

  const message = `sandboxElimination invariant violation (${context}):` + '\n' + errors.join('\n');

  console.error(message);

  if (isTestEnv) {
    throw new Error(message);
  }
}

/**
 * Core elimination helper operating directly on the board and players.
 * This mirrors the logic in ClientSandboxEngine.forceEliminateCap but is
 * pure with respect to GameState, returning updated structures and the
 * number of rings eliminated.
 *
 * Per RR-CANON-R022, R122, R145, R100:
 * - 'line': Eliminate exactly ONE ring from the top (any controlled stack is eligible)
 * - 'territory': Eliminate entire cap (only eligible stacks: multicolor or height > 1)
 * - 'forced': Eliminate entire cap (any controlled stack is eligible)
 */
export function forceEliminateCapOnBoard(
  board: BoardState,
  players: Player[],
  playerNumber: number,
  stacks: RingStack[],
  eliminationContext: 'line' | 'territory' | 'forced' = 'forced'
): ForcedEliminationResult {
  const player = players.find((p) => p.playerNumber === playerNumber);
  if (!player) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  if (stacks.length === 0) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  const stack = stacks.find((s) => s.capHeight > 0) ?? stacks[0];
  const capHeight = calculateCapHeight(stack.rings);
  if (capHeight <= 0) {
    return { board, players, totalRingsEliminatedDelta: 0 };
  }

  // Determine how many rings to eliminate based on context (RR-CANON-R022, R122):
  // - 'line': Eliminate exactly ONE ring (per RR-CANON-R122)
  // - 'territory' or 'forced': Eliminate entire cap (per RR-CANON-R145, R100)
  const ringsToEliminate = eliminationContext === 'line' ? 1 : capHeight;

  if (TERRITORY_TRACE_DEBUG) {
    // eslint-disable-next-line no-console
    console.log('[sandboxElimination.forceEliminateCapOnBoard]', {
      playerNumber,
      stackPosition: stack.position,
      capHeight,
      stackHeight: stack.stackHeight,
      eliminationContext,
      ringsToEliminate,
    });
  }

  const remainingRings = stack.rings.slice(ringsToEliminate);

  const updatedEliminatedRings = { ...board.eliminatedRings };
  updatedEliminatedRings[playerNumber] =
    (updatedEliminatedRings[playerNumber] || 0) + ringsToEliminate;

  const updatedPlayers = players.map((p) =>
    p.playerNumber === playerNumber
      ? { ...p, eliminatedRings: p.eliminatedRings + ringsToEliminate }
      : p
  );

  const nextBoard: BoardState = {
    ...board,
    stacks: new Map(board.stacks),
    markers: new Map(board.markers),
    collapsedSpaces: new Map(board.collapsedSpaces),
    territories: new Map(board.territories),
    formedLines: [...board.formedLines],
    eliminatedRings: updatedEliminatedRings,
  };

  if (remainingRings.length > 0) {
    const newStack: RingStack = {
      ...stack,
      rings: remainingRings,
      stackHeight: remainingRings.length,
      capHeight: calculateCapHeight(remainingRings),
      controllingPlayer: remainingRings[0],
    };
    const key = positionToString(stack.position);
    nextBoard.stacks.set(key, newStack);
  } else {
    const key = positionToString(stack.position);
    nextBoard.stacks.delete(key);
  }

  const result: ForcedEliminationResult = {
    board: nextBoard,
    players: updatedPlayers,
    totalRingsEliminatedDelta: ringsToEliminate,
  };

  assertForcedEliminationConsistency(
    'forceEliminateCapOnBoard',
    { board, players },
    { board: result.board, players: result.players, delta: result.totalRingsEliminatedDelta },
    playerNumber
  );

  return result;
}
