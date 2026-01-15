/**
 * Tutorial Annotations Hook
 *
 * Generates explanatory annotations for valid moves during "Learn the Basics" mode.
 * Helps new players understand what each move does before making it.
 */

import { useMemo } from 'react';
import type { GameState, Move, Position } from '../../shared/types/game';
import { positionToString } from '../../shared/types/game';

export interface MoveAnnotation {
  /** Position of the annotation on the board */
  position: Position;
  /** Brief explanation of the move */
  explanation: string;
  /** Whether this is a recommended move for beginners */
  isRecommended: boolean;
  /** Move type for styling purposes */
  moveType: 'placement' | 'movement' | 'capture';
}

interface UseTutorialAnnotationsOptions {
  /** Current game state */
  gameState: GameState | null;
  /** List of valid moves for the current player */
  validMoves: Move[];
  /** Whether tutorial mode is active */
  isLearnMode: boolean;
  /** Currently selected position (for context-aware annotations) */
  selectedPosition?: Position;
}

/**
 * Generates move annotations for the board.
 *
 * In tutorial mode, this provides helpful explanations for each valid move target,
 * helping new players understand the consequences of their actions.
 */
export function useTutorialAnnotations({
  gameState,
  validMoves,
  isLearnMode,
  selectedPosition,
}: UseTutorialAnnotationsOptions): MoveAnnotation[] {
  return useMemo(() => {
    if (!isLearnMode || !gameState || validMoves.length === 0) {
      return [];
    }

    const currentPhase = gameState.currentPhase;
    const board = gameState.board;
    const annotations: MoveAnnotation[] = [];

    // Group moves by target position for deduplication
    const movesByTarget = new Map<string, Move[]>();
    for (const move of validMoves) {
      const targetKey = positionToString(move.to);
      const existing = movesByTarget.get(targetKey) || [];
      existing.push(move);
      movesByTarget.set(targetKey, existing);
    }

    for (const [_targetKey, moves] of movesByTarget) {
      const move = moves[0]; // Use first move for the position
      const target = move.to;
      const targetCell = board.grid.get(positionToString(target));

      if (currentPhase === 'ring_placement') {
        annotations.push(generatePlacementAnnotation(target, targetCell, board, gameState));
      } else if (currentPhase === 'movement' && selectedPosition) {
        annotations.push(
          generateMovementAnnotation(target, targetCell, selectedPosition, board, gameState)
        );
      } else if (
        (currentPhase === 'capture' || currentPhase === 'chain_capture') &&
        selectedPosition
      ) {
        // For captures, the move involves jumping over an enemy
        annotations.push(
          generateCaptureAnnotation(target, targetCell, selectedPosition, board, gameState)
        );
      }
    }

    return annotations;
  }, [gameState, validMoves, isLearnMode, selectedPosition]);
}

function generatePlacementAnnotation(
  target: Position,
  targetCell: ReturnType<Map<string, unknown>['get']>,
  board: GameState['board'],
  gameState: GameState
): MoveAnnotation {
  const currentPlayer = gameState.currentPlayer;

  // Check if this is an empty cell or existing stack
  const isEmpty = !targetCell || (targetCell as { stack?: unknown[] }).stack?.length === 0;

  // Check for adjacent friendly stacks
  const adjacentFriendlyCount = countAdjacentFriendlyStacks(target, board, currentPlayer);

  let explanation: string;
  let isRecommended = false;

  if (isEmpty) {
    if (adjacentFriendlyCount > 0) {
      explanation = `Start new stack (${adjacentFriendlyCount} friendly neighbor${adjacentFriendlyCount > 1 ? 's' : ''})`;
      isRecommended = true; // Building connected groups is a good strategy
    } else {
      explanation = 'Start new stack';
    }
  } else {
    explanation = 'Add ring to existing stack';
    // Stacking on existing positions can be strategic
  }

  return {
    position: target,
    explanation,
    isRecommended,
    moveType: 'placement',
  };
}

function generateMovementAnnotation(
  target: Position,
  targetCell: ReturnType<Map<string, unknown>['get']>,
  from: Position,
  board: GameState['board'],
  _gameState: GameState
): MoveAnnotation {
  const fromCell = board.grid.get(positionToString(from)) as { stack?: unknown[] } | undefined;
  const stackHeight = fromCell?.stack?.length ?? 1;

  // Calculate distance moved (simplified - actual distance calc would need board geometry)
  const hasMarker = targetCell && (targetCell as { hasMarker?: boolean }).hasMarker;

  let explanation: string;
  let isRecommended = false;

  if (hasMarker) {
    explanation = `Move stack (landing on marker - top ring eliminated)`;
    isRecommended = false; // Landing on markers loses a ring
  } else {
    explanation = `Move stack (height ${stackHeight})`;
    isRecommended = true;
  }

  return {
    position: target,
    explanation,
    isRecommended,
    moveType: 'movement',
  };
}

function generateCaptureAnnotation(
  target: Position,
  _targetCell: ReturnType<Map<string, unknown>['get']>,
  from: Position,
  board: GameState['board'],
  _gameState: GameState
): MoveAnnotation {
  // Find the captured stack position (midpoint between from and to)
  const capturedPos = {
    row: Math.floor((from.row + target.row) / 2),
    col: Math.floor((from.col + target.col) / 2),
  };
  const capturedCell = board.grid.get(positionToString(capturedPos)) as
    | { stack?: unknown[] }
    | undefined;
  const capturedStackHeight = capturedCell?.stack?.length ?? 0;

  const explanation = `Capture stack (+${capturedStackHeight} ring${capturedStackHeight !== 1 ? 's' : ''})`;

  return {
    position: target,
    explanation,
    isRecommended: true, // Captures are generally good
    moveType: 'capture',
  };
}

function countAdjacentFriendlyStacks(
  pos: Position,
  board: GameState['board'],
  currentPlayer: number
): number {
  const adjacentOffsets = [
    { row: -1, col: 0 },
    { row: 1, col: 0 },
    { row: 0, col: -1 },
    { row: 0, col: 1 },
    { row: -1, col: -1 },
    { row: -1, col: 1 },
    { row: 1, col: -1 },
    { row: 1, col: 1 },
  ];

  let count = 0;
  for (const offset of adjacentOffsets) {
    const adjacentPos = { row: pos.row + offset.row, col: pos.col + offset.col };
    const cell = board.grid.get(positionToString(adjacentPos)) as
      | { stack?: Array<{ owner: number }> }
      | undefined;
    if (cell?.stack && cell.stack.length > 0) {
      const topRing = cell.stack[cell.stack.length - 1];
      if (topRing.owner === currentPlayer) {
        count++;
      }
    }
  }
  return count;
}
