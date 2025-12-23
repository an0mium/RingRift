import { useState, useCallback } from 'react';
import type { Position } from '../../shared/types/game';

/**
 * Board selection state managed by the hook.
 */
export interface BoardSelectionState {
  /** Currently selected cell position (e.g., clicked stack) */
  selectedCell: Position | null;
  /** Cells highlighted as valid targets for the current selection */
  highlightedCells: Position[];
}

/**
 * Actions for managing board selection.
 */
export interface BoardSelectionActions {
  /** Set the currently selected cell */
  setSelectedCell: (cell: Position | null) => void;
  /** Set the highlighted target cells */
  setHighlightedCells: (cells: Position[]) => void;
  /** Clear all selection state (selected cell + highlights) */
  clearSelection: () => void;
}

/**
 * Return type for useSandboxBoardSelection hook.
 */
export type UseSandboxBoardSelectionReturn = [BoardSelectionState, BoardSelectionActions];

/**
 * Custom hook for managing sandbox board selection state.
 *
 * Handles:
 * - Currently selected cell (e.g., clicked stack for movement)
 * - Highlighted cells (valid move targets)
 * - Clearing all selection state at once
 *
 * Extracted from SandboxGameHost to reduce component complexity.
 *
 * @returns Tuple of [state, actions] for board selection management
 */
export function useSandboxBoardSelection(): UseSandboxBoardSelectionReturn {
  // Currently selected cell on the board (using undefined internally to match prior state,
  // but exposed as null for cleaner external API)
  const [selected, setSelected] = useState<Position | undefined>(undefined);

  // Valid target positions for the current selection
  const [validTargets, setValidTargets] = useState<Position[]>([]);

  // Set the selected cell (converts null to undefined internally)
  const setSelectedCell = useCallback((cell: Position | null) => {
    setSelected(cell ?? undefined);
  }, []);

  // Set highlighted cells
  const setHighlightedCells = useCallback((cells: Position[]) => {
    setValidTargets(cells);
  }, []);

  // Clear all selection state
  const clearSelection = useCallback(() => {
    setSelected(undefined);
    setValidTargets([]);
  }, []);

  const state: BoardSelectionState = {
    selectedCell: selected ?? null,
    highlightedCells: validTargets,
  };

  const actions: BoardSelectionActions = {
    setSelectedCell,
    setHighlightedCells,
    clearSelection,
  };

  return [state, actions];
}

export default useSandboxBoardSelection;
