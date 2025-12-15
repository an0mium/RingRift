import { useState, useCallback } from 'react';
import type { Position, RingStack } from '../../shared/types/game';

export interface CellInfoState {
  position: Position;
  stack?: RingStack | null;
  isCollapsed?: boolean;
  collapsedOwner?: number;
  isValidTarget?: boolean;
}

export interface UseCellInfoToastReturn {
  /** Current cell info to display, or null if hidden */
  cellInfo: CellInfoState | null;
  /** Show the cell info toast for a specific cell */
  showCellInfo: (info: CellInfoState) => void;
  /** Hide the cell info toast */
  hideCellInfo: () => void;
}

/**
 * Hook for managing the cell info toast state.
 * Used to show cell details on long-press in touch interfaces.
 *
 * The long-press detection happens in BoardView's touch handlers;
 * this hook just manages the visibility state of the info toast.
 *
 * @example
 * ```tsx
 * const { cellInfo, showCellInfo, hideCellInfo } = useCellInfoToast();
 *
 * // In BoardView's onCellContextMenu (triggered by long-press):
 * const handleContextMenu = (pos: Position) => {
 *   const stack = board.stacks.get(positionToString(pos));
 *   showCellInfo({ position: pos, stack });
 * };
 *
 * // Render the toast:
 * {cellInfo && (
 *   <CellInfoToast {...cellInfo} onDismiss={hideCellInfo} />
 * )}
 * ```
 */
export function useCellInfoToast(): UseCellInfoToastReturn {
  const [cellInfo, setCellInfo] = useState<CellInfoState | null>(null);

  const showCellInfo = useCallback((info: CellInfoState) => {
    setCellInfo(info);
  }, []);

  const hideCellInfo = useCallback(() => {
    setCellInfo(null);
  }, []);

  return {
    cellInfo,
    showCellInfo,
    hideCellInfo,
  };
}

export default useCellInfoToast;
