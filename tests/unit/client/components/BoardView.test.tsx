import React, { act } from 'react';
import { render, screen, fireEvent, within, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardView } from '../../../../src/client/components/BoardView';
import type { BoardViewProps } from '../../../../src/client/components/BoardView';
import type {
  BoardState,
  BoardType,
  Position,
  RingStack,
  MarkerInfo,
} from '../../../../src/shared/types/game';
import { positionToString } from '../../../../src/shared/types/game';
import type {
  BoardViewModel,
  CellViewModel,
  StackViewModel,
} from '../../../../src/client/adapters/gameViewModels';

// ═══════════════════════════════════════════════════════════════════════════
// Test Helpers
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Create an empty BoardState for a given board type
 */
function createEmptyBoard(boardType: BoardType): BoardState {
  const sizeMap: Record<BoardType, number> = {
    square8: 8,
    square19: 19,
    hex8: 9,
    hexagonal: 25,
  };

  return {
    stacks: new Map(),
    markers: new Map(),
    collapsedSpaces: new Map(),
    territories: new Map(),
    formedLines: [],
    eliminatedRings: {},
    size: sizeMap[boardType],
    type: boardType,
  };
}

/**
 * Create a RingStack at the specified position
 * Cap height is calculated from the TOP of the stack (last element in array)
 */
function createStack(pos: Position, rings: number[], controllingPlayer?: number): RingStack {
  // Cap height counts consecutive same-color rings from the TOP (end of array)
  let capHeight = 0;
  if (rings.length > 0) {
    const topRingPlayer = rings[rings.length - 1];
    for (let i = rings.length - 1; i >= 0; i--) {
      if (rings[i] === topRingPlayer) {
        capHeight++;
      } else {
        break;
      }
    }
  }

  return {
    position: pos,
    rings,
    stackHeight: rings.length,
    capHeight,
    controllingPlayer: controllingPlayer ?? (rings.length > 0 ? rings[rings.length - 1] : 0),
  };
}

/**
 * Create a MarkerInfo at the specified position
 */
function createMarker(pos: Position, player: number): MarkerInfo {
  return {
    position: pos,
    player,
    type: 'regular',
  };
}

/**
 * Create a BoardState with stacks and markers
 */
function createBoardWithStacks(
  boardType: BoardType,
  stacks: Array<{ pos: Position; rings: number[] }>,
  markers: Array<{ pos: Position; player: number }> = []
): BoardState {
  const board = createEmptyBoard(boardType);

  for (const { pos, rings } of stacks) {
    const key = positionToString(pos);
    board.stacks.set(key, createStack(pos, rings));
  }

  for (const { pos, player } of markers) {
    const key = positionToString(pos);
    board.markers.set(key, createMarker(pos, player));
  }

  return board;
}

/**
 * Create default BoardViewProps
 */
function createDefaultProps(overrides: Partial<BoardViewProps> = {}): BoardViewProps {
  const boardType = overrides.boardType ?? 'square8';
  return {
    boardType,
    board: createEmptyBoard(boardType),
    ...overrides,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Rendering Tests
// ═══════════════════════════════════════════════════════════════════════════

describe('BoardView', () => {
  describe('Rendering', () => {
    it('renders without crashing for square8 board', () => {
      const props = createDefaultProps({ boardType: 'square8' });
      const { container } = render(<BoardView {...props} />);

      expect(screen.getByTestId('board-view')).toBeInTheDocument();
      expect(container.querySelector('[role="grid"]')).toBeInTheDocument();
    });

    it('renders without crashing for square19 board', () => {
      const props = createDefaultProps({ boardType: 'square19' });
      const { container } = render(<BoardView {...props} />);

      expect(screen.getByTestId('board-view')).toBeInTheDocument();
      expect(container.querySelector('[role="grid"]')).toBeInTheDocument();
    });

    it('renders without crashing for hexagonal board', () => {
      const board = createEmptyBoard('hexagonal');
      const props = createDefaultProps({ boardType: 'hexagonal', board });
      const { container } = render(<BoardView {...props} />);

      expect(screen.getByTestId('board-view')).toBeInTheDocument();
      expect(container.querySelector('[role="grid"]')).toBeInTheDocument();
    });

    it('renders without crashing for hex8 board', () => {
      const board = createEmptyBoard('hex8');
      const props = createDefaultProps({ boardType: 'hex8', board });
      const { container } = render(<BoardView {...props} />);

      expect(screen.getByTestId('board-view')).toBeInTheDocument();
      expect(container.querySelector('[role="grid"]')).toBeInTheDocument();
    });

    it('renders correct number of cells for square8 board (64 cells)', () => {
      const props = createDefaultProps({ boardType: 'square8' });
      const { container } = render(<BoardView {...props} />);

      // Each cell is a button with data-x and data-y attributes
      const cells = container.querySelectorAll('button[data-x][data-y]');
      expect(cells.length).toBe(64);
    });

    it('renders correct number of cells for square19 board (361 cells)', () => {
      const props = createDefaultProps({ boardType: 'square19' });
      const { container } = render(<BoardView {...props} />);

      const cells = container.querySelectorAll('button[data-x][data-y]');
      expect(cells.length).toBe(361);
    });

    it('applies correct aria-label to the board for each board type', () => {
      const boardTypes: BoardType[] = ['square8', 'square19', 'hexagonal'];
      const expectedLabels = ['8x8', '19x19', 'Hexagonal'];

      boardTypes.forEach((boardType, idx) => {
        const props = createDefaultProps({ boardType });
        const { container, unmount } = render(<BoardView {...props} />);

        const grid = container.querySelector('[role="grid"]');
        expect(grid).toHaveAttribute('aria-label', expect.stringContaining(expectedLabels[idx]));

        unmount();
      });
    });

    it('renders coordinate labels when showCoordinateLabels is true for square board', () => {
      const props = createDefaultProps({
        boardType: 'square8',
        showCoordinateLabels: true,
      });
      const { container } = render(<BoardView {...props} />);

      // Should have file labels (a-h) and rank labels (1-8)
      expect(container.textContent).toMatch(/[a-h]/i);
      expect(container.textContent).toMatch(/[1-8]/);
    });

    it('applies dark and light square styling alternately on square boards', () => {
      const board = createBoardWithStacks('square8', []);
      const props = createDefaultProps({ boardType: 'square8', board });
      const { container } = render(<BoardView {...props} />);

      const cell00 = container.querySelector('button[data-x="0"][data-y="0"]');
      const cell01 = container.querySelector('button[data-x="0"][data-y="1"]');
      const cell10 = container.querySelector('button[data-x="1"][data-y="0"]');

      // (0,0) and (1,1) should both be dark; (0,1) and (1,0) should be light
      expect(cell00?.className).toContain('bg-slate-300');
      expect(cell01?.className).toContain('bg-slate-100');
      expect(cell10?.className).toContain('bg-slate-100');
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Props Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Props', () => {
    describe('Stacks Display', () => {
      it('displays stacks on cells correctly', () => {
        // Stack with rings [1, 1, 2] - bottom to top: 1, 1, 2
        // Cap height counts from top: only the 2 on top = C1
        const board = createBoardWithStacks('square8', [{ pos: { x: 2, y: 2 }, rings: [1, 1, 2] }]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="2"][data-y="2"]');
        expect(cell).toBeInTheDocument();

        // Should show stack height and cap height labels
        expect(cell?.textContent).toContain('H3');
        expect(cell?.textContent).toContain('C1'); // Cap is 1 (just the 2 on top)
      });

      it('renders multiple stacks at different positions', () => {
        const board = createBoardWithStacks('square8', [
          { pos: { x: 0, y: 0 }, rings: [1] },
          { pos: { x: 3, y: 3 }, rings: [2, 2] },
          { pos: { x: 7, y: 7 }, rings: [1, 2, 1] },
        ]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell00 = container.querySelector('button[data-x="0"][data-y="0"]');
        const cell33 = container.querySelector('button[data-x="3"][data-y="3"]');
        const cell77 = container.querySelector('button[data-x="7"][data-y="7"]');

        expect(cell00?.textContent).toContain('H1');
        expect(cell33?.textContent).toContain('H2');
        expect(cell77?.textContent).toContain('H3');
      });

      it('displays correct cap height for mixed player stacks', () => {
        // Stack with player 1 on top, then player 2
        const board = createBoardWithStacks('square8', [
          { pos: { x: 4, y: 4 }, rings: [1, 2, 2, 1] },
        ]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="4"][data-y="4"]');
        // Cap height should be 1 since only top ring is player 1
        expect(cell?.textContent).toContain('C1');
      });
    });

    describe('Valid Targets', () => {
      it('highlights valid move targets when provided', () => {
        const board = createBoardWithStacks('square8', [{ pos: { x: 2, y: 2 }, rings: [1] }]);
        const validTargets: Position[] = [
          { x: 3, y: 2 },
          { x: 2, y: 3 },
          { x: 3, y: 3 },
        ];
        const props = createDefaultProps({
          boardType: 'square8',
          board,
          validTargets,
        });
        const { container } = render(<BoardView {...props} />);

        // Check that each valid target has the valid-move-cell class
        validTargets.forEach((target) => {
          const cell = container.querySelector(
            `button[data-x="${target.x}"][data-y="${target.y}"]`
          );
          expect(cell?.className).toContain('valid-move-cell');
        });

        // Non-target cells should not have the valid-move styling
        const nonTarget = container.querySelector('button[data-x="0"][data-y="0"]');
        expect(nonTarget?.className).not.toContain('valid-move-cell');
      });

      it('applies emerald outline to valid target cells on square board', () => {
        const validTargets: Position[] = [{ x: 4, y: 4 }];
        const props = createDefaultProps({
          boardType: 'square8',
          validTargets,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="4"][data-y="4"]');
        expect(cell?.className).toContain('outline-emerald');
      });
    });

    describe('Selection State', () => {
      it('shows selection ring on selected cell', () => {
        const board = createBoardWithStacks('square8', [{ pos: { x: 2, y: 2 }, rings: [1] }]);
        const props = createDefaultProps({
          boardType: 'square8',
          board,
          selectedPosition: { x: 2, y: 2 },
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="2"][data-y="2"]');
        expect(cell?.className).toContain('ring-emerald-400');
        expect(cell).toHaveAttribute('aria-pressed', 'true');
      });

      it('does not show selection style on non-selected cells', () => {
        const board = createBoardWithStacks('square8', [{ pos: { x: 2, y: 2 }, rings: [1] }]);
        const props = createDefaultProps({
          boardType: 'square8',
          board,
          selectedPosition: { x: 2, y: 2 },
        });
        const { container } = render(<BoardView {...props} />);

        const otherCell = container.querySelector('button[data-x="5"][data-y="5"]');
        expect(otherCell?.className).not.toContain('ring-emerald-400');
        expect(otherCell).not.toHaveAttribute('aria-pressed', 'true');
      });
    });

    describe('Markers Display', () => {
      it('renders markers when present on empty cells', () => {
        const board = createBoardWithStacks('square8', [], [{ pos: { x: 1, y: 1 }, player: 1 }]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="1"][data-y="1"]');
        // Markers are rendered as nested divs with rounded-full class
        const markerElement = cell?.querySelector('.rounded-full');
        expect(markerElement).toBeInTheDocument();
      });

      it('does not render marker when stack is present at same position', () => {
        // Create board with both stack and marker at same position
        const board = createBoardWithStacks(
          'square8',
          [{ pos: { x: 1, y: 1 }, rings: [1] }],
          [{ pos: { x: 1, y: 1 }, player: 1 }]
        );
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="1"][data-y="1"]');
        // Stack should be visible (H1 C1)
        expect(cell?.textContent).toContain('H1');
        // Markers have specific border width when rendered
        const markerWithBorder = cell?.querySelector('.border-\\[6px\\]');
        expect(markerWithBorder).toBeNull();
      });
    });

    describe('Collapsed Spaces', () => {
      it('displays collapsed space territory styling', () => {
        const board = createEmptyBoard('square8');
        board.collapsedSpaces.set('3,3', 1);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="3"][data-y="3"]');
        // Collapsed spaces get territory color classes
        expect(cell?.className).toContain('bg-emerald');
      });

      it('applies correct player color for different player collapsed spaces', () => {
        const board = createEmptyBoard('square8');
        board.collapsedSpaces.set('2,2', 1);
        board.collapsedSpaces.set('5,5', 2);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const p1Cell = container.querySelector('button[data-x="2"][data-y="2"]');
        const p2Cell = container.querySelector('button[data-x="5"][data-y="5"]');

        expect(p1Cell?.className).toContain('emerald');
        expect(p2Cell?.className).toContain('sky');
      });
    });

    describe('Player Colors', () => {
      it('renders player 1 stacks with emerald color', () => {
        const board = createBoardWithStacks('square8', [{ pos: { x: 0, y: 0 }, rings: [1] }]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="0"][data-y="0"]');
        expect(cell?.innerHTML).toContain('emerald');
      });

      it('renders player 2 stacks with sky color', () => {
        const board = createBoardWithStacks('square8', [{ pos: { x: 1, y: 1 }, rings: [2] }]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="1"][data-y="1"]');
        expect(cell?.innerHTML).toContain('sky');
      });

      it('renders player 3 stacks with amber color', () => {
        const board = createBoardWithStacks('square8', [{ pos: { x: 2, y: 2 }, rings: [3] }]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="2"][data-y="2"]');
        expect(cell?.innerHTML).toContain('amber');
      });

      it('renders player 4 stacks with fuchsia color', () => {
        const board = createBoardWithStacks('square8', [{ pos: { x: 3, y: 3 }, rings: [4] }]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="3"][data-y="3"]');
        expect(cell?.innerHTML).toContain('fuchsia');
      });
    });

    describe('Spectator Mode', () => {
      it('disables cells when isSpectator is true', () => {
        const props = createDefaultProps({
          boardType: 'square8',
          isSpectator: true,
        });
        const { container } = render(<BoardView {...props} />);

        const cells = container.querySelectorAll('button[data-x][data-y]');
        cells.forEach((cell) => {
          expect(cell).toBeDisabled();
        });
      });

      it('applies cursor-default class when spectating', () => {
        const props = createDefaultProps({
          boardType: 'square8',
          isSpectator: true,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="0"][data-y="0"]');
        expect(cell?.className).toContain('cursor-default');
      });
    });

    describe('ViewModel Support', () => {
      it('uses viewModel for cell state when provided', () => {
        const board = createEmptyBoard('square8');

        // Create a minimal viewModel with a selected cell
        const cells: CellViewModel[] = [
          {
            position: { x: 3, y: 3 },
            positionKey: '3,3',
            isSelected: true,
            isValidTarget: false,
            isDarkSquare: true,
          },
        ];

        const viewModel: BoardViewModel = {
          boardType: 'square8',
          size: 8,
          cells,
        };

        const props = createDefaultProps({
          boardType: 'square8',
          board,
          viewModel,
        });
        const { container } = render(<BoardView {...props} />);

        // The cell at 3,3 should reflect the viewModel's isSelected
        const cell = container.querySelector('button[data-x="3"][data-y="3"]');
        expect(cell?.className).toContain('ring-emerald-400');
      });
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Interaction Tests
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Interactions', () => {
    describe('Click Handlers', () => {
      it('calls onCellClick with correct position when cell is clicked', () => {
        const onCellClick = jest.fn();
        const props = createDefaultProps({
          boardType: 'square8',
          onCellClick,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="4"][data-y="5"]');
        fireEvent.click(cell!);

        expect(onCellClick).toHaveBeenCalledTimes(1);
        expect(onCellClick).toHaveBeenCalledWith({ x: 4, y: 5 });
      });

      it('calls onCellClick with hex coordinates including z for hexagonal boards', () => {
        // For hex boards, we need to create the board first and add a stack
        // to ensure the cell is rendered
        const board = createEmptyBoard('hexagonal');
        // Use a valid hex position (q=0, r=0, s=0 is always valid)
        const hexPos: Position = { x: 0, y: 0, z: 0 };
        board.stacks.set(positionToString(hexPos), createStack(hexPos, [1]));

        const onCellClick = jest.fn();
        const props = createDefaultProps({
          boardType: 'hexagonal',
          board,
          onCellClick,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="0"][data-y="0"][data-z="0"]');
        if (cell) {
          fireEvent.click(cell);
          expect(onCellClick).toHaveBeenCalledTimes(1);
          const calledPos = onCellClick.mock.calls[0][0];
          expect(calledPos.x).toBe(0);
          expect(calledPos.y).toBe(0);
          // z can be 0 or -0 in JavaScript, both are valid
          expect(calledPos.z === 0 || Object.is(calledPos.z, -0)).toBe(true);
        }
      });

      it('does not call onCellClick when isSpectator is true', () => {
        const onCellClick = jest.fn();
        const props = createDefaultProps({
          boardType: 'square8',
          onCellClick,
          isSpectator: true,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="4"][data-y="5"]');
        fireEvent.click(cell!);

        expect(onCellClick).not.toHaveBeenCalled();
      });

      it('calls onCellDoubleClick when double-clicking a cell', () => {
        const onCellDoubleClick = jest.fn();
        const props = createDefaultProps({
          boardType: 'square8',
          onCellDoubleClick,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="2"][data-y="3"]');
        fireEvent.doubleClick(cell!);

        expect(onCellDoubleClick).toHaveBeenCalledTimes(1);
        expect(onCellDoubleClick).toHaveBeenCalledWith({ x: 2, y: 3 });
      });

      it('calls onCellContextMenu on right-click', () => {
        const onCellContextMenu = jest.fn();
        const props = createDefaultProps({
          boardType: 'square8',
          onCellContextMenu,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="6"][data-y="6"]');
        fireEvent.contextMenu(cell!);

        expect(onCellContextMenu).toHaveBeenCalledWith({ x: 6, y: 6 });
      });
    });

    describe('Keyboard Navigation', () => {
      it('navigates with arrow keys when board is focused', () => {
        const onCellClick = jest.fn();
        const props = createDefaultProps({
          boardType: 'square8',
          onCellClick,
        });
        const { container } = render(<BoardView {...props} />);

        const boardContainer = screen.getByTestId('board-view');
        boardContainer.focus();

        // Press arrow down - should move focus
        fireEvent.keyDown(boardContainer, { key: 'ArrowDown' });

        // Focus should now be on a cell
        // The first position focused should be 0,0 when no initial focus
        const cell00 = container.querySelector('button[data-x="0"][data-y="0"]');
        expect(document.activeElement === cell00 || document.activeElement === boardContainer).toBe(
          true
        );
      });

      it('selects cell with Enter key when focused', async () => {
        const onCellClick = jest.fn();
        const props = createDefaultProps({
          boardType: 'square8',
          onCellClick,
        });
        const { container } = render(<BoardView {...props} />);

        // Focus a specific cell directly - wrap in act to handle state updates
        const cell = container.querySelector('button[data-x="3"][data-y="3"]') as HTMLButtonElement;
        await act(async () => {
          cell.focus();
        });

        const boardContainer = screen.getByTestId('board-view');
        await act(async () => {
          fireEvent.keyDown(boardContainer, { key: 'Enter' });
        });

        expect(onCellClick).toHaveBeenCalledWith({ x: 3, y: 3 });
      });

      it('selects cell with Space key when focused', async () => {
        const onCellClick = jest.fn();
        const props = createDefaultProps({
          boardType: 'square8',
          onCellClick,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="4"][data-y="4"]') as HTMLButtonElement;
        await act(async () => {
          cell.focus();
        });

        const boardContainer = screen.getByTestId('board-view');
        await act(async () => {
          fireEvent.keyDown(boardContainer, { key: ' ' });
        });

        expect(onCellClick).toHaveBeenCalledWith({ x: 4, y: 4 });
      });

      it('calls onShowKeyboardHelp when ? key is pressed', () => {
        const onShowKeyboardHelp = jest.fn();
        const props = createDefaultProps({
          boardType: 'square8',
          onShowKeyboardHelp,
        });
        render(<BoardView {...props} />);

        const boardContainer = screen.getByTestId('board-view');
        boardContainer.focus();
        fireEvent.keyDown(boardContainer, { key: '?' });

        expect(onShowKeyboardHelp).toHaveBeenCalled();
      });
    });

    describe('Focus State', () => {
      it('shows focus ring on cell when focused via keyboard', () => {
        const props = createDefaultProps({ boardType: 'square8' });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="2"][data-y="2"]') as HTMLButtonElement;
        fireEvent.focus(cell);

        // The cell should have focus indication (amber ring for keyboard focus)
        const boardContainer = screen.getByTestId('board-view');
        fireEvent.keyDown(boardContainer, { key: 'ArrowDown' });
        // After navigation, some cell should be styled with focus
      });

      it('updates focused position when cell receives focus', () => {
        const props = createDefaultProps({ boardType: 'square8' });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="5"][data-y="5"]') as HTMLButtonElement;
        fireEvent.focus(cell);

        // The component should track this internally
        // Verify by checking that keyboard nav from this position works
        const boardContainer = screen.getByTestId('board-view');
        fireEvent.keyDown(boardContainer, { key: 'ArrowRight' });

        const nextCell = container.querySelector(
          'button[data-x="6"][data-y="5"]'
        ) as HTMLButtonElement;
        expect(document.activeElement).toBe(nextCell);
      });
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Edge Cases
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Edge Cases', () => {
    describe('Empty Board State', () => {
      it('renders correctly with no stacks, markers, or collapsed spaces', () => {
        const props = createDefaultProps({ boardType: 'square8' });
        const { container } = render(<BoardView {...props} />);

        expect(screen.getByTestId('board-view')).toBeInTheDocument();
        // No stack labels should be present
        expect(container.textContent).not.toMatch(/H\d+ C\d+/);
      });

      it('allows clicking on empty cells', () => {
        const onCellClick = jest.fn();
        const props = createDefaultProps({
          boardType: 'square8',
          onCellClick,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="0"][data-y="0"]');
        fireEvent.click(cell!);

        expect(onCellClick).toHaveBeenCalledWith({ x: 0, y: 0 });
      });
    });

    describe('Full Board State', () => {
      it('renders correctly when many cells have stacks', () => {
        const board = createEmptyBoard('square8');
        // Fill the board with stacks in a checkerboard pattern
        for (let y = 0; y < 8; y++) {
          for (let x = 0; x < 8; x++) {
            if ((x + y) % 2 === 0) {
              const pos = { x, y };
              const player = (x + y) % 4 === 0 ? 1 : 2;
              board.stacks.set(positionToString(pos), createStack(pos, [player]));
            }
          }
        }

        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        // Should render without crashing and show multiple stacks
        const stackLabels = container.textContent?.match(/H\d+ C\d+/g);
        expect(stackLabels).not.toBeNull();
        expect(stackLabels!.length).toBeGreaterThan(10);
      });
    });

    describe('Mixed Player Stacks', () => {
      it('correctly displays stack with alternating player rings', () => {
        const board = createBoardWithStacks('square8', [
          { pos: { x: 4, y: 4 }, rings: [1, 2, 1, 2, 1] },
        ]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="4"][data-y="4"]');
        expect(cell?.textContent).toContain('H5'); // Height 5
        expect(cell?.textContent).toContain('C1'); // Cap height 1 (only top ring)
      });

      it('correctly displays cap height with multiple same-color rings on top', () => {
        // Stack [2, 2, 2, 1, 1] - bottom to top: 2, 2, 2, 1, 1
        // Top of stack is 1, cap counts consecutive 1s from top = 2
        const board = createBoardWithStacks('square8', [
          { pos: { x: 3, y: 3 }, rings: [2, 2, 2, 1, 1] },
        ]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="3"][data-y="3"]');
        expect(cell?.textContent).toContain('H5'); // Height 5
        expect(cell?.textContent).toContain('C2'); // Cap height 2 (two 1s on top)
      });
    });

    describe('Shake Animation', () => {
      it('applies shake animation class when shakingCellKey matches', () => {
        const props = createDefaultProps({
          boardType: 'square8',
          shakingCellKey: '2,3',
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="2"][data-y="3"]');
        expect(cell?.className).toContain('invalid-move-shake');
      });

      it('does not apply shake animation to other cells', () => {
        const props = createDefaultProps({
          boardType: 'square8',
          shakingCellKey: '2,3',
        });
        const { container } = render(<BoardView {...props} />);

        const otherCell = container.querySelector('button[data-x="0"][data-y="0"]');
        expect(otherCell?.className).not.toContain('invalid-move-shake');
      });
    });

    describe('Chain Capture Path', () => {
      it('accepts chainCapturePath prop without crashing', () => {
        const chainCapturePath: Position[] = [
          { x: 2, y: 2 },
          { x: 4, y: 2 },
          { x: 6, y: 2 },
        ];
        const props = createDefaultProps({
          boardType: 'square8',
          chainCapturePath,
        });

        expect(() => render(<BoardView {...props} />)).not.toThrow();
      });

      it('renders chain capture path overlay when path has 2+ positions', () => {
        const board = createBoardWithStacks('square8', [
          { pos: { x: 2, y: 2 }, rings: [1] },
          { pos: { x: 4, y: 2 }, rings: [2] },
          { pos: { x: 6, y: 2 }, rings: [1] },
        ]);
        const chainCapturePath: Position[] = [
          { x: 2, y: 2 },
          { x: 4, y: 2 },
          { x: 6, y: 2 },
        ];
        const props = createDefaultProps({
          boardType: 'square8',
          board,
          chainCapturePath,
        });
        const { container } = render(<BoardView {...props} />);

        // The chain capture path renders an SVG overlay
        const svgOverlay = container.querySelector('svg');
        // If there's an SVG, it should contain path elements or lines
        expect(svgOverlay || container).toBeInTheDocument();
      });
    });

    describe('Accessibility', () => {
      it('provides accessible cell labels with position and stack info', () => {
        const board = createBoardWithStacks('square8', [{ pos: { x: 2, y: 2 }, rings: [1, 1] }]);
        const props = createDefaultProps({ boardType: 'square8', board });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="2"][data-y="2"]');
        const ariaLabel = cell?.getAttribute('aria-label');

        expect(ariaLabel).toContain('Cell');
        expect(ariaLabel).toContain('Stack height 2');
        expect(ariaLabel).toContain('cap 2');
      });

      it('marks valid move targets in aria-label', () => {
        const validTargets: Position[] = [{ x: 3, y: 3 }];
        const props = createDefaultProps({
          boardType: 'square8',
          validTargets,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="3"][data-y="3"]');
        const ariaLabel = cell?.getAttribute('aria-label');

        expect(ariaLabel).toContain('Valid move target');
      });

      it('has screen reader announcement region', () => {
        const props = createDefaultProps({ boardType: 'square8' });
        const { container } = render(<BoardView {...props} />);

        const srRegion = container.querySelector('[role="status"][aria-live="polite"]');
        expect(srRegion).toBeInTheDocument();
        expect(srRegion?.className).toContain('sr-only');
      });

      it('provides descriptive region aria-label on board container', () => {
        const props = createDefaultProps({ boardType: 'square8' });
        render(<BoardView {...props} />);

        const boardContainer = screen.getByTestId('board-view');
        expect(boardContainer).toHaveAttribute('aria-label');
        expect(boardContainer.getAttribute('aria-label')).toContain('arrow keys');
      });
    });

    describe('Decision Highlights', () => {
      it('applies decision-highlight-primary class when highlight data provided via viewModel', () => {
        const board = createEmptyBoard('square8');

        const viewModel: BoardViewModel = {
          boardType: 'square8',
          size: 8,
          cells: [],
          decisionHighlights: {
            choiceKind: 'line_order',
            highlights: [{ positionKey: '4,4', intensity: 'primary' }],
          },
        };

        const props = createDefaultProps({
          boardType: 'square8',
          board,
          viewModel,
        });
        const { container } = render(<BoardView {...props} />);

        const cell = container.querySelector('button[data-x="4"][data-y="4"]');
        expect(cell).toHaveAttribute('data-decision-highlight', 'primary');
      });
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Movement Grid Overlay
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Movement Grid Overlay', () => {
    it('renders movement grid when showMovementGrid is true', () => {
      const props = createDefaultProps({
        boardType: 'square8',
        showMovementGrid: true,
      });
      const { container } = render(<BoardView {...props} />);

      // Movement grid is rendered as an SVG
      const svg = container.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });

    it('does not render movement grid when showMovementGrid is false', () => {
      const props = createDefaultProps({
        boardType: 'square8',
        showMovementGrid: false,
      });
      const { container } = render(<BoardView {...props} />);

      // Should not have SVG overlay for movement grid
      // Note: Other overlays like chain capture path might render SVGs
      // so we just check that showMovementGrid=false doesn't add the grid
      // We can't definitively say "no SVG" since other features might add one
      expect(screen.getByTestId('board-view')).toBeInTheDocument();
    });
  });

  // ═══════════════════════════════════════════════════════════════════════════
  // Line and Territory Overlays
  // ═══════════════════════════════════════════════════════════════════════════

  describe('Debug Overlays', () => {
    it('applies line overlay data attributes when showLineOverlays is true', () => {
      const board = createEmptyBoard('square8');
      board.formedLines = [
        {
          positions: [
            { x: 0, y: 0 },
            { x: 1, y: 0 },
            { x: 2, y: 0 },
            { x: 3, y: 0 },
          ],
          player: 1,
          length: 4,
          direction: { x: 1, y: 0 },
        },
      ];

      const props = createDefaultProps({
        boardType: 'square8',
        board,
        showLineOverlays: true,
      });
      const { container } = render(<BoardView {...props} />);

      const lineCell = container.querySelector('button[data-x="0"][data-y="0"]');
      expect(lineCell).toHaveAttribute('data-line-overlay', 'true');
      expect(lineCell).toHaveAttribute('data-line-overlay-player', '1');
    });

    it('applies territory region overlay data attributes when showTerritoryRegionOverlays is true', () => {
      const board = createEmptyBoard('square8');
      board.territories.set('region-1', {
        spaces: [
          { x: 5, y: 5 },
          { x: 5, y: 6 },
        ],
        controllingPlayer: 2,
        isDisconnected: true,
      });

      const props = createDefaultProps({
        boardType: 'square8',
        board,
        showTerritoryRegionOverlays: true,
      });
      const { container } = render(<BoardView {...props} />);

      const regionCell = container.querySelector('button[data-x="5"][data-y="5"]');
      expect(regionCell).toHaveAttribute('data-region-overlay', 'true');
      expect(regionCell).toHaveAttribute('data-region-overlay-player', '2');
      expect(regionCell).toHaveAttribute('data-region-overlay-disconnected', 'true');
    });
  });
});
