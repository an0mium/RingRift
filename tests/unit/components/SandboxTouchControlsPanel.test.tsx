import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import { SandboxTouchControlsPanel } from '../../../src/client/components/SandboxTouchControlsPanel';

describe('SandboxTouchControlsPanel', () => {
  const defaultProps = {
    selectedPosition: undefined,
    selectedStackDetails: null,
    validTargets: [],
    isCaptureDirectionPending: false,
    captureTargets: [],
    canUndoSegment: false,
    onClearSelection: jest.fn(),
    onUndoSegment: jest.fn(),
    onApplyMove: jest.fn(),
    showMovementGrid: false,
    onToggleMovementGrid: jest.fn(),
    showValidTargets: false,
    onToggleValidTargets: jest.fn(),
    phaseLabel: 'movement',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders the panel with test id', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByTestId('sandbox-touch-controls')).toBeInTheDocument();
    });

    it('renders Touch Controls heading', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByText('Touch Controls')).toBeInTheDocument();
    });

    it('renders phase label', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} phaseLabel="capture" />);

      expect(screen.getByText('Phase: capture')).toBeInTheDocument();
    });

    it('renders phase hint when provided', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} phaseHint="Select a target" />);

      expect(screen.getByText('Select a target')).toBeInTheDocument();
    });

    it('renders selection section', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByText('Selection')).toBeInTheDocument();
    });

    it('renders Visual aids section', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByText('Visual aids')).toBeInTheDocument();
    });
  });

  describe('selection display', () => {
    it('shows None when no position selected', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByText(/Tap any stack or empty cell to begin/i)).toBeInTheDocument();
    });

    it('shows selected position coordinates (2D)', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} selectedPosition={{ x: 3, y: 5 }} />);

      expect(screen.getByText('(3, 5)')).toBeInTheDocument();
    });

    it('shows selected position coordinates (3D with z)', () => {
      render(
        <SandboxTouchControlsPanel {...defaultProps} selectedPosition={{ x: 2, y: 4, z: 1 }} />
      );

      expect(screen.getByText('(2, 4, 1)')).toBeInTheDocument();
    });

    it('shows stack summary when stack details provided', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          selectedPosition={{ x: 3, y: 5 }}
          selectedStackDetails={{ height: 4, cap: 2, controllingPlayer: 0 }}
        />
      );

      expect(screen.getByText('H4 · C2 · P0')).toBeInTheDocument();
    });

    it('shows empty cell message when position selected without stack', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          selectedPosition={{ x: 3, y: 5 }}
          selectedStackDetails={null}
        />
      );

      expect(screen.getByText(/Empty cell – select a highlighted target/i)).toBeInTheDocument();
    });

    it('shows target count', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          validTargets={[
            { x: 1, y: 1 },
            { x: 2, y: 2 },
            { x: 3, y: 3 },
          ]}
        />
      );

      expect(screen.getByText('Targets: 3')).toBeInTheDocument();
    });

    it('shows capture continuation message when pending', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          selectedPosition={{ x: 3, y: 5 }}
          isCaptureDirectionPending={true}
        />
      );

      expect(screen.getByText(/Capture continuation available/i)).toBeInTheDocument();
    });
  });

  describe('action buttons', () => {
    it('renders Clear selection button', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Clear selection/i })).toBeInTheDocument();
    });

    it('renders Undo last segment button', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Undo last segment/i })).toBeInTheDocument();
    });

    it('renders Finish move button', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Finish move/i })).toBeInTheDocument();
    });

    it('disables Clear selection when no selection', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Clear selection/i })).toBeDisabled();
    });

    it('enables Clear selection when position selected', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} selectedPosition={{ x: 1, y: 1 }} />);

      expect(screen.getByRole('button', { name: /Clear selection/i })).not.toBeDisabled();
    });

    it('disables Clear selection during capture direction pending', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          selectedPosition={{ x: 1, y: 1 }}
          isCaptureDirectionPending={true}
        />
      );

      expect(screen.getByRole('button', { name: /Clear selection/i })).toBeDisabled();
    });

    it('disables Undo button when canUndoSegment is false', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} canUndoSegment={false} />);

      expect(screen.getByRole('button', { name: /Undo last segment/i })).toBeDisabled();
    });

    it('enables Undo button when canUndoSegment is true', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} canUndoSegment={true} />);

      expect(screen.getByRole('button', { name: /Undo last segment/i })).not.toBeDisabled();
    });

    it('disables Finish move when no selection', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByRole('button', { name: /Finish move/i })).toBeDisabled();
    });

    it('disables Finish move when no valid targets', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          selectedPosition={{ x: 1, y: 1 }}
          validTargets={[]}
        />
      );

      expect(screen.getByRole('button', { name: /Finish move/i })).toBeDisabled();
    });

    it('disables Finish move during capture direction pending', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          selectedPosition={{ x: 1, y: 1 }}
          validTargets={[{ x: 2, y: 2 }]}
          isCaptureDirectionPending={true}
        />
      );

      expect(screen.getByRole('button', { name: /Finish move/i })).toBeDisabled();
    });

    it('enables Finish move with selection and targets', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          selectedPosition={{ x: 1, y: 1 }}
          validTargets={[{ x: 2, y: 2 }]}
        />
      );

      expect(screen.getByRole('button', { name: /Finish move/i })).not.toBeDisabled();
    });
  });

  describe('button callbacks', () => {
    it('calls onClearSelection when Clear button clicked', async () => {
      const onClearSelection = jest.fn();
      const user = userEvent.setup();
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          selectedPosition={{ x: 1, y: 1 }}
          onClearSelection={onClearSelection}
        />
      );

      await user.click(screen.getByRole('button', { name: /Clear selection/i }));

      expect(onClearSelection).toHaveBeenCalledTimes(1);
    });

    it('calls onUndoSegment when Undo button clicked', async () => {
      const onUndoSegment = jest.fn();
      const user = userEvent.setup();
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          canUndoSegment={true}
          onUndoSegment={onUndoSegment}
        />
      );

      await user.click(screen.getByRole('button', { name: /Undo last segment/i }));

      expect(onUndoSegment).toHaveBeenCalledTimes(1);
    });

    it('calls onApplyMove when Finish move button clicked', async () => {
      const onApplyMove = jest.fn();
      const user = userEvent.setup();
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          selectedPosition={{ x: 1, y: 1 }}
          validTargets={[{ x: 2, y: 2 }]}
          onApplyMove={onApplyMove}
        />
      );

      await user.click(screen.getByRole('button', { name: /Finish move/i }));

      expect(onApplyMove).toHaveBeenCalledTimes(1);
    });
  });

  describe('visual aid toggles', () => {
    it('renders Show valid targets checkbox', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByLabelText(/Show valid targets/i)).toBeInTheDocument();
    });

    it('renders Show movement grid checkbox', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.getByLabelText(/Show movement grid/i)).toBeInTheDocument();
    });

    it('shows valid targets checkbox as checked when enabled', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} showValidTargets={true} />);

      expect(screen.getByLabelText(/Show valid targets/i)).toBeChecked();
    });

    it('shows movement grid checkbox as checked when enabled', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} showMovementGrid={true} />);

      expect(screen.getByLabelText(/Show movement grid/i)).toBeChecked();
    });

    it('calls onToggleValidTargets when checkbox changed', async () => {
      const onToggleValidTargets = jest.fn();
      const user = userEvent.setup();
      render(
        <SandboxTouchControlsPanel {...defaultProps} onToggleValidTargets={onToggleValidTargets} />
      );

      await user.click(screen.getByLabelText(/Show valid targets/i));

      expect(onToggleValidTargets).toHaveBeenCalledWith(true);
    });

    it('calls onToggleMovementGrid when checkbox changed', async () => {
      const onToggleMovementGrid = jest.fn();
      const user = userEvent.setup();
      render(
        <SandboxTouchControlsPanel {...defaultProps} onToggleMovementGrid={onToggleMovementGrid} />
      );

      await user.click(screen.getByLabelText(/Show movement grid/i));

      expect(onToggleMovementGrid).toHaveBeenCalledWith(true);
    });
  });

  describe('debug overlays section', () => {
    it('renders debug overlays when toggle handlers provided', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          onToggleLineOverlays={jest.fn()}
          onToggleTerritoryOverlays={jest.fn()}
        />
      );

      expect(screen.getByText('Debug overlays')).toBeInTheDocument();
    });

    it('does not render debug overlays when handlers not provided', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.queryByText('Debug overlays')).not.toBeInTheDocument();
    });

    it('renders line overlay toggle', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          showLineOverlays={true}
          onToggleLineOverlays={jest.fn()}
          onToggleTerritoryOverlays={jest.fn()}
        />
      );

      expect(screen.getByLabelText(/Show detected lines/i)).toBeInTheDocument();
    });

    it('renders territory overlay toggle', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          showTerritoryOverlays={true}
          onToggleLineOverlays={jest.fn()}
          onToggleTerritoryOverlays={jest.fn()}
        />
      );

      expect(screen.getByLabelText(/Show territory regions/i)).toBeInTheDocument();
    });
  });

  describe('skip buttons', () => {
    it('renders Skip territory button when available', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          canSkipTerritoryProcessing={true}
          onSkipTerritoryProcessing={jest.fn()}
        />
      );

      expect(screen.getByTestId('sandbox-skip-territory-button')).toBeInTheDocument();
      expect(screen.getByText('Skip territory processing')).toBeInTheDocument();
    });

    it('does not render Skip territory button when not available', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.queryByTestId('sandbox-skip-territory-button')).not.toBeInTheDocument();
    });

    it('calls onSkipTerritoryProcessing when clicked', async () => {
      const onSkipTerritoryProcessing = jest.fn();
      const user = userEvent.setup();
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          canSkipTerritoryProcessing={true}
          onSkipTerritoryProcessing={onSkipTerritoryProcessing}
        />
      );

      await user.click(screen.getByTestId('sandbox-skip-territory-button'));

      expect(onSkipTerritoryProcessing).toHaveBeenCalledTimes(1);
    });

    it('renders Skip capture button when available', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          canSkipCapture={true}
          onSkipCapture={jest.fn()}
        />
      );

      expect(screen.getByTestId('sandbox-skip-capture-button')).toBeInTheDocument();
      expect(screen.getByText('Skip capture')).toBeInTheDocument();
    });

    it('calls onSkipCapture when clicked', async () => {
      const onSkipCapture = jest.fn();
      const user = userEvent.setup();
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          canSkipCapture={true}
          onSkipCapture={onSkipCapture}
        />
      );

      await user.click(screen.getByTestId('sandbox-skip-capture-button'));

      expect(onSkipCapture).toHaveBeenCalledTimes(1);
    });

    it('renders Skip recovery button when available', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          canSkipRecovery={true}
          onSkipRecovery={jest.fn()}
        />
      );

      expect(screen.getByTestId('sandbox-skip-recovery-button')).toBeInTheDocument();
      expect(screen.getByText('Skip recovery')).toBeInTheDocument();
    });

    it('calls onSkipRecovery when clicked', async () => {
      const onSkipRecovery = jest.fn();
      const user = userEvent.setup();
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          canSkipRecovery={true}
          onSkipRecovery={onSkipRecovery}
        />
      );

      await user.click(screen.getByTestId('sandbox-skip-recovery-button'));

      expect(onSkipRecovery).toHaveBeenCalledTimes(1);
    });
  });

  describe('game storage section', () => {
    it('renders game storage section when auto-save handlers provided', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          autoSaveGames={false}
          onToggleAutoSave={jest.fn()}
        />
      );

      expect(screen.getByText('Game storage')).toBeInTheDocument();
    });

    it('does not render game storage when handlers not provided', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} />);

      expect(screen.queryByText('Game storage')).not.toBeInTheDocument();
    });

    it('renders auto-save checkbox', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          autoSaveGames={true}
          onToggleAutoSave={jest.fn()}
        />
      );

      expect(screen.getByLabelText(/Auto-save completed games/i)).toBeInTheDocument();
    });

    it('shows saving status', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          autoSaveGames={true}
          onToggleAutoSave={jest.fn()}
          gameSaveStatus="saving"
        />
      );

      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });

    it('shows saved status', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          autoSaveGames={true}
          onToggleAutoSave={jest.fn()}
          gameSaveStatus="saved"
        />
      );

      expect(screen.getByText('Saved')).toBeInTheDocument();
    });

    it('shows saved locally status', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          autoSaveGames={true}
          onToggleAutoSave={jest.fn()}
          gameSaveStatus="saved-local"
        />
      );

      expect(screen.getByText('Saved locally')).toBeInTheDocument();
    });

    it('shows error status', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          autoSaveGames={true}
          onToggleAutoSave={jest.fn()}
          gameSaveStatus="error"
        />
      );

      expect(screen.getByText('Error')).toBeInTheDocument();
    });
  });

  describe('capture segments section', () => {
    it('renders capture segments when targets available', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          captureTargets={[
            { x: 1, y: 2 },
            { x: 3, y: 4 },
          ]}
        />
      );

      expect(screen.getByText('Capture segments')).toBeInTheDocument();
    });

    it('does not render capture segments when no targets', () => {
      render(<SandboxTouchControlsPanel {...defaultProps} captureTargets={[]} />);

      expect(screen.queryByText('Capture segments')).not.toBeInTheDocument();
    });

    it('displays capture target positions', () => {
      render(
        <SandboxTouchControlsPanel
          {...defaultProps}
          captureTargets={[
            { x: 1, y: 2 },
            { x: 3, y: 4, z: 5 },
          ]}
        />
      );

      expect(screen.getByText('(1, 2)')).toBeInTheDocument();
      expect(screen.getByText('(3, 4, 5)')).toBeInTheDocument();
    });
  });
});
