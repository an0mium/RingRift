import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { BoardControlsOverlay } from '../../src/client/components/BoardControlsOverlay';

describe('BoardControlsOverlay', () => {
  it('renders basic controls and keyboard shortcuts in backend mode', () => {
    const onClose = jest.fn();

    render(<BoardControlsOverlay mode="backend" onClose={onClose} />);

    expect(screen.getByTestId('board-controls-overlay')).toBeInTheDocument();
    expect(screen.getByTestId('board-controls-basic-section')).toBeInTheDocument();
    expect(screen.getByTestId('board-controls-keyboard-section')).toBeInTheDocument();
    expect(screen.getByText(/\?/i)).toBeInTheDocument();
  });

  it('renders sandbox touch controls and touch panel details when enabled', () => {
    const onClose = jest.fn();

    render(<BoardControlsOverlay mode="sandbox" hasTouchControlsPanel onClose={onClose} />);

    expect(screen.getByTestId('board-controls-sandbox-section')).toBeInTheDocument();

    const panelHeadings = screen.getAllByText(/Sandbox touch controls panel/i);
    expect(panelHeadings.length).toBeGreaterThan(0);

    expect(screen.getByText(/Clear selection/i)).toBeInTheDocument();
    expect(screen.getByText(/Finish move/i)).toBeInTheDocument();
    expect(screen.getByText(/Show valid targets/i)).toBeInTheDocument();
    expect(screen.getByText(/Show movement grid/i)).toBeInTheDocument();
  });

  it('invokes onClose when the close button is clicked', () => {
    const onClose = jest.fn();

    render(<BoardControlsOverlay mode="backend" onClose={onClose} />);

    fireEvent.click(screen.getByTestId('board-controls-close-button'));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('renders spectator-specific read-only copy and omits keyboard-play note', () => {
    const onClose = jest.fn();

    render(<BoardControlsOverlay mode="spectator" onClose={onClose} />);

    // Spectator badge should be visible in the header.
    expect(screen.getByText(/Spectator/i)).toBeInTheDocument();

    // Basic controls section should explain read-only behaviour for spectators.
    const basicSection = screen.getByTestId('board-controls-basic-section');
    expect(basicSection).toHaveAttribute('aria-label', 'Basic board controls');
    expect(
      screen.getByText(
        /As a spectator the board is read-only: you can click cells to inspect stacks, but cannot submit moves\./i
      )
    ).toBeInTheDocument();

    // Keyboard section is present, but the "play entirely without a mouse" note
    // should NOT be rendered for spectators.
    const keyboardSection = screen.getByTestId('board-controls-keyboard-section');
    expect(keyboardSection).toHaveAttribute('aria-label', 'Keyboard shortcuts');
    expect(screen.queryByText(/play RingRift entirely without a mouse/i)).not.toBeInTheDocument();

    // Overlay itself should expose dialog semantics.
    const overlay = screen.getByTestId('board-controls-overlay');
    expect(overlay).toHaveAttribute('role', 'dialog');
    expect(overlay).toHaveAttribute('aria-modal', 'true');
    expect(overlay).toHaveAttribute('aria-labelledby', 'board-controls-title');
  });
});
