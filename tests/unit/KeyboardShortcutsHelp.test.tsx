/**
 * Unit tests for KeyboardShortcutsHelp.tsx
 *
 * Tests cover:
 * - Component rendering and visibility
 * - Keyboard shortcuts display
 * - Close functionality (button, Escape, ?, backdrop)
 * - Accessibility features (ARIA, focus management)
 * - Focus trap behavior
 * - Shortcut categories
 *
 * Target: ≥80% coverage for KeyboardShortcutsHelp.tsx
 *
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { KeyboardShortcutsHelp } from '../../src/client/components/KeyboardShortcutsHelp';

describe('KeyboardShortcutsHelp', () => {
  const mockOnClose = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Component rendering', () => {
    it('should render all keyboard shortcuts when visible', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Check sections are rendered
      expect(screen.getByText('Board Navigation')).toBeInTheDocument();
      expect(screen.getByText('Dialog Navigation')).toBeInTheDocument();
      expect(screen.getByText('General')).toBeInTheDocument();
    });

    it('should display shortcut keys correctly', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Check for various shortcut keys (multiple instances exist in different sections)
      expect(screen.getAllByText('↑').length).toBeGreaterThan(0);
      expect(screen.getAllByText('↓').length).toBeGreaterThan(0);
      expect(screen.getAllByText('←').length).toBeGreaterThan(0);
      expect(screen.getAllByText('→').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Enter').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Space').length).toBeGreaterThan(0);
      expect(screen.getAllByText('Escape')).toHaveLength(2); // Board and Dialog sections
      expect(screen.getByText('Esc')).toBeInTheDocument(); // Footer abbreviation
      expect(screen.getAllByText('?').length).toBeGreaterThan(0); // Board section and footer
      expect(screen.getAllByText('Tab').length).toBeGreaterThan(1); // Multiple instances
    });

    it('should display shortcut descriptions', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Check descriptions
      expect(screen.getByText('Navigate between board cells')).toBeInTheDocument();
      expect(screen.getByText('Select current cell')).toBeInTheDocument();
      expect(screen.getByText('Cancel current action / Clear selection')).toBeInTheDocument();
      expect(screen.getByText('Show this help dialog')).toBeInTheDocument();
      expect(screen.getByText('Navigate between options')).toBeInTheDocument();
      expect(screen.getByText('Move focus between elements')).toBeInTheDocument();
      expect(screen.getByText('Navigate to next interactive element')).toBeInTheDocument();
      expect(screen.getByText('Navigate to previous interactive element')).toBeInTheDocument();
    });
  });

  describe('Visibility control', () => {
    it('should be hidden when isOpen is false', () => {
      const { container } = render(<KeyboardShortcutsHelp isOpen={false} onClose={mockOnClose} />);

      expect(container.firstChild).toBeNull();
    });

    it('should show when isOpen is true', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(screen.getByRole('dialog')).toBeInTheDocument();
      expect(screen.getByText('Keyboard Shortcuts')).toBeInTheDocument();
    });

    it('should toggle correctly between visible and hidden', () => {
      const { rerender, container } = render(
        <KeyboardShortcutsHelp isOpen={false} onClose={mockOnClose} />
      );

      expect(container.firstChild).toBeNull();

      rerender(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);
      expect(screen.getByRole('dialog')).toBeInTheDocument();

      rerender(<KeyboardShortcutsHelp isOpen={false} onClose={mockOnClose} />);
      expect(container.firstChild).toBeNull();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const dialog = screen.getByRole('dialog');
      expect(dialog).toHaveAttribute('aria-modal', 'true');
      expect(dialog).toHaveAttribute('aria-labelledby', 'keyboard-shortcuts-title');

      const title = screen.getByText('Keyboard Shortcuts');
      expect(title).toHaveAttribute('id', 'keyboard-shortcuts-title');

      const closeButton = screen.getByRole('button', {
        name: /close keyboard shortcuts help/i,
      });
      expect(closeButton).toBeInTheDocument();
    });

    it('should focus close button on mount', async () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', {
        name: /close keyboard shortcuts help/i,
      });

      await waitFor(() => {
        expect(closeButton).toHaveFocus();
      });
    });

    it('should be keyboard navigable', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', {
        name: /close keyboard shortcuts help/i,
      });

      // Verify button can be focused
      closeButton.focus();
      expect(closeButton).toHaveFocus();

      // Tab key should work for navigation (tested in focus trap tests)
      fireEvent.keyDown(document, { key: 'Tab' });
    });
  });

  describe('Close behavior', () => {
    it('should call onClose when close button clicked', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', {
        name: /close keyboard shortcuts help/i,
      });
      fireEvent.click(closeButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should close on Escape key press', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      fireEvent.keyDown(document, { key: 'Escape' });

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should close on ? key press', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      fireEvent.keyDown(document, { key: '?' });

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should close when backdrop is clicked', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const backdrop = screen.getByRole('presentation');
      fireEvent.click(backdrop);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should not close when dialog content is clicked', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const dialog = screen.getByRole('dialog');
      fireEvent.click(dialog);

      expect(mockOnClose).not.toHaveBeenCalled();
    });

    it('should prevent default on Escape key', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const event = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true });
      const preventDefaultSpy = jest.spyOn(event, 'preventDefault');

      document.dispatchEvent(event);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });

    it('should prevent default on ? key', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const event = new KeyboardEvent('keydown', { key: '?', bubbles: true });
      const preventDefaultSpy = jest.spyOn(event, 'preventDefault');

      document.dispatchEvent(event);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });
  });

  describe('Shortcut organization', () => {
    it('should group shortcuts by category', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Verify category headers exist
      const boardNav = screen.getByText('Board Navigation');
      const dialogNav = screen.getByText('Dialog Navigation');
      const general = screen.getByText('General');

      expect(boardNav).toBeInTheDocument();
      expect(dialogNav).toBeInTheDocument();
      expect(general).toBeInTheDocument();

      // Verify they have the correct styling class
      expect(boardNav.className).toContain('uppercase');
    });

    it('should display all defined shortcuts', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Board shortcuts (4 items)
      expect(screen.getByText('Navigate between board cells')).toBeInTheDocument();
      expect(screen.getByText('Select current cell')).toBeInTheDocument();
      expect(screen.getByText('Cancel current action / Clear selection')).toBeInTheDocument();
      expect(screen.getByText('Show this help dialog')).toBeInTheDocument();

      // Dialog shortcuts (4 items)
      expect(screen.getByText('Navigate between options')).toBeInTheDocument();
      expect(screen.getByText('Select focused option')).toBeInTheDocument();
      expect(screen.getByText('Move focus between elements')).toBeInTheDocument();
      expect(screen.getByText('Close dialog (if cancellable)')).toBeInTheDocument();

      // General shortcuts (2 items)
      expect(screen.getByText('Navigate to next interactive element')).toBeInTheDocument();
      expect(screen.getByText('Navigate to previous interactive element')).toBeInTheDocument();
    });
  });

  describe('Focus trap', () => {
    it('should trap focus within dialog on Tab', async () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', {
        name: /close keyboard shortcuts help/i,
      });

      await waitFor(() => {
        expect(closeButton).toHaveFocus();
      });

      // Since there's only one focusable element (close button),
      // Tab should keep focus on it
      fireEvent.keyDown(document, { key: 'Tab' });

      // Focus should stay on the close button or cycle back to it
      await waitFor(() => {
        expect(closeButton).toHaveFocus();
      });
    });

    it('should trap focus with Shift+Tab', async () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', {
        name: /close keyboard shortcuts help/i,
      });

      await waitFor(() => {
        expect(closeButton).toHaveFocus();
      });

      // Shift+Tab should also keep focus within the dialog
      fireEvent.keyDown(document, { key: 'Tab', shiftKey: true });

      await waitFor(() => {
        expect(closeButton).toHaveFocus();
      });
    });

    it('should not interfere with other key presses', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Press a non-Tab key
      fireEvent.keyDown(document, { key: 'a' });

      // Should not call onClose
      expect(mockOnClose).not.toHaveBeenCalled();
    });
  });

  describe('Event cleanup', () => {
    it('should remove event listeners on unmount', () => {
      const { unmount } = render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const removeEventListenerSpy = jest.spyOn(document, 'removeEventListener');

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('keydown', expect.any(Function));

      removeEventListenerSpy.mockRestore();
    });

    it('should remove event listeners when isOpen changes to false', () => {
      const { rerender } = render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const removeEventListenerSpy = jest.spyOn(document, 'removeEventListener');

      rerender(<KeyboardShortcutsHelp isOpen={false} onClose={mockOnClose} />);

      expect(removeEventListenerSpy).toHaveBeenCalledWith('keydown', expect.any(Function));

      removeEventListenerSpy.mockRestore();
    });

    it('should add event listeners when isOpen changes to true', () => {
      const { rerender } = render(<KeyboardShortcutsHelp isOpen={false} onClose={mockOnClose} />);

      const addEventListenerSpy = jest.spyOn(document, 'addEventListener');

      rerender(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      expect(addEventListenerSpy).toHaveBeenCalledWith('keydown', expect.any(Function));

      addEventListenerSpy.mockRestore();
    });
  });

  describe('Footer instructions', () => {
    it('should display footer with close instructions', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Check for footer text
      const footer = screen.getByText(/Press/i);
      expect(footer).toBeInTheDocument();

      // Verify it mentions both Esc and ? keys
      expect(footer.textContent).toContain('?');
      expect(footer.textContent).toContain('Esc');
      expect(footer.textContent).toContain('close this dialog');
    });
  });

  describe('Multiple shortcuts rendering', () => {
    it('should render shortcuts with multiple keys correctly', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Check for Shift + Tab shortcut
      const shiftElements = screen.getAllByText('Shift');
      expect(shiftElements.length).toBeGreaterThan(0);

      const plusSigns = screen.getAllByText('+');
      expect(plusSigns.length).toBeGreaterThan(0);
    });

    it('should render arrow keys correctly', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // All four arrow keys should be present (appear in multiple sections)
      expect(screen.getAllByText('↑').length).toBeGreaterThan(0);
      expect(screen.getAllByText('↓').length).toBeGreaterThan(0);
      expect(screen.getAllByText('←').length).toBeGreaterThan(0);
      expect(screen.getAllByText('→').length).toBeGreaterThan(0);
    });

    it('should render Enter and Space keys correctly', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      // Both should appear multiple times in different sections
      const enterElements = screen.getAllByText('Enter');
      const spaceElements = screen.getAllByText('Space');

      expect(enterElements.length).toBeGreaterThan(1);
      expect(spaceElements.length).toBeGreaterThan(1);
    });
  });

  describe('Dialog structure', () => {
    it('should have correct modal backdrop', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const backdrop = screen.getByRole('presentation');
      expect(backdrop).toHaveClass('fixed', 'inset-0', 'z-50');
      expect(backdrop).toHaveClass('bg-black/70');
    });

    it('should have scrollable content area', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const dialog = screen.getByRole('dialog');
      const scrollArea = dialog.querySelector('.overflow-y-auto');

      expect(scrollArea).toBeInTheDocument();
      expect(scrollArea).toHaveClass('max-h-[60vh]');
    });

    it('should render dialog title correctly', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const title = screen.getByText('Keyboard Shortcuts');
      expect(title.tagName).toBe('H2');
      expect(title).toHaveClass('text-lg', 'font-semibold');
    });

    it('should render close button with SVG icon', () => {
      render(<KeyboardShortcutsHelp isOpen={true} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', {
        name: /close keyboard shortcuts help/i,
      });

      const svg = closeButton.querySelector('svg');
      expect(svg).toBeInTheDocument();
      expect(svg).toHaveAttribute('aria-hidden', 'true');
    });
  });
});
