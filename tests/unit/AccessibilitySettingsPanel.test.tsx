import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { AccessibilitySettingsPanel } from '../../src/client/components/AccessibilitySettingsPanel';
import * as AccessibilityContext from '../../src/client/contexts/AccessibilityContext';

// Mock the useAccessibility hook
const mockSetPreference = jest.fn();
const mockResetPreferences = jest.fn();
const mockGetPlayerColor = jest.fn().mockImplementation((index) => {
  const colors = ['#10b981', '#0ea5e9', '#f59e0b', '#d946ef'];
  return colors[index] || '#64748b';
});
const mockGetPlayerColorClass = jest.fn().mockReturnValue('bg-emerald-500');

const defaultMockContext: AccessibilityContext.AccessibilityContextValue = {
  highContrastMode: false,
  colorVisionMode: 'normal',
  reducedMotion: false,
  largeText: false,
  systemPrefersReducedMotion: false,
  effectiveReducedMotion: false,
  setPreference: mockSetPreference,
  resetPreferences: mockResetPreferences,
  getPlayerColor: mockGetPlayerColor,
  getPlayerColorClass: mockGetPlayerColorClass,
};

// Helper to render with mocked context
function renderWithContext(
  contextOverrides: Partial<AccessibilityContext.AccessibilityContextValue> = {},
  props: Partial<React.ComponentProps<typeof AccessibilitySettingsPanel>> = {}
) {
  const mockContext = { ...defaultMockContext, ...contextOverrides };
  jest.spyOn(AccessibilityContext, 'useAccessibility').mockReturnValue(mockContext);

  return render(<AccessibilitySettingsPanel {...props} />);
}

describe('AccessibilitySettingsPanel', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Rendering', () => {
    it('renders without crashing', () => {
      renderWithContext();

      expect(screen.getByText('Accessibility')).toBeInTheDocument();
    });

    it('displays all accessibility toggle options', () => {
      renderWithContext();

      expect(screen.getByText('High Contrast Mode')).toBeInTheDocument();
      expect(screen.getByText('Reduce Motion')).toBeInTheDocument();
      expect(screen.getByText('Large Text')).toBeInTheDocument();
      expect(screen.getByText('Color Vision Mode')).toBeInTheDocument();
    });

    it('displays reset to defaults button', () => {
      renderWithContext();

      expect(screen.getByText('Reset to defaults')).toBeInTheDocument();
    });

    it('displays keyboard shortcuts hint', () => {
      renderWithContext();

      expect(screen.getByText(/Press/)).toBeInTheDocument();
      expect(screen.getByText('?')).toBeInTheDocument();
    });

    it('displays player color preview', () => {
      renderWithContext();

      expect(screen.getByText('Player color preview:')).toBeInTheDocument();
      // Check for player indicators (1-4)
      expect(screen.getByText('1')).toBeInTheDocument();
      expect(screen.getByText('2')).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument();
      expect(screen.getByText('4')).toBeInTheDocument();
    });
  });

  describe('Toggle Interactions', () => {
    it('calls setPreference when high contrast toggle is clicked', () => {
      renderWithContext();

      const toggle = screen.getByRole('switch', { name: /High Contrast Mode/i });
      fireEvent.click(toggle);

      expect(mockSetPreference).toHaveBeenCalledWith('highContrastMode', true);
    });

    it('calls setPreference when reduce motion toggle is clicked', () => {
      renderWithContext();

      const toggle = screen.getByRole('switch', { name: /Reduce Motion/i });
      fireEvent.click(toggle);

      expect(mockSetPreference).toHaveBeenCalledWith('reducedMotion', true);
    });

    it('calls setPreference when large text toggle is clicked', () => {
      renderWithContext();

      const toggle = screen.getByRole('switch', { name: /Large Text/i });
      fireEvent.click(toggle);

      expect(mockSetPreference).toHaveBeenCalledWith('largeText', true);
    });

    it('toggles from on to off when already enabled', () => {
      renderWithContext({ highContrastMode: true });

      const toggle = screen.getByRole('switch', { name: /High Contrast Mode/i });
      fireEvent.click(toggle);

      expect(mockSetPreference).toHaveBeenCalledWith('highContrastMode', false);
    });
  });

  describe('Color Vision Mode Selection', () => {
    it('renders color vision mode dropdown with all options', () => {
      renderWithContext();

      const select = screen.getByRole('combobox');
      expect(select).toBeInTheDocument();

      expect(screen.getByText('Standard Colors')).toBeInTheDocument();
    });

    it('calls setPreference when color vision mode is changed', () => {
      renderWithContext();

      const select = screen.getByRole('combobox');
      fireEvent.change(select, { target: { value: 'deuteranopia' } });

      expect(mockSetPreference).toHaveBeenCalledWith('colorVisionMode', 'deuteranopia');
    });

    it('displays current color vision mode as selected', () => {
      renderWithContext({ colorVisionMode: 'protanopia' });

      const select = screen.getByRole('combobox') as HTMLSelectElement;
      expect(select.value).toBe('protanopia');
    });
  });

  describe('Reset Functionality', () => {
    it('calls resetPreferences when reset button is clicked', () => {
      renderWithContext();

      const resetButton = screen.getByText('Reset to defaults');
      fireEvent.click(resetButton);

      expect(mockResetPreferences).toHaveBeenCalledTimes(1);
    });
  });

  describe('Callback Props', () => {
    it('calls onSettingsChange when a setting is changed', () => {
      const onSettingsChange = jest.fn();
      renderWithContext({}, { onSettingsChange });

      const toggle = screen.getByRole('switch', { name: /High Contrast Mode/i });
      fireEvent.click(toggle);

      expect(onSettingsChange).toHaveBeenCalledTimes(1);
    });

    it('calls onSettingsChange when reset is clicked', () => {
      const onSettingsChange = jest.fn();
      renderWithContext({}, { onSettingsChange });

      const resetButton = screen.getByText('Reset to defaults');
      fireEvent.click(resetButton);

      expect(onSettingsChange).toHaveBeenCalledTimes(1);
    });
  });

  describe('Compact Mode', () => {
    it('shows shorter descriptions in compact mode', () => {
      renderWithContext({}, { compact: true });

      // In compact mode, High Contrast should show shorter description
      expect(screen.getByText('Stronger borders and colors')).toBeInTheDocument();
    });

    it('shows full descriptions in normal mode', () => {
      renderWithContext({}, { compact: false });

      expect(
        screen.getByText(/Increases visual distinction with thicker borders/)
      ).toBeInTheDocument();
    });
  });

  describe('System Reduced Motion', () => {
    it('shows system preference message when systemPrefersReducedMotion is true', () => {
      renderWithContext({ systemPrefersReducedMotion: true });

      expect(screen.getByText(/Your system prefers reduced motion/)).toBeInTheDocument();
    });

    it('shows warning when system prefers reduced motion but user setting is off', () => {
      renderWithContext({
        systemPrefersReducedMotion: true,
        reducedMotion: false,
      });

      expect(screen.getByText(/System reduced motion is enabled/)).toBeInTheDocument();
    });
  });

  describe('ARIA Accessibility', () => {
    it('has proper role="switch" attributes on toggles', () => {
      renderWithContext();

      const switches = screen.getAllByRole('switch');
      expect(switches.length).toBeGreaterThanOrEqual(3); // High Contrast, Reduce Motion, Large Text

      switches.forEach((toggle) => {
        expect(toggle).toHaveAttribute('aria-checked');
      });
    });

    it('toggles have proper aria-checked state', () => {
      renderWithContext({ highContrastMode: true });

      const toggle = screen.getByRole('switch', { name: /High Contrast Mode/i });
      expect(toggle).toHaveAttribute('aria-checked', 'true');
    });

    it('labels are associated with their controls', () => {
      renderWithContext();

      // The combobox should have an associated label
      const select = screen.getByRole('combobox');
      expect(select).toHaveAttribute('id', 'color-vision');
    });
  });
});