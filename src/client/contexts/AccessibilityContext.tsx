/**
 * AccessibilityContext - Manages user accessibility preferences
 *
 * Provides settings for:
 * - High contrast mode (increased borders, stronger colors)
 * - Colorblind-friendly palette (patterns + distinct colors)
 * - Reduced motion (respects prefers-reduced-motion, plus manual override)
 *
 * Preferences are persisted to localStorage and exposed via useAccessibility hook.
 */

import React, { createContext, useContext, useCallback, useEffect, useState, useMemo } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ColorVisionMode = 'normal' | 'deuteranopia' | 'protanopia' | 'tritanopia';

export interface AccessibilityPreferences {
  /** High contrast mode - stronger borders, higher color contrast */
  highContrastMode: boolean;
  /** Color vision deficiency mode */
  colorVisionMode: ColorVisionMode;
  /** Reduced motion - disables animations */
  reducedMotion: boolean;
  /** Large text mode - increases base font sizes */
  largeText: boolean;
}

export interface AccessibilityContextValue extends AccessibilityPreferences {
  /** Update a single preference */
  setPreference: <K extends keyof AccessibilityPreferences>(
    key: K,
    value: AccessibilityPreferences[K]
  ) => void;
  /** Reset all preferences to defaults */
  resetPreferences: () => void;
  /** Whether system prefers reduced motion */
  systemPrefersReducedMotion: boolean;
  /** Effective reduced motion (user setting OR system preference) */
  effectiveReducedMotion: boolean;
  /** Get player color class based on current color vision mode */
  getPlayerColorClass: (playerIndex: number, type: 'bg' | 'text' | 'border' | 'ring') => string;
  /** Get player color for SVG/canvas (hex value) */
  getPlayerColor: (playerIndex: number) => string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STORAGE_KEY = 'ringrift-accessibility-preferences';

const DEFAULT_PREFERENCES: AccessibilityPreferences = {
  highContrastMode: false,
  colorVisionMode: 'normal',
  reducedMotion: false,
  largeText: false,
};

/**
 * Player color palettes optimized for different color vision deficiencies.
 *
 * Normal: Standard emerald/sky/amber/fuchsia palette
 * Deuteranopia/Protanopia: Blue/orange high-contrast palette (avoids red-green confusion)
 * Tritanopia: Magenta/cyan palette (avoids blue-yellow confusion)
 */
const PLAYER_COLOR_PALETTES: Record<ColorVisionMode, string[]> = {
  normal: ['#10b981', '#0ea5e9', '#f59e0b', '#d946ef'], // emerald-500, sky-500, amber-500, fuchsia-500
  deuteranopia: ['#2563eb', '#ea580c', '#0891b2', '#7c3aed'], // blue-600, orange-600, cyan-600, violet-600
  protanopia: ['#2563eb', '#ea580c', '#0891b2', '#7c3aed'], // Same as deuteranopia
  tritanopia: ['#db2777', '#06b6d4', '#84cc16', '#f97316'], // pink-600, cyan-500, lime-500, orange-500
};

/**
 * Tailwind class mappings for player colors by vision mode and type.
 */
const PLAYER_COLOR_CLASSES: Record<ColorVisionMode, Record<string, string[]>> = {
  normal: {
    bg: ['bg-emerald-500', 'bg-sky-500', 'bg-amber-500', 'bg-fuchsia-500'],
    text: ['text-emerald-500', 'text-sky-500', 'text-amber-500', 'text-fuchsia-500'],
    border: ['border-emerald-500', 'border-sky-500', 'border-amber-500', 'border-fuchsia-500'],
    ring: ['ring-emerald-500', 'ring-sky-500', 'ring-amber-500', 'ring-fuchsia-500'],
  },
  deuteranopia: {
    bg: ['bg-blue-600', 'bg-orange-600', 'bg-cyan-600', 'bg-violet-600'],
    text: ['text-blue-600', 'text-orange-600', 'text-cyan-600', 'text-violet-600'],
    border: ['border-blue-600', 'border-orange-600', 'border-cyan-600', 'border-violet-600'],
    ring: ['ring-blue-600', 'ring-orange-600', 'ring-cyan-600', 'ring-violet-600'],
  },
  protanopia: {
    bg: ['bg-blue-600', 'bg-orange-600', 'bg-cyan-600', 'bg-violet-600'],
    text: ['text-blue-600', 'text-orange-600', 'text-cyan-600', 'text-violet-600'],
    border: ['border-blue-600', 'border-orange-600', 'border-cyan-600', 'border-violet-600'],
    ring: ['ring-blue-600', 'ring-orange-600', 'ring-cyan-600', 'ring-violet-600'],
  },
  tritanopia: {
    bg: ['bg-pink-600', 'bg-cyan-500', 'bg-lime-500', 'bg-orange-500'],
    text: ['text-pink-600', 'text-cyan-500', 'text-lime-500', 'text-orange-500'],
    border: ['border-pink-600', 'border-cyan-500', 'border-lime-500', 'border-orange-500'],
    ring: ['ring-pink-600', 'ring-cyan-500', 'ring-lime-500', 'ring-orange-500'],
  },
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const AccessibilityContext = createContext<AccessibilityContextValue | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

interface AccessibilityProviderProps {
  children: React.ReactNode;
}

export function AccessibilityProvider({
  children,
}: AccessibilityProviderProps): React.ReactElement {
  // Load initial preferences from localStorage
  const [preferences, setPreferences] = useState<AccessibilityPreferences>(() => {
    if (typeof window === 'undefined') return DEFAULT_PREFERENCES;
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as Partial<AccessibilityPreferences>;
        return { ...DEFAULT_PREFERENCES, ...parsed };
      }
    } catch {
      // Ignore parse errors
    }
    return DEFAULT_PREFERENCES;
  });

  // Track system preference for reduced motion
  const [systemPrefersReducedMotion, setSystemPrefersReducedMotion] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });

  // Listen for system preference changes
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const handler = (e: MediaQueryListEvent) => setSystemPrefersReducedMotion(e.matches);

    mediaQuery.addEventListener('change', handler);
    return () => mediaQuery.removeEventListener('change', handler);
  }, []);

  // Persist preferences to localStorage
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    } catch {
      // Ignore storage errors
    }
  }, [preferences]);

  // Apply CSS class to document root for global styling
  useEffect(() => {
    if (typeof document === 'undefined') return;

    const root = document.documentElement;

    // High contrast mode
    root.classList.toggle('high-contrast', preferences.highContrastMode);

    // Color vision mode
    root.dataset.colorVision = preferences.colorVisionMode;

    // Reduced motion (explicit user preference overrides system)
    const effectiveReducedMotion = preferences.reducedMotion || systemPrefersReducedMotion;
    root.classList.toggle('reduce-motion', effectiveReducedMotion);

    // Large text
    root.classList.toggle('large-text', preferences.largeText);
  }, [preferences, systemPrefersReducedMotion]);

  const setPreference = useCallback(
    <K extends keyof AccessibilityPreferences>(key: K, value: AccessibilityPreferences[K]) => {
      setPreferences((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const resetPreferences = useCallback(() => {
    setPreferences(DEFAULT_PREFERENCES);
  }, []);

  const effectiveReducedMotion = preferences.reducedMotion || systemPrefersReducedMotion;

  const getPlayerColorClass = useCallback(
    (playerIndex: number, type: 'bg' | 'text' | 'border' | 'ring'): string => {
      const palette = PLAYER_COLOR_CLASSES[preferences.colorVisionMode];
      const classes = palette[type];
      const safeIndex = Math.max(0, Math.min(playerIndex, classes.length - 1));
      return classes[safeIndex];
    },
    [preferences.colorVisionMode]
  );

  const getPlayerColor = useCallback(
    (playerIndex: number): string => {
      const palette = PLAYER_COLOR_PALETTES[preferences.colorVisionMode];
      const safeIndex = Math.max(0, Math.min(playerIndex, palette.length - 1));
      return palette[safeIndex];
    },
    [preferences.colorVisionMode]
  );

  const value = useMemo<AccessibilityContextValue>(
    () => ({
      ...preferences,
      setPreference,
      resetPreferences,
      systemPrefersReducedMotion,
      effectiveReducedMotion,
      getPlayerColorClass,
      getPlayerColor,
    }),
    [
      preferences,
      setPreference,
      resetPreferences,
      systemPrefersReducedMotion,
      effectiveReducedMotion,
      getPlayerColorClass,
      getPlayerColor,
    ]
  );

  return <AccessibilityContext.Provider value={value}>{children}</AccessibilityContext.Provider>;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Access accessibility preferences and helpers.
 *
 * @example
 * ```tsx
 * const { highContrastMode, setPreference, getPlayerColorClass } = useAccessibility();
 *
 * // Toggle high contrast
 * setPreference('highContrastMode', !highContrastMode);
 *
 * // Get player color class
 * const bgClass = getPlayerColorClass(playerIndex, 'bg');
 * ```
 */
export function useAccessibility(): AccessibilityContextValue {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export { PLAYER_COLOR_PALETTES, PLAYER_COLOR_CLASSES };
