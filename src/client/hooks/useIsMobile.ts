/**
 * ═══════════════════════════════════════════════════════════════════════════
 * useIsMobile Hook
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Provides consistent mobile detection across the client application using
 * CSS media query matching. This aligns with Tailwind's responsive breakpoints
 * (md: 768px) for a unified mobile/desktop experience.
 *
 * Usage:
 *   const isMobile = useIsMobile();
 *   // Renders mobile-optimized layout when true
 */

import { useState, useEffect } from 'react';

/** Tailwind md: breakpoint - below this is considered mobile */
const MOBILE_BREAKPOINT = 768;

/**
 * Hook to detect if the current viewport is mobile-sized.
 * Uses window.matchMedia for efficient, reactive breakpoint detection.
 *
 * @returns true if viewport width is below the mobile breakpoint (768px)
 */
export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState<boolean>(() => {
    // SSR-safe: default to false if window is undefined
    if (typeof window === 'undefined') return false;
    return window.innerWidth < MOBILE_BREAKPOINT;
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const mediaQuery = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`);

    // Set initial state based on media query
    setIsMobile(mediaQuery.matches);

    // Handler for media query changes
    const handleChange = (event: MediaQueryListEvent) => {
      setIsMobile(event.matches);
    };

    // Modern browsers use addEventListener
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
    } else {
      // Fallback for older browsers
      mediaQuery.addListener(handleChange);
    }

    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener('change', handleChange);
      } else {
        mediaQuery.removeListener(handleChange);
      }
    };
  }, []);

  return isMobile;
}

/**
 * Hook to detect if the current device supports touch input.
 * Useful for showing/hiding touch-specific controls.
 *
 * @returns true if the device has touch capability
 */
export function useIsTouchDevice(): boolean {
  const [isTouch, setIsTouch] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    // Check for touch capability
    const hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    setIsTouch(hasTouch);
  }, []);

  return isTouch;
}

/**
 * Combined hook for mobile-responsive rendering decisions.
 * Returns both viewport size and touch capability information.
 */
export interface MobileState {
  /** True if viewport is below mobile breakpoint */
  isMobile: boolean;
  /** True if device supports touch input */
  isTouch: boolean;
  /** True if both mobile viewport AND touch device (typical mobile phone) */
  isMobileTouch: boolean;
}

export function useMobileState(): MobileState {
  const isMobile = useIsMobile();
  const isTouch = useIsTouchDevice();

  return {
    isMobile,
    isTouch,
    isMobileTouch: isMobile && isTouch,
  };
}
