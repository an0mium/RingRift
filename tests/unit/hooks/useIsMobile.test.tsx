/**
 * Tests for useIsMobile hook
 *
 * Validates that the hook correctly detects mobile viewports using
 * window.matchMedia and responds to breakpoint changes.
 */

import React from 'react';
import { render, screen, act } from '@testing-library/react';
import {
  useIsMobile,
  useIsTouchDevice,
  useMobileState,
} from '../../../src/client/hooks/useIsMobile';

// ═══════════════════════════════════════════════════════════════════════════
// Test Utilities
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Creates a mock matchMedia that can be controlled in tests.
 */
function createMockMatchMedia(initialMatches: boolean) {
  const listeners: Array<(e: MediaQueryListEvent) => void> = [];
  let currentMatches = initialMatches;

  const mockMediaQueryList = {
    get matches() {
      return currentMatches;
    },
    media: '(max-width: 767px)',
    onchange: null as ((this: MediaQueryList, ev: MediaQueryListEvent) => unknown) | null,
    addListener: jest.fn((listener: (e: MediaQueryListEvent) => void) => {
      listeners.push(listener);
    }),
    removeListener: jest.fn((listener: (e: MediaQueryListEvent) => void) => {
      const index = listeners.indexOf(listener);
      if (index >= 0) listeners.splice(index, 1);
    }),
    addEventListener: jest.fn((event: string, listener: EventListener) => {
      if (event === 'change') listeners.push(listener as (e: MediaQueryListEvent) => void);
    }),
    removeEventListener: jest.fn((event: string, listener: EventListener) => {
      if (event === 'change') {
        const index = listeners.indexOf(listener as (e: MediaQueryListEvent) => void);
        if (index >= 0) listeners.splice(index, 1);
      }
    }),
    dispatchEvent: jest.fn(() => true),
  };

  const trigger = (matches: boolean) => {
    currentMatches = matches;
    const event = { matches, media: mockMediaQueryList.media } as MediaQueryListEvent;
    listeners.forEach((listener) => listener(event));
  };

  return { mockMediaQueryList: mockMediaQueryList as unknown as MediaQueryList, trigger };
}

/**
 * Test component that renders the result of useIsMobile
 */
function IsMobileTestComponent() {
  const isMobile = useIsMobile();
  return <div data-testid="is-mobile">{isMobile ? 'mobile' : 'desktop'}</div>;
}

/**
 * Test component that renders the result of useIsTouchDevice
 */
function IsTouchTestComponent() {
  const isTouch = useIsTouchDevice();
  return <div data-testid="is-touch">{isTouch ? 'touch' : 'no-touch'}</div>;
}

/**
 * Test component that renders the result of useMobileState
 */
function MobileStateTestComponent() {
  const { isMobile, isTouch, isMobileTouch } = useMobileState();
  return (
    <div>
      <div data-testid="mobile-state-mobile">{isMobile ? 'mobile' : 'desktop'}</div>
      <div data-testid="mobile-state-touch">{isTouch ? 'touch' : 'no-touch'}</div>
      <div data-testid="mobile-state-combo">
        {isMobileTouch ? 'mobile-touch' : 'not-mobile-touch'}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests: useIsMobile
// ═══════════════════════════════════════════════════════════════════════════

describe('useIsMobile', () => {
  let originalMatchMedia: typeof window.matchMedia;
  let originalInnerWidth: number;

  beforeEach(() => {
    originalMatchMedia = window.matchMedia;
    originalInnerWidth = window.innerWidth;
  });

  afterEach(() => {
    window.matchMedia = originalMatchMedia;
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      value: originalInnerWidth,
    });
  });

  it('returns true when viewport is below mobile breakpoint (768px)', () => {
    const { mockMediaQueryList } = createMockMatchMedia(true);
    window.matchMedia = jest.fn().mockReturnValue(mockMediaQueryList);
    Object.defineProperty(window, 'innerWidth', { writable: true, value: 375 });

    render(<IsMobileTestComponent />);

    expect(screen.getByTestId('is-mobile')).toHaveTextContent('mobile');
  });

  it('returns false when viewport is at or above mobile breakpoint', () => {
    const { mockMediaQueryList } = createMockMatchMedia(false);
    window.matchMedia = jest.fn().mockReturnValue(mockMediaQueryList);
    Object.defineProperty(window, 'innerWidth', { writable: true, value: 1024 });

    render(<IsMobileTestComponent />);

    expect(screen.getByTestId('is-mobile')).toHaveTextContent('desktop');
  });

  it('updates when media query changes from desktop to mobile', () => {
    const { mockMediaQueryList, trigger } = createMockMatchMedia(false);
    window.matchMedia = jest.fn().mockReturnValue(mockMediaQueryList);
    Object.defineProperty(window, 'innerWidth', { writable: true, value: 1024 });

    render(<IsMobileTestComponent />);
    expect(screen.getByTestId('is-mobile')).toHaveTextContent('desktop');

    // Simulate resize to mobile
    act(() => {
      trigger(true);
    });

    expect(screen.getByTestId('is-mobile')).toHaveTextContent('mobile');
  });

  it('updates when media query changes from mobile to desktop', () => {
    const { mockMediaQueryList, trigger } = createMockMatchMedia(true);
    window.matchMedia = jest.fn().mockReturnValue(mockMediaQueryList);
    Object.defineProperty(window, 'innerWidth', { writable: true, value: 375 });

    render(<IsMobileTestComponent />);
    expect(screen.getByTestId('is-mobile')).toHaveTextContent('mobile');

    // Simulate resize to desktop
    act(() => {
      trigger(false);
    });

    expect(screen.getByTestId('is-mobile')).toHaveTextContent('desktop');
  });

  it('cleans up event listener on unmount', () => {
    const { mockMediaQueryList } = createMockMatchMedia(false);
    window.matchMedia = jest.fn().mockReturnValue(mockMediaQueryList);

    const { unmount } = render(<IsMobileTestComponent />);
    unmount();

    expect(mockMediaQueryList.removeEventListener).toHaveBeenCalled();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Tests: useIsTouchDevice
// ═══════════════════════════════════════════════════════════════════════════

describe('useIsTouchDevice', () => {
  let originalOntouchstart: typeof window.ontouchstart;
  let originalMaxTouchPoints: number;

  beforeEach(() => {
    originalOntouchstart = window.ontouchstart;
    originalMaxTouchPoints = navigator.maxTouchPoints;
  });

  afterEach(() => {
    // Restore original values
    if (originalOntouchstart !== undefined) {
      window.ontouchstart = originalOntouchstart;
    } else {
      delete (window as { ontouchstart?: unknown }).ontouchstart;
    }
    Object.defineProperty(navigator, 'maxTouchPoints', {
      writable: true,
      value: originalMaxTouchPoints,
    });
  });

  it('returns true when ontouchstart is available', () => {
    (window as { ontouchstart?: unknown }).ontouchstart = null;
    Object.defineProperty(navigator, 'maxTouchPoints', { writable: true, value: 0 });

    render(<IsTouchTestComponent />);

    expect(screen.getByTestId('is-touch')).toHaveTextContent('touch');
  });

  it('returns true when maxTouchPoints > 0', () => {
    delete (window as { ontouchstart?: unknown }).ontouchstart;
    Object.defineProperty(navigator, 'maxTouchPoints', { writable: true, value: 5 });

    render(<IsTouchTestComponent />);

    expect(screen.getByTestId('is-touch')).toHaveTextContent('touch');
  });

  it('returns false when no touch capability', () => {
    delete (window as { ontouchstart?: unknown }).ontouchstart;
    Object.defineProperty(navigator, 'maxTouchPoints', { writable: true, value: 0 });

    render(<IsTouchTestComponent />);

    expect(screen.getByTestId('is-touch')).toHaveTextContent('no-touch');
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// Tests: useMobileState
// ═══════════════════════════════════════════════════════════════════════════

describe('useMobileState', () => {
  let originalMatchMedia: typeof window.matchMedia;
  let originalInnerWidth: number;
  let originalMaxTouchPoints: number;

  beforeEach(() => {
    originalMatchMedia = window.matchMedia;
    originalInnerWidth = window.innerWidth;
    originalMaxTouchPoints = navigator.maxTouchPoints;
  });

  afterEach(() => {
    window.matchMedia = originalMatchMedia;
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      value: originalInnerWidth,
    });
    Object.defineProperty(navigator, 'maxTouchPoints', {
      writable: true,
      value: originalMaxTouchPoints,
    });
    delete (window as { ontouchstart?: unknown }).ontouchstart;
  });

  it('returns isMobileTouch: true when both mobile AND touch', () => {
    const { mockMediaQueryList } = createMockMatchMedia(true);
    window.matchMedia = jest.fn().mockReturnValue(mockMediaQueryList);
    Object.defineProperty(window, 'innerWidth', { writable: true, value: 375 });
    (window as { ontouchstart?: unknown }).ontouchstart = null;

    render(<MobileStateTestComponent />);

    expect(screen.getByTestId('mobile-state-mobile')).toHaveTextContent('mobile');
    expect(screen.getByTestId('mobile-state-touch')).toHaveTextContent('touch');
    expect(screen.getByTestId('mobile-state-combo')).toHaveTextContent('mobile-touch');
  });

  it('returns isMobileTouch: false when mobile but NOT touch (desktop emulating mobile)', () => {
    const { mockMediaQueryList } = createMockMatchMedia(true);
    window.matchMedia = jest.fn().mockReturnValue(mockMediaQueryList);
    Object.defineProperty(window, 'innerWidth', { writable: true, value: 375 });
    Object.defineProperty(navigator, 'maxTouchPoints', { writable: true, value: 0 });

    render(<MobileStateTestComponent />);

    expect(screen.getByTestId('mobile-state-mobile')).toHaveTextContent('mobile');
    expect(screen.getByTestId('mobile-state-touch')).toHaveTextContent('no-touch');
    expect(screen.getByTestId('mobile-state-combo')).toHaveTextContent('not-mobile-touch');
  });

  it('returns isMobileTouch: false when touch but NOT mobile (tablets, large touch screens)', () => {
    const { mockMediaQueryList } = createMockMatchMedia(false);
    window.matchMedia = jest.fn().mockReturnValue(mockMediaQueryList);
    Object.defineProperty(window, 'innerWidth', { writable: true, value: 1024 });
    (window as { ontouchstart?: unknown }).ontouchstart = null;

    render(<MobileStateTestComponent />);

    expect(screen.getByTestId('mobile-state-mobile')).toHaveTextContent('desktop');
    expect(screen.getByTestId('mobile-state-touch')).toHaveTextContent('touch');
    expect(screen.getByTestId('mobile-state-combo')).toHaveTextContent('not-mobile-touch');
  });
});
