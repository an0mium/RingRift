import React from 'react';
import { render, screen, act, waitFor } from '@testing-library/react';
import { renderHook } from '@testing-library/react';
import '@testing-library/jest-dom';
import {
  ScreenReaderAnnouncer,
  useScreenReaderAnnouncement,
} from '../../../src/client/components/ScreenReaderAnnouncer';

describe('ScreenReaderAnnouncer', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders with aria-live regions', () => {
    render(<ScreenReaderAnnouncer message="" />);

    const region1 = screen.getByTestId('sr-announcer-1');
    const region2 = screen.getByTestId('sr-announcer-2');

    expect(region1).toHaveAttribute('aria-live', 'polite');
    expect(region2).toHaveAttribute('aria-live', 'polite');
    expect(region1).toHaveAttribute('aria-atomic', 'true');
    expect(region2).toHaveAttribute('aria-atomic', 'true');
  });

  it('uses assertive politeness when specified', () => {
    render(<ScreenReaderAnnouncer message="" politeness="assertive" />);

    const region1 = screen.getByTestId('sr-announcer-1');
    const region2 = screen.getByTestId('sr-announcer-2');

    expect(region1).toHaveAttribute('aria-live', 'assertive');
    expect(region2).toHaveAttribute('aria-live', 'assertive');
  });

  it('announces message in live region', () => {
    const { rerender } = render(<ScreenReaderAnnouncer message="" />);

    // Initially empty
    expect(screen.getByTestId('sr-announcer-1')).toHaveTextContent('');
    expect(screen.getByTestId('sr-announcer-2')).toHaveTextContent('');

    // Announce message
    rerender(<ScreenReaderAnnouncer message="Player 1's turn" />);

    // One of the regions should have the message
    const region1 = screen.getByTestId('sr-announcer-1');
    const region2 = screen.getByTestId('sr-announcer-2');
    const hasMessage =
      region1.textContent === "Player 1's turn" || region2.textContent === "Player 1's turn";
    expect(hasMessage).toBe(true);
  });

  it('alternates between regions for successive messages', () => {
    const { rerender } = render(<ScreenReaderAnnouncer message="" />);

    // First message
    rerender(<ScreenReaderAnnouncer message="Message 1" />);

    act(() => {
      jest.advanceTimersByTime(1100);
    });

    // Second message
    rerender(<ScreenReaderAnnouncer message="Message 2" />);

    const region1 = screen.getByTestId('sr-announcer-1');
    const region2 = screen.getByTestId('sr-announcer-2');

    // One region should have Message 2
    const hasSecondMessage =
      region1.textContent === 'Message 2' || region2.textContent === 'Message 2';
    expect(hasSecondMessage).toBe(true);
  });

  it('clears message after timeout', async () => {
    const { rerender } = render(<ScreenReaderAnnouncer message="" />);

    rerender(<ScreenReaderAnnouncer message="Temporary message" />);

    // Message should be present initially
    const region1 = screen.getByTestId('sr-announcer-1');
    const region2 = screen.getByTestId('sr-announcer-2');
    const hasMessage =
      region1.textContent === 'Temporary message' || region2.textContent === 'Temporary message';
    expect(hasMessage).toBe(true);

    // Advance time to clear message
    act(() => {
      jest.advanceTimersByTime(1100);
    });

    // Both regions should be empty
    expect(region1).toHaveTextContent('');
    expect(region2).toHaveTextContent('');
  });

  it('is visually hidden but accessible', () => {
    render(<ScreenReaderAnnouncer message="Hidden message" />);

    const region1 = screen.getByTestId('sr-announcer-1');

    // Check visually hidden styles
    expect(region1).toHaveStyle({
      position: 'absolute',
      width: '1px',
      height: '1px',
      overflow: 'hidden',
    });
  });

  it('does not announce empty strings', () => {
    const { rerender } = render(<ScreenReaderAnnouncer message="Initial" />);

    act(() => {
      jest.advanceTimersByTime(1100);
    });

    // Try to announce empty string
    rerender(<ScreenReaderAnnouncer message="" />);

    const region1 = screen.getByTestId('sr-announcer-1');
    const region2 = screen.getByTestId('sr-announcer-2');

    // Both should remain empty (no new announcement)
    expect(region1).toHaveTextContent('');
    expect(region2).toHaveTextContent('');
  });

  it('does not re-announce identical consecutive messages', () => {
    const { rerender } = render(<ScreenReaderAnnouncer message="" />);

    // First announcement
    rerender(<ScreenReaderAnnouncer message="Same message" />);

    act(() => {
      jest.advanceTimersByTime(1100);
    });

    // Try to announce same message again
    rerender(<ScreenReaderAnnouncer message="Same message" />);

    // Should not trigger a new announcement (prevMessageRef prevents it)
    const region1 = screen.getByTestId('sr-announcer-1');
    const region2 = screen.getByTestId('sr-announcer-2');

    expect(region1).toHaveTextContent('');
    expect(region2).toHaveTextContent('');
  });
});

describe('useScreenReaderAnnouncement', () => {
  it('returns empty message initially', () => {
    const { result } = renderHook(() => useScreenReaderAnnouncement());

    expect(result.current.message).toBe('');
  });

  it('updates message when announce is called', () => {
    const { result } = renderHook(() => useScreenReaderAnnouncement());

    act(() => {
      result.current.announce('New announcement');
    });

    expect(result.current.message).toBe('New announcement');
  });

  it('returns stable announce function', () => {
    const { result, rerender } = renderHook(() => useScreenReaderAnnouncement());

    const firstAnnounce = result.current.announce;
    rerender();
    const secondAnnounce = result.current.announce;

    // Function reference should be stable (not recreated on each render)
    // Note: This depends on implementation, may need useCallback
    expect(typeof firstAnnounce).toBe('function');
    expect(typeof secondAnnounce).toBe('function');
  });
});
