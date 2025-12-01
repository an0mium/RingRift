import React, { useEffect, useState, useRef } from 'react';

export interface ScreenReaderAnnouncerProps {
  /**
   * Message to announce. When this changes, the new message is announced.
   * Set to empty string to clear the announcement.
   */
  message: string;
  /**
   * Politeness level for the announcement.
   * - 'polite': Waits for user to finish their current task (default)
   * - 'assertive': Interrupts immediately (use sparingly for urgent info)
   */
  politeness?: 'polite' | 'assertive';
}

/**
 * Visually hidden component that announces messages to screen readers.
 *
 * Uses aria-live regions to communicate dynamic content changes.
 * The component is positioned off-screen but remains accessible to
 * assistive technology.
 *
 * Usage:
 * ```tsx
 * const [announcement, setAnnouncement] = useState('');
 *
 * // When turn changes:
 * setAnnouncement(`It's now Player 2's turn`);
 *
 * <ScreenReaderAnnouncer message={announcement} />
 * ```
 */
export function ScreenReaderAnnouncer({
  message,
  politeness = 'polite',
}: ScreenReaderAnnouncerProps) {
  // Use alternating regions to ensure announcement is triggered on duplicate messages
  const [currentMessage, setCurrentMessage] = useState('');
  const [isFirst, setIsFirst] = useState(true);
  const prevMessageRef = useRef<string>('');

  useEffect(() => {
    let timeout: ReturnType<typeof setTimeout> | undefined;

    if (message && message !== prevMessageRef.current) {
      setCurrentMessage(message);
      setIsFirst((prev) => !prev);
      prevMessageRef.current = message;

      // Clear message after announcement to allow re-announcement of same message
      timeout = setTimeout(() => {
        setCurrentMessage('');
      }, 1000);
    }

    // Always return a cleanup function so all code paths are covered and
    // React's EffectCallback contract (void | () => void) is satisfied.
    return () => {
      if (timeout !== undefined) {
        clearTimeout(timeout);
      }
    };
  }, [message]);

  // Visually hidden but accessible to screen readers
  const hiddenStyle: React.CSSProperties = {
    position: 'absolute',
    width: '1px',
    height: '1px',
    padding: 0,
    margin: '-1px',
    overflow: 'hidden',
    clip: 'rect(0, 0, 0, 0)',
    whiteSpace: 'nowrap',
    border: 0,
  };

  return (
    <>
      {/* Two alternating live regions ensure announcements are always triggered */}
      <div
        aria-live={politeness}
        aria-atomic="true"
        style={hiddenStyle}
        data-testid="sr-announcer-1"
      >
        {isFirst ? currentMessage : ''}
      </div>
      <div
        aria-live={politeness}
        aria-atomic="true"
        style={hiddenStyle}
        data-testid="sr-announcer-2"
      >
        {!isFirst ? currentMessage : ''}
      </div>
    </>
  );
}

/**
 * Hook to manage screen reader announcements.
 *
 * Returns an announce function that can be called to trigger announcements,
 * and the current message to pass to ScreenReaderAnnouncer.
 *
 * Usage:
 * ```tsx
 * const { message, announce } = useScreenReaderAnnouncement();
 *
 * // When something happens:
 * announce(`Game over! Player 1 wins by elimination.`);
 *
 * <ScreenReaderAnnouncer message={message} />
 * ```
 */
export function useScreenReaderAnnouncement() {
  const [message, setMessage] = useState('');

  const announce = (text: string) => {
    setMessage(text);
  };

  return { message, announce };
}
