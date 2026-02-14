import { useEffect } from 'react';

const BASE_TITLE = 'RingRift';

/**
 * Sets the document title for the current page.
 * Resets to the base title on unmount.
 */
export function useDocumentTitle(subtitle?: string) {
  useEffect(() => {
    document.title = subtitle
      ? `${subtitle} - ${BASE_TITLE}`
      : `${BASE_TITLE} - Multiplayer Strategy Game`;
    return () => {
      document.title = `${BASE_TITLE} - Multiplayer Strategy Game`;
    };
  }, [subtitle]);
}
