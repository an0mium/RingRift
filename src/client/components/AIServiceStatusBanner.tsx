/**
 * @fileoverview AIServiceStatusBanner - Shows AI service connection status in sandbox mode.
 *
 * This component displays a warning banner when the AI service is unavailable
 * and the sandbox is using local heuristics as a fallback. Provides a retry
 * option for users to attempt reconnection.
 */

import React from 'react';
import { StatusBanner } from './ui/StatusBanner';
import { Button } from './ui/Button';
import type { AIServiceStatus } from '../hooks/useSandboxAIServiceStatus';

export interface AIServiceStatusBannerProps {
  /** Current AI service status */
  status: AIServiceStatus;
  /** User-friendly status message */
  message: string | null;
  /** Whether the service is configured for this environment */
  isServiceConfigured: boolean;
  /** Callback to retry connection */
  onRetry: () => Promise<boolean>;
  /** Callback to dismiss the banner */
  onDismiss: () => void;
  /** Whether retry is in progress */
  isRetrying?: boolean;
}

/**
 * Banner component that shows AI service connection status.
 *
 * Displays:
 * - Warning when AI service is in fallback mode (using local heuristics)
 * - Retry button to attempt reconnection
 * - Dismiss button to hide the banner
 *
 * Does not display anything when:
 * - Service is connected normally
 * - Service is not configured for this environment (expected in production)
 * - No message to show
 */
export const AIServiceStatusBanner: React.FC<AIServiceStatusBannerProps> = ({
  status,
  message,
  isServiceConfigured,
  onRetry,
  onDismiss,
  isRetrying = false,
}) => {
  const [isRetryingLocal, setIsRetryingLocal] = React.useState(false);

  // Don't show anything if connected or unavailable by design
  if (status === 'connected' || status === 'unknown') {
    return null;
  }

  // Don't show anything if service isn't configured (expected in production)
  if (!isServiceConfigured) {
    return null;
  }

  // Don't show anything if no message
  if (!message) {
    return null;
  }

  const handleRetry = async () => {
    setIsRetryingLocal(true);
    try {
      await onRetry();
    } finally {
      setIsRetryingLocal(false);
    }
  };

  const isLoading = isRetrying || isRetryingLocal;

  return (
    <StatusBanner
      variant="warning"
      title="AI Service Fallback"
      actions={
        <>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={handleRetry}
            disabled={isLoading}
            className="text-[11px] px-3 py-1"
          >
            {isLoading ? 'Retrying...' : 'Retry'}
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={onDismiss}
            className="text-[11px] px-3 py-1"
          >
            Dismiss
          </Button>
        </>
      }
    >
      <span className="text-xs">{message}</span>
    </StatusBanner>
  );
};

export default AIServiceStatusBanner;
