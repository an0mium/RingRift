import React from 'react';
import { Button } from './ui/Button';

export interface EmptyStateProps {
  /** Main heading text */
  title: string;
  /** Optional description below the title */
  description?: string;
  /** Optional icon component to display above the title */
  icon?: React.ReactNode;
  /** Optional call-to-action button */
  action?: {
    label: string;
    onClick: () => void;
  };
  /** Additional CSS classes */
  className?: string;
}

/**
 * EmptyState - A reusable component for displaying empty states.
 *
 * Use this when:
 * - A list has no items
 * - Search returns no results
 * - A feature is not yet configured
 * - Data is unavailable
 */
export function EmptyState({ title, description, icon, action, className = '' }: EmptyStateProps) {
  return (
    <div
      className={`flex flex-col items-center justify-center p-6 text-center ${className}`}
      role="status"
      aria-live="polite"
    >
      {icon && <div className="mb-3 text-slate-500">{icon}</div>}
      <h3 className="text-sm font-medium text-slate-300">{title}</h3>
      {description && <p className="mt-1 text-xs text-slate-500 max-w-xs">{description}</p>}
      {action && (
        <Button
          type="button"
          onClick={action.onClick}
          className="mt-4 min-h-[44px] touch-manipulation"
        >
          {action.label}
        </Button>
      )}
    </div>
  );
}

export default EmptyState;
