import React, { useState, useRef } from 'react';
import { Button } from './ui/Button';
import { Dialog } from './ui/Dialog';

export interface ResignButtonProps {
  /** Callback when resignation is confirmed */
  onResign: () => void;
  /** Whether the button is disabled (e.g., during API call) */
  disabled?: boolean;
  /** Whether a resignation is currently in progress */
  isResigning?: boolean;
  /**
   * Optional controlled flag for opening the confirmation dialog.
   * When omitted, the component manages its own open state.
   */
  isConfirmOpen?: boolean;
  /** Optional controlled state setter for the confirmation dialog. */
  onConfirmOpenChange?: (isOpen: boolean) => void;
}

/**
 * Resign button with confirmation dialog.
 *
 * Accessibility features:
 * - Dialog has role="alertdialog" for urgent confirmation
 * - Escape key cancels resignation
 * - Focus is restored to the trigger button on close
 */
export function ResignButton({
  onResign,
  disabled,
  isResigning,
  isConfirmOpen: controlledIsConfirmOpen,
  onConfirmOpenChange,
}: ResignButtonProps) {
  const [uncontrolledIsConfirmOpen, setUncontrolledIsConfirmOpen] = useState(false);
  const isConfirmOpen = controlledIsConfirmOpen ?? uncontrolledIsConfirmOpen;
  const setIsConfirmOpen = (next: boolean) => {
    if (onConfirmOpenChange) {
      onConfirmOpenChange(next);
      return;
    }
    setUncontrolledIsConfirmOpen(next);
  };
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const cancelButtonRef = useRef<HTMLButtonElement | null>(null);

  const handleConfirm = () => {
    setIsConfirmOpen(false);
    onResign();
  };

  const handleCancel = () => {
    setIsConfirmOpen(false);
  };

  return (
    <>
      <Button
        ref={triggerRef}
        variant="danger"
        size="sm"
        onClick={() => setIsConfirmOpen(true)}
        disabled={disabled || isResigning}
        aria-haspopup="dialog"
        data-testid="resign-button"
      >
        {isResigning ? 'Resigning...' : 'Resign'}
      </Button>

      <Dialog
        isOpen={isConfirmOpen}
        onClose={handleCancel}
        role="alertdialog"
        labelledBy="resign-dialog-title"
        describedBy="resign-dialog-description"
        initialFocusRef={cancelButtonRef}
        overlayTestId="resign-dialog-overlay"
        className="bg-slate-900 border border-slate-700 rounded-lg shadow-xl max-w-sm w-full mx-4 p-6 space-y-4"
      >
        <h2 id="resign-dialog-title" className="text-xl font-bold text-slate-100">
          Resign Game?
        </h2>
        <p id="resign-dialog-description" className="text-sm text-slate-300">
          Are you sure you want to resign? This will end the game and count as a loss. Your opponent
          will be declared the winner.
        </p>
        <div className="flex gap-3 justify-end">
          <Button
            ref={cancelButtonRef}
            variant="secondary"
            size="sm"
            onClick={handleCancel}
            data-testid="resign-cancel-button"
          >
            Cancel
          </Button>
          <Button
            variant="danger"
            size="sm"
            onClick={handleConfirm}
            data-testid="resign-confirm-button"
          >
            Yes, Resign
          </Button>
        </div>
      </Dialog>
    </>
  );
}
