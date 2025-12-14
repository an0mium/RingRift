import React, { useRef } from 'react';
import { Dialog } from './ui/Dialog';
import { Button } from './ui/Button';

export interface RecoveryLineChoiceDialogProps {
  isOpen: boolean;
  onChooseOption1: () => void;
  onChooseOption2: () => void;
  onClose: () => void;
}

export function RecoveryLineChoiceDialog({
  isOpen,
  onChooseOption1,
  onChooseOption2,
  onClose,
}: RecoveryLineChoiceDialogProps) {
  const option2Ref = useRef<HTMLButtonElement | null>(null);

  return (
    <Dialog
      isOpen={isOpen}
      onClose={onClose}
      labelledBy="recovery-choice-title"
      describedBy="recovery-choice-description"
      initialFocusRef={option2Ref}
      className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-lg p-6 space-y-4"
      overlayTestId="recovery-choice-overlay"
    >
      <div className="space-y-2">
        <h2 id="recovery-choice-title" className="text-lg font-bold text-slate-100">
          Overlength Recovery Line
        </h2>
        <p id="recovery-choice-description" className="text-sm text-slate-300">
          Choose how to resolve the overlength recovery line.
        </p>
      </div>

      <div className="grid gap-3">
        <button
          type="button"
          onClick={onChooseOption2}
          ref={option2Ref}
          className="rounded-xl border border-emerald-500/40 bg-emerald-500/10 px-4 py-3 text-left text-sm text-emerald-100 hover:bg-emerald-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400"
        >
          <div className="font-semibold">Option 2 (Free)</div>
          <div className="text-xs text-emerald-100/80">
            Collapse minimum markers. No buried ring extracted.
          </div>
        </button>

        <button
          type="button"
          onClick={onChooseOption1}
          className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-left text-sm text-amber-100 hover:bg-amber-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
        >
          <div className="font-semibold">Option 1 (Cost)</div>
          <div className="text-xs text-amber-100/80">
            Collapse all markers. Extract 1 buried ring.
          </div>
        </button>
      </div>

      <div className="flex justify-end pt-1">
        <Button type="button" variant="secondary" size="sm" onClick={onClose}>
          Cancel
        </Button>
      </div>
    </Dialog>
  );
}
