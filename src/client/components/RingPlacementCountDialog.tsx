import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Dialog } from './ui/Dialog';
import { Button } from './ui/Button';

export interface RingPlacementCountDialogProps {
  isOpen: boolean;
  maxCount: number;
  defaultCount?: number;
  isStackPlacement?: boolean;
  onClose: () => void;
  onConfirm: (count: number) => void;
}

export function RingPlacementCountDialog({
  isOpen,
  maxCount,
  defaultCount,
  isStackPlacement,
  onClose,
  onConfirm,
}: RingPlacementCountDialogProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [value, setValue] = useState('');
  const [error, setError] = useState<string | null>(null);

  const clampedDefault = useMemo(() => {
    const fallback = Math.min(2, Math.max(1, maxCount));
    const raw = defaultCount ?? fallback;
    if (!Number.isFinite(raw)) return fallback;
    return Math.max(1, Math.min(maxCount, Math.floor(raw)));
  }, [defaultCount, maxCount]);

  useEffect(() => {
    if (!isOpen) return;
    setValue(String(clampedDefault));
    setError(null);
  }, [isOpen, clampedDefault]);

  const hint = isStackPlacement
    ? 'On stacks, canonical placement is 1 ring.'
    : `Place 1–${maxCount} rings on this cell.`;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    setError(null);

    const parsed = Number.parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed < 1 || parsed > maxCount) {
      setError(`Enter a number from 1 to ${maxCount}.`);
      return;
    }

    onConfirm(parsed);
  };

  const parsedValue = Number.parseInt(value, 10);
  const canSubmit =
    Number.isFinite(parsedValue) && parsedValue >= 1 && parsedValue <= maxCount && !error;

  return (
    <Dialog
      isOpen={isOpen}
      onClose={onClose}
      labelledBy="ring-placement-count-title"
      describedBy="ring-placement-count-description"
      initialFocusRef={inputRef}
      className="bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-sm p-6 space-y-4"
      overlayTestId="ring-placement-count-overlay"
    >
      <div className="space-y-1">
        <h2 id="ring-placement-count-title" className="text-lg font-bold text-slate-100">
          Place Rings
        </h2>
        <p id="ring-placement-count-description" className="text-sm text-slate-300">
          {hint}
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-3">
        <div className="space-y-1">
          <label
            htmlFor="ring-placement-count-input"
            className="text-sm font-medium text-slate-200"
          >
            Number of rings
          </label>
          <input
            ref={inputRef}
            id="ring-placement-count-input"
            type="number"
            min={1}
            max={Math.max(1, maxCount)}
            inputMode="numeric"
            value={value}
            onChange={(e) => {
              setValue(e.target.value);
              setError(null);
            }}
            className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
          />
          {error && (
            <p className="text-sm text-red-300" role="alert">
              {error}
            </p>
          )}
        </div>

        <div className="flex justify-end gap-3 pt-2">
          <Button type="button" variant="secondary" size="sm" onClick={onClose}>
            Cancel
          </Button>
          <Button type="submit" size="sm" disabled={!canSubmit}>
            Place
          </Button>
        </div>
      </form>
    </Dialog>
  );
}
