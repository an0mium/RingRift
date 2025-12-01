import React, { useState, useRef, useEffect } from 'react';
import type { GameState } from '../../shared/types/game';
import type { LoadableScenario } from '../sandbox/scenarioTypes';
import { saveCurrentGameState, exportScenarioToFile } from '../sandbox/statePersistence';

const FOCUSABLE_SELECTORS =
  'a[href], button:not([disabled]), textarea, input, select, [tabindex]:not([tabindex="-1"])';

export interface SaveStateDialogProps {
  isOpen: boolean;
  onClose: () => void;
  gameState: GameState | null;
  onSaved?: (scenario: LoadableScenario) => void;
}

export const SaveStateDialog: React.FC<SaveStateDialogProps> = ({
  isOpen,
  onClose,
  gameState,
  onSaved,
}) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [exportToFile, setExportToFile] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const nameInputRef = useRef<HTMLInputElement | null>(null);

  // Focus name input when dialog opens
  useEffect(() => {
    if (isOpen && nameInputRef.current) {
      nameInputRef.current.focus();
    }
  }, [isOpen]);

  // Reset form when dialog opens
  useEffect(() => {
    if (isOpen) {
      setName('');
      setDescription('');
      setExportToFile(false);
      setError(null);
    }
  }, [isOpen]);

  // Focus trap and keyboard handling
  useEffect(() => {
    if (!isOpen) return;

    const dialogEl = dialogRef.current;
    if (!dialogEl) return;

    const focusable = Array.from(dialogEl.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTORS));
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
        return;
      }

      if (event.key !== 'Tab' || focusable.length === 0) return;

      const active = document.activeElement as HTMLElement | null;
      if (!active) return;

      const isShift = event.shiftKey;

      if (isShift && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!isShift && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    dialogEl.addEventListener('keydown', handleKeyDown);
    return () => dialogEl.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const handleSave = async () => {
    if (!gameState) {
      setError('No game state available to save');
      return;
    }

    if (!name.trim()) {
      setError('Please enter a name for the saved state');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const scenario = saveCurrentGameState(gameState, {
        name: name.trim(),
        description: description.trim() || undefined,
      });

      if (exportToFile) {
        exportScenarioToFile(scenario);
      }

      onSaved?.(scenario);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save game state');
    } finally {
      setSaving(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSave();
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60"
      role="dialog"
      aria-modal="true"
      aria-labelledby="save-state-title"
    >
      <div
        ref={dialogRef}
        className="bg-slate-900 rounded-2xl border border-slate-700 w-full max-w-md p-6 shadow-2xl"
      >
        <h2 id="save-state-title" className="text-xl font-bold text-white mb-4">
          Save Game State
        </h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="save-name" className="block text-sm text-slate-400 mb-1">
              Name <span className="text-red-400">*</span>
            </label>
            <input
              ref={nameInputRef}
              id="save-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My saved game"
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              disabled={saving}
            />
          </div>

          <div>
            <label htmlFor="save-description" className="block text-sm text-slate-400 mb-1">
              Description (optional)
            </label>
            <textarea
              id="save-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe this game state..."
              rows={3}
              className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-600 text-white placeholder-slate-400 resize-none focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              disabled={saving}
            />
          </div>

          <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={exportToFile}
              onChange={(e) => setExportToFile(e.target.checked)}
              className="rounded border-slate-600 bg-slate-800 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-0"
              disabled={saving}
            />
            Also export as JSON file
          </label>

          {error && (
            <div className="p-3 rounded-lg bg-red-900/30 border border-red-700 text-red-300 text-sm">
              {error}
            </div>
          )}

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-800 transition-colors disabled:opacity-50"
              disabled={saving}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={saving || !name.trim()}
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default SaveStateDialog;
