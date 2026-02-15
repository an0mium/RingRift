/**
 * SettingsModal - Global settings modal accessible from the navigation bar
 *
 * Provides access to accessibility settings and other user preferences.
 * Can be opened via the gear icon in the navbar or keyboard shortcut.
 */

import { useRef } from 'react';
import { AccessibilitySettingsPanel } from './AccessibilitySettingsPanel';
import { SoundSettingsPanel } from './SoundSettingsPanel';
import { ThemeSettingsPanel } from './ThemeSettingsPanel';
import { Dialog } from './ui/Dialog';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  return (
    <Dialog
      isOpen={isOpen}
      onClose={onClose}
      labelledBy="settings-modal-title"
      initialFocusRef={closeButtonRef}
      className="relative w-full max-w-lg mx-4 max-h-[90vh] overflow-y-auto rounded-xl bg-slate-900 border border-slate-700 shadow-2xl"
    >
      {/* Header */}
      <div className="sticky top-0 z-10 flex items-center justify-between px-4 py-3 border-b border-slate-700 bg-slate-900">
        <h2 id="settings-modal-title" className="text-lg font-semibold text-slate-100">
          Settings
        </h2>
        <button
          ref={closeButtonRef}
          type="button"
          onClick={onClose}
          className="p-1.5 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800 hover:scale-110 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500 transition-all duration-200"
          aria-label="Close settings"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-6">
        <ThemeSettingsPanel />
        <hr className="border-slate-700" />
        <SoundSettingsPanel />
        <hr className="border-slate-700" />
        <AccessibilitySettingsPanel onSettingsChange={() => {}} />
      </div>
    </Dialog>
  );
}

export default SettingsModal;
