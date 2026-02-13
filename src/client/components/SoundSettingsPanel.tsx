import { useSound } from '../contexts/SoundContext';

export function SoundSettingsPanel() {
  const { muted, volume, turnStartSound, toggleMute, setVolume, setPreference, audioAvailable } =
    useSound();

  if (!audioAvailable) {
    return (
      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-slate-200">Sound</h3>
        <p className="text-xs text-slate-400">
          Audio is not available in this browser. Interact with the page (click or press a key) to
          enable sound.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-slate-200">Sound</h3>

      {/* Mute toggle */}
      <label className="flex items-center justify-between gap-3 cursor-pointer select-none">
        <span className="text-sm text-slate-300">Enable sounds</span>
        <button
          type="button"
          role="switch"
          aria-checked={!muted}
          onClick={toggleMute}
          className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 ${
            !muted ? 'bg-emerald-600' : 'bg-slate-600'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 rounded-full bg-white transition-transform duration-200 ${
              !muted ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </label>

      {/* Volume slider */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <label htmlFor="volume-slider" className="text-sm text-slate-300">
            Volume
          </label>
          <span className="text-xs text-slate-400 tabular-nums">{Math.round(volume * 100)}%</span>
        </div>
        <input
          id="volume-slider"
          type="range"
          min={0}
          max={100}
          step={1}
          value={Math.round(volume * 100)}
          onChange={(e) => setVolume(Number(e.target.value) / 100)}
          disabled={muted}
          className="w-full h-1.5 rounded-full bg-slate-700 appearance-none cursor-pointer accent-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-emerald-400 [&::-webkit-slider-thumb]:appearance-none"
        />
      </div>

      {/* Turn start sound */}
      <label className="flex items-center justify-between gap-3 cursor-pointer select-none">
        <div>
          <span className="text-sm text-slate-300">Turn start notification</span>
          <p className="text-xs text-slate-500">Play a chime when it's your turn</p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={turnStartSound}
          onClick={() => setPreference('turnStartSound', !turnStartSound)}
          disabled={muted}
          className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900 disabled:opacity-40 disabled:cursor-not-allowed ${
            turnStartSound && !muted ? 'bg-emerald-600' : 'bg-slate-600'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 rounded-full bg-white transition-transform duration-200 ${
              turnStartSound && !muted ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </label>
    </div>
  );
}
