import { useTheme } from '../contexts/ThemeContext';

export function ThemeSettingsPanel() {
  const { theme, setTheme } = useTheme();

  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-200 mb-3">Appearance</h3>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setTheme('dark')}
          className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition-colors ${
            theme === 'dark'
              ? 'bg-slate-700 border-emerald-500 text-white'
              : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600'
          }`}
        >
          <svg
            className="w-4 h-4 mx-auto mb-1"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
            />
          </svg>
          Dark
        </button>
        <button
          type="button"
          onClick={() => setTheme('light')}
          className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition-colors ${
            theme === 'light'
              ? 'bg-slate-700 border-emerald-500 text-white'
              : 'bg-slate-800 border-slate-700 text-slate-400 hover:border-slate-600'
          }`}
        >
          <svg
            className="w-4 h-4 mx-auto mb-1"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
            />
          </svg>
          Light
        </button>
      </div>
      {theme === 'light' && (
        <p className="text-xs text-amber-500/80 mt-2">
          Light mode is experimental. Some pages may not render perfectly yet.
        </p>
      )}
    </div>
  );
}
