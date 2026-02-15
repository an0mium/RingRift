import { useState } from 'react';
import { Link } from 'react-router-dom';

const DISMISSED_KEY = 'ringrift:welcome-dismissed';

export function WelcomeBanner() {
  const [dismissed, setDismissed] = useState(() => localStorage.getItem(DISMISSED_KEY) === '1');

  if (dismissed) return null;

  const handleDismiss = () => {
    localStorage.setItem(DISMISSED_KEY, '1');
    setDismissed(true);
  };

  return (
    <div className="relative rounded-2xl border border-emerald-500/30 bg-gradient-to-br from-emerald-900/30 via-slate-900/80 to-sky-900/20 p-6 md:p-8 overflow-hidden">
      <button
        type="button"
        onClick={handleDismiss}
        className="absolute top-3 right-3 text-slate-500 hover:text-slate-300 transition-colors"
        aria-label="Dismiss welcome banner"
      >
        <svg
          className="w-5 h-5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>

      <h2 className="text-xl font-bold text-white mb-2">Welcome to RingRift!</h2>
      <p className="text-slate-300 text-sm mb-5 max-w-lg">
        New here? Start with a quick sandbox game to learn the basics, then jump into online play
        when you're ready.
      </p>

      <div className="flex flex-col sm:flex-row gap-3">
        <Step
          number={1}
          title="Try the Sandbox"
          description="Learn the rules with a practice game against AI"
          to="/sandbox?preset=learn-basics"
          highlight
        />
        <Step
          number={2}
          title="Read the Rules"
          description="Quick overview of how to play"
          to="/help/how-to-play"
        />
        <Step
          number={3}
          title="Play Online"
          description="Join the lobby for rated matches"
          to="/lobby"
        />
      </div>
    </div>
  );
}

function Step({
  number,
  title,
  description,
  to,
  highlight,
}: {
  number: number;
  title: string;
  description: string;
  to: string;
  highlight?: boolean;
}) {
  return (
    <Link
      to={to}
      className={`flex-1 rounded-xl border px-4 py-3 transition-all duration-200 hover:scale-[1.02] ${
        highlight
          ? 'border-emerald-500/50 bg-emerald-900/30 hover:border-emerald-400/70'
          : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
      }`}
    >
      <div className="flex items-start gap-3">
        <span
          className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold flex-shrink-0 mt-0.5 ${
            highlight ? 'bg-emerald-500 text-white' : 'bg-slate-700 text-slate-300'
          }`}
        >
          {number}
        </span>
        <div>
          <p className="text-sm font-semibold text-white">{title}</p>
          <p className="text-xs text-slate-400 mt-0.5">{description}</p>
        </div>
      </div>
    </Link>
  );
}
