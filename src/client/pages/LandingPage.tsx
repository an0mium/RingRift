import { Link } from 'react-router-dom';
import { ButtonLink } from '../components/ui/ButtonLink';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 app-bg">
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="container mx-auto px-4 pt-20 pb-16 sm:pt-28 sm:pb-24 text-center">
          <img
            src="/ringrift-icon.png"
            alt=""
            aria-hidden="true"
            className="mx-auto mb-6 w-20 h-20 sm:w-28 sm:h-28 drop-shadow-lg"
          />
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight">
            <span className="text-white">Ring</span>
            <span className="text-emerald-400">Rift</span>
          </h1>
          <p className="mt-4 text-lg sm:text-xl text-slate-300 max-w-2xl mx-auto">
            A multiplayer territory-control strategy game. Place rings, form lines, claim territory,
            and outplay opponents on dynamic board geometries.
          </p>
          <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-3">
            <ButtonLink to="/sandbox" size="lg" className="min-w-[160px]">
              Play Now
            </ButtonLink>
            <ButtonLink to="/login" variant="outline" size="lg" className="min-w-[160px]">
              Sign In
            </ButtonLink>
          </div>
        </div>
      </section>

      {/* Feature highlights */}
      <section className="container mx-auto px-4 py-16 sm:py-20">
        <h2 className="text-center text-2xl sm:text-3xl font-bold text-white mb-10">
          Why RingRift?
        </h2>
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 max-w-5xl mx-auto">
          {/* Board Geometries */}
          <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-6 hover:border-emerald-500/50 transition-colors">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex-shrink-0 rounded-lg bg-emerald-500/10 p-2">
                <svg
                  className="w-6 h-6 text-emerald-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white">Board Geometries</h3>
            </div>
            <p className="text-sm text-slate-300">
              Play on square or hexagonal boards of varying sizes. From quick 8x8 matches to deep
              19x19 strategy sessions.
            </p>
          </div>

          {/* Neural Network AI */}
          <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-6 hover:border-sky-500/50 transition-colors">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex-shrink-0 rounded-lg bg-sky-500/10 p-2">
                <svg
                  className="w-6 h-6 text-sky-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white">Neural Network AI</h3>
            </div>
            <p className="text-sm text-slate-300">
              Challenge AI opponents trained via self-play with MCTS. Models range from
              beginner-friendly to expert-level across all board types.
            </p>
          </div>

          {/* Real-Time Multiplayer */}
          <div className="rounded-2xl border border-slate-700 bg-slate-900/70 p-6 hover:border-amber-500/50 transition-colors sm:col-span-2 lg:col-span-1">
            <div className="mb-3 flex items-center gap-3">
              <div className="flex-shrink-0 rounded-lg bg-amber-500/10 p-2">
                <svg
                  className="w-6 h-6 text-amber-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={1.5}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white">Real-Time Multiplayer</h3>
            </div>
            <p className="text-sm text-slate-300">
              Create or join live games supporting 2-4 players. Match with friends or random
              opponents through the lobby with matchmaking.
            </p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="border-t border-slate-800 bg-slate-900/30">
        <div className="container mx-auto px-4 py-16 sm:py-20">
          <h2 className="text-center text-2xl sm:text-3xl font-bold text-white mb-10">
            How It Works
          </h2>
          <div className="grid gap-8 sm:grid-cols-3 max-w-4xl mx-auto text-center">
            <div>
              <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-lg">
                1
              </div>
              <h3 className="font-semibold text-white mb-1">Place Rings</h3>
              <p className="text-sm text-slate-400">
                Take turns placing rings on the board. Stack on top of opponents to capture them.
              </p>
            </div>
            <div>
              <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-lg">
                2
              </div>
              <h3 className="font-semibold text-white mb-1">Form Lines</h3>
              <p className="text-sm text-slate-400">
                Connect rings in a row to score bonus points and trigger special collapse decisions.
              </p>
            </div>
            <div>
              <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400 font-bold text-lg">
                3
              </div>
              <h3 className="font-semibold text-white mb-1">Claim Territory</h3>
              <p className="text-sm text-slate-400">
                Surround regions of the board to earn territory points. The player with the most
                points wins.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer CTA */}
      <section className="border-t border-slate-800">
        <div className="container mx-auto px-4 py-12 sm:py-16 text-center">
          <h2 className="text-xl sm:text-2xl font-bold text-white mb-4">Ready to play?</h2>
          <p className="text-slate-400 mb-6 max-w-md mx-auto">
            Jump straight into the sandbox — no account required. Or sign up to play rated
            multiplayer games.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
            <ButtonLink to="/sandbox" size="lg" className="min-w-[180px]">
              Try the Sandbox
            </ButtonLink>
            <ButtonLink to="/register" variant="outline" size="lg" className="min-w-[180px]">
              Create Account
            </ButtonLink>
          </div>
        </div>
      </section>

      {/* Minimal footer */}
      <footer className="border-t border-slate-800 py-6 text-center text-xs text-slate-500">
        <div className="container mx-auto px-4 flex flex-col sm:flex-row items-center justify-center gap-3">
          <span>RingRift</span>
          <span className="hidden sm:inline">&middot;</span>
          <Link to="/help" className="hover:text-slate-300 transition-colors">
            Help
          </Link>
          <span className="hidden sm:inline">&middot;</span>
          <Link to="/sandbox" className="hover:text-slate-300 transition-colors">
            Sandbox
          </Link>
        </div>
      </footer>
    </div>
  );
}
