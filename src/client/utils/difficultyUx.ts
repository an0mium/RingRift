export interface DifficultyDescriptor {
  /** Numeric ladder level (1–10). Matches the difficulty field on AI profiles. */
  id: number;
  /** Short, player-facing label, e.g. "Challenging" or "Expert". */
  name: string;
  /** One-line summary used in dropdowns and compact selectors. */
  shortDescription: string;
  /** Richer text used in "About difficulty levels" help surfaces. */
  detailedDescription: string;
  /** Who this level is primarily for, phrased in human skill terms. */
  recommendedAudience: string;
  /** Optional implementation or calibration notes (not usually shown directly to players). */
  notes?: string;
}

/**
 * Ordered metadata for AI difficulty levels D1–D10 as exposed in the client.
 *
 * This table is intentionally small and stable so that:
 * - Client UX can show meaningful labels instead of bare numbers.
 * - Calibration UX can reference the same wording as
 *   docs/ai/AI_HUMAN_CALIBRATION_GUIDE.md.
 *
 * Engine behaviour is still defined by the canonical ladder:
 * - Python: ai-service/app/config/ladder_config.py
 * - TypeScript: server/game/ai/AIEngine.ts + AI_DIFFICULTY_PRESETS.
 */
export const DIFFICULTY_DESCRIPTORS: readonly DifficultyDescriptor[] = [
  {
    id: 1,
    name: 'Beginner (Random)',
    shortDescription: 'Very weak, mostly-random play for learning the rules.',
    detailedDescription:
      'Uses a mostly random policy with only light safeguards against illegal or impossible moves. Intended purely for learning the interface and basic rules, not for meaningful competition.',
    recommendedAudience:
      'First-time strategy players and testers who want a stress-free way to learn RingRift basics.',
    notes:
      'Not part of the calibrated ladder. Corresponds to legacy D1 random play and should be treated as unrated.',
  },
  {
    id: 2,
    name: 'Learner (D2 – Casual / Learning)',
    shortDescription:
      'Gentle heuristic AI that plays legal, coherent moves but misses many tactics.',
    detailedDescription:
      'Backed by the square8 2-player heuristic profile used for the canonical D2 ladder tier. Makes legal, coherent moves and pursues simple goals but still overlooks many short tactics and long-term plans. New players should be able to win some games after a brief warm-up.',
    recommendedAudience:
      'Brand-new RingRift players and casual board-gamers who want a forgiving opponent while they internalise the rules.',
    notes:
      'Primary calibration anchor for new players on square8 2-player (see AI_HUMAN_CALIBRATION_GUIDE.md Template A).',
  },
  {
    id: 3,
    name: 'Casual (D3 – Minimax)',
    shortDescription:
      'Entry-level minimax that sees short tactics but still leaves chances to outplay it.',
    detailedDescription:
      'Backed by the canonical D3 minimax profile (heuristic evaluation only). Sees short tactical sequences and avoids many one-move blunders, but remains beatable through deeper planning, resource management, and complex multi-threat positions.',
    recommendedAudience:
      'Players who can reliably beat D2 and want a slightly tougher but still relaxed experience.',
    notes:
      'Canonical ladder tier: minimax (non-neural). Search budget and randomness are tuned per board/profile.',
  },
  {
    id: 4,
    name: 'Challenging (D4 – Intermediate)',
    shortDescription:
      'Minimax with NNUE evaluation that punishes obvious mistakes and wins many close games.',
    detailedDescription:
      'Backed by the canonical D4 minimax profile (NNUE evaluation). Sees short tactics, avoids most outright blunders, and reliably converts clear material or territory advantages. Stronger casual players should find this engaging but still winnable.',
    recommendedAudience:
      'Intermediate RingRift players and experienced abstract-board-game players (e.g. club-night Chess or Go strength).',
    notes:
      'Key calibration anchor for “intermediate” difficulty on square8 2-player (see AI_HUMAN_CALIBRATION_GUIDE.md Template B).',
  },
  {
    id: 5,
    name: 'Tough (D5 – MCTS)',
    shortDescription:
      'First MCTS tier: tougher conversion and stronger tactical awareness than minimax.',
    detailedDescription:
      'Backed by the canonical D5 MCTS profile (heuristic rollouts; no neural net). Plays with stronger global search than minimax tiers, improving midgame conversion and reducing unforced mistakes while remaining fast enough for casual sandbox play.',
    recommendedAudience:
      'Players who beat D4 consistently and want a demanding but not yet “expert” opponent.',
    notes: 'Canonical ladder tier: MCTS (non-neural).',
  },
  {
    id: 6,
    name: 'Advanced (D6 – Neural MCTS)',
    shortDescription: 'Neural-guided MCTS that punishes shallow plans and weak structures.',
    detailedDescription:
      'Backed by the canonical D6 MCTS profile (neural value/policy guidance). Avoids most obvious tactical shots, punishes over-extensions, and steadily converts small advantages over many moves. Games often feel tense and technical.',
    recommendedAudience:
      'Strong club-level abstract-game players and advanced RingRift players who already handle D4 comfortably.',
    notes:
      'Primary “advanced” calibration anchor on square8 2-player (see AI_HUMAN_CALIBRATION_GUIDE.md Templates B/C).',
  },
  {
    id: 7,
    name: 'Expert (D7 – Neural MCTS)',
    shortDescription: 'Neural-guided MCTS with higher budget; mistakes are punished quickly.',
    detailedDescription:
      'Backed by the canonical D7 MCTS profile (neural guidance with higher search budget). Plays with high tactical awareness and strong conversion, and will quickly capitalise on poor trades or slow play.',
    recommendedAudience:
      'Very strong regular players who beat D6 at a healthy rate but are not yet ready for the full D8 challenge.',
    notes: 'Canonical ladder tier: MCTS (neural).',
  },
  {
    id: 8,
    name: 'Strong Expert (D8 – Near‑expert)',
    shortDescription: 'Strong search-based AI intended as the top calibrated tier for most humans.',
    detailedDescription:
      'Backed by the canonical D8 MCTS profile (neural guidance with large search budget). Rarely blunders outright, converts small advantages, and punishes greedy territory or elimination attempts. Even very strong human players should find sustaining a high win rate difficult.',
    recommendedAudience:
      'Very strong RingRift players and serious abstract-game enthusiasts who want a near‑expert challenge.',
    notes:
      'Top calibrated tier in the current square8 2-player ladder (see AI_HUMAN_CALIBRATION_GUIDE.md Template C).',
  },
  {
    id: 9,
    name: 'Master (Experimental)',
    shortDescription:
      'Experimental high-end AI; behaviour and strength may change between releases.',
    detailedDescription:
      'Represents experimental high-difficulty configurations (often based on stronger MCTS or Descent-style search). Not currently part of the human-calibrated ladder and primarily used for internal testing, tournaments, or stress testing.',
    recommendedAudience:
      'Internal testers and expert players who explicitly want to probe experimental, potentially unstable AI behaviour.',
    notes:
      'Treated as experimental / unrated. Do not use as a calibration anchor; ladder and engine settings may change without notice.',
  },
  {
    id: 10,
    name: 'Grandmaster (Experimental)',
    shortDescription: 'Maximum experimental difficulty; not guaranteed to be fun or fair.',
    detailedDescription:
      'Reserved for maximum-strength experimental profiles, typically used to explore the limits of search or new engine variants. May feel inconsistent or unbalanced on some boards and is not calibrated against human skill bands.',
    recommendedAudience:
      'AI and engine developers, or expert players explicitly stress-testing the system.',
    notes:
      'Explicitly unrated and outside the human calibration guide. Use with caution in user-facing surfaces; prefer D2/D4/D6/D8 for meaningful difficulty settings.',
  },
];

/**
 * Convenience helper for looking up a difficulty descriptor by its id.
 */
export function getDifficultyDescriptor(id: number): DifficultyDescriptor | undefined {
  return DIFFICULTY_DESCRIPTORS.find((descriptor) => descriptor.id === id);
}
