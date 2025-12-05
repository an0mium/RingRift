import React, { useState, useEffect } from 'react';
import type { GamePhase, Move } from '../../shared/types/game';

export type TeachingTopic =
  | 'ring_placement'
  | 'stack_movement'
  | 'capturing'
  | 'chain_capture'
  | 'line_bonus'
  | 'territory'
  | 'active_no_moves'
  | 'forced_elimination'
  | 'victory_elimination'
  | 'victory_territory'
  | 'victory_stalemate';

export interface TeachingContent {
  title: string;
  icon: string;
  description: string;
  tips: string[];
  relatedPhases?: GamePhase[];
}

const TEACHING_CONTENT: Record<TeachingTopic, TeachingContent> = {
  ring_placement: {
    title: 'Ring Placement',
    icon: '◯',
    description:
      'Players take turns placing rings from their hand onto empty board spaces. Rings stack on top of any existing rings at that position.',
    tips: [
      'Placing adjacent to your existing rings helps build territory',
      'Stacking on opponents can set up future captures',
      'The placement phase continues until all players have placed all rings',
    ],
    relatedPhases: ['ring_placement'],
  },
  stack_movement: {
    title: 'Stack Movement',
    icon: '→',
    description:
      'Move a stack you control (your ring on top) in a straight line at least as many spaces as the stack’s height. You can go farther if the path has no stacks or territory spaces blocking you; markers are allowed and may eliminate your top ring when you land on them.',
    tips: [
      'Taller stacks can threaten long moves and future captures',
      'You cannot move through other stacks or territory spaces',
      'Landing on a marker can eliminate the top ring of your stack',
    ],
    relatedPhases: ['movement'],
  },
  capturing: {
    title: 'Capturing',
    icon: '×',
    description:
      'To capture, jump over an adjacent opponent stack in a straight line and land on the empty or marker space just beyond it. You take the top ring from the jumped stack and add it to the bottom of your own stack – captured rings stay in play.',
    tips: [
      'Captures move rings into your stacks; only later eliminations remove rings from the game',
      'Your stack’s cap height must be at least as high as the stack you jump over',
      'Capturing can build powerful stacks but also makes them tempting targets',
    ],
    relatedPhases: ['capture'],
  },
  chain_capture: {
    title: 'Chain Capture',
    icon: '⇉',
    description:
      'After an overtaking capture, if your new stack can capture again, you are in a chain capture. Starting the first capture is optional, but once the chain begins you must keep capturing while any capture exists, choosing which target to jump over each time.',
    tips: [
      'Plan chain captures to traverse multiple enemy stacks in a single turn',
      'You choose which capture to take when several are available, but you cannot stop early while any capture remains',
      'The chain ends only when no legal capture segments remain',
    ],
    relatedPhases: ['chain_capture'],
  },
  line_bonus: {
    title: 'Line Bonus',
    icon: '═',
    description:
      'Lines are built from your markers. When a straight line of your markers reaches the minimum scoring length for this board, it becomes a scoring line: you collapse markers in that line into permanent Territory and, on many boards, must pay a ring-elimination cost from a stack you control.',
    tips: [
      'Lines are formed from markers, not rings – horizontal, vertical, and diagonal lines all count.',
      'Exact-length lines always collapse fully into Territory and usually require you to eliminate a ring from one of your stacks.',
      'Overlength lines can trade safety for value: you may collapse a shorter scoring segment with no elimination, or collapse the full line and pay the ring cost.',
    ],
    relatedPhases: ['line_processing'],
  },
  territory: {
    title: 'Territory Control',
    icon: '▣',
    description:
      'Territory spaces are collapsed cells that you permanently own. When a disconnected region of your pieces is processed, all of its spaces become your Territory and its rings are eliminated, often at the cost of eliminating a ring from one of your other stacks. If your Territory passes more than half of the board, you win immediately.',
    tips: [
      'Territory comes from collapsing marker lines and resolving disconnected regions.',
      'Once a space becomes Territory it cannot be captured back or undone.',
      'Crossing the territory threshold ends the game immediately, even if other wins were possible.',
    ],
    relatedPhases: ['territory_processing'],
  },
  active_no_moves: {
    title: 'When you have no legal moves',
    icon: '⛔',
    description:
      'Sometimes it is your turn but there are no legal placements, movements, or captures available. This is an Active–No–Moves state: the rules engine will either trigger forced elimination of your stacks, or, if no eliminations are possible, treat you as structurally stuck for Last Player Standing and plateau detection.',
    tips: [
      'Active–No–Moves only looks at real moves: placements, movements, and captures. Forced elimination and automatic line/territory processing do not count as real moves for Last Player Standing.',
      'If you still control stacks but have no placements or movements, the game applies forced elimination caps until a real move becomes available or your stacks are exhausted.',
      'On some boards a full plateau can occur where no player has real moves or forced eliminations; in that case the game ends and the final score comes from territory and eliminated rings, not further play.',
    ],
    relatedPhases: ['movement', 'line_processing', 'territory_processing'],
  },
  forced_elimination: {
    title: 'Forced Elimination (FE)',
    icon: '💥',
    description:
      'Forced Elimination happens when you control stacks but have no legal placements, movements, or captures. Caps are removed from your stacks automatically until either a real move becomes available or your stacks are gone. These eliminations are mandatory and follow the rules, not player choice.',
    tips: [
      'Rings removed by forced elimination are permanently eliminated and count toward global Ring Elimination victory, just like eliminations from movement onto markers, line rewards, or territory processing.',
      'Forced elimination does not count as a “real move” for Last Player Standing – it is an automatic clean‑up step the engine applies when you are blocked but still have material.',
      'You cannot skip forced elimination when its conditions are met; the sequence and which caps are removed are fully determined by the rules.',
    ],
    relatedPhases: ['movement', 'territory_processing'],
  },
  victory_elimination: {
    title: 'Victory: Elimination',
    icon: '💎',
    description:
      'Win by eliminating more than half of all rings in the game – not just one opponent’s set. Eliminated rings are permanently removed; captured rings you carry in stacks do not count toward this threshold.',
    tips: [
      'Track eliminated rings globally, across all players',
      'Eliminations come from movement onto markers, line rewards, territory processing, and forced eliminations',
      'A large line or territory resolution can suddenly push you over the elimination threshold',
    ],
  },
  victory_territory: {
    title: 'Victory: Territory',
    icon: '🏰',
    description:
      'Win by owning more than half of all board spaces as Territory. Territory comes from collapsing marker lines and resolving disconnected regions, and once a space becomes Territory it can’t be captured back.',
    tips: [
      'Territory is permanent: once claimed, those spaces never return to neutral.',
      'On an 8×8 board, you need at least 33 Territory spaces to win.',
      'Territory victories often come from big line/region resolutions rather than single moves.',
    ],
  },
  victory_stalemate: {
    title: 'Victory: Last Player Standing',
    icon: '👑',
    description:
      'Last Player Standing happens when, after a full round of turns, you are the only player who can still make real moves (placements, movements, or captures). Forced eliminations and automatic territory processing do not prevent LPS.',
    tips: [
      'Real moves are placements, movements, and captures; forced eliminations do not count.',
      'Opponents who are out of real moves for a full round are considered out for LPS.',
      'Structural stalemate is different: it happens when no players have real moves or forced eliminations available.',
    ],
  },
};

export interface TeachingOverlayProps {
  /** The topic to display */
  topic: TeachingTopic;
  /** Whether the overlay is visible */
  isOpen: boolean;
  /** Callback when overlay is closed */
  onClose: () => void;
  /** Position hint for the overlay */
  position?: 'center' | 'bottom-right';
  /** Additional CSS classes */
  className?: string;
}

/**
 * Overlay component that displays teaching content for game concepts.
 * Used during analysis, replay, or when players need help understanding a mechanic.
 */
export function TeachingOverlay({
  topic,
  isOpen,
  onClose,
  position = 'center',
  className = '',
}: TeachingOverlayProps) {
  const content = TEACHING_CONTENT[topic];

  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !content) return null;

  const positionClasses =
    position === 'center'
      ? 'fixed inset-0 flex items-center justify-center'
      : 'fixed bottom-4 right-4';

  return (
    <div className={positionClasses} style={{ zIndex: 60 }}>
      {/* Backdrop for center position */}
      {position === 'center' && (
        <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      )}

      {/* Content Card */}
      <div
        className={`relative bg-slate-900 border border-slate-600 rounded-xl shadow-2xl max-w-md w-full mx-4 overflow-hidden ${className}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="teaching-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 bg-slate-800/80 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{content.icon}</span>
            <h2 id="teaching-title" className="text-lg font-bold text-slate-100">
              {content.title}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
            aria-label="Close"
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

        {/* Description */}
        <div className="px-4 py-3">
          <p className="text-sm text-slate-300 leading-relaxed">{content.description}</p>
        </div>

        {/* Tips */}
        <div className="px-4 pb-4">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">
            Tips
          </h3>
          <ul className="space-y-2">
            {content.tips.map((tip, idx) => (
              <li key={idx} className="flex items-start gap-2 text-sm text-slate-400">
                <span className="text-emerald-400 mt-0.5">•</span>
                <span>{tip}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Related Phases Badge */}
        {content.relatedPhases && content.relatedPhases.length > 0 && (
          <div className="px-4 pb-3 flex items-center gap-2">
            <span className="text-[10px] text-slate-500 uppercase">Applies to:</span>
            {content.relatedPhases.map((phase) => (
              <span
                key={phase}
                className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700"
              >
                {phase.replace('_', ' ')}
              </span>
            ))}
          </div>
        )}

        {/* Footer hint */}
        <div className="px-4 py-2 bg-slate-800/50 border-t border-slate-700/50 text-center">
          <span className="text-[10px] text-slate-500">
            Press <kbd className="px-1 py-0.5 bg-slate-700 rounded text-[9px]">Esc</kbd> or click
            outside to close
          </span>
        </div>
      </div>
    </div>
  );
}

/**
 * Hook to manage teaching overlay state
 */
export function useTeachingOverlay() {
  const [currentTopic, setCurrentTopic] = useState<TeachingTopic | null>(null);

  const showTopic = (topic: TeachingTopic) => setCurrentTopic(topic);
  const hideTopic = () => setCurrentTopic(null);

  return {
    currentTopic,
    isOpen: currentTopic !== null,
    showTopic,
    hideTopic,
  };
}

/**
 * Determine which teaching topic is relevant for a given move
 */
export function getTeachingTopicForMove(move: Move): TeachingTopic | null {
  switch (move.type) {
    case 'place_ring':
    case 'skip_placement':
      return 'ring_placement';
    case 'move_stack':
    case 'move_ring':
    case 'build_stack':
      return 'stack_movement';
    case 'overtaking_capture':
      return 'capturing';
    case 'continue_capture_segment':
      return 'chain_capture';
    case 'process_line':
    case 'choose_line_reward':
      return 'line_bonus';
    case 'process_territory_region':
    case 'eliminate_rings_from_stack':
    case 'skip_territory_processing':
      return 'territory';
    default:
      return null;
  }
}

/**
 * Quick-access buttons for all teaching topics
 */
export function TeachingTopicButtons({
  onSelectTopic,
  className = '',
}: {
  onSelectTopic: (topic: TeachingTopic) => void;
  className?: string;
}) {
  const topics: { topic: TeachingTopic; label: string; icon: string }[] = [
    { topic: 'ring_placement', label: 'Placement', icon: '◯' },
    { topic: 'stack_movement', label: 'Movement', icon: '→' },
    { topic: 'capturing', label: 'Capture', icon: '×' },
    { topic: 'chain_capture', label: 'Chain', icon: '⇉' },
    { topic: 'line_bonus', label: 'Lines', icon: '═' },
    { topic: 'territory', label: 'Territory', icon: '▣' },
  ];

  const victoryTopics: { topic: TeachingTopic; label: string; icon: string }[] = [
    { topic: 'victory_elimination', label: 'Elimination', icon: '💎' },
    { topic: 'victory_territory', label: 'Territory', icon: '🏰' },
    { topic: 'victory_stalemate', label: 'Stalemate', icon: '👑' },
  ];

  return (
    <div className={`space-y-2 ${className}`}>
      <div className="text-[10px] text-slate-500 uppercase tracking-wide">Game Mechanics</div>
      <div className="flex flex-wrap gap-1">
        {topics.map(({ topic, label, icon }) => (
          <button
            key={topic}
            type="button"
            onClick={() => onSelectTopic(topic)}
            className="px-2 py-1 rounded text-[11px] bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
          >
            {icon} {label}
          </button>
        ))}
      </div>

      <div className="text-[10px] text-slate-500 uppercase tracking-wide mt-3">
        Victory Conditions
      </div>
      <div className="flex flex-wrap gap-1">
        {victoryTopics.map(({ topic, label, icon }) => (
          <button
            key={topic}
            type="button"
            onClick={() => onSelectTopic(topic)}
            className="px-2 py-1 rounded text-[11px] bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition-colors"
          >
            {icon} {label}
          </button>
        ))}
      </div>
    </div>
  );
}
