import type { PlayerChoiceType } from '../../shared/types/game';

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Decision / PlayerChoice → UX Mapping
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This module is the single mapping layer from low-level PlayerChoiceType
 * values to user-facing copy and high-level semantics for decision phases.
 *
 * It is intentionally UI--framework agnostic: the view models here are
 * simple data objects that can be consumed by React components (ChoiceDialog,
 * GameHUD, spectator overlays, etc.) without importing engine or transport
 * details.
 */

export type ChoiceKind =
  | 'line_order'
  | 'line_reward'
  | 'ring_elimination'
  | 'territory_region_order'
  | 'capture_direction'
  | 'other';

export interface ChoiceCopy {
  /** Main title for the acting player's dialog/HUD. */
  title: string;
  /** Optional, more detailed explanation shown below the title. */
  description?: string;
  /**
   * Optional strategic tip for newer players. Should explain the tradeoffs
   * or tactical considerations involved in this choice.
   */
  strategicTip?: string;
  /** Compact label suitable for chips/badges (e.g. in HUD headers). */
  shortLabel: string;
  /**
   * Spectator-oriented status text. The acting player name may be injected
   * by the caller.
   */
  spectatorLabel: (ctx: { actingPlayerName: string }) => string;
}

export interface ChoiceTimeoutBehavior {
  /** Whether the countdown UI should be shown for this choice type. */
  showCountdown: boolean;
  /** Optional soft warning threshold (e.g. < 5s) for styling. */
  warningThresholdMs?: number;
}

export interface ChoiceViewModel {
  /** Underlying low-level discriminant. */
  type: PlayerChoiceType;
  /** High-level semantic grouping of the choice. */
  kind: ChoiceKind;
  /** Titles, descriptions, and spectator copy for this decision. */
  copy: ChoiceCopy;
  /** Timeout UI semantics. Actual deadline comes from PlayerChoice.timeoutMs. */
  timeout: ChoiceTimeoutBehavior;
}

interface ChoiceViewModelConfig extends Omit<ChoiceViewModel, 'type'> {}

const DEFAULT_TIMEOUT_BEHAVIOR: ChoiceTimeoutBehavior = {
  showCountdown: true,
  warningThresholdMs: 5000,
};

/**
 * Canonical mapping from PlayerChoiceType → high-level UX semantics.
 *
 * NOTE: This table is intentionally the SSOT for decision labels. All
 * components (ChoiceDialog, HUD, spectator views, logs) should ultimately
 * derive human-readable labels from here rather than hard-coding per-type
 * copy in multiple places.
 */
const CHOICE_VIEW_MODEL_MAP: Record<PlayerChoiceType, ChoiceViewModelConfig> = {
  line_order: {
    kind: 'line_order',
    copy: {
      title: 'Multiple Lines Formed!',
      description:
        'You created more than one scoring line. Pick which one to process first. Each line will collapse its markers into permanent territory.',
      strategicTip:
        'Processing order matters! If lines share markers, choosing one may alter or dissolve the other. Consider which territory placement benefits your position most.',
      shortLabel: 'Line order',
      spectatorLabel: ({ actingPlayerName }) =>
        `${actingPlayerName} is choosing which line to score first`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
  line_reward_option: {
    kind: 'line_reward',
    copy: {
      title: 'Overlength Line! Choose Your Reward',
      description:
        'Your markers formed a line longer than required. You have a choice: collapse ALL markers into territory (costs 1 ring), or collapse only the minimum required (costs nothing, but less territory).',
      strategicTip:
        "More territory helps win via Territory Control, but eliminating your rings helps win via Ring Elimination. Consider which victory path you're pursuing and how many rings you can afford to lose.",
      shortLabel: 'Line reward',
      spectatorLabel: ({ actingPlayerName }) => `${actingPlayerName} is choosing their line reward`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
  ring_elimination: {
    kind: 'ring_elimination',
    copy: {
      title: 'Remove a Ring',
      description:
        'Choose which of your stacks to remove the top ring from. This ring is eliminated from play, bringing you closer to the Ring Elimination victory threshold.',
      strategicTip:
        'Pick a stack where losing control hurts least. Losing your top ring from a tall stack keeps you in control of it; losing from a 1-ring stack gives it to your opponent (or makes it neutral).',
      shortLabel: 'Ring elimination',
      spectatorLabel: ({ actingPlayerName }) =>
        `${actingPlayerName} is choosing which ring to remove`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
  region_order: {
    kind: 'territory_region_order',
    copy: {
      title: 'Territory Captured!',
      description:
        'You isolated one or more regions with your markers. Choose which region to claim first. All spaces inside become your territory, and any enemy pieces trapped inside are eliminated.',
      strategicTip:
        'Each region you claim costs your ENTIRE CAP (all your rings) from one stack OUTSIDE the region. Make sure you have a stack to pay with! Larger regions may be worth the cost; smaller ones might not be.',
      shortLabel: 'Territory region',
      spectatorLabel: ({ actingPlayerName }) =>
        `${actingPlayerName} is choosing which territory to claim`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
  capture_direction: {
    kind: 'capture_direction',
    copy: {
      title: 'Chain Capture! Keep Jumping',
      description:
        'You started a capture chain. Choose your next jump direction—captures are mandatory and must continue until no more jumps are possible.',
      strategicTip:
        "Each capture takes the TOP RING from the enemy stack you jump over. Plan your chain to maximize rings captured, but watch out—you might be forced into a bad position if you don't think ahead!",
      shortLabel: 'Capture direction',
      spectatorLabel: ({ actingPlayerName }) =>
        `${actingPlayerName} is choosing their next capture`,
    },
    timeout: DEFAULT_TIMEOUT_BEHAVIOR,
  },
};

/**
 * Return the ChoiceViewModel for a given PlayerChoiceType.
 */
export function getChoiceViewModelForType(type: PlayerChoiceType): ChoiceViewModel {
  const config = CHOICE_VIEW_MODEL_MAP[type];

  if (!config) {
    // Fallback that remains safe for unknown/experimental types while still
    // surfacing a reasonable label in the UI.
    const fallback: ChoiceViewModel = {
      type,
      kind: 'other',
      copy: {
        title: 'Decision Required',
        description: 'A decision is required to continue this phase.',
        shortLabel: 'Decision',
        spectatorLabel: ({ actingPlayerName }) =>
          `Waiting for ${actingPlayerName} to make a decision`,
      },
      timeout: DEFAULT_TIMEOUT_BEHAVIOR,
    };
    return fallback;
  }

  return {
    type,
    ...config,
  };
}

/**
 * Convenience helper for callers that already have a PlayerChoice instance
 * and just need the corresponding view model.
 */
export function getChoiceViewModel(choice: { type: PlayerChoiceType }): ChoiceViewModel {
  return getChoiceViewModelForType(choice.type);
}
