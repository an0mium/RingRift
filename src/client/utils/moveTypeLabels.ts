import type { MoveType, LegacyMoveType } from '../../shared/types/game';
import {
  isLegacyMoveType,
  normalizeLegacyMoveType,
} from '../../shared/engine/legacy/legacyMoveTypes';

const CANONICAL_LABELS: Record<string, string> = {
  place_ring: 'Place Ring',
  skip_placement: 'Pass',
  no_placement_action: 'No Action',
  move_stack: 'Move',
  overtaking_capture: 'Capture',
  continue_capture_segment: 'Continue Capture',
  chain_capture: 'Chain Capture',
  skip_capture: 'End Capture',
  no_movement_action: 'No Action',
  process_line: 'Score Line',
  choose_line_option: 'Line Reward',
  no_line_action: 'No Action',
  choose_territory_option: 'Claim Territory',
  skip_territory_processing: 'Skip Territory',
  no_territory_action: 'No Action',
  eliminate_rings_from_stack: 'Remove Ring',
  forced_elimination: 'Sacrifice',
  swap_sides: 'Swap Sides',
  recovery_slide: 'Recovery',
  skip_recovery: 'Skip Recovery',
};

const LEGACY_LABELS: Record<LegacyMoveType, string> = {
  move_ring: 'Move',
  build_stack: 'Build Stack',
  choose_line_reward: 'Line Reward',
  process_territory_region: 'Claim Territory',
  line_formation: 'Line Scored',
  territory_claim: 'Territory Claimed',
};

const LEGACY_CATEGORY_LABELS: Partial<Record<LegacyMoveType, string>> = {
  line_formation: 'Line',
  territory_claim: 'Territory',
};

export function formatMoveTypeLabel(moveType: MoveType): string {
  if (isLegacyMoveType(moveType)) {
    const legacyLabel = LEGACY_LABELS[moveType] ?? moveType.replace(/_/g, ' ');
    return `${legacyLabel} (legacy)`;
  }

  const canonicalType = normalizeLegacyMoveType(moveType);
  return CANONICAL_LABELS[canonicalType] ?? canonicalType.replace(/_/g, ' ');
}

export function getMoveCategoryLabel(moveType: MoveType): string {
  if (isLegacyMoveType(moveType)) {
    const legacyCategory = LEGACY_CATEGORY_LABELS[moveType];
    if (legacyCategory) {
      return legacyCategory;
    }
  }

  const canonicalType = normalizeLegacyMoveType(moveType);
  switch (canonicalType) {
    case 'place_ring':
      return 'Placement';
    case 'skip_placement':
    case 'no_placement_action':
      return 'Placement (pass)';
    case 'move_stack':
    case 'no_movement_action':
      return 'Movement';
    case 'overtaking_capture':
    case 'continue_capture_segment':
      return 'Capture';
    case 'skip_capture':
      return 'End Capture';
    case 'process_line':
    case 'choose_line_option':
    case 'no_line_action':
      return 'Line';
    case 'eliminate_rings_from_stack':
      return 'Remove Ring';
    case 'choose_territory_option':
    case 'skip_territory_processing':
    case 'no_territory_action':
      return 'Territory';
    case 'forced_elimination':
      return 'Sacrifice';
    case 'swap_sides':
      return 'Swap Sides';
    case 'recovery_slide':
    case 'skip_recovery':
      return 'Recovery';
    default:
      return canonicalType.replace(/_/g, ' ');
  }
}
