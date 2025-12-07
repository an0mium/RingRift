import type { GameResult } from '../../shared/types/game';

export function getGameOverBannerText(reason: GameResult['reason']): string {
  switch (reason) {
    case 'ring_elimination':
      return 'Game over – victory by ring elimination.';
    case 'territory_control':
      return 'Game over – victory by territory control.';
    case 'last_player_standing':
      return 'Game over – victory by last player standing.';
    case 'game_completed':
      return 'Game over – structural stalemate. Winner decided by tiebreak ladder: territory spaces, eliminated rings (including rings in hand), markers, then last real action.';
    case 'timeout':
      return 'Game over – victory on time.';
    case 'resignation':
      return 'Game over – victory by resignation.';
    case 'abandonment':
      return 'Game over – game abandoned.';
    case 'draw':
      return 'Game over – draw.';
    default:
      return 'Game over.';
  }
}
