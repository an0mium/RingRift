import type { GameResult } from '../../shared/types/game';

export function getGameOverBannerText(reason: GameResult['reason']): string {
  switch (reason) {
    case 'ring_elimination':
      return 'Victory! All opponent rings eliminated.';
    case 'territory_control':
      return 'Victory! Territory dominance achieved.';
    case 'last_player_standing':
      return 'Victory! Last player standing wins.';
    case 'game_completed':
      return 'Game over — no moves remain. Winner decided by score: territory, then rings eliminated, then markers.';
    case 'timeout':
      return 'Victory! Opponent ran out of time.';
    case 'resignation':
      return 'Victory! Opponent resigned.';
    case 'abandonment':
      return 'Game abandoned.';
    case 'draw':
      return 'Draw — evenly matched!';
    default:
      return 'Game over.';
  }
}
