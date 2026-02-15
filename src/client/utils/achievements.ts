/**
 * Client-side achievement definitions.
 *
 * Achievements are computed from the user's profile data (rating, gamesPlayed,
 * gamesWon) and recent game history. This avoids needing a backend endpoint
 * for now — the backend Achievement interface in shared/types/user.ts is
 * compatible if we add server-side tracking later.
 */

import type { User } from '../../shared/types/user';
import type { GameSummary } from '../services/api';

export interface AchievementDef {
  id: string;
  name: string;
  description: string;
  icon: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  check: (user: User, games: GameSummary[]) => boolean;
}

export interface UnlockedAchievement {
  id: string;
  name: string;
  description: string;
  icon: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
}

const BOARD_TYPES = ['square8', 'square19', 'hex8', 'hexagonal'] as const;

export const ACHIEVEMENT_DEFS: AchievementDef[] = [
  // --- Common ---
  {
    id: 'first_game',
    name: 'First Steps',
    description: 'Play your first game',
    icon: '🎮',
    rarity: 'common',
    check: (u) => u.gamesPlayed >= 1,
  },
  {
    id: 'first_win',
    name: 'First Victory',
    description: 'Win your first game',
    icon: '🏆',
    rarity: 'common',
    check: (u) => u.gamesWon >= 1,
  },
  {
    id: 'ten_games',
    name: 'Dedicated',
    description: 'Play 10 games',
    icon: '⭐',
    rarity: 'common',
    check: (u) => u.gamesPlayed >= 10,
  },
  {
    id: 'five_wins',
    name: 'On a Roll',
    description: 'Win 5 games',
    icon: '🔥',
    rarity: 'common',
    check: (u) => u.gamesWon >= 5,
  },

  // --- Rare ---
  {
    id: 'fifty_games',
    name: 'Veteran',
    description: 'Play 50 games',
    icon: '🎖️',
    rarity: 'rare',
    check: (u) => u.gamesPlayed >= 50,
  },
  {
    id: 'twenty_wins',
    name: 'Sharpshooter',
    description: 'Win 20 games',
    icon: '🎯',
    rarity: 'rare',
    check: (u) => u.gamesWon >= 20,
  },
  {
    id: 'board_explorer',
    name: 'Board Explorer',
    description: 'Play on all 4 board types',
    icon: '🗺️',
    rarity: 'rare',
    check: (_u, games) => {
      const types = new Set(games.map((g) => g.boardType));
      return BOARD_TYPES.every((t) => types.has(t));
    },
  },
  {
    id: 'multiplayer_win',
    name: 'Crowd Pleaser',
    description: 'Win a 4-player game',
    icon: '👥',
    rarity: 'rare',
    check: (u, games) => games.some((g) => g.maxPlayers >= 4 && g.winnerId === u.id),
  },
  {
    id: 'win_rate_60',
    name: 'Consistent',
    description: 'Maintain a 60%+ win rate (min 10 games)',
    icon: '📈',
    rarity: 'rare',
    check: (u) => u.gamesPlayed >= 10 && u.gamesWon / u.gamesPlayed >= 0.6,
  },

  // --- Epic ---
  {
    id: 'hundred_games',
    name: 'Centurion',
    description: 'Play 100 games',
    icon: '💯',
    rarity: 'epic',
    check: (u) => u.gamesPlayed >= 100,
  },
  {
    id: 'rating_1500',
    name: 'Ring Master',
    description: 'Reach a rating of 1500',
    icon: '💎',
    rarity: 'epic',
    check: (u) => u.rating >= 1500,
  },
  {
    id: 'fifty_wins',
    name: 'Commander',
    description: 'Win 50 games',
    icon: '⚔️',
    rarity: 'epic',
    check: (u) => u.gamesWon >= 50,
  },

  // --- Legendary ---
  {
    id: 'rating_2000',
    name: 'Grandmaster',
    description: 'Reach a rating of 2000',
    icon: '👑',
    rarity: 'legendary',
    check: (u) => u.rating >= 2000,
  },
  {
    id: 'five_hundred_games',
    name: 'Legend',
    description: 'Play 500 games',
    icon: '🌟',
    rarity: 'legendary',
    check: (u) => u.gamesPlayed >= 500,
  },
];

export function computeAchievements(user: User, games: GameSummary[]): UnlockedAchievement[] {
  return ACHIEVEMENT_DEFS.filter((def) => def.check(user, games)).map(
    ({ id, name, description, icon, rarity }) => ({
      id,
      name,
      description,
      icon,
      rarity,
    })
  );
}

export const RARITY_COLORS: Record<string, string> = {
  common: 'border-slate-600 bg-slate-800/60',
  rare: 'border-sky-600 bg-sky-950/40',
  epic: 'border-purple-600 bg-purple-950/40',
  legendary: 'border-amber-500 bg-amber-950/40',
};

export const RARITY_TEXT: Record<string, string> = {
  common: 'text-slate-400',
  rare: 'text-sky-400',
  epic: 'text-purple-400',
  legendary: 'text-amber-400',
};
