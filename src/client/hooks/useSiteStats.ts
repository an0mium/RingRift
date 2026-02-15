import { useState, useEffect } from 'react';

interface SiteStats {
  playersOnline: number;
  activeGames: number;
  gamesPlayed: number;
}

/**
 * Fetches public site stats from /api/stats.
 * Polls every 60s while mounted. Returns null until first successful fetch.
 */
export function useSiteStats(): SiteStats | null {
  const [stats, setStats] = useState<SiteStats | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetchStats = async () => {
      try {
        const res = await fetch('/api/stats');
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) {
          setStats({
            playersOnline: data.playersOnline ?? 0,
            activeGames: data.activeGames ?? 0,
            gamesPlayed: data.gamesPlayed ?? 0,
          });
        }
      } catch {
        // Silently fail — stats are non-critical
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 60_000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return stats;
}
