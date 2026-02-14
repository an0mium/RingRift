import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { gameApi } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import LoadingSpinner from '../components/LoadingSpinner';
import { Button } from '../components/ui/Button';
import { extractErrorMessage } from '../utils/errorReporting';

export default function JoinByInvitePage() {
  const { inviteCode } = useParams<{ inviteCode: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const [gameInfo, setGameInfo] = useState<{
    id: string;
    boardType: string;
    maxPlayers: number;
    status: string;
    playerCount: number;
    players: { id: string; username: string; rating: number }[];
  } | null>(null);
  const [isJoining, setIsJoining] = useState(false);

  useEffect(() => {
    if (!inviteCode) {
      setError('No invite code provided');
      return;
    }

    if (!user) {
      // Not authenticated - redirect to login with return path
      navigate('/login', { replace: true, state: { from: `/join/${inviteCode}` } });
      return;
    }

    // Fetch game info and auto-join
    let cancelled = false;

    async function joinViaInvite() {
      try {
        setIsJoining(true);
        // First fetch game info to show loading state
        const info = await gameApi.getGameByInvite(inviteCode!);
        if (cancelled) return;
        setGameInfo(info);

        // If the game is not waiting, show info without joining
        if (info.status !== 'waiting') {
          // If user is already a player, navigate to the game
          const isPlayer = info.players.some((p) => p.id === user?.id);
          if (isPlayer) {
            navigate(`/game/${info.id}`, { replace: true });
            return;
          }
          setError(
            info.status === 'active'
              ? 'This game is already in progress.'
              : 'This game is no longer accepting players.'
          );
          setIsJoining(false);
          return;
        }

        // Auto-join the game
        const result = await gameApi.joinByInvite(inviteCode!);
        if (cancelled) return;
        navigate(`/game/${result.id}`, { replace: true });
      } catch (err: unknown) {
        if (cancelled) return;
        setIsJoining(false);
        setError(extractErrorMessage(err, 'Failed to join game'));
      }
    }

    joinViaInvite();
    return () => {
      cancelled = true;
    };
  }, [inviteCode, user, navigate]);

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (isJoining && !error) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <LoadingSpinner size="lg" />
        <p className="text-slate-300 text-lg">Joining game...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="bg-slate-800/70 rounded-xl border border-slate-700 p-8 max-w-md w-full text-center space-y-4">
          <div className="text-5xl">
            {error.includes('full') ? '🚫' : error.includes('progress') ? '🎮' : '⚠️'}
          </div>
          <h2 className="text-xl font-semibold text-white">Cannot Join Game</h2>
          <p className="text-slate-300">{error}</p>
          {gameInfo && (
            <div className="text-sm text-slate-400 space-y-1">
              <p>
                Board: {gameInfo.boardType} | Players: {gameInfo.playerCount}/{gameInfo.maxPlayers}
              </p>
            </div>
          )}
          <div className="flex gap-3 justify-center pt-2">
            <Button type="button" onClick={() => navigate('/lobby')}>
              Go to Lobby
            </Button>
            {gameInfo && gameInfo.status === 'active' && (
              <Button
                type="button"
                variant="secondary"
                onClick={() => navigate(`/spectate/${gameInfo.id}`)}
              >
                Spectate
              </Button>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center">
      <LoadingSpinner size="lg" />
    </div>
  );
}
