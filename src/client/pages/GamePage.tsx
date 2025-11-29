import React from 'react';
import { useParams } from 'react-router-dom';
import { BackendGameHost } from './BackendGameHost';
import { SandboxGameHost } from './SandboxGameHost';

/**
 * GamePage
 *
 * Thin shell responsible for:
 * - Selecting the appropriate host based on the current route.
 * - Keeping a stable entry point for existing routes/tests.
 *
 * Host selection:
 * - /game/:gameId and /spectate/:gameId → BackendGameHost (server-backed games)
 * - /sandbox → SandboxGameHost (local sandbox, orchestrator-first)
 *
 * Providers (AuthProvider, GameProvider, SandboxProvider, etc.) are wired at the
 * application root (see index.tsx). This component only decides which host
 * should render within that provider tree.
 */
export default function GamePage() {
  const params = useParams<{ gameId?: string }>();
  const routeGameId = params.gameId;

  if (routeGameId) {
    return <BackendGameHost gameId={routeGameId} />;
  }

  return <SandboxGameHost />;
}
