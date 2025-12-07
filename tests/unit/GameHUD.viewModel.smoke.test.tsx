import React from 'react';
import { render, screen } from '@testing-library/react';
import { GameHUD } from '../../src/client/components/GameHUD';
import type { HUDViewModel, PlayerViewModel } from '../../src/client/adapters/gameViewModels';
import type { TimeControl } from '../../src/shared/types/game';

// Stub out telemetry to avoid side effects during render.
jest.mock('../../src/client/utils/rulesUxTelemetry', () => ({
  sendRulesUxEvent: jest.fn(),
  logRulesUxEvent: jest.fn(),
  logHelpOpenEvent: jest.fn(),
  newHelpSessionId: jest.fn(() => 'help-session'),
  newOverlaySessionId: jest.fn(() => 'overlay-session'),
}));

// Stub TeachingOverlay to avoid pulling in large copy maps.
jest.mock('../../src/client/components/TeachingOverlay', () => {
  const React = require('react');
  return {
    TeachingOverlay: ({ children }: { children: React.ReactNode }) =>
      React.createElement('div', { 'data-testid': 'teaching-overlay' }, children),
    useTeachingOverlay: () => ({
      currentTopic: null,
      isOpen: false,
      showTopic: jest.fn(),
      hideTopic: jest.fn(),
    }),
  };
});

function buildPlayer(
  overrides: Partial<PlayerViewModel> & Pick<PlayerViewModel, 'playerNumber' | 'username'>
): PlayerViewModel {
  return {
    id: `p${overrides.playerNumber}`,
    playerNumber: overrides.playerNumber,
    username: overrides.username,
    isCurrentPlayer: false,
    isUserPlayer: false,
    colorClass: 'text-red-500',
    ringStats: { inHand: 2, onBoard: 3, eliminated: 1, total: 6 },
    territorySpaces: 0,
    aiInfo: { isAI: false },
    ...overrides,
  };
}

function buildViewModel(): HUDViewModel {
  const players = [
    buildPlayer({ playerNumber: 1, username: 'Alice', isCurrentPlayer: true, isUserPlayer: true }),
    buildPlayer({ playerNumber: 2, username: 'Bot', aiInfo: { isAI: true, difficulty: 5 } }),
  ];

  return {
    phase: {
      phaseKey: 'territory_processing',
      label: 'Territory',
      description: 'Process disconnected regions',
      colorClass: 'text-amber-400',
      icon: '🏰',
      actionHint: 'Select a region to process',
      spectatorHint: 'Watching territory processing',
    },
    players,
    turnNumber: 3,
    moveNumber: 5,
    pieRuleSummary: undefined,
    instruction: 'Your move',
    connectionStatus: 'connected',
    isConnectionStale: false,
    isSpectator: false,
    spectatorCount: 0,
    subPhaseDetail: 'Processing captures',
    decisionPhase: {
      isActive: true,
      actingPlayerNumber: 1,
      actingPlayerName: 'Alice',
      isLocalActor: true,
      label: 'Choose line reward',
      shortLabel: 'Line reward',
      description: 'Select how to collapse the overlength line',
      timeRemainingMs: 4_000,
      showCountdown: true,
      isServerCapped: false,
      canSkip: true,
      statusChip: { text: 'Forced elimination pending', tone: 'attention' },
    },
    weirdState: {
      type: 'forced-elimination',
      title: 'Forced elimination required',
      body: 'No legal moves available; you must eliminate one of your stacks.',
      tone: 'warning',
    },
  };
}

const TIME_CONTROL: TimeControl = {
  type: 'rapid',
  initialTime: 300,
  increment: 2,
};

describe('GameHUD (view model path)', () => {
  it('renders phase, players, decision banner, and time control summary', () => {
    const viewModel = buildViewModel();

    render(<GameHUD viewModel={viewModel} timeControl={TIME_CONTROL} />);

    expect(screen.getByTestId('game-hud')).toBeInTheDocument();
    expect(screen.getByTestId('phase-indicator')).toHaveTextContent(/Territory/i);
    expect(screen.getByTestId('hud-time-control-summary')).toHaveTextContent('Rapid • 5+2');
    expect(screen.getByTestId('decision-phase-banner')).toBeInTheDocument();
    expect(screen.getByTestId('decision-phase-countdown')).toHaveAttribute(
      'data-severity',
      'warning'
    );
    expect(screen.getByTestId('hud-decision-status-chip')).toHaveTextContent(/Forced elimination/i);
    expect(screen.getByTestId('hud-decision-skip-hint')).toBeInTheDocument();
    expect(screen.getByTestId('hud-weird-state-banner')).toBeInTheDocument();
    expect(screen.getByTestId('player-card-p1')).toHaveTextContent('Alice');
    expect(screen.getByTestId('player-card-p2')).toHaveTextContent('Bot');
  });
});
