import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MobileGameHUD } from '../../../src/client/components/MobileGameHUD';
import type { HUDViewModel } from '../../../src/client/adapters/gameViewModels';

function createBaseViewModel(overrides: Partial<HUDViewModel> = {}): HUDViewModel {
  return {
    phase: {
      phaseKey: 'movement' as any,
      label: 'Movement',
      description: 'Move a stack',
      icon: '⚡',
      colorClass: 'bg-green-500',
      actionHint: 'Select a stack then a destination',
      spectatorHint: 'Player is choosing a move',
    },
    players: [],
    turnNumber: 1,
    moveNumber: 0,
    connectionStatus: 'connected' as any,
    isConnectionStale: false,
    isSpectator: false,
    spectatorCount: 0,
    subPhaseDetail: undefined,
    decisionPhase: undefined,
    weirdState: undefined,
    ...overrides,
  };
}

describe('MobileGameHUD', () => {
  it('renders spectator badge with viewer count when spectating', () => {
    const vm = createBaseViewModel({
      isSpectator: true,
      spectatorCount: 3,
    });

    render(<MobileGameHUD viewModel={vm} />);

    const badge = screen.getByTestId('mobile-spectator-badge');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent('Spectating');
    expect(badge).toHaveTextContent('3 viewer');
  });
});
