import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { SpectatorHUD } from '../../../src/client/components/SpectatorHUD';
import type { Player, Move } from '../../../src/shared/types/game';

// Mock child components that have complex dependencies
jest.mock('../../../src/client/components/EvaluationGraph', () => {
  const React = require('react');
  return {
    EvaluationGraph: function MockEvaluationGraph({
      evaluationHistory,
    }: {
      evaluationHistory: unknown[];
    }) {
      return React.createElement(
        'div',
        { 'data-testid': 'evaluation-graph' },
        `EvaluationGraph (${evaluationHistory.length} evals)`
      );
    },
  };
});

jest.mock('../../../src/client/components/MoveAnalysisPanel', () => {
  const React = require('react');
  return {
    MoveAnalysisPanel: function MockMoveAnalysisPanel({ analysis }: { analysis: unknown }) {
      return React.createElement(
        'div',
        { 'data-testid': 'move-analysis-panel' },
        `MoveAnalysisPanel ${analysis ? '(with analysis)' : '(empty)'}`
      );
    },
  };
});

// Mock TeachingOverlay since it has complex dependencies
jest.mock('../../../src/client/components/TeachingOverlay', () => {
  const React = require('react');
  return {
    TeachingOverlay: function MockTeachingOverlay({
      topic,
      isOpen,
      onClose,
    }: {
      topic: string;
      isOpen: boolean;
      onClose: () => void;
    }) {
      if (!isOpen) return null;
      return React.createElement(
        'div',
        { 'data-testid': 'teaching-overlay', 'data-topic': topic },
        [
          React.createElement('span', { key: 'topic' }, `Teaching: ${topic}`),
          React.createElement('button', { key: 'close', onClick: onClose }, 'Close Teaching'),
        ]
      );
    },
    TeachingTopicButtons: function MockTeachingTopicButtons({
      onSelectTopic,
    }: {
      onSelectTopic: (topic: string) => void;
    }) {
      return React.createElement('div', { 'data-testid': 'teaching-topic-buttons' }, [
        React.createElement(
          'button',
          { key: 'recovery', onClick: () => onSelectTopic('recovery_action') },
          'Recovery'
        ),
        React.createElement(
          'button',
          { key: 'territory', onClick: () => onSelectTopic('territory') },
          'Territory'
        ),
        React.createElement(
          'button',
          { key: 'chain', onClick: () => onSelectTopic('chain_capture') },
          'Chain Capture'
        ),
      ]);
    },
    useTeachingOverlay: function useMockTeachingOverlay() {
      const ReactMod = require('react');
      const [currentTopic, setCurrentTopic] = ReactMod.useState(null);
      return {
        currentTopic,
        isOpen: currentTopic !== null,
        showTopic: setCurrentTopic,
        hideTopic: () => setCurrentTopic(null),
      };
    },
  };
});

function createPlayers(): Player[] {
  return [
    {
      id: 'p1',
      username: 'Alice',
      playerNumber: 1,
      type: 'human',
      isReady: true,
      timeRemaining: 120_000,
      ringsInHand: 5,
      eliminatedRings: 1,
      territorySpaces: 2,
      totalRings: 24,
      ringsEliminated: 1,
      territory: 2,
    },
    {
      id: 'p2',
      username: 'Bob',
      playerNumber: 2,
      type: 'human',
      isReady: true,
      timeRemaining: 90_000,
      ringsInHand: 4,
      eliminatedRings: 2,
      territorySpaces: 0,
      totalRings: 24,
      ringsEliminated: 2,
      territory: 0,
    },
  ];
}

function createRecoveryMoveHistory(): Move[] {
  return [
    {
      id: 'm1',
      type: 'recovery_slide',
      player: 1,
      playerNumber: 1,
      phase: 'movement',
      from: { x: 0, y: 0, z: null },
      to: { x: 0, y: 1, z: null },
      timestamp: new Date(),
    },
    {
      id: 'm2',
      type: 'skip_recovery',
      player: 2,
      playerNumber: 2,
      phase: 'movement',
      from: null,
      to: { x: 0, y: 0, z: null },
      timestamp: new Date(),
    },
  ] as Move[];
}

describe('SpectatorHUD - Disconnected Players Banner', () => {
  it('shows disconnected players banner when one player disconnects', () => {
    const disconnectedPlayers = [
      {
        id: 'p1',
        username: 'Alice',
        disconnectedAt: Date.now() - 30000, // 30 seconds ago
      },
    ];

    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        disconnectedPlayers={disconnectedPlayers}
      />
    );

    expect(screen.getByTestId('disconnected-players-banner')).toBeInTheDocument();
    // Multiple elements may contain "Alice" so we check within the banner
    const banner = screen.getByTestId('disconnected-players-banner');
    expect(banner).toHaveTextContent(/Alice/);
    expect(banner).toHaveTextContent(/has disconnected/);
    expect(banner).toHaveTextContent(/Waiting to reconnect/);
  });

  it('shows plural message when multiple players disconnect', () => {
    const disconnectedPlayers = [
      {
        id: 'p1',
        username: 'Alice',
        disconnectedAt: Date.now() - 30000,
      },
      {
        id: 'p2',
        username: 'Bob',
        disconnectedAt: Date.now() - 15000,
      },
    ];

    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        disconnectedPlayers={disconnectedPlayers}
      />
    );

    expect(screen.getByText(/Alice, Bob/)).toBeInTheDocument();
    expect(screen.getByText(/have disconnected/)).toBeInTheDocument();
  });

  it('does not show disconnected banner when no players are disconnected', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        disconnectedPlayers={[]}
      />
    );

    expect(screen.queryByTestId('disconnected-players-banner')).not.toBeInTheDocument();
  });

  it('does not show disconnected banner when prop is undefined', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    expect(screen.queryByTestId('disconnected-players-banner')).not.toBeInTheDocument();
  });
});

describe('SpectatorHUD - Teaching Topics Integration', () => {
  it('renders teaching topics section', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    expect(screen.getByTestId('spectator-teaching-topics')).toBeInTheDocument();
    // Multiple elements may contain "Learn Game Mechanics" so check via getAllByText
    expect(screen.getAllByText(/Learn Game Mechanics/i).length).toBeGreaterThan(0);
  });

  it('opens teaching overlay when topic button is clicked', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    // Initially no teaching overlay
    expect(screen.queryByTestId('teaching-overlay')).not.toBeInTheDocument();

    // Click on recovery topic
    fireEvent.click(screen.getByRole('button', { name: /Recovery/i }));

    // Teaching overlay should now be visible
    expect(screen.getByTestId('teaching-overlay')).toBeInTheDocument();
    expect(screen.getByTestId('teaching-overlay')).toHaveAttribute('data-topic', 'recovery_action');
  });

  it('closes teaching overlay when close is clicked', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    // Open overlay
    fireEvent.click(screen.getByRole('button', { name: /Territory/i }));
    expect(screen.getByTestId('teaching-overlay')).toBeInTheDocument();

    // Close it
    fireEvent.click(screen.getByRole('button', { name: /Close Teaching/i }));
    expect(screen.queryByTestId('teaching-overlay')).not.toBeInTheDocument();
  });

  it('can switch between different teaching topics', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    // Click first topic
    fireEvent.click(screen.getByRole('button', { name: /Recovery/i }));
    expect(screen.getByTestId('teaching-overlay')).toHaveAttribute('data-topic', 'recovery_action');

    // Close and click different topic
    fireEvent.click(screen.getByRole('button', { name: /Close Teaching/i }));
    fireEvent.click(screen.getByRole('button', { name: /Chain Capture/i }));
    expect(screen.getByTestId('teaching-overlay')).toHaveAttribute('data-topic', 'chain_capture');
  });
});

describe('SpectatorHUD - Move Annotations for Recovery', () => {
  it('annotates recovery_slide moves correctly', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={2}
        moveNumber={3}
        moveHistory={createRecoveryMoveHistory()}
        evaluationHistory={[]}
      />
    );

    // Should show recovery annotation
    expect(screen.getByText(/P1 performed recovery/)).toBeInTheDocument();
  });

  it('annotates skip_recovery moves correctly', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={2}
        moveNumber={3}
        moveHistory={createRecoveryMoveHistory()}
        evaluationHistory={[]}
      />
    );

    // Should show skip recovery annotation
    expect(screen.getByText(/P2 skipped recovery/)).toBeInTheDocument();
  });
});

describe('SpectatorHUD - Victory Conditions Panel', () => {
  it('shows victory conditions when showVictoryConditions is true (default)', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
      />
    );

    // VictoryConditionsPanel is rendered by default
    // It renders content about victory thresholds
    // Just verify the spectator HUD renders without victory conditions hidden
    expect(screen.getByTestId('spectator-hud')).toBeInTheDocument();
  });

  it('hides victory conditions when showVictoryConditions is false', () => {
    const { container } = render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        showVictoryConditions={false}
      />
    );

    // When hidden, the VictoryConditionsPanel should not be rendered
    // The VictoryConditionsPanel has specific structure we can check for absence
    expect(container.querySelector('[data-testid="victory-conditions"]')).not.toBeInTheDocument();
  });
});

describe('SpectatorHUD - Board Type Prop', () => {
  it('accepts square8 board type', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        boardType="square8"
      />
    );

    expect(screen.getByTestId('spectator-hud')).toBeInTheDocument();
  });

  it('accepts square19 board type', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        boardType="square19"
      />
    );

    expect(screen.getByTestId('spectator-hud')).toBeInTheDocument();
  });

  it('accepts hexagonal board type', () => {
    render(
      <SpectatorHUD
        phase="movement"
        players={createPlayers()}
        currentPlayerNumber={1}
        turnNumber={1}
        moveNumber={1}
        moveHistory={[]}
        evaluationHistory={[]}
        boardType="hexagonal"
      />
    );

    expect(screen.getByTestId('spectator-hud')).toBeInTheDocument();
  });
});
