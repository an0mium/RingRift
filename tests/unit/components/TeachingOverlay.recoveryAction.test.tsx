import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import {
  TeachingOverlay,
  TeachingTopicButtons,
  getTeachingTopicForMove,
  useTeachingOverlay,
  TEACHING_TOPICS,
  parseTeachingTopic,
  type TeachingTopic,
} from '../../../src/client/components/TeachingOverlay';
import type { Move } from '../../../src/shared/types/game';

// Mock telemetry to avoid side effects
jest.mock('../../../src/client/utils/rulesUxTelemetry', () => {
  const actual = jest.requireActual('../../../src/client/utils/rulesUxTelemetry');
  return {
    ...actual,
    logRulesUxEvent: jest.fn().mockResolvedValue(undefined),
    newTeachingFlowId: jest.fn(() => 'test-flow-id'),
  };
});

describe('TeachingOverlay - Recovery Action Topic', () => {
  const onClose = jest.fn();

  beforeEach(() => {
    onClose.mockClear();
  });

  it('renders recovery action topic with correct title and icon', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /recovery action/i })).toBeInTheDocument();
    // Recovery icon is 🔄
    expect(screen.getByText('🔄')).toBeInTheDocument();
  });

  it('displays recovery eligibility tips', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    // Check for key recovery eligibility tips
    expect(screen.getByText(/WHEN CAN YOU RECOVER\?/)).toBeInTheDocument();
    expect(screen.getByText(/you control NO stacks/i)).toBeInTheDocument();
    // Multiple elements contain "markers" so use getAllByText
    expect(screen.getAllByText(/markers/i).length).toBeGreaterThan(0);
    // Multiple elements contain "buried ring" so use getAllByText
    expect(screen.getAllByText(/buried ring/i).length).toBeGreaterThan(0);
  });

  it('displays rings in hand clarification (GAP-RECOV-01)', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    // GAP-RECOV-01: Clarify that rings in hand don't prevent recovery
    expect(screen.getByText(/RINGS IN HAND/i)).toBeInTheDocument();
    expect(screen.getByText(/does NOT prevent recovery eligibility/i)).toBeInTheDocument();
  });

  it('displays recovery action mechanics (GAP-RECOV-02)', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    // GAP-RECOV-02: Recovery action mechanics
    expect(screen.getByText(/HOW TO RECOVER/i)).toBeInTheDocument();
    expect(screen.getByText(/Slide one of your markers/i)).toBeInTheDocument();
  });

  it('displays fallback recovery explanation (GAP-RECOV-02)', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    // GAP-RECOV-02: Fallback recovery when no line can form
    // Multiple elements may contain these patterns so use getAllByText
    expect(screen.getAllByText(/FALLBACK RECOVERY/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/stack-strike/i).length).toBeGreaterThan(0);
  });

  it('displays recovery cost information (GAP-RECOV-03)', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    // GAP-RECOV-03: Recovery cost explanation
    expect(screen.getByText(/RECOVERY COST/i)).toBeInTheDocument();
    // Multiple elements contain "buried ring" so use getAllByText
    expect(screen.getAllByText(/buried ring/i).length).toBeGreaterThan(0);
  });

  it('displays temporary vs permanent elimination distinction (GAP-RECOV-03)', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    // GAP-RECOV-03: Temporary vs permanent elimination
    // Multiple elements may contain these patterns so use getAllByText
    expect(screen.getAllByText(/TEMPORARILY ELIMINATED/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/PERMANENTLY ELIMINATED/i).length).toBeGreaterThan(0);
  });

  it('displays recovery vs LPS guidance (GAP-RECOV-04)', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    // GAP-RECOV-04: Recovery actions don't count for LPS
    expect(screen.getByText(/RECOVERY AND LPS/i)).toBeInTheDocument();
    expect(screen.getByText(/do NOT count as "real actions"/i)).toBeInTheDocument();
  });

  it('displays skip recovery option', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    expect(screen.getByText(/SKIP RECOVERY/i)).toBeInTheDocument();
    expect(screen.getByText(/preserving your buried rings/i)).toBeInTheDocument();
  });

  it('displays recovery action description and tips', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    // Verify main description and key tips are rendered (content based on RR-CANON-R110–R115)
    expect(screen.getByText(/sliding markers to form lines/i)).toBeInTheDocument();
    expect(screen.getByText(/Recovery gives temporarily eliminated/i)).toBeInTheDocument();
  });

  it('closes on close button click', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    fireEvent.click(screen.getByRole('button', { name: /close/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('shows movement as related phase for recovery action', () => {
    render(
      <TeachingOverlay topic="recovery_action" isOpen={true} onClose={onClose} position="center" />
    );

    // Recovery is related to movement phase
    expect(screen.getByText('movement')).toBeInTheDocument();
    expect(screen.getByText(/Applies to:/i)).toBeInTheDocument();
  });
});

describe('getTeachingTopicForMove - Recovery actions', () => {
  it('returns recovery_action for recovery_slide move type', () => {
    const move = {
      type: 'recovery_slide',
    } as Move;

    expect(getTeachingTopicForMove(move)).toBe('recovery_action');
  });

  it('returns recovery_action for skip_recovery move type', () => {
    const move = {
      type: 'skip_recovery',
    } as Move;

    expect(getTeachingTopicForMove(move)).toBe('recovery_action');
  });
});

describe('TeachingTopicButtons - Recovery button', () => {
  it('includes recovery action button in game mechanics section', () => {
    const onSelect = jest.fn();
    render(<TeachingTopicButtons onSelectTopic={onSelect} />);

    const recoveryButton = screen.getByRole('button', { name: /🔄 Recovery/i });
    expect(recoveryButton).toBeInTheDocument();
  });

  it('calls onSelectTopic with recovery_action when clicked', () => {
    const onSelect = jest.fn();
    render(<TeachingTopicButtons onSelectTopic={onSelect} />);

    const recoveryButton = screen.getByRole('button', { name: /🔄 Recovery/i });
    fireEvent.click(recoveryButton);

    expect(onSelect).toHaveBeenCalledWith('recovery_action');
  });
});

describe('TEACHING_TOPICS constant', () => {
  it('includes recovery_action in the list of teaching topics', () => {
    expect(TEACHING_TOPICS).toContain('recovery_action');
  });

  it('has 13 total teaching topics', () => {
    // ring_placement, stack_movement, capturing, chain_capture, line_bonus,
    // line_territory_order, territory, active_no_moves, forced_elimination,
    // recovery_action, victory_elimination, victory_territory, victory_stalemate
    expect(TEACHING_TOPICS.length).toBe(13);
  });
});

describe('parseTeachingTopic', () => {
  it('parses recovery_action as valid topic', () => {
    expect(parseTeachingTopic('recovery_action')).toBe('recovery_action');
  });

  it('returns null for invalid topic', () => {
    expect(parseTeachingTopic('invalid_topic')).toBeNull();
  });

  it('returns null for null input', () => {
    expect(parseTeachingTopic(null)).toBeNull();
  });

  it('returns null for undefined input', () => {
    expect(parseTeachingTopic(undefined)).toBeNull();
  });
});

describe('useTeachingOverlay hook', () => {
  function TestComponent() {
    const teaching = useTeachingOverlay();

    return (
      <div>
        <span data-testid="is-open">{String(teaching.isOpen)}</span>
        <span data-testid="current-topic">{teaching.currentTopic ?? 'none'}</span>
        <button type="button" onClick={() => teaching.showTopic('recovery_action')}>
          Show Recovery
        </button>
        <button type="button" onClick={teaching.hideTopic}>
          Hide
        </button>
      </div>
    );
  }

  it('starts with no topic selected', () => {
    render(<TestComponent />);

    expect(screen.getByTestId('is-open')).toHaveTextContent('false');
    expect(screen.getByTestId('current-topic')).toHaveTextContent('none');
  });

  it('shows recovery action topic when showTopic is called', () => {
    render(<TestComponent />);

    fireEvent.click(screen.getByRole('button', { name: /show recovery/i }));

    expect(screen.getByTestId('is-open')).toHaveTextContent('true');
    expect(screen.getByTestId('current-topic')).toHaveTextContent('recovery_action');
  });

  it('hides topic when hideTopic is called', () => {
    render(<TestComponent />);

    fireEvent.click(screen.getByRole('button', { name: /show recovery/i }));
    expect(screen.getByTestId('is-open')).toHaveTextContent('true');

    fireEvent.click(screen.getByRole('button', { name: /hide/i }));
    expect(screen.getByTestId('is-open')).toHaveTextContent('false');
    expect(screen.getByTestId('current-topic')).toHaveTextContent('none');
  });
});

describe('TeachingOverlay - FSM context tips', () => {
  const onClose = jest.fn();

  beforeEach(() => {
    onClose.mockClear();
  });

  it('displays FSM-aware tips when fsmContext is active', () => {
    const fsmContext = {
      isActive: true,
      phase: 'chain_capture' as const,
      decisionType: 'chain_capture' as const,
      summary: 'You must continue the capture chain.',
      actionHint: 'Select a target to continue capturing.',
      pendingLineCount: 0,
      pendingRegionCount: 0,
      chainContinuationCount: 3,
      forcedEliminationCount: 0,
    };

    render(
      <TeachingOverlay
        topic="chain_capture"
        isOpen={true}
        onClose={onClose}
        position="center"
        fsmContext={fsmContext}
      />
    );

    // Should show FSM section
    expect(screen.getByText(/Current Situation \(FSM\)/i)).toBeInTheDocument();
    expect(screen.getByText(/CURRENT STATE:/i)).toBeInTheDocument();
    expect(screen.getByText(/You must continue the capture chain/)).toBeInTheDocument();
    expect(screen.getByText(/WHAT TO DO:/i)).toBeInTheDocument();
  });

  it('shows chain continuation count tip', () => {
    const fsmContext = {
      isActive: true,
      phase: 'chain_capture' as const,
      decisionType: 'chain_capture' as const,
      summary: 'Continue chain.',
      actionHint: 'Pick a target.',
      pendingLineCount: 0,
      pendingRegionCount: 0,
      chainContinuationCount: 2,
      forcedEliminationCount: 0,
    };

    render(
      <TeachingOverlay
        topic="chain_capture"
        isOpen={true}
        onClose={onClose}
        position="center"
        fsmContext={fsmContext}
      />
    );

    expect(screen.getByText(/You have 2 possible capture targets/)).toBeInTheDocument();
  });

  it('does not show FSM section when fsmContext is inactive', () => {
    const fsmContext = {
      isActive: false,
      phase: 'movement' as const,
      decisionType: undefined,
      summary: '',
      actionHint: '',
      pendingLineCount: 0,
      pendingRegionCount: 0,
      chainContinuationCount: 0,
      forcedEliminationCount: 0,
    };

    render(
      <TeachingOverlay
        topic="territory"
        isOpen={true}
        onClose={onClose}
        position="center"
        fsmContext={fsmContext}
      />
    );

    expect(screen.queryByText(/Current Situation \(FSM\)/i)).not.toBeInTheDocument();
  });

  it('does not show FSM section when fsmContext is null', () => {
    render(
      <TeachingOverlay
        topic="territory"
        isOpen={true}
        onClose={onClose}
        position="center"
        fsmContext={null}
      />
    );

    expect(screen.queryByText(/Current Situation \(FSM\)/i)).not.toBeInTheDocument();
  });
});

describe('TeachingOverlay - bottom-right position', () => {
  it('renders in bottom-right position without Dialog wrapper', () => {
    const onClose = jest.fn();

    const { container } = render(
      <TeachingOverlay
        topic="recovery_action"
        isOpen={true}
        onClose={onClose}
        position="bottom-right"
      />
    );

    // Should have fixed positioning for bottom-right
    const fixedContainer = container.querySelector('.fixed.bottom-4.right-4');
    expect(fixedContainer).toBeInTheDocument();
  });

  it('closes on Escape key in bottom-right position', () => {
    const onClose = jest.fn();

    render(
      <TeachingOverlay
        topic="recovery_action"
        isOpen={true}
        onClose={onClose}
        position="bottom-right"
      />
    );

    fireEvent.keyDown(window, { key: 'Escape', code: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
