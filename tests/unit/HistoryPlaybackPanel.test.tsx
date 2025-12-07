import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { HistoryPlaybackPanel, type HistoryPlaybackPanelProps } from '../../src/client/components/HistoryPlaybackPanel';

describe('HistoryPlaybackPanel', () => {
  const mockOnMoveIndexChange = jest.fn();
  const mockOnExitHistoryView = jest.fn();
  const mockOnEnterHistoryView = jest.fn();

  const defaultProps: HistoryPlaybackPanelProps = {
    totalMoves: 10,
    currentMoveIndex: 5,
    isViewingHistory: true,
    onMoveIndexChange: mockOnMoveIndexChange,
    onExitHistoryView: mockOnExitHistoryView,
    onEnterHistoryView: mockOnEnterHistoryView,
    visible: true,
    hasSnapshots: true,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Rendering', () => {
    it('renders without crashing', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      expect(screen.getByText('History Playback')).toBeInTheDocument();
    });

    it('does not render when visible is false', () => {
      const { container } = render(
        <HistoryPlaybackPanel {...defaultProps} visible={false} />
      );

      expect(container.firstChild).toBeNull();
    });

    it('does not render when totalMoves is 0', () => {
      const { container } = render(
        <HistoryPlaybackPanel {...defaultProps} totalMoves={0} />
      );

      expect(container.firstChild).toBeNull();
    });

    it('displays current move information', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      expect(screen.getByText('Move 5 / 10')).toBeInTheDocument();
    });

    it('displays keyboard hints', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      expect(screen.getByText('Step')).toBeInTheDocument();
      expect(screen.getByText('Play/Pause')).toBeInTheDocument();
      expect(screen.getByText('Exit')).toBeInTheDocument();
    });

    it('shows "Return to Live" button when viewing history', () => {
      render(<HistoryPlaybackPanel {...defaultProps} isViewingHistory={true} />);

      expect(screen.getByText('Return to Live')).toBeInTheDocument();
    });

    it('does not show "Return to Live" button when not viewing history', () => {
      render(<HistoryPlaybackPanel {...defaultProps} isViewingHistory={false} />);

      expect(screen.queryByText('Return to Live')).not.toBeInTheDocument();
    });
  });

  describe('Playback Controls', () => {
    it('calls onMoveIndexChange when step forward is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      const stepForward = screen.getByRole('button', { name: /step forward/i });
      fireEvent.click(stepForward);

      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(6);
    });

    it('calls onMoveIndexChange when step back is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      const stepBack = screen.getByRole('button', { name: /step back/i });
      fireEvent.click(stepBack);

      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(4);
    });

    it('calls onMoveIndexChange with 0 when jump to start is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      const jumpToStart = screen.getByRole('button', { name: /jump to start/i });
      fireEvent.click(jumpToStart);

      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(0);
    });

    it('calls onMoveIndexChange with totalMoves when jump to end is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      const jumpToEnd = screen.getByRole('button', { name: /jump to end/i });
      fireEvent.click(jumpToEnd);

      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(10);
    });

    it('disables step back when at move 0 and viewing history', () => {
      render(
        <HistoryPlaybackPanel
          {...defaultProps}
          currentMoveIndex={0}
          isViewingHistory={true}
        />
      );

      const stepBack = screen.getByRole('button', { name: /step back/i });
      expect(stepBack).toBeDisabled();
    });

    it('disables step forward when at the end', () => {
      render(
        <HistoryPlaybackPanel
          {...defaultProps}
          currentMoveIndex={10}
          totalMoves={10}
        />
      );

      const stepForward = screen.getByRole('button', { name: /step forward/i });
      expect(stepForward).toBeDisabled();
    });
  });

  describe('Play/Pause Functionality', () => {
    it('shows play button when not playing', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('starts playback when play is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      const playButton = screen.getByRole('button', { name: /play/i });
      fireEvent.click(playButton);

      // After clicking, the button should change to pause
      expect(screen.getByRole('button', { name: /pause/i })).toBeInTheDocument();
    });

    it('pauses when pause is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      // Start playing
      const playButton = screen.getByRole('button', { name: /play/i });
      fireEvent.click(playButton);

      // Then pause
      const pauseButton = screen.getByRole('button', { name: /pause/i });
      fireEvent.click(pauseButton);

      // Should be back to play button
      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('auto-advances moves during playback', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      const playButton = screen.getByRole('button', { name: /play/i });
      fireEvent.click(playButton);

      // Advance timers
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      expect(mockOnMoveIndexChange).toHaveBeenCalled();
    });

    it('stops playback when reaching the end', () => {
      render(
        <HistoryPlaybackPanel
          {...defaultProps}
          currentMoveIndex={9}
          totalMoves={10}
        />
      );

      const playButton = screen.getByRole('button', { name: /play/i });
      fireEvent.click(playButton);

      // Advance timers to trigger move
      act(() => {
        jest.advanceTimersByTime(1000);
      });

      // Rerender with updated index
      render(
        <HistoryPlaybackPanel
          {...defaultProps}
          currentMoveIndex={10}
          totalMoves={10}
        />
      );

      // Should auto-stop at end
      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(10);
    });
  });

  describe('Scrubber', () => {
    it('renders a range input for scrubbing', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      const scrubber = screen.getByRole('slider', { name: /move scrubber/i });
      expect(scrubber).toBeInTheDocument();
    });

    it('scrubber value reflects current move index when viewing history', () => {
      render(
        <HistoryPlaybackPanel
          {...defaultProps}
          currentMoveIndex={3}
          isViewingHistory={true}
        />
      );

      const scrubber = screen.getByRole('slider', { name: /move scrubber/i }) as HTMLInputElement;
      expect(scrubber.value).toBe('3');
    });

    it('scrubber value reflects totalMoves when not viewing history', () => {
      render(
        <HistoryPlaybackPanel
          {...defaultProps}
          isViewingHistory={false}
          totalMoves={10}
        />
      );

      const scrubber = screen.getByRole('slider', { name: /move scrubber/i }) as HTMLInputElement;
      expect(scrubber.value).toBe('10');
    });

    it('calls onMoveIndexChange when scrubber is changed', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      const scrubber = screen.getByRole('slider', { name: /move scrubber/i });
      fireEvent.change(scrubber, { target: { value: '7' } });

      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(7);
    });

    it('enters history view when scrubber is changed while not viewing history', () => {
      render(
        <HistoryPlaybackPanel
          {...defaultProps}
          isViewingHistory={false}
        />
      );

      const scrubber = screen.getByRole('slider', { name: /move scrubber/i });
      fireEvent.change(scrubber, { target: { value: '5' } });

      expect(mockOnEnterHistoryView).toHaveBeenCalled();
    });
  });

  describe('Speed Control', () => {
    it('displays speed control buttons', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      expect(screen.getByText('0.5x')).toBeInTheDocument();
      expect(screen.getByText('1x')).toBeInTheDocument();
      expect(screen.getByText('2x')).toBeInTheDocument();
      expect(screen.getByText('5x')).toBeInTheDocument();
    });

    it('changes playback speed when speed button is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      // Click 2x speed
      const speed2x = screen.getByText('2x');
      fireEvent.click(speed2x);

      // Start playing
      const playButton = screen.getByRole('button', { name: /play/i });
      fireEvent.click(playButton);

      // At 2x speed, should trigger move faster
      act(() => {
        jest.advanceTimersByTime(500);
      });

      expect(mockOnMoveIndexChange).toHaveBeenCalled();
    });
  });

  describe('History View Navigation', () => {
    it('calls onExitHistoryView when Return to Live is clicked', () => {
      render(<HistoryPlaybackPanel {...defaultProps} isViewingHistory={true} />);

      const returnButton = screen.getByText('Return to Live');
      fireEvent.click(returnButton);

      expect(mockOnExitHistoryView).toHaveBeenCalled();
    });

    it('enters history view when stepping while not in history view', () => {
      render(
        <HistoryPlaybackPanel
          {...defaultProps}
          isViewingHistory={false}
        />
      );

      const stepBack = screen.getByRole('button', { name: /step back/i });
      fireEvent.click(stepBack);

      expect(mockOnEnterHistoryView).toHaveBeenCalled();
    });
  });

  describe('Keyboard Shortcuts', () => {
    it('responds to ArrowLeft for step back', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      fireEvent.keyDown(window, { key: 'ArrowLeft' });

      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(4);
    });

    it('responds to ArrowRight for step forward', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      fireEvent.keyDown(window, { key: 'ArrowRight' });

      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(6);
    });

    it('responds to Space for play/pause', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      fireEvent.keyDown(window, { key: ' ' });

      // Should start playing
      expect(screen.getByRole('button', { name: /pause/i })).toBeInTheDocument();
    });

    it('responds to Home for jump to start', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      fireEvent.keyDown(window, { key: 'Home' });

      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(0);
    });

    it('responds to End for jump to end', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      fireEvent.keyDown(window, { key: 'End' });

      expect(mockOnMoveIndexChange).toHaveBeenCalledWith(10);
    });

    it('responds to Escape for exiting history view', () => {
      render(<HistoryPlaybackPanel {...defaultProps} isViewingHistory={true} />);

      fireEvent.keyDown(window, { key: 'Escape' });

      expect(mockOnExitHistoryView).toHaveBeenCalled();
    });

    it('ignores keyboard events when focused on input', () => {
      render(
        <>
          <input type="text" data-testid="text-input" />
          <HistoryPlaybackPanel {...defaultProps} />
        </>
      );

      const input = screen.getByTestId('text-input');
      input.focus();

      fireEvent.keyDown(input, { key: 'ArrowLeft' });

      expect(mockOnMoveIndexChange).not.toHaveBeenCalled();
    });
  });

  describe('No Snapshots State', () => {
    it('shows disabled state message when hasSnapshots is false', () => {
      render(<HistoryPlaybackPanel {...defaultProps} hasSnapshots={false} />);

      expect(
        screen.getByText(/History scrubbing is unavailable for this scenario/)
      ).toBeInTheDocument();
    });

    it('disables controls when hasSnapshots is false', () => {
      render(<HistoryPlaybackPanel {...defaultProps} hasSnapshots={false} />);

      const stepBack = screen.getByRole('button', { name: /step back/i });
      const stepForward = screen.getByRole('button', { name: /step forward/i });
      const playButton = screen.getByRole('button', { name: /play/i });
      const scrubber = screen.getByRole('slider', { name: /move scrubber/i });

      expect(stepBack).toBeDisabled();
      expect(stepForward).toBeDisabled();
      expect(playButton).toBeDisabled();
      expect(scrubber).toBeDisabled();
    });

    it('does not call handlers when hasSnapshots is false', () => {
      render(<HistoryPlaybackPanel {...defaultProps} hasSnapshots={false} />);

      const stepForward = screen.getByRole('button', { name: /step forward/i });
      fireEvent.click(stepForward);

      expect(mockOnMoveIndexChange).not.toHaveBeenCalled();
    });
  });

  describe('Accessibility', () => {
    it('has accessible button labels', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      expect(screen.getByRole('button', { name: /jump to start/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /step back/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /step forward/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /jump to end/i })).toBeInTheDocument();
    });

    it('has accessible slider label', () => {
      render(<HistoryPlaybackPanel {...defaultProps} />);

      expect(screen.getByRole('slider', { name: /move scrubber/i })).toBeInTheDocument();
    });
  });
});