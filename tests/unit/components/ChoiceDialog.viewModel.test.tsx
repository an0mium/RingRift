import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ChoiceDialog } from '../../../src/client/components/ChoiceDialog';
import type { ChoiceViewModel } from '../../../src/client/adapters/choiceViewModels';
import type { LineOrderChoice } from '../../../src/shared/types/game';

const choice: LineOrderChoice = {
  id: 'line-order-1',
  gameId: 'g1',
  playerNumber: 1,
  type: 'line_order',
  prompt: 'Pick a line',
  timeoutMs: 10_000,
  options: [
    {
      lineId: 'line-a',
      markerPositions: [
        { x: 0, y: 0 },
        { x: 1, y: 1 },
        { x: 2, y: 2 },
      ],
      moveId: 'line-a-move',
    },
    {
      lineId: 'line-b',
      markerPositions: [
        { x: 0, y: 1 },
        { x: 1, y: 2 },
        { x: 2, y: 3 },
      ],
      moveId: 'line-b-move',
    },
  ],
};

const customViewModel: ChoiceViewModel = {
  type: 'line_order',
  kind: 'line_order',
  copy: {
    title: 'Custom Line Order Title',
    description: 'Custom line ordering description.',
    shortLabel: 'Custom short label',
    spectatorLabel: () => 'Waiting on player',
  },
  timeout: {
    showCountdown: true,
    warningThresholdMs: 5_000,
  },
};

describe('ChoiceDialog with explicit view model', () => {
  it('renders provided view model copy and disables options after submission', async () => {
    const onSelect = jest.fn();

    render(
      <ChoiceDialog
        choice={choice}
        choiceViewModel={customViewModel}
        deadline={Date.now() + 10_000}
        timeRemainingMs={10_000}
        isServerCapped={false}
        onSelectOption={onSelect}
      />
    );

    // Uses provided view model copy instead of defaults.
    expect(screen.getByText('Custom Line Order Title')).toBeInTheDocument();
    expect(screen.getByText(/Custom line ordering description/)).toBeInTheDocument();
    expect(screen.getByText(/Custom short label/i)).toBeInTheDocument();

    const options = screen.getAllByRole('option');
    expect(options).toHaveLength(2);

    fireEvent.click(options[0]);
    expect(onSelect).toHaveBeenCalledTimes(1);
    const [choiceArg, optionArg] = onSelect.mock.calls[0];
    expect(choiceArg).toEqual(choice);
    expect(optionArg).toEqual(choice.options[0]);

    // After submission, buttons should be disabled.
    await waitFor(() => {
      expect(options[0]).toBeDisabled();
      expect(options[1]).toBeDisabled();
    });

    // Further clicks should not trigger additional calls.
    fireEvent.click(options[1]);
    expect(onSelect).toHaveBeenCalledTimes(1);
  });
});
