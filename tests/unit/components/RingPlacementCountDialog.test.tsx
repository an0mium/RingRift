import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { RingPlacementCountDialog } from '../../../src/client/components/RingPlacementCountDialog';

describe('RingPlacementCountDialog', () => {
  it('shows the default count and confirms valid input', async () => {
    const onClose = jest.fn();
    const onConfirm = jest.fn();

    render(
      <RingPlacementCountDialog
        isOpen={true}
        maxCount={5}
        defaultCount={2}
        isStackPlacement={false}
        onClose={onClose}
        onConfirm={onConfirm}
      />
    );

    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveTextContent('Place 1–5 rings');

    const input = screen.getByLabelText('Number of rings') as HTMLInputElement;
    expect(input.value).toBe('2');

    await waitFor(() => {
      expect(input).toHaveFocus();
    });

    fireEvent.change(input, { target: { value: '3' } });
    fireEvent.submit(input.closest('form')!);

    expect(onConfirm).toHaveBeenCalledWith(3);
    expect(onClose).not.toHaveBeenCalled();
  });

  it('rejects out-of-range input and renders an error', () => {
    const onClose = jest.fn();
    const onConfirm = jest.fn();

    render(
      <RingPlacementCountDialog
        isOpen={true}
        maxCount={2}
        defaultCount={2}
        isStackPlacement={false}
        onClose={onClose}
        onConfirm={onConfirm}
      />
    );

    const input = screen.getByLabelText('Number of rings') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '3' } });
    fireEvent.submit(input.closest('form')!);

    expect(screen.getByRole('alert')).toHaveTextContent('Enter a number from 1 to 2.');
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it('calls onClose when Cancel is clicked', () => {
    const onClose = jest.fn();
    const onConfirm = jest.fn();

    render(
      <RingPlacementCountDialog
        isOpen={true}
        maxCount={4}
        defaultCount={1}
        isStackPlacement={true}
        onClose={onClose}
        onConfirm={onConfirm}
      />
    );

    fireEvent.click(screen.getByText('Cancel'));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
  });
});
