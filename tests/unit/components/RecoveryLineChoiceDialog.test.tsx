import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { RecoveryLineChoiceDialog } from '../../../src/client/components/RecoveryLineChoiceDialog';

describe('RecoveryLineChoiceDialog', () => {
  it('focuses Option 2 by default and calls callbacks', async () => {
    const onChooseOption1 = jest.fn();
    const onChooseOption2 = jest.fn();
    const onClose = jest.fn();

    render(
      <RecoveryLineChoiceDialog
        isOpen={true}
        onChooseOption1={onChooseOption1}
        onChooseOption2={onChooseOption2}
        onClose={onClose}
      />
    );

    const option2 = screen.getByText('Option 2 (Free)').closest('button')!;

    await waitFor(() => {
      expect(option2).toHaveFocus();
    });

    fireEvent.click(option2);
    expect(onChooseOption2).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByText('Option 1 (Cost)').closest('button')!);
    expect(onChooseOption1).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByText('Cancel'));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
