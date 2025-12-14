import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ResignButton } from '../../../src/client/components/ResignButton';

describe('ResignButton', () => {
  it('renders resign button', () => {
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} />);

    expect(screen.getByTestId('resign-button')).toBeInTheDocument();
    expect(screen.getByText('Resign')).toBeInTheDocument();
  });

  it('shows confirmation dialog when clicked', async () => {
    const user = userEvent.setup();
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} />);

    await user.click(screen.getByTestId('resign-button'));

    expect(screen.getByRole('alertdialog')).toBeInTheDocument();
    expect(screen.getByText('Resign Game?')).toBeInTheDocument();
    expect(screen.getByText(/Are you sure you want to resign/)).toBeInTheDocument();
    expect(screen.getByTestId('resign-cancel-button')).toBeInTheDocument();
    expect(screen.getByTestId('resign-confirm-button')).toBeInTheDocument();
  });

  it('calls onResign when confirmed', async () => {
    const user = userEvent.setup();
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} />);

    await user.click(screen.getByTestId('resign-button'));
    await user.click(screen.getByTestId('resign-confirm-button'));

    expect(onResign).toHaveBeenCalledTimes(1);
    // Dialog should close after confirmation
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
  });

  it('does not call onResign when cancelled', async () => {
    const user = userEvent.setup();
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} />);

    await user.click(screen.getByTestId('resign-button'));
    await user.click(screen.getByTestId('resign-cancel-button'));

    expect(onResign).not.toHaveBeenCalled();
    // Dialog should close after cancellation
    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
  });

  it('closes dialog on escape key', async () => {
    const user = userEvent.setup();
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} />);

    await user.click(screen.getByTestId('resign-button'));
    expect(screen.getByRole('alertdialog')).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole('alertdialog'), { key: 'Escape' });

    await waitFor(() => {
      expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
    });
    expect(onResign).not.toHaveBeenCalled();
  });

  it('closes dialog on backdrop click', async () => {
    const user = userEvent.setup();
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} />);

    await user.click(screen.getByTestId('resign-button'));
    expect(screen.getByRole('alertdialog')).toBeInTheDocument();

    // Click the fullscreen overlay/backdrop, not the dialog content.
    fireEvent.click(screen.getByTestId('resign-dialog-overlay'));

    await waitFor(() => {
      expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
    });
    expect(onResign).not.toHaveBeenCalled();
  });

  it('disables button when disabled prop is true', () => {
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} disabled />);

    const button = screen.getByTestId('resign-button');
    expect(button).toBeDisabled();
  });

  it('shows loading state when isResigning is true', () => {
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} isResigning />);

    expect(screen.getByText('Resigning...')).toBeInTheDocument();
    expect(screen.getByTestId('resign-button')).toBeDisabled();
  });

  it('has proper accessibility attributes', async () => {
    const user = userEvent.setup();
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} />);

    // Button should have aria-haspopup
    const button = screen.getByTestId('resign-button');
    expect(button).toHaveAttribute('aria-haspopup', 'dialog');

    // Open dialog
    await user.click(button);

    const dialog = screen.getByRole('alertdialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-labelledby', 'resign-dialog-title');
    expect(dialog).toHaveAttribute('aria-describedby', 'resign-dialog-description');
  });

  it('opens confirmation dialog when activated via keyboard', async () => {
    const user = userEvent.setup();
    const onResign = jest.fn();
    render(<ResignButton onResign={onResign} />);

    const button = screen.getByTestId('resign-button');
    button.focus();
    expect(button).toHaveFocus();

    await user.keyboard('{Enter}');
    expect(screen.getByRole('alertdialog')).toBeInTheDocument();
  });
});
