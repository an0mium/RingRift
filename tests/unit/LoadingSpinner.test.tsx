import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import LoadingSpinner from '../../src/client/components/LoadingSpinner';

describe('LoadingSpinner', () => {
  it('renders with default medium size and accessibility attributes', () => {
    render(<LoadingSpinner />);
    const spinner = screen.getByRole('status', { name: /loading/i });

    expect(spinner).toBeInTheDocument();
    expect(spinner).toHaveClass('spinner');
    expect(spinner).toHaveClass('w-8', 'h-8');
  });

  it('supports small and large size variants', () => {
    const { rerender } = render(<LoadingSpinner size="sm" />);
    let spinner = screen.getByRole('status', { name: /loading/i });
    expect(spinner).toHaveClass('w-4', 'h-4');

    rerender(<LoadingSpinner size="lg" />);
    spinner = screen.getByRole('status', { name: /loading/i });
    expect(spinner).toHaveClass('w-12', 'h-12');
  });

  it('merges custom className with base classes', () => {
    render(<LoadingSpinner className="text-emerald-500" />);
    const spinner = screen.getByRole('status', { name: /loading/i });

    expect(spinner).toHaveClass('spinner');
    expect(spinner).toHaveClass('text-emerald-500');
  });
});
