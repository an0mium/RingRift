import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { EmptyState } from '../../../src/client/components/EmptyState';

describe('EmptyState', () => {
  it('renders title text', () => {
    render(<EmptyState title="No items found" />);

    expect(screen.getByText('No items found')).toBeInTheDocument();
  });

  it('renders description when provided', () => {
    render(<EmptyState title="No results" description="Try adjusting your search criteria" />);

    expect(screen.getByText('No results')).toBeInTheDocument();
    expect(screen.getByText('Try adjusting your search criteria')).toBeInTheDocument();
  });

  it('does not render description when not provided', () => {
    render(<EmptyState title="Empty" />);

    expect(screen.getByText('Empty')).toBeInTheDocument();
    expect(screen.queryByRole('paragraph')).not.toBeInTheDocument();
  });

  it('renders icon when provided', () => {
    const TestIcon = () => <svg data-testid="test-icon" />;

    render(<EmptyState title="No data" icon={<TestIcon />} />);

    expect(screen.getByTestId('test-icon')).toBeInTheDocument();
  });

  it('does not render icon container when icon not provided', () => {
    const { container } = render(<EmptyState title="Empty" />);

    // Icon container has mb-3 class
    const iconContainer = container.querySelector('.mb-3');
    expect(iconContainer).not.toBeInTheDocument();
  });

  it('renders action button when provided', () => {
    const mockOnClick = jest.fn();

    render(<EmptyState title="No items" action={{ label: 'Add Item', onClick: mockOnClick }} />);

    const button = screen.getByRole('button', { name: 'Add Item' });
    expect(button).toBeInTheDocument();
  });

  it('calls action onClick when button is clicked', () => {
    const mockOnClick = jest.fn();

    render(<EmptyState title="No items" action={{ label: 'Create New', onClick: mockOnClick }} />);

    fireEvent.click(screen.getByRole('button', { name: 'Create New' }));

    expect(mockOnClick).toHaveBeenCalledTimes(1);
  });

  it('does not render button when action not provided', () => {
    render(<EmptyState title="Empty state" />);

    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(<EmptyState title="Test" className="custom-class" />);

    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toHaveClass('custom-class');
  });

  it('has proper ARIA attributes for accessibility', () => {
    render(<EmptyState title="No data available" />);

    const status = screen.getByRole('status');
    expect(status).toHaveAttribute('aria-live', 'polite');
  });

  it('renders complete empty state with all props', () => {
    const mockOnClick = jest.fn();
    const CustomIcon = () => <span data-testid="custom-icon">📭</span>;

    render(
      <EmptyState
        title="Your inbox is empty"
        description="No messages to display"
        icon={<CustomIcon />}
        action={{ label: 'Compose', onClick: mockOnClick }}
        className="test-wrapper"
      />
    );

    expect(screen.getByText('Your inbox is empty')).toBeInTheDocument();
    expect(screen.getByText('No messages to display')).toBeInTheDocument();
    expect(screen.getByTestId('custom-icon')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Compose' })).toBeInTheDocument();
  });

  it('button has correct styling classes', () => {
    render(<EmptyState title="Empty" action={{ label: 'Click me', onClick: () => {} }} />);

    const button = screen.getByRole('button');
    expect(button).toHaveClass('bg-emerald-600');
    expect(button).toHaveClass('hover:bg-emerald-500');
    expect(button).toHaveClass('text-white');
  });
});
