import React from 'react';
import { render } from '@testing-library/react';

import { Badge } from '../../src/client/components/ui/Badge';

describe('Badge', () => {
  it('renders default variant with base classes', () => {
    const { getByText } = render(<Badge>Default</Badge>);

    const badge = getByText('Default');
    expect(badge.tagName.toLowerCase()).toBe('span');
    expect(badge.className).toContain('inline-flex');
    expect(badge.className).toContain('rounded-full');
  });

  it('applies variant-specific classes for primary and success', () => {
    const { getByText } = render(
      <>
        <Badge variant="primary">Primary</Badge>
        <Badge variant="success">Success</Badge>
      </>
    );

    const primary = getByText('Primary');
    const success = getByText('Success');

    expect(primary.className).toContain('bg-blue-600');
    expect(primary.className).toContain('text-white');

    expect(success.className).toContain('bg-emerald-600');
    expect(success.className).toContain('text-white');
  });
});
