import React from 'react';
import { MemoryRouter } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from '../../src/client/App';

// Mock backend and sandbox hosts to observe which one is mounted.
// Define mock components at module scope so Jest's mock factory does not
// close over JSX helpers (jsx_runtime_1). Names are prefixed with "mock"
// to satisfy Jest's out-of-scope variable rules.
const mockBackendGameHost = (props: { gameId: string }) => (
  <div data-testid="backend-host">backend-{props.gameId}</div>
);

const mockSandboxGameHost = () => <div data-testid="sandbox-host">sandbox</div>;

jest.mock('../../src/client/pages/BackendGameHost', () => ({
  __esModule: true,
  BackendGameHost: mockBackendGameHost,
}));

jest.mock('../../src/client/pages/SandboxGameHost', () => ({
  __esModule: true,
  SandboxGameHost: mockSandboxGameHost,
}));

// Simplify auth for routing tests: treat user as authenticated (so protected
// /game/:gameId routes resolve) and not loading.
jest.mock('../../src/client/contexts/AuthContext', () => ({
  __esModule: true,
  useAuth: () => ({ user: { id: 'test-user' }, isLoading: false }),
}));

describe('App host selection (backend vs sandbox)', () => {
  it('uses BackendGameHost for /game/:gameId', async () => {
    render(
      <MemoryRouter initialEntries={['/game/abc123']}>
        <App />
      </MemoryRouter>
    );

    expect(await screen.findByTestId('backend-host')).toHaveTextContent('backend-abc123');
    expect(screen.queryByTestId('sandbox-host')).not.toBeInTheDocument();
  });

  it('uses BackendGameHost for /spectate/:gameId', async () => {
    render(
      <MemoryRouter initialEntries={['/spectate/xyz789']}>
        <App />
      </MemoryRouter>
    );

    expect(await screen.findByTestId('backend-host')).toHaveTextContent('backend-xyz789');
    expect(screen.queryByTestId('sandbox-host')).not.toBeInTheDocument();
  });

  it('uses SandboxGameHost for /sandbox', async () => {
    render(
      <MemoryRouter initialEntries={['/sandbox']}>
        <App />
      </MemoryRouter>
    );

    expect(await screen.findByTestId('sandbox-host')).toBeInTheDocument();
    expect(screen.queryByTestId('backend-host')).not.toBeInTheDocument();
  });
});
