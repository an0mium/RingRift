import React from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import GamePage from '../../src/client/pages/GamePage';

// Mock the host components at module scope
const mockBackendGameHost = jest.fn(({ gameId }: { gameId: string }) => (
  <div data-testid="backend-host">backend-{gameId}</div>
));

const mockSandboxGameHost = jest.fn(() => <div data-testid="sandbox-host">sandbox</div>);

jest.mock('../../src/client/pages/BackendGameHost', () => ({
  __esModule: true,
  BackendGameHost: (props: { gameId: string }) => mockBackendGameHost(props),
}));

jest.mock('../../src/client/pages/SandboxGameHost', () => ({
  __esModule: true,
  SandboxGameHost: () => mockSandboxGameHost(),
}));

describe('GamePage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders BackendGameHost when gameId param is present', () => {
    render(
      <MemoryRouter initialEntries={['/game/test-game-123']}>
        <Routes>
          <Route path="/game/:gameId" element={<GamePage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByTestId('backend-host')).toHaveTextContent('backend-test-game-123');
    expect(screen.queryByTestId('sandbox-host')).not.toBeInTheDocument();
    expect(mockBackendGameHost).toHaveBeenCalledWith({ gameId: 'test-game-123' });
  });

  it('renders SandboxGameHost when no gameId param is present', () => {
    render(
      <MemoryRouter initialEntries={['/sandbox']}>
        <Routes>
          <Route path="/sandbox" element={<GamePage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByTestId('sandbox-host')).toBeInTheDocument();
    expect(screen.queryByTestId('backend-host')).not.toBeInTheDocument();
    expect(mockSandboxGameHost).toHaveBeenCalled();
  });

  it('renders BackendGameHost for spectate routes with gameId', () => {
    render(
      <MemoryRouter initialEntries={['/spectate/spectate-game-456']}>
        <Routes>
          <Route path="/spectate/:gameId" element={<GamePage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(screen.getByTestId('backend-host')).toHaveTextContent('backend-spectate-game-456');
    expect(mockBackendGameHost).toHaveBeenCalledWith({ gameId: 'spectate-game-456' });
  });

  it('passes correct gameId to BackendGameHost for UUID-style game IDs', () => {
    const uuid = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';
    render(
      <MemoryRouter initialEntries={[`/game/${uuid}`]}>
        <Routes>
          <Route path="/game/:gameId" element={<GamePage />} />
        </Routes>
      </MemoryRouter>
    );

    expect(mockBackendGameHost).toHaveBeenCalledWith({ gameId: uuid });
  });
});
