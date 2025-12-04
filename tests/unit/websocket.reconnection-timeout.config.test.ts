import { Server as HTTPServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import { config as appConfig } from '../../src/server/config';
import { WebSocketServer } from '../../src/server/websocket/server';

jest.mock('socket.io', () => {
  const actual = jest.requireActual('socket.io');
  const MockServer = jest.fn().mockImplementation((_httpServer: HTTPServer, options: any) => {
    // Return a lightweight stub; tests will inspect the constructor
    // options via the mocked Server function.
    return {
      __options: options,
      use: jest.fn(),
      on: jest.fn(),
      emit: jest.fn(),
    };
  });

  return {
    ...actual,
    Server: MockServer,
  };
});

describe('WebSocket reconnection timeout configuration', () => {
  it('uses the configured WS reconnection timeout value', () => {
    const server = new HTTPServer();

    // Sanity check that the config exposes a positive timeout value.
    expect(appConfig.server.wsReconnectionTimeoutMs).toBeGreaterThan(0);

    // Constructing the WebSocketServer should pick up the timeout from config.
    const wsServer = new WebSocketServer(server as any);
    expect(wsServer).toBeDefined();

    const MockedSocketServer = SocketIOServer as unknown as jest.Mock;
    expect(MockedSocketServer).toHaveBeenCalled();

    const options = MockedSocketServer.mock.calls[0][1] as any;
    expect(options.pingTimeout).toBe(appConfig.server.wsReconnectionTimeoutMs);
  });
});
