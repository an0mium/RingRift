/**
 * Topology enforcement tests
 *
 * Covers the branches in src/server/config/topology.ts to ensure that:
 * - Single-instance topology logs an informational message and does not throw.
 * - multi-unsafe topology throws in production but only warns in non-production.
 * - multi-sticky topology logs a warning and does not throw.
 */

import { enforceAppTopology } from '../../src/server/config/topology';

interface MinimalAppConfig {
  nodeEnv: string;
  isProduction: boolean;
  app: {
    topology: 'single' | 'multi-unsafe' | 'multi-sticky';
  };
}

function makeConfig(overrides: Partial<MinimalAppConfig> = {}): MinimalAppConfig {
  return {
    nodeEnv: overrides.nodeEnv ?? 'development',
    isProduction: overrides.isProduction ?? false,
    app: {
      topology: overrides.app?.topology ?? 'single',
    },
  };
}

describe('enforceAppTopology', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('logs info and does not throw for single topology', () => {
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    const config = makeConfig({ app: { topology: 'single' } });

    expect(() => enforceAppTopology(config as any)).not.toThrow();
    expect(logSpy).toHaveBeenCalledWith(
      expect.stringContaining('App topology: single-instance mode')
    );
  });

  it('throws and logs error for multi-unsafe topology in production', () => {
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    const config = makeConfig({
      nodeEnv: 'production',
      isProduction: true,
      app: { topology: 'multi-unsafe' },
    });

    expect(() => enforceAppTopology(config as any)).toThrow(
      /Unsupported app topology "multi-unsafe" in production/
    );
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining('Refusing to start in NODE_ENV=production')
    );
  });

  it('logs warning but does not throw for multi-unsafe topology in non-production', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const config = makeConfig({
      nodeEnv: 'development',
      isProduction: false,
      app: { topology: 'multi-unsafe' },
    });

    expect(() => enforceAppTopology(config as any)).not.toThrow();
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('RINGRIFT_APP_TOPOLOGY=multi-unsafe')
    );
  });

  it('logs warning and does not throw for multi-sticky topology', () => {
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const config = makeConfig({
      nodeEnv: 'production',
      isProduction: true,
      app: { topology: 'multi-sticky' },
    });

    expect(() => enforceAppTopology(config as any)).not.toThrow();
    expect(warnSpy).toHaveBeenCalledWith(
      expect.stringContaining('RINGRIFT_APP_TOPOLOGY=multi-sticky')
    );
  });
});
