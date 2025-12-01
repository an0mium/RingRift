/**
 * Application Topology Enforcement
 *
 * This module enforces deployment topology constraints and logs
 * appropriate warnings/errors based on the configured topology mode.
 *
 * Topology modes:
 * - `single`: Single instance, all state in memory (default)
 * - `multi-unsafe`: Multiple instances without sticky sessions (NOT RECOMMENDED)
 * - `multi-sticky`: Multiple instances with infrastructure-enforced sticky sessions
 *
 * Note: This module uses console logging to avoid circular dependencies with
 * the logger module which imports config.
 */

import type { AppConfig } from './unified';

/**
 * Enforce application topology constraints at startup.
 *
 * This function validates that the configured topology is appropriate
 * for the current environment and logs warnings/errors accordingly.
 *
 * @param config - Application configuration
 * @throws Error if topology is unsupported in production
 */
export function enforceAppTopology(config: AppConfig): void {
  const topology = config.app.topology;
  const nodeEnv = config.nodeEnv;

  // Use console logging to avoid circular dependency with logger module
  // This function is only called at startup, so console is acceptable here

  if (topology === 'single') {
    // eslint-disable-next-line no-console
    console.log(
      '[config] App topology: single-instance mode (RINGRIFT_APP_TOPOLOGY=single). ' +
        'The server assumes it is the only app instance talking to this database and Redis ' +
        'for authoritative game sessions.'
    );
    return;
  }

  const logContext = JSON.stringify({ topology, nodeEnv });

  if (topology === 'multi-unsafe') {
    const message =
      'RINGRIFT_APP_TOPOLOGY=multi-unsafe: multi-instance deployment without sticky sessions ' +
      'or shared state is unsupported. Each instance will maintain its own in-process game sessions.';

    if (config.isProduction) {
      console.error(
        `[config] ${message} Refusing to start in NODE_ENV=production. ` +
          `Configure infrastructure-enforced sticky sessions and/or shared game state before using multiple app instances. ${logContext}`
      );
      throw new Error(
        'Unsupported app topology "multi-unsafe" in production. Refusing to start with multiple app instances.'
      );
    }

    console.warn(
      `[config] ${message} Continuing because NODE_ENV=${nodeEnv}. ` +
        `This mode is intended only for development/experimentation. ${logContext}`
    );
    return;
  }

  if (topology === 'multi-sticky') {
    console.warn(
      '[config] RINGRIFT_APP_TOPOLOGY=multi-sticky: backend assumes infrastructure-enforced sticky sessions ' +
        '(HTTP + WebSocket) for all game-affecting traffic to a given game. ' +
        `Correctness is not guaranteed if sticky sessions are misconfigured or absent. ${logContext}`
    );
  }
}

/**
 * Topology mode type
 */
export type { AppTopology } from './env';
