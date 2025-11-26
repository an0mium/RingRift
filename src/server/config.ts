/**
 * Configuration Re-export for Backward Compatibility
 *
 * This file maintains backward compatibility for imports like:
 *   import { config } from './config';
 *   import { config } from '../config';
 *
 * The canonical configuration module is now at `./config/index.ts`.
 * New code should import from `./config` (the directory) instead.
 *
 * @deprecated Use `import { config } from './config'` (directory) instead.
 */

// Re-export everything from the canonical config module
export * from './config/index';
