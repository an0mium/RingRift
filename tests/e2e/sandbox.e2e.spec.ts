import { test, expect } from '@playwright/test';
import { goToSandbox } from './helpers/test-utils';
import { GamePage } from './pages';

/**
 * E2E Test Suite: Sandbox Host Flow
 * ============================================================================
 *
 * Verifies that the `/sandbox` route can launch a playable game by:
 * - Rendering the pre-game sandbox setup view.
 * - Navigating to a backend game when "Launch Game" is clicked.
 * - Showing a fully ready game view (board + HUD connection + turn indicator).
 *
 * RUN COMMAND: npm run test:e2e -- sandbox.e2e.spec.ts
 */

test.describe('Sandbox host E2E', () => {
  test.setTimeout(120_000);

  test('Launch Game from sandbox navigates to backend game and renders board', async ({ page }) => {
    // Navigate to the sandbox pre-game setup.
    await goToSandbox(page);

    // Click the Launch Game button in the sandbox host.
    await page.getByRole('button', { name: /Launch Game/i }).click();

    // On a healthy backend, the sandbox host first attempts to create a real
    // backend game and navigates to /game/:gameId on success.
    await page.waitForURL('**/game/**', { timeout: 30_000 });

    const gamePage = new GamePage(page);
    await gamePage.waitForReady();

    // Sanity-check core game UI elements.
    await expect(gamePage.boardView).toBeVisible();
    await gamePage.assertConnected();
    await expect(gamePage.turnIndicator).toBeVisible();
  });
});
