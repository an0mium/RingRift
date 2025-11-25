import { test, expect, Page } from '@playwright/test';

/**
 * E2E Test Suite: Game Flow
 * ============================================================================
 *
 * This suite tests the core game functionality happy path:
 * - Creating a game from the lobby
 * - Game board rendering with correct data-testid selectors
 * - Making moves and seeing them logged
 * - State persistence after page reload
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for game persistence)
 * - Redis running (for WebSocket sessions and game state)
 * - Dev server running on http://localhost:5173
 * - Backend WebSocket server listening
 *
 * SELECTORS:
 * - data-testid="board-view" on the BoardView component
 * - Valid move targets have class "outline-emerald-300/90" (outline highlight)
 * - "Game log" heading in GameEventLog component
 * - "Recent moves" subheading when moves exist
 *
 * RUN COMMAND: npm run test:e2e -- --timeout 60000
 */

/**
 * Generates unique user credentials for test isolation.
 */
function generateUserCredentials() {
  const timestamp = Date.now();
  const random = Math.floor(Math.random() * 1_000_000);
  const slug = `${timestamp}-${random}`;
  const email = `e2e+${slug}@example.com`;
  const username = `e2e-game-${slug}`;
  const password = 'E2E_test_password_123!';
  return { email, username, password };
}

/**
 * Registers a new user and ensures they're logged in.
 * Returns the user credentials for reference.
 */
async function registerAndLogin(page: Page) {
  const { email, username, password } = generateUserCredentials();

  await page.goto('/register');
  await expect(page.getByRole('heading', { name: /create an account/i })).toBeVisible();

  await page.getByLabel('Email').fill(email);
  await page.getByLabel('Username').fill(username);
  await page.getByLabel('Password', { exact: true }).fill(password);
  await page.getByLabel('Confirm password').fill(password);

  await page.getByRole('button', { name: /create account/i }).click();

  // Wait for redirect and auth state
  await page.waitForURL('**/', { timeout: 30_000 });
  await expect(page.getByText(username)).toBeVisible({ timeout: 10_000 });

  return { email, username, password };
}

/**
 * Creates a backend AI game from the lobby and waits for the game page to load.
 * Returns the game URL for reference.
 *
 * The flow is:
 * 1. Register and login
 * 2. Navigate to lobby
 * 3. Open create game form
 * 4. Submit with default settings (creates AI opponent game)
 * 5. Wait for game page with board
 */
async function createBackendGameFromLobby(page: Page): Promise<string> {
  await registerAndLogin(page);

  // Navigate to lobby
  await page.getByRole('link', { name: /lobby/i }).click();
  await page.waitForURL('**/lobby', { timeout: 15_000 });

  // Verify lobby page loaded
  await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible({ timeout: 10_000 });

  // Open create game form
  await page.getByRole('button', { name: /\+ Create Game/i }).click();
  await expect(page.getByRole('heading', { name: /Create Backend Game/i })).toBeVisible({
    timeout: 5_000,
  });

  // Submit game creation with default settings (human vs AI)
  await page.getByRole('button', { name: /^Create Game$/i }).click();

  // Wait for redirect to game page
  await page.waitForURL('**/game/**', { timeout: 30_000 });

  // Verify board is rendered - use data-testid for stability
  await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 15_000 });

  return page.url();
}

test.describe('Backend game flow E2E', () => {
  // Increase timeout for game operations that involve WebSocket and DB
  test.setTimeout(120_000);

  test('creates AI game from lobby and renders board + HUD', async ({ page }) => {
    await createBackendGameFromLobby(page);

    // Verify core game UI elements are present
    // Board should be visible with data-testid
    await expect(page.getByTestId('board-view')).toBeVisible();

    // GameHUD should show connection status (WebSocket connected indicator)
    // The HUD displays "Connection:" followed by status
    await expect(page.locator('text=/Connection/i')).toBeVisible({ timeout: 10_000 });

    // Turn indicator should be visible
    await expect(page.locator('text=/Turn/i')).toBeVisible();

    // Game log section header
    await expect(page.locator('text=/Game log/i')).toBeVisible();
  });

  test('game board has interactive cells during ring placement', async ({ page }) => {
    await createBackendGameFromLobby(page);

    // During ring placement phase, valid targets should be highlighted
    // Valid cells have outline-emerald-300 class for visual feedback
    // Wait for game to initialize and show valid moves
    const boardView = page.getByTestId('board-view');
    await expect(boardView).toBeVisible();

    // Find cells that are clickable (all cells are buttons in BoardView)
    const cells = boardView.locator('button');
    const cellCount = await cells.count();

    // Board should have cells (8x8=64 for default board type)
    expect(cellCount).toBeGreaterThan(0);
  });

  test('submits a ring placement move and logs it', async ({ page }) => {
    await createBackendGameFromLobby(page);

    // Wait for board to be fully ready and game to initialize
    await expect(page.getByTestId('board-view')).toBeVisible();

    // In ring placement phase, find a valid target cell (highlighted with outline-emerald)
    // The styling uses: outline-emerald-300/90
    const validTargetSelector = '[data-testid="board-view"] button[class*="outline-emerald"]';

    // Wait for valid targets to appear (game may need time to sync state)
    await page.waitForSelector(validTargetSelector, {
      state: 'visible',
      timeout: 25_000,
    });

    const targetCell = page.locator(validTargetSelector).first();
    await targetCell.click();

    // After making a move, the game log should update
    // First the log section, then "Recent moves" appears when there are moves
    await expect(page.locator('text=/Game log/i')).toBeVisible();

    // Wait for move to be logged - should show "Recent moves" section
    await expect(page.locator('text=/Recent moves/i')).toBeVisible({ timeout: 15_000 });

    // The move entry should mention P1 (player 1) for the first human move
    const moveEntry = page.locator('li').filter({ hasText: /P1/ }).first();
    await expect(moveEntry).toBeVisible({ timeout: 10_000 });
  });

  test('resyncs game state after full page reload', async ({ page }) => {
    const initialUrl = await createBackendGameFromLobby(page);

    // Verify initial state
    await expect(page.getByTestId('board-view')).toBeVisible();

    // Reload the page completely
    await page.reload();

    // Should return to the same game URL
    await page.waitForURL(initialUrl, { timeout: 15_000 });

    // Board should re-render after WebSocket reconnects
    await expect(page.getByTestId('board-view')).toBeVisible({ timeout: 20_000 });

    // HUD should show connection restored
    await expect(page.locator('text=/Connection/i')).toBeVisible({ timeout: 10_000 });

    // Turn indicator should be restored
    await expect(page.locator('text=/Turn/i')).toBeVisible();
  });

  test('can navigate back to lobby from game page', async ({ page }) => {
    await createBackendGameFromLobby(page);

    // Verify we're on a game page
    await expect(page.getByTestId('board-view')).toBeVisible();

    // Navigate back to lobby via nav link
    await page.getByRole('link', { name: /lobby/i }).click();
    await page.waitForURL('**/lobby', { timeout: 10_000 });

    // Lobby should show
    await expect(page.getByRole('heading', { name: /Game Lobby/i })).toBeVisible();
  });
});
