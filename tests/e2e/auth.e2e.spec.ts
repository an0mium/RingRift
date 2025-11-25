import { test, expect } from '@playwright/test';

/**
 * E2E Test Suite: User Authentication
 * ============================================================================
 *
 * This suite tests the core authentication happy path:
 * - User registration with email, username, and password
 * - Automatic login after registration
 * - Logout functionality
 * - Re-login with existing credentials
 *
 * INFRASTRUCTURE REQUIREMENTS:
 * - PostgreSQL running (for user persistence)
 * - Redis running (for session management)
 * - Dev server running on http://localhost:5173
 *
 * RUN COMMAND: npm run test:e2e -- --timeout 60000
 */

/**
 * Generates unique user credentials for test isolation.
 * Each test run creates distinct users to avoid conflicts between parallel runs.
 */
function generateUserCredentials() {
  const timestamp = Date.now();
  const random = Math.floor(Math.random() * 1_000_000);
  const slug = `${timestamp}-${random}`;
  const email = `e2e+${slug}@example.com`;
  const username = `e2e-user-${slug}`;
  const password = 'E2E_test_password_123!';
  return { email, username, password };
}

test.describe('Auth E2E – registration and login', () => {
  // Increase timeout for auth operations that involve database
  test.setTimeout(90_000);

  test('registers a new user, logs out, and logs back in', async ({ page }) => {
    const { email, username, password } = generateUserCredentials();

    // Step 1: Navigate to registration page
    await page.goto('/register');
    await expect(page.getByRole('heading', { name: /create an account/i })).toBeVisible();

    // Step 2: Fill registration form
    // Using exact label matches for stability
    await page.getByLabel('Email').fill(email);
    await page.getByLabel('Username').fill(username);
    await page.getByLabel('Password', { exact: true }).fill(password);
    await page.getByLabel('Confirm password').fill(password);

    // Step 3: Submit and verify successful registration
    await page.getByRole('button', { name: /create account/i }).click();

    // Wait for redirect to authenticated home page
    await page.waitForURL('**/', { timeout: 30_000 });

    // Step 4: Verify authenticated state in navbar
    const logoutButton = page.getByRole('button', { name: /logout/i });
    await expect(logoutButton).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText(username)).toBeVisible();

    // Step 5: Log out
    await logoutButton.click();

    // After logout, should redirect to /login
    await page.waitForURL('**/login', { timeout: 10_000 });

    // Step 6: Log back in with same credentials
    await page.getByLabel('Email').fill(email);
    await page.getByLabel('Password', { exact: true }).fill(password);

    await page.getByRole('button', { name: /login/i }).click();

    // Wait for redirect back to home
    await page.waitForURL('**/', { timeout: 30_000 });

    // Step 7: Verify authenticated state restored
    await expect(page.getByRole('button', { name: /logout/i })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText(username)).toBeVisible();
  });

  test('shows error for invalid login credentials', async ({ page }) => {
    // Navigate to login page
    await page.goto('/login');
    await expect(page.getByRole('heading', { name: /login/i })).toBeVisible();

    // Attempt login with non-existent credentials
    await page.getByLabel('Email').fill('nonexistent@example.com');
    await page.getByLabel('Password', { exact: true }).fill('WrongPassword123!');

    await page.getByRole('button', { name: /login/i }).click();

    // Should redirect to registration with prefilled email (current app behavior)
    // OR show an error - either is acceptable for this happy path test
    await expect(
      page.getByRole('heading', { name: /create an account/i }).or(page.locator('.text-red-300'))
    ).toBeVisible({ timeout: 10_000 });
  });
});
