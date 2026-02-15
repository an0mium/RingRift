import { test, expect } from '@playwright/test';

test.describe('PWA Support', () => {
  test('serves web manifest', async ({ page }) => {
    const response = await page.goto('/manifest.webmanifest');
    expect(response?.status()).toBe(200);

    const manifest = await response?.json();
    expect(manifest.name).toBe('RingRift - Multiplayer Strategy Game');
    expect(manifest.short_name).toBe('RingRift');
    expect(manifest.display).toBe('standalone');
    expect(manifest.start_url).toBe('/sandbox');
    expect(manifest.icons).toBeDefined();
    expect(manifest.icons.length).toBeGreaterThanOrEqual(2);
  });

  test('serves PWA icons', async ({ page }) => {
    const icon192 = await page.goto('/pwa-192x192.png');
    expect(icon192?.status()).toBe(200);
    expect(icon192?.headers()['content-type']).toContain('image/png');

    const icon512 = await page.goto('/pwa-512x512.png');
    expect(icon512?.status()).toBe(200);
  });

  test('serves apple touch icon', async ({ page }) => {
    const icon = await page.goto('/apple-touch-icon.png');
    expect(icon?.status()).toBe(200);
    expect(icon?.headers()['content-type']).toContain('image/png');
  });

  test('HTML includes apple-touch-icon link', async ({ page }) => {
    await page.goto('/');
    const link = page.locator('link[rel="apple-touch-icon"]');
    await expect(link).toHaveAttribute('href', '/apple-touch-icon.png');
  });

  test('HTML includes theme-color meta', async ({ page }) => {
    await page.goto('/');
    const meta = page.locator('meta[name="theme-color"]');
    await expect(meta).toHaveAttribute('content', '#0f172a');
  });
});
