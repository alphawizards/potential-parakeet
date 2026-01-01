
import { test, expect } from '@playwright/test';

test.describe('Truth Engine', () => {
  test.beforeEach(async ({ page }) => {
    // Enable console logging
    page.on('console', msg => console.log(`[Browser Console] ${msg.type()}: ${msg.text()}`));
    page.on('pageerror', err => console.log(`[Browser PageError]: ${err.message}`));

    // Mock API responses to ensure the Dashboard loads correctly (legacy endpoints)
    await page.route('**/api/trades/*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ trades: [], total: 0 }),
      });
    });

    await page.route('**/api/trades/metrics/dashboard*', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ portfolio: { total_value: 100000 }, open_positions: 0 }),
      });
    });

    await page.route('**/api/trades/metrics/portfolio*', async (route) => {
       await route.fulfill({ status: 200, body: JSON.stringify({}) });
    });

    // Start at home
    await page.goto('http://localhost:3000/');
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status !== testInfo.expectedStatus) {
      const screenshotPath = `test-results/${testInfo.title.replace(/\s+/g, '-')}-failure.png`;
      await page.screenshot({ path: screenshotPath, fullPage: true });
      console.log(`[Test] Screenshot saved to ${screenshotPath}`);
    }
  });

  test('Navigate to Truth Engine', async ({ page }) => {
    const link = page.getByRole('link', { name: 'Truth Engine' });
    await expect(link).toBeVisible();
    await link.click();
    await expect(page).toHaveURL(/.*\/truth-engine/);
    await expect(page.getByRole('heading', { name: 'Truth Engine' })).toBeVisible();
  });

  test('Scenario B (The Gem): Valid Strategy', async ({ page }) => {
    await page.goto('http://localhost:3000/truth-engine');

    // Select "Statistical Arbitrage Pairs" (The Gem)
    await page.getByRole('combobox').selectOption('stat-arb-pairs');

    // Assert "VALID" badge
    await expect(page.getByText('VALID', { exact: true })).toBeVisible();
    await expect(page.getByText('OVERFIT WARNING')).not.toBeVisible();

    // Check Metrics (DSR 1.65, Trials 8)
    await expect(page.getByText('1.65')).toBeVisible();
    await expect(page.getByText('8', { exact: true })).toBeVisible();
  });

  test('Scenario A (The Fraud): Overfit Strategy', async ({ page }) => {
    await page.goto('http://localhost:3000/truth-engine');

    // Select "ML Return Predictor" (The Fraud)
    await page.getByRole('combobox').selectOption('ml-predictor');

    // Assert "OVERFIT WARNING" badge
    await expect(page.getByText('OVERFIT WARNING')).toBeVisible();

    // Check Metrics (DSR 0.32, Trials 1000)
    await expect(page.getByText('0.32')).toBeVisible();
    await expect(page.getByText('1000')).toBeVisible();
  });

  test('Forensic Charts Rendering', async ({ page }) => {
    await page.goto('http://localhost:3000/truth-engine');

    // Check for chart titles
    await expect(page.getByText('Information Coefficient (IC) Decay')).toBeVisible();
    await expect(page.getByText('Factor Attribution')).toBeVisible();
    await expect(page.getByText('Execution Surface (Slippage vs VIX)')).toBeVisible();

    // Verify Recharts containers are present
    await expect(page.locator('.recharts-surface').first()).toBeVisible();
    // We expect at least 3 charts
    expect(await page.locator('.recharts-surface').count()).toBeGreaterThanOrEqual(3);
  });

  test('Execution Surface High Volatility Handling', async ({ page }) => {
     await page.goto('http://localhost:3000/truth-engine');
     // Select the strategy with high vol execution data
     await page.getByRole('combobox').selectOption('ml-predictor');

     // Verify the chart title is still visible
     await expect(page.getByText('Execution Surface (Slippage vs VIX)')).toBeVisible();
  });
});
