import { expect, test } from "@playwright/test";

test("loads and explores a complete sparse match in a real browser", async ({ page }) => {
  const browserErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
  page.on("pageerror", (error) => browserErrors.push(error.message));

  await page.goto("/?view=match");
  await expect(page).toHaveTitle("Blob Match Microscope");

  const explorer = page.locator("blob-match-explorer");
  await expect(explorer.locator("[data-status]"))
    .toContainText("Local replay");
  await expect(explorer.locator("[data-run-title]"))
    .not.toHaveText("—");
  await expect(explorer.locator("[data-frame-label]"))
    .toHaveText("0 / 12");

  await explorer.getByRole("button", { name: "Next event" }).click();
  await expect(explorer.locator("[data-frame-label]"))
    .toHaveText("1 / 12");
  await expect(explorer.locator("[data-sequence]"))
    .not.toHaveText("initial");

  const cells = explorer.getByRole("combobox", { name: "Choose a cell" });
  await expect(cells.locator("option")).toHaveCount(8);
  await cells.selectOption({ index: 1 });
  await expect(explorer.locator("[data-cell-title]"))
    .not.toHaveText("Select a cell");

  await explorer.getByRole("button", { name: "Last event" }).click();
  await expect(explorer.locator("[data-frame-label]"))
    .toHaveText("12 / 12");
  await explorer.getByRole("button", { name: "Population" }).click();
  await explorer.locator("[data-layer]").selectOption("signal");

  for (const canvas of ["[data-board]", "[data-chart-canvas]"]) {
    const bounds = await explorer.locator(canvas).boundingBox();
    expect(bounds?.width).toBeGreaterThan(100);
    expect(bounds?.height).toBeGreaterThan(100);
  }
  expect(browserErrors).toEqual([]);
});

test("fails closed when local JSON claims trusted status", async ({ page }) => {
  await page.goto("/?view=match");
  const message = await page.locator("blob-match-explorer").evaluate((element) => {
    const forged = structuredClone(element.bundle);
    forged.run.server_verified = true;
    try {
      element.data = forged;
      return "accepted";
    } catch (error) {
      return error.message;
    }
  });
  expect(message).toContain("cannot claim a verified or committed result");
});
