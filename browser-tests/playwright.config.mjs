import { defineConfig } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  testDir: path.join(directory, "tests"),
  outputDir: path.join(directory, "test-results"),
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: process.env.CI
    ? [["line"], ["html", { outputFolder: path.join(directory, "playwright-report"), open: "never" }]]
    : "line",
  use: {
    baseURL: "http://127.0.0.1:4173",
    browserName: "chromium",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "python3 scripts/telemetry_viewer_server.py",
    cwd: path.resolve(directory, ".."),
    url: "http://127.0.0.1:4173/?view=match",
    reuseExistingServer: !process.env.CI,
    timeout: 15_000,
  },
});
