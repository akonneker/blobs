import { expect, test } from "@playwright/test";

test("sparse indexing matches a dense oracle through arbitrary rewinds", async ({ page }) => {
  await page.goto("/?view=match");
  const result = await page.evaluate(async () => {
    const { SparseMatchHistory } = await import("/match-history.js");
    const bundle = await (await fetch("/reports/match-explorer.json")).json();
    // Exercise repeated writes, birth and death within the same event.
    const key = "extra";
    bundle.presentation.cell_teams[key] = bundle.teams[0].id;
    bundle.events.push({ time: 100000, tiles: [{ index: 0, loose: 12 }, { index: 0, loose: 7 }], cells: [
      { key, x: 0, y: 0, core: 10, assimilated: 5 }, { key, assimilated: 9 }, { key, remove: true },
    ] });
    const history = await SparseMatchHistory.build(bundle);
    const tiles = bundle.initial.tiles.map((tile) => ({ ...structuredClone(tile), signals: structuredClone(tile.signals ?? [0, 0, 0, 0]) }));
    const cells = new Map(bundle.initial.cells.map((cell) => [String(cell.key), { ...cell, key: String(cell.key) }]));
    const snapshots = [];
    const capture = () => ({ tiles: structuredClone(tiles), cells: [...cells].sort(([a], [b]) => a.localeCompare(b)).map(([key, cell]) => [key, { ...cell, team: bundle.presentation.cell_teams[key] }]) });
    snapshots.push(capture());
    for (const event of bundle.events) {
      for (const patch of event.tiles ?? []) tiles[patch.index] = { ...tiles[patch.index], ...patch };
      for (const patch of event.cells ?? []) {
        const key = String(patch.key);
        if (patch.remove || patch.alive === false) cells.delete(key);
        else cells.set(key, { ...cells.get(key), ...patch, key });
      }
      snapshots.push(capture());
    }
    const canonical = (value) => JSON.stringify(value, (_, item) => item && typeof item === "object" && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item);
    const errors = [];
    for (const target of [snapshots.length - 1, 2, 0, 7, 1, snapshots.length - 1, 0]) {
      history.seek(target);
      const actual = { tiles: history.state.tiles, cells: [...history.state.cells].sort(([a], [b]) => a.localeCompare(b)) };
      if (canonical(actual) !== canonical(snapshots[target])) errors.push(`state ${target}`);
    }
    const cellEnergy = (c) => ["core", "assimilated", "gut", "carried", "escrow"].reduce((sum, k) => sum + Number(c[k] ?? 0), 0);
    for (let index = 0; index < snapshots.length; index++) {
      const snapshot = snapshots[index];
      const metric = history.metrics[index];
      const environment = snapshot.tiles.reduce((sum, t) => sum + Number(t.plant ?? 0) + Number(t.loose ?? 0) + Number(t.diffuse ?? 0), 0);
      if (environment !== metric.environment) errors.push(`environment ${index}`);
      for (const team of bundle.teams) {
        const members = snapshot.cells.map(([, c]) => c).filter((c) => String(c.team) === String(team.id));
        const actual = metric.teams.get(String(team.id));
        if (members.length !== actual.population || members.reduce((sum, c) => sum + cellEnergy(c), 0) !== actual.energy) errors.push(`team ${index}`);
      }
    }
    return errors;
  });
  expect(result).toEqual([]);
});

test("long sparse histories retain changes rather than periodic full worlds", async ({ page }) => {
  await page.goto("/?view=match");
  const result = await page.evaluate(async () => {
    const { SparseMatchHistory } = await import("/match-history.js");
    const bundle = {
      initial: { time: 0, tiles: Array.from({ length: 256 * 256 }, () => ({ loose: 0, signals: [0, 0, 0, 0] })), cells: [] },
      teams: [], presentation: { cell_teams: {} },
      events: Array.from({ length: 10000 }, (_, i) => ({ time: i + 1, tiles: [{ index: i, loose: 1 }] })),
    };
    let yielded = false;
    setTimeout(() => { yielded = true; }, 0);
    const history = await SparseMatchHistory.build(bundle);
    history.seek(10000);
    const energy = history.metrics[10000].environment;
    history.seek(0);
    return { retained: history.changedValues, undoTiles: history.undo.reduce((n, e) => n + e.tiles.length, 0), energy, initial: history.state.tiles[5000].loose, yielded };
  });
  expect(result).toEqual({ retained: 10000, undoTiles: 10000, energy: 10000, initial: 0, yielded: true });
});

test("a new load cancels indexing without replacing the newer match", async ({ page }) => {
  await page.goto("/?view=match");
  const explorer = page.locator("blob-match-explorer");
  await expect(explorer.locator("[data-status]")).toContainText("Local replay");
  const result = await explorer.evaluate(async (element) => {
    const older = structuredClone(element.bundle);
    older.events = Array.from({ length: 2000 }, (_, index) => ({ time: index + 1, tiles: [], cells: [] }));
    const newer = structuredClone(element.bundle);
    newer.events = newer.events.slice(0, 2);
    const first = element.acceptBundle(older, false);
    const second = element.acceptBundle(newer, false);
    await Promise.all([first, second]);
    return { count: element.bundle.events.length, frames: element.frames.length };
  });
  expect(result).toEqual({ count: 2, frames: 3 });
});
