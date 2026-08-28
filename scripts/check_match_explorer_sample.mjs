import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const source = process.argv[2]
  ? resolve(process.argv[2])
  : fileURLToPath(new URL("../blob_web/viewer/sample-match.json", import.meta.url));
const bundle = JSON.parse(await readFile(source, "utf8"));
if (bundle.schema_version !== 1) throw new Error("sample must use match explorer schema 1");

const { width, height } = bundle.run.board;
const area = width * height;
if (bundle.initial.tiles.length !== area) throw new Error(`expected ${area} initial tiles`);

const teams = new Set(bundle.teams.map((team) => String(team.id)));
const assignment = bundle.presentation.cell_teams;
const cells = new Map(bundle.initial.cells.map((cell) => [String(cell.key), { ...cell }]));
let previousTime = Number(bundle.initial.time);

function checkCells(sequence) {
  const occupied = new Set();
  for (const cell of cells.values()) {
    if (Object.hasOwn(cell, "team")) throw new Error(`event ${sequence}: cell state embeds a team`);
    if (!teams.has(String(assignment[String(cell.key)]))) throw new Error(`event ${sequence}: cell ${cell.key} has an unknown team`);
    if (!Number.isInteger(cell.x) || !Number.isInteger(cell.y) || cell.x < 0 || cell.y < 0 || cell.x >= width || cell.y >= height) {
      throw new Error(`event ${sequence}: cell ${cell.key} is outside the board`);
    }
    const tile = `${cell.x},${cell.y}`;
    if (occupied.has(tile)) throw new Error(`event ${sequence}: duplicate occupancy at ${tile}`);
    occupied.add(tile);
  }
}

checkCells("initial");
for (const event of bundle.events) {
  const time = Number(event.time);
  if (!Number.isFinite(time) || time < previousTime) throw new Error(`event ${event.sequence}: time is not monotonic`);
  previousTime = time;
  for (const action of event.resolved_actions ?? []) {
    if (!teams.has(String(assignment[String(action.actor)]))) throw new Error(`event ${event.sequence}: action actor ${action.actor} has an unknown team`);
    if (action.consumed_energy < 0 || (action.family !== "consume" && action.consumed_energy !== 0)) throw new Error(`event ${event.sequence}: invalid consumed energy`);
  }
  for (const tile of event.tiles ?? []) {
    if (!Number.isInteger(tile.index) || tile.index < 0 || tile.index >= area) throw new Error(`event ${event.sequence}: invalid tile patch`);
  }
  for (const patch of event.cells ?? []) {
    if (Object.hasOwn(patch, "team")) throw new Error(`event ${event.sequence}: cell patch embeds a team`);
    const key = String(patch.key);
    if (patch.alive === false || patch.remove === true) cells.delete(key);
    else cells.set(key, { ...(cells.get(key) ?? {}), ...patch, key });
  }
  checkCells(event.sequence);
}

if (bundle.run.canonical_replay) {
  if (bundle.run.canonical_replay.events !== bundle.events.length) throw new Error("canonical replay event count diverges from presentation trace");
  const replay = await readFile(join(dirname(source), bundle.run.canonical_replay.file));
  const hash = createHash("sha256").update(replay).digest("hex");
  if (hash !== bundle.run.canonical_replay.sha256) throw new Error("canonical replay SHA-256 does not match its binding");
}

console.log(`match explorer sample: ${bundle.events.length} events, ${area} tiles, valid sparse history`);
