// Reversible sparse presentation history. Objects are replaced, never mutated,
// so undo records retain only the values actually changed by each event.
const number = (value) => Number(value ?? 0);
const cellEnergy = (cell) => cell ? number(cell.core) + number(cell.assimilated) + number(cell.gut) + number(cell.carried) + number(cell.escrow) : 0;
const tileEnergy = (tile) => number(tile.plant) + number(tile.loose) + number(tile.diffuse);
const copyTile = (tile) => ({ ...tile, signals: [...(tile.signals ?? [0, 0, 0, 0])] });

export class SparseMatchHistory {
  static async build(bundle, signal) {
    const history = new SparseMatchHistory(bundle);
    for (let start = 0; start < bundle.events.length; start += 256) {
      signal?.throwIfAborted();
      const end = Math.min(bundle.events.length, start + 256);
      for (let index = start; index < end; index++) history.indexEvent(bundle.events[index]);
      // Allow loading cancellation and UI painting during large histories.
      if (end < bundle.events.length) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    signal?.throwIfAborted();
    history.seek(0);
    return history;
  }

  constructor(bundle) {
    this.bundle = bundle;
    this.state = {
      cells: new Map(bundle.initial.cells.map((cell) => [String(cell.key), { ...cell, key: String(cell.key), team: this.team(cell.key) }])),
      tiles: bundle.initial.tiles.map(copyTile),
    };
    this.frames = [{ sequence: "initial", time: number(bundle.initial.time), summary: "Initial state", actions: 0 }, ...bundle.events];
    this.frameIndex = 0;
    this.undo = [];
    this.changedValues = 0;
    this.aggregate = {
      environment: this.state.tiles.reduce((sum, tile) => sum + tileEnergy(tile), 0),
      teams: new Map(bundle.teams.map((team) => [String(team.id), {
        energy: 0, population: 0, actions: {}, foodOpportunities: 0, consumeSelections: 0, consumedEnergy: 0,
      }])),
    };
    for (const cell of this.state.cells.values()) this.countCell(cell, 1);
    this.metrics = [this.measure(this.frames[0])];
  }

  team(key) { return this.bundle.presentation.cell_teams[String(key)]; }

  countCell(cell, direction) {
    const value = cell && this.aggregate.teams.get(String(cell.team));
    if (value) { value.energy += direction * cellEnergy(cell); value.population += direction; }
  }

  apply(event, record = false) {
    const undo = { tiles: [], cells: [] };
    for (const patch of event.tiles ?? []) {
      const previous = this.state.tiles[patch.index];
      if (!previous) continue;
      const next = { ...previous, ...patch, signals: [...(patch.signals ?? previous.signals)] };
      if (record) {
        undo.tiles.push([patch.index, previous]);
        this.aggregate.environment += tileEnergy(next) - tileEnergy(previous);
      }
      this.state.tiles[patch.index] = next;
    }
    for (const patch of event.cells ?? []) {
      const key = String(patch.key);
      const previous = this.state.cells.get(key);
      const next = patch.alive === false || patch.remove === true ? undefined : { ...previous, ...patch, key, team: this.team(key) };
      if (record) { undo.cells.push([key, previous]); this.countCell(previous, -1); this.countCell(next, 1); }
      if (next) this.state.cells.set(key, next);
      else this.state.cells.delete(key);
    }
    return undo;
  }

  indexEvent(event) {
    const undo = this.apply(event, true);
    this.undo.push(undo);
    this.changedValues += undo.tiles.length + undo.cells.length;
    for (const action of event.resolved_actions ?? []) {
      const totals = this.aggregate.teams.get(String(this.team(action.actor)));
      if (!totals) continue;
      const family = String(action.family ?? "unknown");
      totals.actions[family] = number(totals.actions[family]) + 1;
      if (action.food_at_origin) totals.foodOpportunities++;
      if (family === "consume") totals.consumeSelections++;
      totals.consumedEnergy += number(action.consumed_energy);
    }
    this.metrics.push(this.measure(event));
    this.frameIndex++;
  }

  measure(frame) {
    return {
      time: number(frame.time), actions: number(frame.actions), births: (frame.births ?? []).length, deaths: (frame.deaths ?? []).length,
      environment: this.aggregate.environment,
      teams: new Map([...this.aggregate.teams].map(([key, value]) => [key, { ...value, actions: { ...value.actions } }])),
    };
  }

  seek(target) {
    target = Math.max(0, Math.min(this.frames.length - 1, Math.trunc(Number(target) || 0)));
    while (this.frameIndex > target) {
      const undo = this.undo[this.frameIndex - 1];
      for (let i = undo.cells.length - 1; i >= 0; i--) {
        const [key, previous] = undo.cells[i];
        if (previous) this.state.cells.set(key, previous);
        else this.state.cells.delete(key);
      }
      for (let i = undo.tiles.length - 1; i >= 0; i--) {
        const [index, previous] = undo.tiles[i];
        this.state.tiles[index] = previous;
      }
      this.frameIndex--;
    }
    while (this.frameIndex < target) this.apply(this.frames[++this.frameIndex]);
    return this.state;
  }
}
