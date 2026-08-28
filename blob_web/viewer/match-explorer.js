const COLORS = ["#65e6a8", "#ffb55e", "#72b7ff", "#e987ff", "#f56f7d", "#d8e66a"];
const CHECKPOINT_INTERVAL = 128;

const cloneState = (state) => ({
  cells: new Map([...state.cells].map(([key, value]) => [key, { ...value }])),
  tiles: state.tiles.map((tile) => ({ ...tile, signals: [...(tile.signals ?? [0, 0, 0, 0])] })),
});

const number = (value) => Number(value ?? 0);
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" })[character]);
const safeColor = (value, fallback) => /^#[0-9a-f]{6}$/i.test(value ?? "") ? value : fallback;
const totalCellEnergy = (cell) => number(cell.core) + number(cell.assimilated) + number(cell.gut) + number(cell.carried) + number(cell.escrow);
const totalTileEnergy = (tile) => number(tile.plant) + number(tile.loose) + number(tile.diffuse);
const formatEnergy = (value) => Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value);

class BlobMatchExplorer extends HTMLElement {
  static observedAttributes = ["src"];

  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this.bundle = null;
    this.frames = [];
    this.metrics = [];
    this.checkpoints = new Map();
    this.frameIndex = 0;
    this.state = null;
    this.selectedCell = null;
    this.playTimer = null;
    this.speed = 4;
    this.layer = "energy";
    this.chartMetric = "energy";
    this.camera = { scale: 1, x: 0, y: 0 };
    this.drag = null;
    this.verified = false;
    this.resizeObserver = new ResizeObserver(() => this.renderCanvases());
  }

  connectedCallback() {
    this.renderShell();
    this.bindEvents();
    this.resizeObserver.observe(this.shadowRoot.querySelector(".board-wrap"));
    this.resizeObserver.observe(this.shadowRoot.querySelector(".chart-wrap"));
    this.load(this.getAttribute("src"));
  }

  disconnectedCallback() {
    this.stop();
    this.resizeObserver.disconnect();
  }

  attributeChangedCallback(name, oldValue, newValue) {
    if (name === "src" && oldValue !== newValue && this.isConnected) this.load(newValue);
  }

  set data(value) {
    this.acceptBundle(value, false);
  }

  setVerifiedMatch(value, verification) {
    if (!verification?.verified) throw new Error("A successful replay verification result is required");
    this.acceptBundle(value, true);
  }

  async load(src) {
    if (!src) return;
    this.setStatus("Loading match…", "busy");
    try {
      const response = await fetch(src, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.acceptBundle(await response.json(), false);
    } catch (error) {
      this.setStatus(`Could not load match: ${error.message}`, "error");
    }
  }

  acceptBundle(bundle, verified) {
    this.validate(bundle, verified);
    this.stop();
    this.bundle = bundle;
    this.verified = verified;
    this.frameIndex = 0;
    this.selectedCell = null;
    this.camera = { scale: 1, x: 0, y: 0 };
    this.buildIndex();
    this.seek(0, true);
    this.renderHeader();
    const localStatus = bundle.run.canonical_replay
      ? `Local replay · ${bundle.run.canonical_replay.events} canonical events`
      : "Local replay · presentation trace";
    this.setStatus(verified ? "Server-attested replay" : localStatus, verified ? "verified" : "local");
  }

  validate(bundle, verified) {
    if (!bundle || bundle.schema_version !== 1) throw new Error("Expected match explorer schema 1");
    if (!bundle.run?.board || !Number.isInteger(bundle.run.board.width) || !Number.isInteger(bundle.run.board.height)) {
      throw new Error("Match board dimensions are missing");
    }
    if (!Array.isArray(bundle.teams) || !Array.isArray(bundle.initial?.cells) || !Array.isArray(bundle.initial?.tiles)) {
      throw new Error("Match teams and initial state are required");
    }
    if (!bundle.presentation?.cell_teams || typeof bundle.presentation.cell_teams !== "object" || Array.isArray(bundle.presentation.cell_teams)) {
      throw new Error("A presentation-only cell_teams sidecar is required");
    }
    if (!Array.isArray(bundle.events)) throw new Error("Match events must be an array");
    const area = bundle.run.board.width * bundle.run.board.height;
    if (area > 16_000_000 || bundle.events.length > 1_000_000) throw new Error("Monolithic match bundle exceeds the local viewer limits");
    if (!verified && (bundle.run.server_verified || bundle.run.replay_committed || bundle.presentation?.verified)) {
      throw new Error("Browser-loaded JSON cannot claim a verified or committed result");
    }
    if (bundle.initial.tiles.length !== area) throw new Error(`Expected ${area} initial tiles`);
    const ids = new Set(bundle.teams.map((team) => String(team.id)));
    for (const cell of [...bundle.initial.cells, ...bundle.events.flatMap((event) => event.cells ?? [])]) {
      if (Object.hasOwn(cell, "team")) throw new Error(`Cell ${cell.key} embeds a team; use presentation.cell_teams`);
      if (!ids.has(String(bundle.presentation.cell_teams[String(cell.key)]))) throw new Error(`Unknown presentation team for cell ${cell.key}`);
    }
  }

  teamForKey(key) {
    return this.bundle.presentation.cell_teams[String(key)];
  }

  buildIndex() {
    const initial = {
      cells: new Map(this.bundle.initial.cells.map((cell) => [String(cell.key), { ...cell, key: String(cell.key), team: this.teamForKey(cell.key) }])),
      tiles: this.bundle.initial.tiles.map((tile) => ({ ...tile, signals: [...(tile.signals ?? [0, 0, 0, 0])] })),
    };
    this.frames = [{ sequence: "initial", time: number(this.bundle.initial.time), summary: "Initial state", actions: 0 }, ...this.bundle.events];
    this.checkpoints = new Map([[0, cloneState(initial)]]);
    this.metrics = [];
    const actionTotals = new Map(this.bundle.teams.map((team) => [String(team.id), {
      families: {}, foodOpportunities: 0, consumeSelections: 0, consumedEnergy: 0,
    }]));
    let working = initial;
    this.metrics.push(this.measure(working, this.frames[0], actionTotals));
    this.bundle.events.forEach((event, eventIndex) => {
      this.applyEvent(working, event);
      for (const action of event.resolved_actions ?? []) {
        const totals = actionTotals.get(String(this.teamForKey(action.actor)));
        if (!totals) continue;
        const family = String(action.family ?? "unknown");
        totals.families[family] = number(totals.families[family]) + 1;
        if (action.food_at_origin) totals.foodOpportunities += 1;
        if (family === "consume") totals.consumeSelections += 1;
        totals.consumedEnergy += number(action.consumed_energy);
      }
      const frameIndex = eventIndex + 1;
      this.metrics.push(this.measure(working, event, actionTotals));
      if (frameIndex % CHECKPOINT_INTERVAL === 0) this.checkpoints.set(frameIndex, cloneState(working));
    });
  }

  applyEvent(state, event) {
    for (const patch of event.tiles ?? []) {
      const previous = state.tiles[patch.index];
      if (!previous) continue;
      state.tiles[patch.index] = { ...previous, ...patch, signals: [...(patch.signals ?? previous.signals ?? [0, 0, 0, 0])] };
    }
    for (const patch of event.cells ?? []) {
      const key = String(patch.key);
      if (patch.alive === false || patch.remove === true) state.cells.delete(key);
      else state.cells.set(key, { ...(state.cells.get(key) ?? {}), ...patch, key, team: this.teamForKey(key) });
    }
  }

  measure(state, frame, actionTotals) {
    const teams = new Map(this.bundle.teams.map((team) => {
      const totals = actionTotals.get(String(team.id));
      return [String(team.id), {
        energy: 0,
        population: 0,
        actions: { ...(totals?.families ?? {}) },
        foodOpportunities: number(totals?.foodOpportunities),
        consumeSelections: number(totals?.consumeSelections),
        consumedEnergy: number(totals?.consumedEnergy),
      }];
    }));
    for (const cell of state.cells.values()) {
      const value = teams.get(String(cell.team));
      if (value) {
        value.energy += totalCellEnergy(cell);
        value.population += 1;
      }
    }
    return {
      time: number(frame.time),
      actions: number(frame.actions),
      births: (frame.births ?? []).length,
      deaths: (frame.deaths ?? []).length,
      environment: state.tiles.reduce((sum, tile) => sum + totalTileEnergy(tile), 0),
      teams,
    };
  }

  seek(index, force = false) {
    const target = Math.max(0, Math.min(this.frames.length - 1, number(index)));
    if (!force && target === this.frameIndex) return;
    const checkpointIndex = Math.floor(target / CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL;
    let state = cloneState(this.checkpoints.get(checkpointIndex) ?? this.checkpoints.get(0));
    for (let cursor = checkpointIndex + 1; cursor <= target; cursor += 1) this.applyEvent(state, this.frames[cursor]);
    this.state = state;
    this.frameIndex = target;
    if (this.selectedCell && !state.cells.has(this.selectedCell)) {
      // Keep the identity selected so its death is visible in the inspector.
    }
    this.updateFrame();
  }

  step(delta) {
    this.seek(this.frameIndex + delta);
    if (this.frameIndex === this.frames.length - 1) this.stop();
  }

  play() {
    if (this.playTimer) return this.stop();
    if (this.frameIndex === this.frames.length - 1) this.seek(0);
    const button = this.shadowRoot.querySelector("[data-action=play]");
    button.textContent = "Pause";
    button.setAttribute("aria-label", "Pause match");
    this.playTimer = setInterval(() => this.step(1), 1000 / this.speed);
  }

  stop() {
    clearInterval(this.playTimer);
    this.playTimer = null;
    const button = this.shadowRoot?.querySelector("[data-action=play]");
    if (button) {
      button.textContent = "Play";
      button.setAttribute("aria-label", "Play match");
    }
  }

  renderShell() {
    this.shadowRoot.innerHTML = `
      <style>${this.styles()}</style>
      <main>
        <header>
          <div>
            <p class="eyebrow">Blob observatory / match explorer</p>
            <h1>Match microscope</h1>
            <p class="lede">Replay the field, follow individual lives, and find the moments where policy becomes behavior.</p>
          </div>
          <div class="header-meta">
            <span class="status" data-status>Loading…</span>
            <nav aria-label="Observatory views">
              <a href="./">Field notes</a><a href="?view=controls">Control matrix</a><a href="?view=adjudication">Adjudication</a>
            </nav>
          </div>
        </header>
        <section class="runbar">
          <div><span>Match</span><strong data-run-title>—</strong></div>
          <div><span>Rules</span><strong data-rules>—</strong></div>
          <div><span>Outcome</span><strong data-outcome>—</strong></div>
          <label class="file-button">Open JSON<input type="file" accept="application/json,.json" data-file /></label>
        </section>
        <section class="metrics" data-metric-cards></section>
        <section class="workspace">
          <article class="panel field-panel">
            <div class="panel-head">
              <div><p class="kicker">Live field</p><h2 data-frame-title>Initial state</h2></div>
              <div class="field-tools">
                <label>Layer<select data-layer><option value="energy">Energy</option><option value="signal">Signals</option><option value="elevation">Elevation</option><option value="plain">Teams</option></select></label>
                <button data-action="zoom-out" aria-label="Zoom out">−</button>
                <button data-action="fit" aria-label="Fit board">Fit</button>
                <button data-action="zoom-in" aria-label="Zoom in">+</button>
              </div>
            </div>
            <div class="board-wrap"><canvas data-board aria-label="Interactive match board"></canvas><div class="board-help">Scroll to zoom · drag to pan · select a cell to inspect</div></div>
            <div class="legend" data-legend></div>
          </article>
          <aside class="panel inspector">
            <div class="panel-head"><div><p class="kicker">Cell lens</p><h2 data-cell-title>Select a cell</h2></div><div class="cell-tools"><select data-cell-picker aria-label="Choose a cell"><option value="">Choose cell</option></select><button class="quiet" data-action="follow" disabled>Follow</button></div></div>
            <div data-cell-detail class="empty">Choose any occupied tile to track that cell through the full replay.</div>
            <div class="mini-chart"><canvas data-cell-chart aria-label="Selected cell energy history"></canvas></div>
          </aside>
        </section>
        <section class="transport panel">
          <div class="transport-buttons">
            <button data-action="first" aria-label="First event">|‹</button><button data-action="prev" aria-label="Previous event">‹</button>
            <button class="primary" data-action="play" aria-label="Play match">Play</button>
            <button data-action="next" aria-label="Next event">›</button><button data-action="last" aria-label="Last event">›|</button>
          </div>
          <label class="scrubber"><span data-time-label>t = 0</span><input type="range" min="0" value="0" data-scrub aria-label="Match history" /><span data-frame-label>0 / 0</span></label>
          <label class="speed">Speed<select data-speed><option value="1">1×</option><option value="4" selected>4×</option><option value="12">12×</option><option value="30">30×</option></select></label>
        </section>
        <section class="lower-grid">
          <article class="panel chart-panel">
            <div class="panel-head"><div><p class="kicker">Match trajectory</p><h2>Team and field dynamics</h2></div><div class="segmented"><button class="active" data-chart="energy">Energy</button><button data-chart="population">Population</button></div></div>
            <div class="chart-wrap"><canvas data-chart-canvas aria-label="Match metric timeline"></canvas></div>
            <div class="legend" data-chart-legend></div>
          </article>
          <article class="panel event-panel">
            <div class="panel-head"><div><p class="kicker">Resolution trace</p><h2>Current event</h2></div><span class="event-sequence" data-sequence>initial</span></div>
            <div data-event-detail></div>
          </article>
        </section>
      </main>`;
  }

  bindEvents() {
    const root = this.shadowRoot;
    root.addEventListener("click", (event) => {
      const action = event.target.closest("[data-action]")?.dataset.action;
      if (!action) return;
      if (action === "first") this.seek(0);
      if (action === "prev") this.step(-1);
      if (action === "play") this.play();
      if (action === "next") this.step(1);
      if (action === "last") this.seek(this.frames.length - 1);
      if (action === "zoom-in") this.zoom(1.25);
      if (action === "zoom-out") this.zoom(0.8);
      if (action === "fit") { this.camera = { scale: 1, x: 0, y: 0 }; this.renderBoard(); }
      if (action === "follow") this.centerSelected();
    });
    root.querySelector("[data-scrub]").addEventListener("input", (event) => this.seek(event.target.value));
    root.querySelector("[data-speed]").addEventListener("change", (event) => {
      this.speed = number(event.target.value); if (this.playTimer) { this.stop(); this.play(); }
    });
    root.querySelector("[data-layer]").addEventListener("change", (event) => { this.layer = event.target.value; this.renderBoard(); });
    root.querySelector("[data-cell-picker]").addEventListener("change", (event) => {
      this.selectedCell = event.target.value || null;
      this.renderCell();
      this.renderBoard();
    });
    root.querySelectorAll("[data-chart]").forEach((button) => button.addEventListener("click", () => {
      this.chartMetric = button.dataset.chart;
      root.querySelectorAll("[data-chart]").forEach((item) => item.classList.toggle("active", item === button));
      this.renderMetricChart();
    }));
    root.querySelector("[data-file]").addEventListener("change", async (event) => {
      const file = event.target.files?.[0]; if (!file) return;
      try {
        if (file.size > 256 * 1024 * 1024) throw new Error("Match JSON exceeds the 256 MiB local file limit");
        this.acceptBundle(JSON.parse(await file.text()), false);
      } catch (error) { this.setStatus(error.message, "error"); }
      event.target.value = "";
    });
    const canvas = root.querySelector("[data-board]");
    canvas.addEventListener("wheel", (event) => { event.preventDefault(); this.zoom(event.deltaY < 0 ? 1.15 : 0.87); }, { passive: false });
    canvas.addEventListener("pointerdown", (event) => { canvas.setPointerCapture(event.pointerId); this.drag = { x: event.clientX, y: event.clientY, moved: false }; });
    canvas.addEventListener("pointermove", (event) => {
      if (!this.drag) return;
      const dx = event.clientX - this.drag.x; const dy = event.clientY - this.drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) this.drag.moved = true;
      this.camera.x += dx; this.camera.y += dy; this.drag.x = event.clientX; this.drag.y = event.clientY; this.renderBoard();
    });
    canvas.addEventListener("pointerup", (event) => { const moved = this.drag?.moved; this.drag = null; if (!moved) this.selectAt(event); });
    root.host.addEventListener("keydown", (event) => {
      if (event.target.matches("input, select")) return;
      if (event.key === "ArrowLeft") this.step(-1);
      if (event.key === "ArrowRight") this.step(1);
      if (event.key === " ") { event.preventDefault(); this.play(); }
    });
    root.host.tabIndex = 0;
  }

  updateFrame() {
    if (!this.bundle || !this.state) return;
    const frame = this.frames[this.frameIndex]; const metric = this.metrics[this.frameIndex]; const root = this.shadowRoot;
    root.querySelector("[data-scrub]").max = String(this.frames.length - 1);
    root.querySelector("[data-scrub]").value = String(this.frameIndex);
    root.querySelector("[data-time-label]").textContent = `t = ${formatEnergy(metric.time)}`;
    root.querySelector("[data-frame-label]").textContent = `${this.frameIndex} / ${this.frames.length - 1}`;
    root.querySelector("[data-frame-title]").textContent = frame.summary ?? `Event ${frame.sequence}`;
    this.renderMetricCards(metric);
    this.renderEvent(frame, metric);
    this.renderCell();
    this.renderCanvases();
  }

  renderHeader() {
    const run = this.bundle.run;
    this.shadowRoot.querySelector("[data-run-title]").textContent = run.title ?? run.id ?? "Untitled match";
    this.shadowRoot.querySelector("[data-rules]").textContent = run.ruleset ?? "Unlabeled ruleset";
    this.shadowRoot.querySelector("[data-outcome]").textContent = run.outcome ?? "In progress";
    const identities = new Map(this.bundle.initial.cells.map((cell) => [String(cell.key), cell]));
    for (const event of this.bundle.events) for (const cell of event.cells ?? []) if (!identities.has(String(cell.key))) identities.set(String(cell.key), cell);
    this.shadowRoot.querySelector("[data-cell-picker]").innerHTML = `<option value="">Choose cell</option>${[...identities].map(([key, cell]) => {
      const team = this.bundle.teams.find((item) => String(item.id) === String(this.teamForKey(key)));
      return `<option value="${escapeHtml(key)}">${escapeHtml(team?.name ?? "Team")} · ${escapeHtml(key)}</option>`;
    }).join("")}`;
    this.shadowRoot.querySelector("[data-legend]").innerHTML = this.bundle.teams.map((team, index) => `<span><i style="--color:${safeColor(team.color, COLORS[index % COLORS.length])}"></i>${escapeHtml(team.name)}</span>`).join("") + `<span><i class="plant"></i>Plant energy</span>`;
  }

  renderMetricCards(metric) {
    const teamEnergy = [...metric.teams.values()].reduce((sum, team) => sum + team.energy, 0);
    const population = [...metric.teams.values()].reduce((sum, team) => sum + team.population, 0);
    const cards = [
      ["Simulation time", formatEnergy(metric.time)], ["Population", population], ["Cell energy", formatEnergy(teamEnergy)],
      ["Field energy", formatEnergy(metric.environment)], ["Resolved actions", metric.actions], ["Births / deaths", `${metric.births} / ${metric.deaths}`],
    ];
    this.shadowRoot.querySelector("[data-metric-cards]").innerHTML = cards.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
  }

  renderEvent(frame, metric) {
    const root = this.shadowRoot;
    root.querySelector("[data-sequence]").textContent = String(frame.sequence ?? "initial");
    const outcomes = frame.outcomes ?? {};
    const rows = [
      ["Completed at", formatEnergy(frame.time)], ["Attempted actions", number(frame.actions)],
      ["Completed", number(outcomes.completed)], ["Contested", number(outcomes.contested)], ["Frustrated", number(outcomes.frustrated)],
      ["Rejected", number(outcomes.rejected)], ["Interrupted", number(outcomes.interrupted)],
      ["Births", (frame.births ?? []).length], ["Deaths", (frame.deaths ?? []).length],
    ];
    const current = frame.resolved_actions ?? [];
    const currentFood = current.filter((action) => action.food_at_origin).length;
    const currentConsumes = current.filter((action) => action.family === "consume").length;
    const currentExtracted = current.reduce((sum, action) => sum + number(action.consumed_energy), 0);
    if (current.length) rows.push(["Food opportunities", currentFood], ["Consume selections", currentConsumes], ["Energy extracted", currentExtracted]);
    const distributions = this.bundle.teams.map((team, index) => {
      const totals = metric.teams.get(String(team.id));
      const families = Object.entries(totals?.actions ?? {}).sort((left, right) => right[1] - left[1]);
      const actions = families.length
        ? families.map(([family, count]) => `<span>${escapeHtml(family)} <b>${escapeHtml(count)}</b></span>`).join("")
        : `<span>no resolved actions</span>`;
      return `<div class="action-team"><h3><i style="--color:${safeColor(team.color, COLORS[index % COLORS.length])}"></i>${escapeHtml(team.name)}</h3><div class="action-chips">${actions}</div><p>${escapeHtml(totals?.consumeSelections ?? 0)} consumes / ${escapeHtml(totals?.foodOpportunities ?? 0)} food-origin actions · ${escapeHtml(formatEnergy(totals?.consumedEnergy ?? 0))} extracted</p></div>`;
    }).join("");
    root.querySelector("[data-event-detail]").innerHTML = `<p class="event-copy">${escapeHtml(frame.note ?? frame.summary ?? "Canonical initial state before the first decision.")}</p><dl>${rows.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl><div class="action-distribution"><h3>Cumulative action distribution</h3>${distributions}</div>`;
  }

  renderCell() {
    const root = this.shadowRoot; const cell = this.selectedCell ? this.state.cells.get(this.selectedCell) : null;
    root.querySelector("[data-cell-picker]").value = this.selectedCell ?? "";
    const follow = root.querySelector("[data-action=follow]"); follow.disabled = !cell;
    if (!this.selectedCell) {
      root.querySelector("[data-cell-title]").textContent = "Select a cell";
      root.querySelector("[data-cell-detail]").innerHTML = "Choose any occupied tile to track that cell through the full replay.";
      root.querySelector("[data-cell-detail]").className = "empty";
      this.renderCellChart(null); return;
    }
    const team = this.bundle.teams.find((item) => String(item.id) === String(cell?.team ?? this.findCellTeam(this.selectedCell)));
    root.querySelector("[data-cell-title]").textContent = `${team?.name ?? "Unknown team"} · ${this.selectedCell}`;
    if (!cell) {
      root.querySelector("[data-cell-detail]").innerHTML = `<div class="death-note">Not alive at this event</div><p>This identity remains selected so you can scrub back to its lifetime.</p>`;
      root.querySelector("[data-cell-detail]").className = "cell-detail";
      this.renderCellChart(this.selectedCell); return;
    }
    const values = [["Position", `${cell.x}, ${cell.y}`], ["Core mass", formatEnergy(cell.core)], ["Assimilated / health", formatEnergy(cell.assimilated)], ["Gut", formatEnergy(cell.gut)], ["Carried", formatEnergy(cell.carried)], ["Escrow", formatEnergy(cell.escrow)], ["Total mass-energy", formatEnergy(totalCellEnergy(cell))]];
    if (cell.age !== undefined) values.push(["Age", formatEnergy(cell.age)]);
    values.push(["State", cell.guarded ? "Guarded" : (cell.pending ?? cell.role ?? "Active")]);
    const lastAction = this.lastActionFor(this.selectedCell);
    if (lastAction) {
      values.push(["Last action", lastAction.family], ["Action outcome", lastAction.status]);
      if (lastAction.family === "consume") values.push(["Energy extracted", formatEnergy(lastAction.consumed_energy)]);
    }
    root.querySelector("[data-cell-detail]").innerHTML = `<dl>${values.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>`;
    root.querySelector("[data-cell-detail]").className = "cell-detail";
    this.renderCellChart(this.selectedCell);
  }

  findCellTeam(key) {
    return this.teamForKey(key);
  }

  lastActionFor(key) {
    for (let frameIndex = this.frameIndex; frameIndex > 0; frameIndex -= 1) {
      const actions = this.frames[frameIndex].resolved_actions ?? [];
      for (let index = actions.length - 1; index >= 0; index -= 1) {
        if (String(actions[index].actor) === String(key)) return actions[index];
      }
    }
    return null;
  }

  renderCanvases() { this.renderBoard(); this.renderMetricChart(); this.renderCellChart(this.selectedCell); }

  canvasContext(canvas) {
    const rect = canvas.getBoundingClientRect(); const ratio = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * ratio)); const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const context = canvas.getContext("2d"); context.setTransform(ratio, 0, 0, ratio, 0, 0); return { context, width: rect.width, height: rect.height };
  }

  boardTransform(width, height) {
    const board = this.bundle.run.board; const base = Math.min((width - 28) / board.width, (height - 28) / board.height);
    const size = Math.max(2, base * this.camera.scale); const boardWidth = size * board.width; const boardHeight = size * board.height;
    return { size, left: (width - boardWidth) / 2 + this.camera.x, top: (height - boardHeight) / 2 + this.camera.y };
  }

  renderBoard() {
    if (!this.state) return;
    const canvas = this.shadowRoot.querySelector("[data-board]"); const { context, width, height } = this.canvasContext(canvas); const transform = this.boardTransform(width, height); const board = this.bundle.run.board;
    context.clearRect(0, 0, width, height); context.fillStyle = "#07100d"; context.fillRect(0, 0, width, height);
    const elevations = this.state.tiles.map((tile) => number(tile.elevation)); const maxElevation = Math.max(1, ...elevations); const maxTileEnergy = Math.max(1, ...this.state.tiles.map(totalTileEnergy));
    for (let index = 0; index < this.state.tiles.length; index += 1) {
      const tile = this.state.tiles[index]; const x = index % board.width; const y = Math.floor(index / board.width); const px = transform.left + x * transform.size; const py = transform.top + y * transform.size;
      let color = "#11241d";
      if (this.layer === "energy") { const intensity = Math.min(1, totalTileEnergy(tile) / maxTileEnergy); color = `rgb(${13 + intensity * 26}, ${31 + intensity * 75}, ${25 + intensity * 26})`; }
      if (this.layer === "elevation") { const intensity = number(tile.elevation) / maxElevation; color = `rgb(${18 + intensity * 75}, ${28 + intensity * 60}, ${25 + intensity * 42})`; }
      if (this.layer === "signal") { const signals = tile.signals ?? [0, 0, 0, 0]; const max = Math.max(1, ...signals); color = `rgb(${18 + 110 * signals[0] / max}, ${25 + 110 * signals[1] / max}, ${25 + 130 * signals[2] / max})`; }
      context.fillStyle = color; context.fillRect(px, py, transform.size + 0.5, transform.size + 0.5);
      if (number(tile.plant) > 0 && transform.size > 5) { context.fillStyle = "#79d97c"; context.beginPath(); context.arc(px + transform.size * .5, py + transform.size * .5, Math.max(1.3, transform.size * .12), 0, Math.PI * 2); context.fill(); }
      if (transform.size > 9) { context.strokeStyle = "rgba(173,220,198,.08)"; context.strokeRect(px + .5, py + .5, transform.size, transform.size); }
    }
    const teamIndex = new Map(this.bundle.teams.map((team, index) => [String(team.id), index]));
    for (const cell of this.state.cells.values()) {
      const px = transform.left + (number(cell.x) + .5) * transform.size; const py = transform.top + (number(cell.y) + .5) * transform.size; const color = this.bundle.teams[teamIndex.get(String(cell.team))]?.color ?? COLORS[teamIndex.get(String(cell.team)) % COLORS.length];
      context.fillStyle = color; context.beginPath(); context.arc(px, py, Math.max(2.4, transform.size * .34), 0, Math.PI * 2); context.fill();
      context.strokeStyle = cell.guarded ? "#fff7bd" : "rgba(4,12,9,.8)"; context.lineWidth = cell.guarded ? 2.4 : 1; context.stroke();
      if (String(cell.key) === this.selectedCell) { context.strokeStyle = "#ffffff"; context.lineWidth = 2; context.beginPath(); context.arc(px, py, Math.max(5, transform.size * .48), 0, Math.PI * 2); context.stroke(); }
    }
  }

  renderMetricChart() {
    if (!this.bundle || !this.metrics.length) return;
    const canvas = this.shadowRoot.querySelector("[data-chart-canvas]"); const { context, width, height } = this.canvasContext(canvas); const pad = { left: 44, right: 18, top: 16, bottom: 28 }; const plotW = width - pad.left - pad.right; const plotH = height - pad.top - pad.bottom;
    context.clearRect(0, 0, width, height); context.fillStyle = "#0b1713"; context.fillRect(0, 0, width, height);
    const series = this.bundle.teams.map((team, teamIndex) => ({ name: team.name, color: safeColor(team.color, COLORS[teamIndex % COLORS.length]), values: this.metrics.map((metric) => metric.teams.get(String(team.id))?.[this.chartMetric] ?? 0) }));
    if (this.chartMetric === "energy") series.push({ name: "Environment", color: "#7f978a", values: this.metrics.map((metric) => metric.environment) });
    const max = Math.max(1, ...series.flatMap((item) => item.values));
    context.font = "11px ui-monospace, monospace"; context.fillStyle = "#81958c"; context.strokeStyle = "rgba(164,200,183,.12)"; context.lineWidth = 1;
    for (let line = 0; line <= 4; line += 1) { const y = pad.top + plotH * line / 4; context.beginPath(); context.moveTo(pad.left, y); context.lineTo(width - pad.right, y); context.stroke(); context.fillText(formatEnergy(max * (1 - line / 4)), 4, y + 4); }
    for (const item of series) { context.strokeStyle = item.color; context.lineWidth = 2; context.beginPath(); item.values.forEach((value, index) => { const x = pad.left + plotW * index / Math.max(1, item.values.length - 1); const y = pad.top + plotH * (1 - value / max); if (index === 0) context.moveTo(x, y); else context.lineTo(x, y); }); context.stroke(); }
    const playhead = pad.left + plotW * this.frameIndex / Math.max(1, this.metrics.length - 1); context.strokeStyle = "rgba(255,255,255,.72)"; context.setLineDash([3, 4]); context.beginPath(); context.moveTo(playhead, pad.top); context.lineTo(playhead, pad.top + plotH); context.stroke(); context.setLineDash([]);
    this.shadowRoot.querySelector("[data-chart-legend]").innerHTML = series.map((item) => `<span><i style="--color:${item.color}"></i>${escapeHtml(item.name)}</span>`).join("");
  }

  renderCellChart(key) {
    const canvas = this.shadowRoot.querySelector("[data-cell-chart]"); const { context, width, height } = this.canvasContext(canvas); context.clearRect(0, 0, width, height); context.fillStyle = "#0b1713"; context.fillRect(0, 0, width, height); if (!key || !this.bundle) return;
    const values = []; let value = null;
    const initial = this.bundle.initial.cells.find((cell) => String(cell.key) === key); if (initial) value = { ...initial };
    values.push(value ? totalCellEnergy(value) : null);
    for (const event of this.bundle.events) { const patch = (event.cells ?? []).find((cell) => String(cell.key) === key); if (patch) value = patch.alive === false || patch.remove ? null : { ...(value ?? {}), ...patch }; values.push(value ? totalCellEnergy(value) : null); }
    const max = Math.max(1, ...values.filter((item) => item !== null)); context.strokeStyle = "#65e6a8"; context.lineWidth = 2; context.beginPath(); let started = false;
    values.forEach((item, index) => { if (item === null) { started = false; return; } const x = 10 + (width - 20) * index / Math.max(1, values.length - 1); const y = 8 + (height - 16) * (1 - item / max); if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y); }); context.stroke();
    const x = 10 + (width - 20) * this.frameIndex / Math.max(1, values.length - 1); context.strokeStyle = "rgba(255,255,255,.65)"; context.setLineDash([2, 3]); context.beginPath(); context.moveTo(x, 5); context.lineTo(x, height - 5); context.stroke(); context.setLineDash([]);
  }

  zoom(factor) { this.camera.scale = Math.max(.6, Math.min(12, this.camera.scale * factor)); this.renderBoard(); }
  centerSelected() { const cell = this.state.cells.get(this.selectedCell); if (!cell) return; const canvas = this.shadowRoot.querySelector("[data-board]"); const { width, height } = canvas.getBoundingClientRect(); const transform = this.boardTransform(width, height); this.camera.x += width / 2 - (transform.left + (number(cell.x) + .5) * transform.size); this.camera.y += height / 2 - (transform.top + (number(cell.y) + .5) * transform.size); this.renderBoard(); }
  selectAt(event) { if (!this.state) return; const canvas = this.shadowRoot.querySelector("[data-board]"); const rect = canvas.getBoundingClientRect(); const transform = this.boardTransform(rect.width, rect.height); const x = Math.floor((event.clientX - rect.left - transform.left) / transform.size); const y = Math.floor((event.clientY - rect.top - transform.top) / transform.size); const cell = [...this.state.cells.values()].find((item) => number(item.x) === x && number(item.y) === y); this.selectedCell = cell ? String(cell.key) : null; this.renderCell(); this.renderBoard(); }
  setStatus(message, kind) { const node = this.shadowRoot?.querySelector("[data-status]"); if (node) { node.textContent = message; node.className = `status ${kind}`; } }

  styles() { return `
    :host{display:block;min-height:100vh;color:#e4f3eb;background:radial-gradient(circle at 13% 4%,#183229 0,transparent 26rem),#09120f;font-family:Inter,ui-sans-serif,system-ui,sans-serif;outline:none}*{box-sizing:border-box}main{width:min(1540px,calc(100% - 32px));margin:auto;padding:28px 0 54px}header{display:flex;justify-content:space-between;gap:28px;align-items:flex-end;margin-bottom:20px}h1,h2,p{margin:0}h1{font-size:clamp(2rem,5vw,4.3rem);line-height:.94;letter-spacing:-.055em;font-weight:650}h2{font-size:1.05rem;letter-spacing:-.015em}.eyebrow,.kicker{color:#66dba5;text-transform:uppercase;letter-spacing:.15em;font:700 .68rem ui-monospace,monospace;margin-bottom:8px}.lede{color:#9aafa4;max-width:680px;margin-top:12px}.header-meta{text-align:right}.status{display:inline-block;border:1px solid #315346;border-radius:99px;padding:7px 11px;color:#a8bcb2;font:650 .67rem ui-monospace,monospace;text-transform:uppercase;letter-spacing:.08em}.status.verified{color:#65e6a8;border-color:#3c8b69}.status.error{color:#ff8f8f;border-color:#7a3d3d}nav{margin-top:15px;display:flex;gap:14px;justify-content:flex-end}a{color:#9eb5aa;text-decoration:none;font-size:.76rem}a:hover{color:#fff}.runbar,.panel,.metrics>div{background:rgba(12,27,22,.91);border:1px solid #203b31;box-shadow:0 16px 50px rgba(0,0,0,.16)}.runbar{display:grid;grid-template-columns:1.4fr 1fr 1fr auto;border-radius:12px;padding:13px 16px;align-items:center;gap:18px}.runbar div{display:flex;flex-direction:column;min-width:0}.runbar span,.metrics span{color:#728b7e;text-transform:uppercase;letter-spacing:.1em;font:650 .62rem ui-monospace,monospace}.runbar strong{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.84rem;margin-top:3px}.file-button,button,select{border:1px solid #315044;background:#122820;color:#d8e9df;border-radius:7px;padding:8px 11px;font:650 .72rem inherit;cursor:pointer}.file-button input{display:none}.metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:9px;margin:10px 0}.metrics>div{padding:12px 14px;border-radius:9px}.metrics strong{display:block;font:650 1.15rem ui-monospace,monospace;margin-top:5px}.workspace{display:grid;grid-template-columns:minmax(0,2.25fr) minmax(280px,.75fr);gap:10px}.panel{border-radius:12px;overflow:hidden}.panel-head{height:64px;display:flex;justify-content:space-between;align-items:center;padding:12px 14px;border-bottom:1px solid #203b31;gap:15px}.panel-head .kicker{margin-bottom:4px}.field-tools{display:flex;align-items:center;gap:6px}.field-tools label,.speed{color:#80968a;font-size:.68rem}.field-tools select,.speed select{margin-left:6px;padding:6px}.field-tools button{padding:7px 10px}.board-wrap{height:min(60vh,610px);min-height:390px;position:relative;overflow:hidden}.board-wrap canvas,.chart-wrap canvas,.mini-chart canvas{width:100%;height:100%;display:block;touch-action:none}.board-help{position:absolute;left:12px;bottom:11px;padding:5px 8px;background:rgba(5,13,10,.74);border-radius:5px;color:#71877c;font:600 .62rem ui-monospace,monospace;pointer-events:none}.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;min-height:38px;padding:9px 14px;border-top:1px solid #203b31;color:#8fa49a;font-size:.7rem}.legend span{display:flex;align-items:center;gap:6px}.legend i{width:9px;height:9px;border-radius:50%;background:var(--color)}.legend i.plant{background:#79d97c;box-shadow:0 0 0 3px rgba(121,217,124,.13)}.inspector{min-height:490px}.quiet{background:transparent}.quiet:disabled{opacity:.4;cursor:default}.empty,.cell-detail{padding:18px;color:#82978d;font-size:.82rem;line-height:1.55}.cell-detail dl,.event-panel dl{margin:0}.cell-detail dl div,.event-panel dl div{display:flex;justify-content:space-between;gap:14px;padding:9px 0;border-bottom:1px solid rgba(73,105,91,.25)}dt{color:#80958a}dd{margin:0;color:#e1eee7;font:650 .78rem ui-monospace,monospace;text-align:right}.death-note{color:#ff8c86;text-transform:uppercase;letter-spacing:.1em;font:700 .68rem ui-monospace,monospace}.mini-chart{height:120px;margin:5px 14px 14px;border:1px solid #1d362c;border-radius:8px;overflow:hidden}.transport{display:grid;grid-template-columns:auto minmax(240px,1fr) auto;align-items:center;gap:18px;margin:10px 0;padding:11px 13px}.transport-buttons{display:flex;gap:5px}.transport .primary{min-width:68px;background:#57d59a;color:#06130e;border-color:#57d59a}.scrubber{display:grid;grid-template-columns:90px 1fr 80px;align-items:center;gap:10px;color:#8ca095;font:650 .68rem ui-monospace,monospace}.scrubber span:last-child{text-align:right}input[type=range]{accent-color:#65e6a8;width:100%}.lower-grid{display:grid;grid-template-columns:minmax(0,1.9fr) minmax(300px,.7fr);gap:10px}.chart-wrap{height:280px}.segmented{display:flex}.segmented button{border-radius:0}.segmented button:first-child{border-radius:7px 0 0 7px}.segmented button:last-child{border-radius:0 7px 7px 0}.segmented .active{background:#315d4b;color:#fff}.event-panel{min-height:380px}.event-sequence{color:#6edba6;font:700 .74rem ui-monospace,monospace}.event-copy{padding:16px 14px 5px;color:#a0b5aa;font-size:.8rem;line-height:1.5}.event-panel dl{padding:5px 14px 15px}.event-panel dl div{padding:8px 0}button:hover,.file-button:hover,select:hover{border-color:#5c8b77}@media(max-width:1050px){.metrics{grid-template-columns:repeat(3,1fr)}.workspace,.lower-grid{grid-template-columns:1fr}.inspector{min-height:0}.board-wrap{height:520px}.runbar{grid-template-columns:1fr 1fr}}@media(max-width:680px){main{width:min(100% - 18px,1540px);padding-top:15px}header{align-items:flex-start;flex-direction:column}.header-meta{text-align:left}nav{justify-content:flex-start;flex-wrap:wrap}.runbar{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.panel-head{height:auto;align-items:flex-start}.field-tools{flex-wrap:wrap;justify-content:flex-end}.board-wrap{height:430px;min-height:320px}.transport{grid-template-columns:1fr}.transport-buttons{justify-content:center}.scrubber{grid-template-columns:68px 1fr 65px}.speed{text-align:center}}
    .cell-tools{display:flex;align-items:center;gap:6px}.cell-tools select{max-width:150px}.action-distribution{border-top:1px solid #203b31;padding:14px}.action-distribution>h3{margin:0 0 12px;color:#728b7e;text-transform:uppercase;letter-spacing:.1em;font:700 .64rem ui-monospace,monospace}.action-team{padding:10px 0;border-top:1px solid rgba(73,105,91,.2)}.action-team:first-of-type{border-top:0}.action-team h3{display:flex;align-items:center;gap:7px;margin:0 0 7px;font-size:.76rem}.action-team h3 i{width:8px;height:8px;border-radius:50%;background:var(--color)}.action-team p{color:#748b7f;font:600 .65rem ui-monospace,monospace;margin-top:7px}.action-chips{display:flex;gap:5px;flex-wrap:wrap}.action-chips span{border:1px solid #29463a;border-radius:99px;padding:4px 7px;color:#93aa9f;font:600 .63rem ui-monospace,monospace}.action-chips b{color:#e0ede6}
  `; }
}

customElements.define("blob-match-explorer", BlobMatchExplorer);
