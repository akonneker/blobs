const ROLE_NAMES = ["Feeder", "Explorer", "Builder", "Defender", "Attacker"];
const SIGNAL_NAMES = ["Plant", "Threat", "Build", "Frontier"];
const OUTCOME_NAMES = ["Move", "Attack", "Guard", "Consume", "Split", "Excavate", "Deposit", "Signal"];

const styles = String.raw`
  :host {
    --ink: #ecf4dd;
    --muted: #92a68f;
    --panel: rgba(18, 32, 27, 0.88);
    --line: rgba(202, 225, 184, 0.14);
    --moss: #a8d46f;
    --amber: #f2b84b;
    --clay: #df735a;
    --water: #6ab8aa;
    display: block;
    min-height: 100vh;
    color: var(--ink);
    background:
      radial-gradient(circle at 8% 5%, rgba(105, 151, 76, 0.20), transparent 28rem),
      radial-gradient(circle at 92% 14%, rgba(75, 132, 122, 0.14), transparent 32rem),
      #09120f;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  * { box-sizing: border-box; }
  button, input { font: inherit; }

  .shell {
    width: min(1480px, 100%);
    margin: 0 auto;
    padding: 28px clamp(18px, 4vw, 56px) 48px;
  }

  .masthead {
    display: flex;
    justify-content: space-between;
    gap: 24px;
    align-items: flex-start;
    padding-bottom: 22px;
    border-bottom: 1px solid var(--line);
  }

  .eyebrow, .label, th {
    color: var(--muted);
    font: 700 11px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: .14em;
    text-transform: uppercase;
  }

  h1 {
    margin: 5px 0 0;
    font: 500 clamp(34px, 5vw, 66px)/.95 Georgia, "Times New Roman", serif;
    letter-spacing: -.04em;
  }

  .lede {
    max-width: 610px;
    margin: 13px 0 0;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.6;
  }

  .controls { display: flex; flex-wrap: wrap; gap: 9px; justify-content: flex-end; }
  button, .upload-label, .control-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-height: 40px;
    padding: 0 15px;
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--ink);
    background: rgba(255,255,255,.035);
    cursor: pointer;
    text-decoration: none;
    transition: border-color .16s ease, background .16s ease, transform .16s ease;
  }
  button:hover, .upload-label:hover, .control-link:hover { border-color: rgba(168,212,111,.5); background: rgba(168,212,111,.08); }
  button:active, .upload-label:active { transform: translateY(1px); }
  button:focus-visible, .upload-label:focus-within, .control-link:focus-visible { outline: 2px solid var(--moss); outline-offset: 3px; }
  input[type=file] { width: 1px; height: 1px; opacity: 0; position: absolute; }

  .runbar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 26px;
    align-items: center;
    padding: 18px 0;
  }
  .run-item { display: flex; gap: 8px; align-items: baseline; font-size: 13px; }
  .run-value { color: var(--ink); }
  .trust {
    margin-left: auto;
    padding: 7px 11px;
    border-radius: 4px;
    background: rgba(242,184,75,.12);
    color: #ffd585;
    font: 700 11px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
    letter-spacing: .05em;
    text-transform: uppercase;
  }
  .trust.verified { color: #c9ef91; background: rgba(168,212,111,.12); }

  .summary {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border: 1px solid var(--line);
    border-radius: 14px;
    overflow: hidden;
    background: rgba(255,255,255,.02);
  }
  .metric { min-height: 132px; padding: 21px 22px; border-right: 1px solid var(--line); }
  .metric:last-child { border-right: 0; }
  .metric-value { margin-top: 18px; font: 500 34px/1 Georgia, serif; }
  .metric-note { margin-top: 8px; color: var(--muted); font-size: 12px; }

  .grid {
    display: grid;
    grid-template-columns: minmax(360px, 1.2fr) minmax(300px, .8fr);
    gap: 16px;
    margin-top: 16px;
  }
  .panel {
    border: 1px solid var(--line);
    border-radius: 14px;
    background: var(--panel);
    overflow: hidden;
  }
  .panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; padding: 19px 20px 0; }
  h2 { margin: 0; font: 500 19px/1.2 Georgia, serif; }
  .panel-note { color: var(--muted); font-size: 11px; }

  .chart { display: block; width: 100%; height: 250px; margin-top: 8px; }
  .legend { display: flex; flex-wrap: wrap; gap: 16px; padding: 0 20px 18px; color: var(--muted); font-size: 11px; }
  .legend i { display: inline-block; width: 8px; height: 8px; margin-right: 6px; border-radius: 50%; }

  .roles, .signals, .outcomes { padding: 20px; display: grid; gap: 14px; }
  .coverage-memory {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin: 0 20px 20px;
    padding: 12px 14px;
    border: 1px solid var(--line);
    border-radius: 7px;
    color: var(--muted);
    font-size: 11px;
  }
  .coverage-memory strong { color: var(--ink); font: 500 16px Georgia, serif; }
  .bar-row { display: grid; grid-template-columns: 78px 1fr 44px; gap: 12px; align-items: center; font-size: 12px; }
  .track { height: 7px; background: rgba(255,255,255,.06); border-radius: 20px; overflow: hidden; }
  .fill { height: 100%; border-radius: inherit; background: var(--moss); }
  .bar-value { color: var(--muted); text-align: right; font-family: ui-monospace, monospace; }

  .lower {
    display: grid;
    grid-template-columns: minmax(320px, .75fr) minmax(420px, 1.25fr);
    gap: 16px;
    margin-top: 16px;
  }
  .map-wrap { padding: 20px; display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: center; }
  .map {
    width: min(100%, 330px);
    aspect-ratio: 1;
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 5px;
  }
  .tile { position: relative; border: 1px solid rgba(255,255,255,.055); border-radius: 5px; background: rgba(255,255,255,.025); }
  .tile.ring { background: rgba(223,115,90,.10); border-color: rgba(223,115,90,.32); }
  .tile.wall { background: rgba(223,115,90,.26); border-color: rgba(242,184,75,.42); }
  .tile.gate { background: rgba(106,184,170,.13); border: 1px dashed rgba(106,184,170,.65); }
  .tile.plant::after { content: ""; position: absolute; inset: 27%; border-radius: 55% 35% 55% 35%; background: var(--moss); transform: rotate(18deg); box-shadow: 0 0 18px rgba(168,212,111,.35); }
  .tile.defender::before { content: "D"; position: absolute; z-index: 1; inset: 50% auto auto 50%; transform: translate(-50%,-50%); color: #fff2c7; font: 700 10px ui-monospace, monospace; }
  .map-key { display: grid; gap: 10px; color: var(--muted); font-size: 11px; }
  .swatch { display: inline-block; width: 11px; height: 11px; margin-right: 7px; vertical-align: -1px; border-radius: 2px; }

  .checks { padding: 13px 20px 20px; }
  .check { display: grid; grid-template-columns: 9px 1fr auto; gap: 11px; align-items: start; padding: 13px 0; border-bottom: 1px solid var(--line); }
  .check:last-child { border-bottom: 0; }
  .dot { width: 8px; height: 8px; margin-top: 5px; border-radius: 50%; background: var(--moss); }
  .dot.warning { background: var(--amber); }
  .dot.exposed { background: var(--clay); }
  .check-name { font-size: 13px; }
  .check-detail { margin-top: 4px; color: var(--muted); font-size: 11px; line-height: 1.45; }
  .check-status { color: var(--muted); font: 700 10px ui-monospace, monospace; text-transform: uppercase; }

  .action-strip { display: grid; grid-template-columns: repeat(4, 1fr); border-top: 1px solid var(--line); }
  .action { padding: 16px 20px; border-right: 1px solid var(--line); }
  .action:last-child { border: 0; }
  .action strong { display: block; margin-top: 7px; font: 500 21px Georgia, serif; }

  .empty, .error { margin: 72px auto; max-width: 540px; padding: 30px; text-align: center; }
  .error { color: #ffd1c6; }
  :host(.drop-active) { outline: 2px dashed var(--moss); outline-offset: -10px; }

  @media (max-width: 900px) {
    .masthead { flex-direction: column; }
    .controls { justify-content: flex-start; }
    .summary { grid-template-columns: repeat(2, 1fr); }
    .metric:nth-child(2) { border-right: 0; }
    .metric:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
    .grid, .lower { grid-template-columns: 1fr; }
  }
  @media (max-width: 560px) {
    .shell { padding-inline: 13px; }
    .summary { grid-template-columns: 1fr; }
    .metric { border-right: 0; border-bottom: 1px solid var(--line); min-height: 110px; }
    .metric:last-child { border-bottom: 0; }
    .runbar { gap: 8px 18px; }
    .trust { width: 100%; margin-left: 0; }
    .map-wrap { grid-template-columns: 1fr; }
    .action-strip { grid-template-columns: repeat(2, 1fr); }
    .action:nth-child(2) { border-right: 0; }
    .action:nth-child(-n+2) { border-bottom: 1px solid var(--line); }
  }
`;

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const number = (value) => {
  const parsed = Number(value ?? 0);
  return Number.isFinite(parsed) ? parsed : 0;
};
const percent = (numerator, denominator) => denominator ? `${Math.round(100 * numerator / denominator)}%` : "—";
const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

class BlobTelemetryViewer extends HTMLElement {
  static observedAttributes = ["src"];

  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._data = null;
    this._verification = null;
    this._resizeObserver = new ResizeObserver(() => this.drawTimeline());
  }

  connectedCallback() {
    this.renderLoading();
    this.installDropTarget();
    if (this.getAttribute("src")) this.load(this.getAttribute("src"));
  }

  disconnectedCallback() { this._resizeObserver.disconnect(); }

  attributeChangedCallback(name, oldValue, newValue) {
    if (name === "src" && oldValue !== newValue && this.isConnected && newValue) this.load(newValue);
  }

  set data(value) { this.setTelemetry(value); }
  get data() { return this._data; }

  async load(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Telemetry request failed (${response.status})`);
      this.setTelemetry(await response.json());
    } catch (error) {
      this.renderError(error);
    }
  }

  setTelemetry(value) {
    this._verification = null;
    this.commitTelemetry(value);
  }

  setVerifiedTelemetry(value, verification) {
    if (!verification || verification.verified !== true) {
      throw new Error("Verified telemetry requires a successful attestation result");
    }
    this._verification = verification;
    this.commitTelemetry(value);
  }

  commitTelemetry(value) {
    if (!value || number(value.schema_version) !== 1 || !value.latest) {
      throw new Error("Expected Blob telemetry schema_version 1 with a latest sample");
    }
    this._data = value;
    this.render();
    this.dispatchEvent(new CustomEvent("blob-telemetry-load", { detail: { data: value } }));
  }

  renderLoading() {
    this.shadowRoot.innerHTML = `<style>${styles}</style><main class="shell"><div class="empty"><p class="eyebrow">Blob telemetry</p><h1>Reading field notes…</h1></div></main>`;
  }

  renderError(error) {
    this.shadowRoot.innerHTML = `<style>${styles}</style><main class="shell"><div class="error"><p class="eyebrow">Could not load telemetry</p><h2>${escapeHtml(error?.message || error)}</h2></div></main>`;
  }

  render() {
    const data = this._data;
    const run = data.run || {};
    const isVerified = this._verification?.verified === true;
    const latest = data.latest;
    const roles = Array.isArray(latest.cells_by_role) ? latest.cells_by_role.map(number) : [];
    const signals = Array.isArray(latest.signal_energy) ? latest.signal_energy.map(number) : [];
    const explorerCoverage = number(latest.explorer_coverage_bits);
    const explorerCoverageAverage = roles[1] ? explorerCoverage / roles[1] : 0;
    const outcomeSuccesses = Array.isArray(latest.outcome_successes) ? latest.outcome_successes.map(number) : [];
    const outcomeSetbacks = Array.isArray(latest.outcome_setbacks) ? latest.outcome_setbacks.map(number) : [];
    const outcomeContentions = Array.isArray(latest.outcome_contentions) ? latest.outcome_contentions.map(number) : [];
    const outcomeReliability = OUTCOME_NAMES.map((_, index) => {
      const successes = outcomeSuccesses[index] || 0;
      const attempts = successes + (outcomeSetbacks[index] || 0) + (outcomeContentions[index] || 0);
      return (successes + 1) / (attempts + 2);
    });
    const digestionAverage = number(latest.digestion_samples)
      ? number(latest.digestion_progress_weighted_sum) / number(latest.digestion_samples)
      : 0;
    const signalRetention = number(latest.signal_samples)
      ? number(latest.signal_retention_weighted_sum_q8) / number(latest.signal_samples) / 256
      : 0;
    const maxRole = Math.max(1, ...roles);
    const maxSignal = Math.max(1, ...signals);
    const checks = Array.isArray(data.adversarial_checks) ? data.adversarial_checks : [];
    const actions = data.actions || {};

    this.shadowRoot.innerHTML = `
      <style>${styles}</style>
      <main class="shell">
        <header class="masthead">
          <div>
            <div class="eyebrow">Tinker / simulation observatory</div>
            <h1>Blob field notes</h1>
            <p class="lede">Inspect colony behavior without crossing the Mind boundary. Private coordinates remain private; world reconciliation and trust status belong to the host.</p>
          </div>
          <div class="controls">
            <a class="control-link" href="?view=controls">Control matrix</a>
            <a class="control-link" href="?view=adjudication">Adjudication lab</a>
            <button id="sample" type="button">Load sample</button>
            <label class="upload-label">Open telemetry<input id="file" type="file" accept="application/json,.json" /></label>
          </div>
        </header>

        <section class="runbar" aria-label="Run identity">
          ${this.runItem("Run", run.id || "unnamed")}
          ${this.runItem("Mind", run.mind || "unknown")}
          ${this.runItem("Rules", run.ruleset || "unknown")}
          ${this.runItem("Board", run.board || "—")}
          ${this.runItem("Step", compact.format(number(run.step)))}
          <span class="trust ${isVerified ? "verified" : ""}">${isVerified ? "Server verified" : "Local · unverified"}</span>
        </section>

        <section class="summary" aria-label="Summary metrics">
          ${this.metric("Plant discovery", percent(number(latest.confirmed_mapped_plant_tiles), number(latest.world_plant_tiles)), `${number(latest.confirmed_mapped_plant_tiles)} of ${number(latest.world_plant_tiles)} canonical plant tiles`)}
          ${this.metric("Raised ring", percent(number(latest.elevated_ring_tiles), number(latest.plant_ring_tiles)), `${number(latest.elevated_ring_tiles)} elevated ring tiles`)}
          ${this.metric("Hard barrier", percent(number(latest.impassable_ring_tiles), number(latest.plant_ring_tiles)), `${Math.max(0, number(latest.plant_ring_tiles) - number(latest.impassable_ring_tiles))} relief / traversable tiles`)}
          ${this.metric("Defenders posted", compact.format(number(latest.defenders_on_station)), `${compact.format(number(latest.colony_cells))} colony cells observed`)}
        </section>

        <section class="grid">
          <article class="panel">
            <div class="panel-head"><h2>Colony trajectory</h2><span class="panel-note">discovery, barrier, staffing</span></div>
            <canvas class="chart" id="timeline" aria-label="Colony telemetry timeline"></canvas>
            <div class="legend">
              <span><i style="background:#a8d46f"></i>Plant discovery</span>
              <span><i style="background:#df735a"></i>Hard barrier</span>
              <span><i style="background:#6ab8aa"></i>Defender coverage</span>
            </div>
          </article>
          <article class="panel">
            <div class="panel-head"><h2>Role ecology</h2><span class="panel-note">private behavior markers</span></div>
            <div class="roles">
              ${ROLE_NAMES.map((name, index) => this.bar(name, roles[index] || 0, maxRole, compact.format(roles[index] || 0))).join("")}
            </div>
            <div class="coverage-memory"><span>Explorer rolling-window density</span><strong>${explorerCoverageAverage.toFixed(1)} / 64</strong></div>
            <div class="panel-head"><h2>Signal field</h2><span class="panel-note">energy by anonymous channel</span></div>
            <div class="signals">
              ${SIGNAL_NAMES.map((name, index) => this.bar(name, signals[index] || 0, maxSignal, compact.format(signals[index] || 0))).join("")}
            </div>
            <div class="panel-head"><h2>Learned reliability</h2><span class="panel-note">private outcome evidence</span></div>
            <div class="outcomes">
              ${OUTCOME_NAMES.map((name, index) => this.bar(name, outcomeReliability[index], 1, `${Math.round(outcomeReliability[index] * 100)}%`)).join("")}
            </div>
            <div class="coverage-memory"><span>Gut progress / decision interval</span><strong>${digestionAverage.toFixed(1)}</strong></div>
            <div class="coverage-memory"><span>Same-tile signal retention</span><strong>${Math.round(signalRetention * 100)}%</strong></div>
            <div class="coverage-memory"><span>Completed private wall plans</span><strong>${compact.format(number(latest.completed_wall_plan_records))}</strong></div>
            <div class="coverage-memory"><span>Excavation rotation pressure</span><strong>${compact.format(number(latest.excavation_pit_pressure))} · max ${compact.format(number(latest.maximum_excavation_pit_pressure))}</strong></div>
          </article>
        </section>

        <section class="lower">
          <article class="panel">
            <div class="panel-head"><h2>Plant perimeter</h2><span class="panel-note">host-reconstructed view</span></div>
            <div class="map-wrap">${this.renderMap(data.spatial)}<div class="map-key">
              <span><i class="swatch" style="background:#a8d46f"></i>plant</span>
              <span><i class="swatch" style="background:rgba(223,115,90,.45)"></i>hard wall</span>
              <span><i class="swatch" style="border:1px dashed #6ab8aa"></i>relief gate</span>
              <span><i class="swatch" style="background:#fff2c7"></i>defender</span>
            </div></div>
            <div class="action-strip">
              ${this.action("Damage dealt", actions.damage_dealt)}
              ${this.action("Terrain lifted", actions.terrain_lifted)}
              ${this.action("Terrain dumped", actions.terrain_dumped)}
              ${this.action("Failed actions", actions.failed)}
            </div>
          </article>
          <article class="panel">
            <div class="panel-head"><h2>Adversarial checks</h2><span class="panel-note">operational assumptions under pressure</span></div>
            <div class="checks">
              ${checks.map((check) => this.check(check)).join("") || '<p class="panel-note">No adversarial checks recorded.</p>'}
            </div>
          </article>
        </section>
      </main>`;

    this.shadowRoot.getElementById("file").addEventListener("change", (event) => this.readFile(event.target.files[0]));
    this.shadowRoot.getElementById("sample").addEventListener("click", () => this.load(this.getAttribute("src") || "./sample-telemetry.json"));
    const canvas = this.shadowRoot.getElementById("timeline");
    this._resizeObserver.disconnect();
    this._resizeObserver.observe(canvas);
    this.drawTimeline();
  }

  runItem(label, value) { return `<span class="run-item"><span class="label">${escapeHtml(label)}</span><span class="run-value">${escapeHtml(value)}</span></span>`; }
  metric(label, value, note) { return `<article class="metric"><div class="label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(value)}</div><div class="metric-note">${escapeHtml(note)}</div></article>`; }
  bar(label, value, maximum, display) { return `<div class="bar-row"><span>${escapeHtml(label)}</span><div class="track"><div class="fill" style="width:${Math.min(100, 100 * value / maximum)}%"></div></div><span class="bar-value">${escapeHtml(display)}</span></div>`; }
  action(label, value) { return `<div class="action"><span class="label">${escapeHtml(label)}</span><strong>${escapeHtml(compact.format(number(value)))}</strong></div>`; }

  check(check) {
    const status = ["passed", "warning", "exposed"].includes(check.status) ? check.status : "warning";
    return `<div class="check"><i class="dot ${status}"></i><div><div class="check-name">${escapeHtml(check.name)}</div><div class="check-detail">${escapeHtml(check.detail)}</div></div><span class="check-status">${escapeHtml(status)}</span></div>`;
  }

  renderMap(spatial = {}) {
    const ring = Array.isArray(spatial.ring) ? spatial.ring : [];
    const states = new Map(ring.map((tile) => [`${number(tile.x)},${number(tile.y)}`, tile]));
    let cells = "";
    for (let y = -2; y <= 2; y += 1) {
      for (let x = -2; x <= 2; x += 1) {
        const tile = states.get(`${x},${y}`);
        const classes = ["tile"];
        if (x === 0 && y === 0) classes.push("plant");
        if (Math.max(Math.abs(x), Math.abs(y)) === 1) classes.push("ring");
        if (tile?.status === "wall") classes.push("wall");
        if (tile?.status === "gate") classes.push("gate");
        if (tile?.occupant === "defender") classes.push("defender");
        cells += `<span class="${classes.join(" ")}" title="relative tile ${x}, ${y}${tile?.elevation != null ? `; elevation ${escapeHtml(tile.elevation)}` : ""}"></span>`;
      }
    }
    return `<div class="map" role="img" aria-label="Five by five plant perimeter map">${cells}</div>`;
  }

  drawTimeline() {
    const canvas = this.shadowRoot?.getElementById("timeline");
    const samples = Array.isArray(this._data?.timeline) ? this._data.timeline : [];
    if (!canvas || !samples.length) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
    const context = canvas.getContext("2d");
    context.scale(dpr, dpr);
    const width = rect.width;
    const height = rect.height;
    const margin = { left: 37, right: 18, top: 20, bottom: 28 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    context.font = "10px ui-monospace, monospace";
    context.fillStyle = "#829488";
    context.strokeStyle = "rgba(202,225,184,.10)";
    context.lineWidth = 1;
    for (let tick = 0; tick <= 4; tick += 1) {
      const y = margin.top + plotHeight * tick / 4;
      context.beginPath(); context.moveTo(margin.left, y); context.lineTo(width - margin.right, y); context.stroke();
      context.fillText(`${100 - tick * 25}%`, 5, y + 3);
    }
    const series = [
      ["plant_discovery", "#a8d46f"],
      ["barrier_coverage", "#df735a"],
      ["defender_coverage", "#6ab8aa"],
    ];
    for (const [key, color] of series) {
      context.beginPath();
      context.strokeStyle = color;
      context.lineWidth = 2;
      samples.forEach((sample, index) => {
        const x = margin.left + (samples.length === 1 ? 0 : plotWidth * index / (samples.length - 1));
        const y = margin.top + plotHeight * (1 - Math.max(0, Math.min(1, number(sample[key]))));
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.stroke();
    }
    context.fillStyle = "#829488";
    context.fillText(`step ${compact.format(number(samples[0].step))}`, margin.left, height - 8);
    const tail = `step ${compact.format(number(samples.at(-1).step))}`;
    context.fillText(tail, width - margin.right - context.measureText(tail).width, height - 8);
  }

  async readFile(file) {
    if (!file) return;
    try { this.setTelemetry(JSON.parse(await file.text())); }
    catch (error) { this.renderError(error); }
  }

  installDropTarget() {
    this.addEventListener("dragover", (event) => { event.preventDefault(); this.classList.add("drop-active"); });
    this.addEventListener("dragleave", () => this.classList.remove("drop-active"));
    this.addEventListener("drop", (event) => {
      event.preventDefault();
      this.classList.remove("drop-active");
      this.readFile(event.dataTransfer?.files?.[0]);
    });
  }
}

customElements.define("blob-telemetry-viewer", BlobTelemetryViewer);
