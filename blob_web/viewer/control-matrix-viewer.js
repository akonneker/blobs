const PROFILE_ORDER = ["simple", "aggressive", "defensive", "explorer"];

const styles = String.raw`
  :host {
    --ink: #ecf4dd; --muted: #92a68f; --panel: rgba(18,32,27,.9);
    --line: rgba(202,225,184,.14); --moss: #a8d46f; --amber: #f2b84b;
    --clay: #df735a; --water: #6ab8aa;
    display:block; min-height:100vh; color:var(--ink);
    background:radial-gradient(circle at 10% 4%,rgba(105,151,76,.2),transparent 28rem),radial-gradient(circle at 90% 12%,rgba(75,132,122,.15),transparent 31rem),#09120f;
    font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  }
  * { box-sizing:border-box; } button,input { font:inherit; }
  .shell { width:min(1480px,100%); margin:0 auto; padding:28px clamp(18px,4vw,56px) 52px; }
  .masthead { display:flex; justify-content:space-between; align-items:flex-start; gap:24px; padding-bottom:22px; border-bottom:1px solid var(--line); }
  .eyebrow,.label,th { color:var(--muted); font:700 11px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.14em; text-transform:uppercase; }
  h1 { margin:5px 0 0; font:500 clamp(34px,5vw,66px)/.95 Georgia,"Times New Roman",serif; letter-spacing:-.04em; }
  h2 { margin:0; font:500 20px/1.2 Georgia,"Times New Roman",serif; }
  .lede { max-width:650px; margin:13px 0 0; color:var(--muted); font-size:14px; line-height:1.6; }
  .controls { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:9px; }
  button,.upload,.nav { display:inline-flex; align-items:center; justify-content:center; min-height:40px; padding:0 15px; border:1px solid var(--line); border-radius:999px; color:var(--ink); background:rgba(255,255,255,.035); cursor:pointer; text-decoration:none; }
  button:hover,.upload:hover,.nav:hover { border-color:rgba(168,212,111,.5); background:rgba(168,212,111,.08); }
  button:focus-visible,.upload:focus-within,.nav:focus-visible { outline:2px solid var(--moss); outline-offset:3px; }
  input[type=file] { position:absolute; width:1px; height:1px; opacity:0; }
  .runbar { display:flex; flex-wrap:wrap; gap:12px 26px; align-items:center; padding:18px 0; }
  .run-item { display:flex; gap:8px; align-items:baseline; font-size:13px; }
  .trust { margin-left:auto; padding:7px 11px; border-radius:4px; color:#ffd585; background:rgba(242,184,75,.12); font:700 11px/1 ui-monospace,monospace; letter-spacing:.05em; text-transform:uppercase; }
  .summary { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); border-radius:14px; overflow:hidden; background:rgba(255,255,255,.02); }
  .metric { min-height:126px; padding:21px 22px; border-right:1px solid var(--line); }
  .metric:last-child { border:0; }
  .metric-value { margin-top:18px; font:500 34px/1 Georgia,serif; }
  .metric-note { margin-top:8px; color:var(--muted); font-size:12px; line-height:1.4; }
  .progress { height:4px; margin-top:16px; overflow:hidden; background:rgba(255,255,255,.05); border-radius:99px; }
  .progress > i { display:block; height:100%; background:linear-gradient(90deg,var(--water),var(--moss)); transition:width .2s ease; }
  .profiles { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin-top:16px; }
  .profile,.panel { border:1px solid var(--line); border-radius:14px; background:var(--panel); overflow:hidden; }
  .profile { padding:19px 20px; } .profile h2 { text-transform:capitalize; }
  .record { margin-top:17px; font:500 27px/1 Georgia,serif; } .record small { color:var(--muted); font:11px/1 ui-monospace,monospace; }
  .stack { display:flex; height:7px; margin-top:14px; overflow:hidden; border-radius:99px; background:rgba(255,255,255,.05); }
  .stack i:nth-child(1) { background:var(--moss); } .stack i:nth-child(2) { background:var(--clay); } .stack i:nth-child(3) { background:var(--amber); }
  .profile-note { margin-top:10px; color:var(--muted); font-size:11px; }
  .panel { margin-top:16px; }
  .panel-head { display:flex; justify-content:space-between; gap:20px; align-items:baseline; padding:19px 20px; border-bottom:1px solid var(--line); }
  .panel-note { color:var(--muted); font-size:11px; }
  .table-wrap { overflow:auto; } table { width:100%; border-collapse:collapse; min-width:880px; }
  th { padding:12px 16px; text-align:left; background:rgba(255,255,255,.018); }
  td { padding:13px 16px; border-top:1px solid var(--line); font-size:12px; vertical-align:middle; }
  tbody tr:hover { background:rgba(168,212,111,.035); }
  .opponent { text-transform:capitalize; } .rate { font:500 18px Georgia,serif; } .seat { color:var(--muted); font:11px/1.55 ui-monospace,monospace; }
  .status-dot { display:inline-block; width:7px; height:7px; margin-right:7px; border-radius:50%; background:var(--moss); }
  .status-dot.live { background:var(--amber); box-shadow:0 0 0 4px rgba(242,184,75,.08); }
  .trust-note { display:grid; grid-template-columns:auto 1fr; gap:13px; margin:16px 0 0; padding:17px 19px; border:1px solid rgba(242,184,75,.22); border-radius:12px; color:var(--muted); font-size:12px; line-height:1.55; background:rgba(242,184,75,.045); }
  .trust-note strong { color:#ffd585; }
  .empty,.error { margin:72px auto; max-width:560px; padding:30px; text-align:center; } .error { color:#ffd1c6; }
  :host(.drop-active) { outline:2px dashed var(--moss); outline-offset:-10px; }
  @media(max-width:1000px) { .profiles { grid-template-columns:repeat(2,1fr); } }
  @media(max-width:850px) { .masthead { flex-direction:column; } .controls { justify-content:flex-start; } .summary { grid-template-columns:repeat(2,1fr); } .metric:nth-child(2) { border-right:0; } .metric:nth-child(-n+2) { border-bottom:1px solid var(--line); } }
  @media(max-width:560px) { .shell { padding-inline:13px; } .summary,.profiles { grid-template-columns:1fr; } .metric { min-height:108px; border-right:0; border-bottom:1px solid var(--line); } .metric:last-child { border-bottom:0; } .trust { width:100%; margin-left:0; } }
`;

const escapeHtml = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
const numeric = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });
const pct = (value) => `${(100 * numeric(value)).toFixed(1)}%`;
const record = (episodes) => {
  const values = { wins: 0, losses: 0, timeouts: 0 };
  for (const episode of episodes || []) {
    if (episode.outcome === "candidate_win") values.wins += 1;
    else if (episode.outcome === "candidate_loss") values.losses += 1;
    else values.timeouts += 1;
  }
  return values;
};
const recordText = (value) => `${value.wins}-${value.losses}-${value.timeouts}`;
const CONTROL_MATRIX_SCHEMA_VERSION = 6;
const SUPPORTED_CONTROL_MATRIX_SCHEMA_VERSIONS = new Set([2, 3, 4, 5, CONTROL_MATRIX_SCHEMA_VERSION]);

class BlobControlMatrixViewer extends HTMLElement {
  static observedAttributes = ["src"];
  constructor() { super(); this.attachShadow({ mode: "open" }); this._data = null; this._source = null; this._pollTimer = null; }
  connectedCallback() { this.installDropTarget(); this.renderLoading(); if (this.getAttribute("src")) this.load(this.getAttribute("src")); }
  disconnectedCallback() { this.stopPolling(); }
  attributeChangedCallback(name, oldValue, newValue) { if (name === "src" && oldValue !== newValue && this.isConnected && newValue) this.load(newValue); }
  set data(value) { this.setReport(value); } get data() { return this._data; }

  async load(url, quiet = false) {
    this.stopPolling(); this._source = url;
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) throw new Error(`Comparison report request failed (${response.status})`);
      this.commitReport(await response.json());
    } catch (error) { if (!quiet) this.renderError(error); else this.schedulePoll(); }
  }

  setReport(value) { this.stopPolling(); this._source = null; this.commitReport(value); }
  commitReport(value) {
    const isFinal = SUPPORTED_CONTROL_MATRIX_SCHEMA_VERSIONS.has(numeric(value?.schema_version)) && Array.isArray(value?.matchups) && Array.isArray(value?.summaries);
    const isProgress = numeric(value?.schema_version) === 1 && value?.report_kind === "blob_control_matrix_progress" && SUPPORTED_CONTROL_MATRIX_SCHEMA_VERSIONS.has(numeric(value?.control_matrix_schema_version)) && Array.isArray(value?.matchups) && Array.isArray(value?.summaries);
    if (!isFinal && !isProgress) throw new Error(`Expected a supported control-matrix report through schema ${CONTROL_MATRIX_SCHEMA_VERSION} or progress schema 1 snapshot`);
    if (value.server_verified === true || value.replay_committed === true) throw new Error("This local comparison viewer does not accept trusted-status claims");
    this._data = value; this.render();
    this.dispatchEvent(new CustomEvent("blob-control-matrix-load", { detail: { data: value } }));
    if (isProgress && value.status === "running" && this._source) this.schedulePoll();
  }
  schedulePoll() { const delay = Math.max(500, numeric(this.getAttribute("poll-interval")) || 2000); this._pollTimer = window.setTimeout(() => this.load(this._source, true), delay); }
  stopPolling() { if (this._pollTimer != null) window.clearTimeout(this._pollTimer); this._pollTimer = null; }
  renderLoading() { this.shadowRoot.innerHTML = `<style>${styles}</style><main class="shell"><div class="empty"><p class="eyebrow">Maintained-Mind controls</p><h1>Reading comparison matrix…</h1></div></main>`; }
  renderError(error) { this.shadowRoot.innerHTML = `<style>${styles}</style><main class="shell"><div class="error"><p class="eyebrow">Could not load comparison report</p><h2>${escapeHtml(error?.message || error)}</h2><p>Generate a matrix or choose a recorded JSON file.</p></div></main>`; }

  render() {
    const data = this._data;
    const isProgress = data.report_kind === "blob_control_matrix_progress";
    const status = isProgress ? data.status : "complete";
    const matchups = data.matchups || []; const summaries = data.summaries || [];
    const completedMatchups = isProgress ? numeric(data.completed_matchups) : matchups.length;
    const totalMatchups = isProgress ? numeric(data.total_matchups) : matchups.length;
    const completedEpisodes = isProgress ? numeric(data.completed_episodes) : summaries.reduce((sum, item) => sum + numeric(item.episodes), 0);
    const totalEpisodes = isProgress ? numeric(data.total_episodes) : completedEpisodes;
    const totalWins = summaries.reduce((sum, item) => sum + numeric(item.colony_wins), 0);
    const totalLosses = summaries.reduce((sum, item) => sum + numeric(item.colony_losses), 0);
    const totalTimeouts = summaries.reduce((sum, item) => sum + numeric(item.timeouts), 0);
    const decisive = totalWins + totalLosses; const seats = this.seatTotals(matchups);
    const deadline = this.deadlineEvidence(matchups);
    const horizon = matchups.find((item) => item.report?.scenario?.victory)?.report?.scenario?.victory?.sim_time_limit_quanta;
    const seatZeroRate = seats.zero.wins / Math.max(1, seats.zero.wins + seats.zero.losses + seats.zero.timeouts);
    const seatOneRate = seats.one.wins / Math.max(1, seats.one.wins + seats.one.losses + seats.one.timeouts);
    const progress = totalMatchups ? completedMatchups / totalMatchups : 0;
    this.shadowRoot.innerHTML = `<style>${styles}</style><main class="shell">
      <header class="masthead"><div><div class="eyebrow">Tinker / simulation observatory</div><h1>Control matrix</h1><p class="lede">Compare the colony against maintained proxy Minds across paired worlds. Every seed is mirrored between both placement seats; all normalization and aggregation stays in the host.</p></div>
      <div class="controls"><a class="nav" href="?view=telemetry">Colony telemetry</a><a class="nav" href="?view=adjudication">Adjudication lab</a><button id="recorded" type="button">Recorded</button><button id="live" type="button">Live</button><button id="refresh" type="button">Refresh</button><label class="upload">Open JSON<input id="file" type="file" accept="application/json,.json" /></label></div></header>
      <section class="runbar" aria-label="Matrix identity">${this.runItem("State", `<i class="status-dot ${status === "running" ? "live" : ""}"></i>${escapeHtml(status)}`)}${this.runItem("Matrix", escapeHtml(String(data.matrix_config_hash || "unknown").slice(0, 12)))}${this.runItem("ABI", escapeHtml(data.mind_abi_version ?? "—"))}${this.runItem("Seeds", escapeHtml((data.seeds || []).length))}${this.runItem("Seats", "mirrored")}${horizon ? this.runItem("Event deadline", compact.format(horizon)) : ""}<span class="trust">Local · unverified</span></section>
      <section class="summary" aria-label="Comparison summary">${this.metric("Progress", `${completedMatchups} / ${totalMatchups}`, `${completedEpisodes} of ${totalEpisodes} episodes`, progress)}${this.metric("Colony wins", compact.format(totalWins), `${totalLosses} losses · ${totalTimeouts} timeouts`)}${this.metric("Decisive win rate", decisive ? pct(totalWins / decisive) : "—", "timeouts excluded from this display")}${this.metric("Seat gap", `${Math.abs(100 * (seatOneRate - seatZeroRate)).toFixed(1)} pp`, `team 0 ${pct(seatZeroRate)} · team 1 ${pct(seatOneRate)}`)}</section>
      <section class="profiles" aria-label="Opponent summaries">${PROFILE_ORDER.map((profile) => this.profile(profile, summaries.find((item) => item.opponent === profile))).join("")}</section>
      <section class="panel"><div class="panel-head"><h2>Regime breakdown</h2><span class="panel-note">W–L–T · normalized to colony side</span></div><div class="table-wrap"><table><thead><tr><th>Regime</th><th>Opponent</th><th>Record</th><th>Win rate</th><th>Mean steps</th><th>Placement seats</th></tr></thead><tbody>${matchups.map((item) => this.matchupRow(item)).join("") || '<tr><td colspan="6">Waiting for the first completed matchup…</td></tr>'}</tbody></table></div></section>
      ${deadline ? `<aside class="trust-note"><strong>Raw deadline evidence</strong><span>${deadline.count} event-time draws. Mean terminal colony/control compartments: core ${deadline.colony.core_mass.toFixed(1)}/${deadline.control.core_mass.toFixed(1)}, assimilated ${deadline.colony.assimilated_energy.toFixed(1)}/${deadline.control.assimilated_energy.toFixed(1)}, gut ${deadline.colony.gut_energy.toFixed(1)}/${deadline.control.gut_energy.toFixed(1)}, carried ${deadline.colony.carried_material_mass.toFixed(1)}/${deadline.control.carried_material_mass.toFixed(1)}, escrow ${deadline.colony.payload_escrow.toFixed(1)}/${deadline.control.payload_escrow.toFixed(1)}. No biomass weighting or gut discount has been imposed.</span></aside>` : ""}
      ${data.error ? `<aside class="trust-note"><strong>Run failed</strong><span>${escapeHtml(data.error)}</span></aside>` : ""}
      <aside class="trust-note"><strong>Not leaderboard evidence</strong><span>This report is deterministic local evaluation, not server execution of submitted Wasm. Browser loading, live polling, and an immutable local matrix cannot set trusted status. A future verifier must bind admitted artifact hashes, private server randomness, canonical replay commitments, and a server signature.</span></aside>
    </main>`;
    this.shadowRoot.getElementById("file").addEventListener("change", (event) => this.readFile(event.target.files[0]));
    this.shadowRoot.getElementById("refresh").addEventListener("click", () => this._source && this.load(this._source));
    this.shadowRoot.getElementById("recorded").addEventListener("click", () => this.load(this.getAttribute("recorded-src") || this.getAttribute("src")));
    this.shadowRoot.getElementById("live").addEventListener("click", () => this.load(this.getAttribute("live-src") || this.getAttribute("src")));
  }
  runItem(label, value) { return `<span class="run-item"><span class="label">${escapeHtml(label)}</span><span>${value}</span></span>`; }
  metric(label, value, note, progress = null) { return `<article class="metric"><div class="label">${escapeHtml(label)}</div><div class="metric-value">${escapeHtml(value)}</div><div class="metric-note">${escapeHtml(note)}</div>${progress == null ? "" : `<div class="progress"><i style="width:${Math.max(0, Math.min(100, progress * 100))}%"></i></div>`}</article>`; }
  profile(name, summary = {}) {
    const wins = numeric(summary.colony_wins); const losses = numeric(summary.colony_losses); const timeouts = numeric(summary.timeouts); const episodes = Math.max(1, numeric(summary.episodes));
    return `<article class="profile"><div class="label">Colony versus</div><h2>${escapeHtml(name)}</h2><div class="record">${wins}–${losses}–${timeouts} <small>W–L–T</small></div><div class="stack"><i style="width:${100 * wins / episodes}%"></i><i style="width:${100 * losses / episodes}%"></i><i style="width:${100 * timeouts / episodes}%"></i></div><div class="profile-note">${numeric(summary.episodes) ? `${pct(summary.colony_win_rate)} raw win rate` : "pending"}</div></article>`;
  }
  matchupRow(item) {
    const report = item.report || item; const aggregate = report.aggregate || {}; const episodes = report.episodes || [];
    const compactSeat = (seat) => { const value = (item.seats || []).find((entry) => entry.seat === seat); return value ? { wins: numeric(value.colony_wins), losses: numeric(value.colony_losses), timeouts: numeric(value.timeouts) } : null; };
    const zero = compactSeat("colony_team_zero") || record(episodes.filter((episode) => episode.seat === "colony_team_zero")); const one = compactSeat("colony_team_one") || record(episodes.filter((episode) => episode.seat === "colony_team_one"));
    return `<tr><td>${escapeHtml(item.variant)}</td><td class="opponent">${escapeHtml(report.opponent)}</td><td><strong>${numeric(aggregate.colony_wins)}–${numeric(aggregate.colony_losses)}–${numeric(aggregate.timeouts)}</strong></td><td class="rate">${pct(aggregate.colony_win_rate)}</td><td>${compact.format(numeric(aggregate.average_environment_steps))}</td><td class="seat">T0 ${recordText(zero)}<br>T1 ${recordText(one)}</td></tr>`;
  }
  seatTotals(matchups) {
    const totals = { zero: { wins: 0, losses: 0, timeouts: 0 }, one: { wins: 0, losses: 0, timeouts: 0 } };
    for (const matchup of matchups) {
      if (Array.isArray(matchup.seats)) {
        for (const seat of matchup.seats) { const target = seat.seat === "colony_team_one" ? totals.one : totals.zero; target.wins += numeric(seat.colony_wins); target.losses += numeric(seat.colony_losses); target.timeouts += numeric(seat.timeouts); }
      } else {
        for (const episode of matchup.report?.episodes || []) { const target = episode.seat === "colony_team_one" ? totals.one : totals.zero; if (episode.outcome === "candidate_win") target.wins += 1; else if (episode.outcome === "candidate_loss") target.losses += 1; else target.timeouts += 1; }
      }
    }
    return totals;
  }
  deadlineEvidence(matchups) {
    const fields = ["core_mass", "assimilated_energy", "gut_energy", "carried_material_mass", "payload_escrow"];
    const result = { count: 0, colony: {}, control: {} }; for (const field of fields) { result.colony[field] = 0; result.control[field] = 0; }
    for (const matchup of matchups) for (const episode of matchup.report?.episodes || []) {
      if (episode.end_reason !== "sim_time_deadline" || !episode.terminal_colony_energy || !episode.terminal_control_energy) continue;
      result.count += 1; for (const field of fields) { result.colony[field] += numeric(episode.terminal_colony_energy[field]); result.control[field] += numeric(episode.terminal_control_energy[field]); }
    }
    if (!result.count) return null;
    for (const field of fields) { result.colony[field] /= result.count; result.control[field] /= result.count; }
    return result;
  }
  async readFile(file) { if (!file) return; try { this.setReport(JSON.parse(await file.text())); } catch (error) { this.renderError(error); } }
  installDropTarget() {
    this.addEventListener("dragover", (event) => { event.preventDefault(); this.classList.add("drop-active"); });
    this.addEventListener("dragleave", () => this.classList.remove("drop-active"));
    this.addEventListener("drop", (event) => { event.preventDefault(); this.classList.remove("drop-active"); this.readFile(event.dataTransfer?.files?.[0]); });
  }
}

customElements.define("blob-control-matrix-viewer", BlobControlMatrixViewer);
