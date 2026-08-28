//! Opt-in canonical match recording for the local web explorer.
//!
//! Normal rollout environments never construct this recorder. An explicitly
//! recorded evaluation enables verified replay hashing, then converts each
//! canonical resolver delta into a compact presentation patch. Team ownership
//! remains a host-only sidecar and is never added to canonical cell state.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use blob_engine::engine::Engine;
use blob_engine::resolution::{
    ActionRequest, BatchReport, CellKey, CellState, OutcomeStatus, ReplayBundle, TileState,
};
use blob_interface::types::{CellId, TeamId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MATCH_EXPLORER_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct MatchExplorerConfig {
    pub run_id: String,
    pub title: String,
    pub ruleset: String,
    pub teams: BTreeMap<u64, MatchExplorerTeamSpec>,
}

#[derive(Debug, Clone)]
pub struct MatchExplorerTeamSpec {
    pub name: String,
    pub mind: String,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerBundle {
    pub schema_version: u32,
    pub run: MatchExplorerRun,
    pub presentation: MatchExplorerPresentation,
    pub teams: Vec<MatchExplorerTeam>,
    pub initial: MatchExplorerInitial,
    pub events: Vec<MatchExplorerEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerRun {
    pub id: String,
    pub title: String,
    pub board: MatchExplorerBoard,
    pub ruleset: String,
    pub compiled_ruleset_hash: String,
    pub outcome: String,
    pub end_reason: String,
    pub server_verified: bool,
    pub replay_committed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_replay: Option<MatchExplorerReplayBinding>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerBoard {
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerReplayBinding {
    pub format: String,
    pub file: String,
    pub sha256: String,
    pub events: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerPresentation {
    pub trust: String,
    pub team_assignment_source: String,
    pub note: String,
    pub cell_teams: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerTeam {
    pub id: u64,
    pub name: String,
    pub mind: String,
    pub color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerInitial {
    pub time: u64,
    pub tiles: Vec<MatchExplorerTile>,
    pub cells: Vec<MatchExplorerCell>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerTile {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    pub elevation: i16,
    pub plant: u64,
    pub plant_capacity: u64,
    pub plant_growth_rate: u64,
    pub loose: u64,
    pub diffuse: u64,
    pub signals: [u64; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerCell {
    pub key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alive: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub y: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub core: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assimilated: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gut: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub carried: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub escrow: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub marker: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guarded: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ready_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pending: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_outcome: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerEvent {
    pub sequence: String,
    pub time: u64,
    pub summary: String,
    pub note: String,
    pub actions: usize,
    #[serde(default)]
    pub resolved_actions: Vec<MatchExplorerAction>,
    pub outcomes: MatchExplorerOutcomes,
    pub births: Vec<String>,
    pub deaths: Vec<String>,
    pub tiles: Vec<MatchExplorerTile>,
    pub cells: Vec<MatchExplorerCell>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_hash: Option<String>,
}

/// Host-side presentation of an authenticated canonical outcome. Actor keys
/// route through `presentation.cell_teams`; team identity is never embedded in
/// canonical cell state or exposed to a Mind.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerAction {
    pub actor: String,
    pub family: String,
    pub status: String,
    pub started_at: u64,
    pub completed_at: u64,
    pub origin: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<usize>,
    pub effort_spent: u64,
    pub payload: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_amount: Option<u64>,
    pub consumed_energy: u64,
    /// Food present at the origin immediately before this resolution batch.
    /// This is an ecological opportunity signal, not a claim that every
    /// parameterized Consume choice was affordable.
    pub food_at_origin: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MatchExplorerOutcomes {
    pub completed: usize,
    pub contested: usize,
    pub frustrated: usize,
    pub rejected: usize,
    pub interrupted: usize,
}

pub struct MatchExplorerRecorder {
    bundle: MatchExplorerBundle,
    width: usize,
    tiles: Vec<TileState>,
}

pub struct RecordedExplorerMatch {
    pub bundle: MatchExplorerBundle,
    replay: ReplayBundle,
}

impl MatchExplorerRecorder {
    pub fn start(engine: &Engine, config: MatchExplorerConfig) -> Result<Self, String> {
        let simulation = engine
            .reference_simulation()
            .ok_or("match explorer recording requires initialized canonical state")?;
        let width = config_board_width(engine);
        let height = config_board_height(engine);
        let mut cell_teams = BTreeMap::new();
        for (cell_id, cell) in &engine.cells {
            cell_teams.insert(cell_key_string(*cell_id)?, team_key(cell.team_id)?);
        }
        for team in cell_teams.values() {
            if !config.teams.contains_key(team) {
                return Err(format!(
                    "host team {team} has no explorer presentation label"
                ));
            }
        }
        let mut teams = config
            .teams
            .into_iter()
            .map(|(id, team)| MatchExplorerTeam {
                id,
                name: team.name,
                mind: team.mind,
                color: team.color,
            })
            .collect::<Vec<_>>();
        teams.sort_unstable_by_key(|team| team.id);
        let cells = simulation
            .cells()
            .iter()
            .map(|(key, cell)| explorer_cell(*key, cell, width))
            .collect();
        let tiles = simulation
            .tiles()
            .iter()
            .map(|tile| explorer_tile(None, tile))
            .collect();
        Ok(Self {
            width,
            tiles: simulation.tiles().to_vec(),
            bundle: MatchExplorerBundle {
                schema_version: MATCH_EXPLORER_SCHEMA_VERSION,
                run: MatchExplorerRun {
                    id: config.run_id,
                    title: config.title,
                    board: MatchExplorerBoard { width, height },
                    ruleset: config.ruleset,
                    compiled_ruleset_hash: simulation.compiled_ruleset_hash().to_hex(),
                    outcome: "in_progress".into(),
                    end_reason: "in_progress".into(),
                    server_verified: false,
                    replay_committed: false,
                    canonical_replay: None,
                },
                presentation: MatchExplorerPresentation {
                    trust: "local_unverified".into(),
                    team_assignment_source: "host_post_match_sidecar".into(),
                    note: "Team assignments are host-only presentation data and were never visible to a Mind.".into(),
                    cell_teams,
                },
                teams,
                initial: MatchExplorerInitial {
                    time: simulation.now().0,
                    tiles,
                    cells,
                },
                events: Vec::new(),
            },
        })
    }

    pub fn observe(&mut self, report: &BatchReport) -> Result<(), String> {
        for (parent, child) in &report.births {
            let team = self
                .bundle
                .presentation
                .cell_teams
                .get(&parent.0.to_string())
                .copied()
                .ok_or_else(|| format!("birth parent {} has no host team assignment", parent.0))?;
            self.bundle
                .presentation
                .cell_teams
                .insert(child.0.to_string(), team);
        }
        let mut outcomes = MatchExplorerOutcomes::default();
        for outcome in &report.outcomes {
            match outcome.status {
                OutcomeStatus::Success => outcomes.completed += 1,
                OutcomeStatus::Contested => outcomes.contested += 1,
                OutcomeStatus::Frustrated => outcomes.frustrated += 1,
                OutcomeStatus::Rejected(_) => outcomes.rejected += 1,
                OutcomeStatus::Interrupted => outcomes.interrupted += 1,
            }
        }
        let action_count = report.outcomes.len();
        let resolved_actions = report
            .outcomes
            .iter()
            .map(|outcome| MatchExplorerAction {
                actor: outcome.actor.0.to_string(),
                family: format!("{:?}", outcome.action).to_lowercase(),
                status: outcome_status(outcome.status),
                started_at: outcome.started_at.0,
                completed_at: outcome.completed_at.0,
                origin: outcome.origin.0,
                target: outcome.target.map(|target| target.0),
                effort_spent: outcome.effort_spent,
                payload: outcome.payload,
                requested_amount: requested_amount(&outcome.request),
                consumed_energy: outcome.consumed_energy,
                food_at_origin: self
                    .tiles
                    .get(outcome.origin.0)
                    .is_some_and(|tile| tile.plant_energy > 0 || tile.loose_energy > 0),
            })
            .collect();
        let summary = if action_count == 0 {
            "Passive field update".into()
        } else {
            format!("{action_count} actions resolved")
        };
        let note = format!(
            "{} completed · {} contested · {} frustrated · {} rejected · {} interrupted",
            outcomes.completed,
            outcomes.contested,
            outcomes.frustrated,
            outcomes.rejected,
            outcomes.interrupted
        );
        self.bundle.events.push(MatchExplorerEvent {
            sequence: (self.bundle.events.len() + 1).to_string(),
            time: report.completed_at.0,
            summary,
            note,
            actions: action_count,
            resolved_actions,
            outcomes,
            births: report
                .births
                .iter()
                .map(|(_, child)| child.0.to_string())
                .collect(),
            deaths: report
                .deaths
                .iter()
                .map(|cell| cell.0.to_string())
                .collect(),
            tiles: report
                .delta
                .tiles
                .iter()
                .map(|delta| explorer_tile(Some(delta.tile.0), &delta.after))
                .collect(),
            cells: report
                .delta
                .cells
                .iter()
                .map(|delta| match delta.after.as_ref() {
                    Some(cell) => explorer_cell(delta.cell, cell, self.width),
                    None => removed_cell(delta.cell),
                })
                .collect(),
            state_hash: report.state_hash().map(|hash| hash.to_hex()),
        });
        for delta in &report.delta.tiles {
            if let Some(tile) = self.tiles.get_mut(delta.tile.0) {
                *tile = delta.after.clone();
            }
        }
        Ok(())
    }

    pub fn finish(
        mut self,
        replay: ReplayBundle,
        outcome: impl Into<String>,
        end_reason: impl Into<String>,
    ) -> RecordedExplorerMatch {
        self.bundle.run.outcome = outcome.into();
        self.bundle.run.end_reason = end_reason.into();
        RecordedExplorerMatch {
            bundle: self.bundle,
            replay,
        }
    }

    pub fn event_count(&self) -> usize {
        self.bundle.events.len()
    }
}

fn requested_amount(request: &ActionRequest) -> Option<u64> {
    match request {
        ActionRequest::Consume { amount } | ActionRequest::Regurgitate { amount, .. } => {
            Some(*amount)
        }
        ActionRequest::Attack { payload, .. } => Some(*payload),
        ActionRequest::Split {
            child_allocation, ..
        } => Some(*child_allocation),
        ActionRequest::Signal { amounts } => {
            Some(amounts.iter().copied().fold(0_u64, u64::saturating_add))
        }
        _ => None,
    }
}

fn outcome_status(status: OutcomeStatus) -> String {
    match status {
        OutcomeStatus::Success => "success".into(),
        OutcomeStatus::Contested => "contested".into(),
        OutcomeStatus::Frustrated => "frustrated".into(),
        OutcomeStatus::Interrupted => "interrupted".into(),
        OutcomeStatus::Rejected(reason) => format!("rejected:{reason:?}").to_lowercase(),
    }
}

impl RecordedExplorerMatch {
    pub fn publish(mut self, output: &Path) -> Result<(PathBuf, PathBuf), String> {
        let stem = output
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or("match explorer output needs a UTF-8 file stem")?;
        let replay_name = format!("{stem}.replay.bin");
        let replay_path = output.with_file_name(&replay_name);
        let replay_bytes = self.replay.to_bytes();
        let replay_hash = hex_sha256(replay_bytes);
        self.bundle.run.canonical_replay = Some(MatchExplorerReplayBinding {
            format: format!(
                "blob-reference-replay-bundle-v{}",
                blob_engine::resolution::REPLAY_FORMAT_VERSION
            ),
            file: replay_name,
            sha256: replay_hash,
            events: self.bundle.events.len(),
        });
        let json = serde_json::to_vec_pretty(&self.bundle)
            .map_err(|error| format!("could not encode match explorer bundle: {error}"))?;
        atomic_write(&replay_path, replay_bytes)?;
        atomic_write(output, &json)?;
        Ok((output.to_path_buf(), replay_path))
    }

    pub fn replay_event_count(&self) -> usize {
        self.replay.archive().event_count()
    }
}

fn explorer_tile(index: Option<usize>, tile: &TileState) -> MatchExplorerTile {
    MatchExplorerTile {
        index,
        elevation: tile.elevation,
        plant: tile.plant_energy,
        plant_capacity: tile.plant_capacity,
        plant_growth_rate: tile.plant_growth_rate,
        loose: tile.loose_energy,
        diffuse: tile.diffuse_energy,
        signals: tile.signal_energy,
    }
}

fn explorer_cell(key: CellKey, cell: &CellState, width: usize) -> MatchExplorerCell {
    let escrow = cell
        .pending_action
        .as_ref()
        .map_or(0, |pending| pending.payload_escrow);
    MatchExplorerCell {
        key: key.0.to_string(),
        alive: None,
        x: Some(cell.position.0 % width),
        y: Some(cell.position.0 / width),
        core: Some(cell.core_mass),
        assimilated: Some(cell.assimilated_energy),
        gut: Some(cell.gut_energy),
        carried: Some(cell.carried_material_mass),
        escrow: Some(escrow),
        marker: Some(cell.marker),
        guarded: Some(cell.guarded),
        ready_at: Some(cell.ready_at.0),
        pending: Some(cell.pending_action.as_ref().map_or_else(
            || "ready".into(),
            |pending| format!("{:?}", pending.request).to_lowercase(),
        )),
        last_outcome: cell
            .last_outcome
            .as_ref()
            .map(|outcome| format!("{outcome:?}").to_lowercase()),
    }
}

fn removed_cell(key: CellKey) -> MatchExplorerCell {
    MatchExplorerCell {
        key: key.0.to_string(),
        alive: Some(false),
        x: None,
        y: None,
        core: None,
        assimilated: None,
        gut: None,
        carried: None,
        escrow: None,
        marker: None,
        guarded: None,
        ready_at: None,
        pending: None,
        last_outcome: None,
    }
}

fn config_board_width(engine: &Engine) -> usize {
    engine.world.dimensions.0
}

fn config_board_height(engine: &Engine) -> usize {
    engine.world.dimensions.1
}

fn cell_key_string(cell: CellId) -> Result<String, String> {
    u64::try_from(cell.0)
        .map(|key| key.to_string())
        .map_err(|_| "host cell ID does not fit a canonical key".into())
}

fn team_key(team: TeamId) -> Result<u64, String> {
    u64::try_from(team.0).map_err(|_| "host team ID does not fit the explorer sidecar".into())
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("output")
    ));
    fs::write(&temporary, bytes)
        .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path)
        .map_err(|error| format!("could not publish {}: {error}", path.display()))
}
