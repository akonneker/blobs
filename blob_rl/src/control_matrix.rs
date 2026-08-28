//! Deterministic colony-versus-maintained-control evaluation.
//!
//! This is intentionally separate from the RL curriculum's synthetic
//! `OpponentProfile` baselines. Each profile below calls the exact native entry
//! point used to build its maintained Wasm artifact. Published reports remain
//! explicitly local/unverified until an online verifier re-executes submitted
//! Wasm and commits a replay.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use blob_interface::reference_mind::{ReferenceMind, ReferenceMindDecision, ReferenceMindInput};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::{EnvConfig, RewardConfig, ScenarioProfile, TrainingConfig};
use crate::env::{BlobEnv, EpisodeEndReason, EpisodeOutcome, OpponentMindFactory};
use crate::sweep_execution::load_validated_sweep;
use crate::telemetry::{
    CellEnergyCompartments, EcologySample, StepTelemetry, TelemetryConfig, TelemetryEpisodeOutcome,
    TrainingTelemetryState, TrainingTelemetrySummary,
};
use crate::viability::{mind_abi_hash, ViabilityOutcome};

pub const CONTROL_MATRIX_SCHEMA_VERSION: u32 = 6;
pub const CONTROL_MATRIX_PROGRESS_SCHEMA_VERSION: u32 = 1;
const MAX_CONTROL_MATRIX_BYTES: u64 = 64 * 1024 * 1024;
static CONTROL_MATRIX_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);
type ControlProgressObserver<'a> =
    dyn Fn(&ControlMatrixProgressReport) -> Result<(), String> + Sync + 'a;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum MaintainedMindProfile {
    Simple,
    Aggressive,
    Defensive,
    Explorer,
    Colony,
    ColonySignalDisabled,
    ColonySignalOneQuantum,
    ColonySignalSidecar,
    ColonySignalCadenced,
}

impl MaintainedMindProfile {
    pub const CONTROLS: [Self; 4] = [
        Self::Simple,
        Self::Aggressive,
        Self::Defensive,
        Self::Explorer,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Aggressive => "aggressive",
            Self::Defensive => "defensive",
            Self::Explorer => "explorer",
            Self::Colony => "colony",
            Self::ColonySignalDisabled => "colony_signal_disabled",
            Self::ColonySignalOneQuantum => "colony_signal_one_quantum",
            Self::ColonySignalSidecar => "colony_signal_sidecar",
            Self::ColonySignalCadenced => "colony_signal_cadenced",
        }
    }

    pub const fn is_colony_candidate(self) -> bool {
        matches!(
            self,
            Self::Colony
                | Self::ColonySignalDisabled
                | Self::ColonySignalOneQuantum
                | Self::ColonySignalSidecar
                | Self::ColonySignalCadenced
        )
    }

    pub fn decide(self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        match self {
            Self::Simple => simple_mind::decide(input),
            Self::Aggressive => aggressive_mind::decide(input),
            Self::Defensive => defensive_mind::decide(input),
            Self::Explorer => explorer_mind::decide(input),
            Self::Colony => colony_mind::decide(input),
            Self::ColonySignalDisabled => {
                colony_mind::decide_with_signal_policy(input, colony_mind::SignalPolicy::Disabled)
            }
            Self::ColonySignalOneQuantum => colony_mind::decide_with_signal_policy(
                input,
                colony_mind::SignalPolicy::OneQuantumSidecar,
            ),
            Self::ColonySignalSidecar => colony_mind::decide_with_signal_policy(
                input,
                colony_mind::SignalPolicy::SemanticSidecar,
            ),
            Self::ColonySignalCadenced => colony_mind::decide_with_signal_policy(
                input,
                colony_mind::SignalPolicy::CadencedSemanticSidecar,
            ),
        }
    }
}

impl std::fmt::Display for MaintainedMindProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

struct MaintainedMind {
    profile: MaintainedMindProfile,
}

impl MaintainedMind {
    const fn new(profile: MaintainedMindProfile) -> Self {
        Self { profile }
    }
}

impl ReferenceMind for MaintainedMind {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        self.profile.decide(input)
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlExecutionMode {
    NativeMaintainedEntryPoint,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlVerificationScope {
    LocalDeterministic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlMatrixProgressStatus {
    Running,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ControlSeatPolicy {
    MirroredTeamZeroAndOne,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ControlSeat {
    ColonyTeamZero,
    ColonyTeamOne,
}

impl ControlSeat {
    const MIRRORED: [Self; 2] = [Self::ColonyTeamZero, Self::ColonyTeamOne];

    const fn colony_is_team_zero(self) -> bool {
        matches!(self, Self::ColonyTeamZero)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ControlEpisode {
    pub seed: u64,
    pub seat: ControlSeat,
    pub outcome: ViabilityOutcome,
    pub end_reason: EpisodeEndReason,
    pub environment_steps: u64,
    pub final_sim_time_quanta: u64,
    pub colony_cells: usize,
    pub control_cells: usize,
    pub initial_tracked_mass_energy: u128,
    pub final_tracked_mass_energy: u128,
    pub terminal_colony_energy: CellEnergyCompartments,
    pub terminal_control_energy: CellEnergyCompartments,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlAggregate {
    pub episodes: usize,
    pub colony_wins: usize,
    pub colony_losses: usize,
    pub timeouts: usize,
    pub colony_win_rate: f64,
    pub average_environment_steps: f64,
    pub average_final_colony_cells: f64,
    pub average_final_control_cells: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlMatchupReport {
    pub matchup_config_hash: String,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub scenario: ScenarioProfile,
    pub candidate: MaintainedMindProfile,
    pub opponent: MaintainedMindProfile,
    pub seat_policy: ControlSeatPolicy,
    pub seeds: Vec<u64>,
    pub telemetry_config: TelemetryConfig,
    pub episodes: Vec<ControlEpisode>,
    pub aggregate: ControlAggregate,
    pub telemetry: TrainingTelemetrySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ControlMatrixVariant {
    pub name: String,
    pub description: Option<String>,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlMatrixMatchup {
    pub variant: String,
    pub report: ControlMatchupReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlProfileSummary {
    pub opponent: MaintainedMindProfile,
    pub episodes: usize,
    pub colony_wins: usize,
    pub colony_losses: usize,
    pub timeouts: usize,
    pub colony_win_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlMatrixReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub mind_abi_version: u16,
    pub mind_abi_hash: String,
    pub manifest_sha256: String,
    pub matrix_config_hash: String,
    pub execution_mode: ControlExecutionMode,
    pub verification_scope: ControlVerificationScope,
    pub server_verified: bool,
    pub replay_committed: bool,
    pub seat_policy: ControlSeatPolicy,
    pub candidate: MaintainedMindProfile,
    pub opponents: Vec<MaintainedMindProfile>,
    pub seeds: Vec<u64>,
    pub variants: Vec<ControlMatrixVariant>,
    pub matchups: Vec<ControlMatrixMatchup>,
    pub summaries: Vec<ControlProfileSummary>,
}

/// Mutable local status for UI polling while an immutable matrix is built.
/// This is never a leaderboard or attestation artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlMatrixProgressReport {
    pub schema_version: u32,
    pub report_kind: String,
    pub control_matrix_schema_version: u32,
    pub status: ControlMatrixProgressStatus,
    pub error: Option<String>,
    pub mind_abi_version: u16,
    pub mind_abi_hash: String,
    pub manifest_sha256: String,
    pub matrix_config_hash: String,
    pub execution_mode: ControlExecutionMode,
    pub verification_scope: ControlVerificationScope,
    pub server_verified: bool,
    pub replay_committed: bool,
    pub seat_policy: ControlSeatPolicy,
    pub candidate: MaintainedMindProfile,
    pub opponents: Vec<MaintainedMindProfile>,
    pub seeds: Vec<u64>,
    pub variants: Vec<ControlMatrixVariant>,
    pub completed_matchups: usize,
    pub total_matchups: usize,
    pub completed_episodes: usize,
    pub total_episodes: usize,
    pub matchups: Vec<ControlMatrixProgressMatchup>,
    pub summaries: Vec<ControlProfileSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ControlSeatAggregate {
    pub seat: ControlSeat,
    pub episodes: usize,
    pub colony_wins: usize,
    pub colony_losses: usize,
    pub timeouts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlMatrixProgressMatchup {
    pub variant: String,
    pub opponent: MaintainedMindProfile,
    pub aggregate: ControlAggregate,
    pub seats: Vec<ControlSeatAggregate>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlMatrixOptions {
    pub candidate: MaintainedMindProfile,
    pub opponents: Vec<MaintainedMindProfile>,
    pub max_parallel: usize,
}

#[derive(Serialize)]
struct MatchupIdentity<'a> {
    mind_abi_version: u16,
    mind_abi_hash: &'a str,
    semantic_ruleset_hash: &'a str,
    compiled_ruleset_hash: &'a str,
    scenario_hash: &'a str,
    candidate: MaintainedMindProfile,
    opponent: MaintainedMindProfile,
    seat_policy: ControlSeatPolicy,
    seeds: &'a [u64],
    telemetry_config: &'a TelemetryConfig,
}

#[derive(Serialize)]
struct MatrixIdentity<'a> {
    mind_abi_version: u16,
    mind_abi_hash: &'a str,
    manifest_sha256: &'a str,
    execution_mode: ControlExecutionMode,
    candidate: MaintainedMindProfile,
    seat_policy: ControlSeatPolicy,
    opponents: &'a [MaintainedMindProfile],
}

#[derive(Clone)]
struct MatrixJob {
    variant: String,
    config: TrainingConfig,
    semantic_ruleset_hash: String,
    compiled_ruleset_hash: String,
    scenario_hash: String,
    opponent: MaintainedMindProfile,
}

fn zero_reward() -> RewardConfig {
    RewardConfig {
        survive_tick: 0.0,
        eat_energy: 0.0,
        damage_enemy: 0.0,
        kill_enemy: 0.0,
        cell_died: 0.0,
        split_success: 0.0,
        team_wins: 0.0,
        team_loses: 0.0,
        move_toward_food: 0.0,
        move_toward_enemy: 0.0,
        proximity_search_radius: 0,
        deadline: Default::default(),
    }
}

fn hash_json(value: &impl Serialize) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| crate::sweep::sha256(&bytes))
        .map_err(|error| format!("failed to encode control-matrix identity: {error}"))
}

fn aggregate_episodes(episodes: &[ControlEpisode]) -> ControlAggregate {
    let colony_wins = episodes
        .iter()
        .filter(|episode| episode.outcome == ViabilityOutcome::CandidateWin)
        .count();
    let colony_losses = episodes
        .iter()
        .filter(|episode| episode.outcome == ViabilityOutcome::CandidateLoss)
        .count();
    let timeouts = episodes.len().saturating_sub(colony_wins + colony_losses);
    let denominator = episodes.len() as f64;
    ControlAggregate {
        episodes: episodes.len(),
        colony_wins,
        colony_losses,
        timeouts,
        colony_win_rate: colony_wins as f64 / denominator,
        average_environment_steps: episodes
            .iter()
            .map(|episode| episode.environment_steps as f64)
            .sum::<f64>()
            / denominator,
        average_final_colony_cells: episodes
            .iter()
            .map(|episode| episode.colony_cells as f64)
            .sum::<f64>()
            / denominator,
        average_final_control_cells: episodes
            .iter()
            .map(|episode| episode.control_cells as f64)
            .sum::<f64>()
            / denominator,
    }
}

fn swap_ecology_sides(sample: &mut EcologySample) {
    std::mem::swap(&mut sample.training_cells, &mut sample.opponent_cells);
    std::mem::swap(
        &mut sample.training_cells_on_plants,
        &mut sample.opponent_cells_on_plants,
    );
    std::mem::swap(
        &mut sample.training_cells_on_major_food,
        &mut sample.opponent_cells_on_major_food,
    );
    std::mem::swap(&mut sample.training_energy, &mut sample.opponent_energy);
    std::mem::swap(
        &mut sample.training_spatial_entropy,
        &mut sample.opponent_spatial_entropy,
    );
}

fn normalize_step_telemetry(step: &mut StepTelemetry, seat: ControlSeat) {
    if seat.colony_is_team_zero() {
        return;
    }
    std::mem::swap(&mut step.training, &mut step.opponents);
    if let Some(sample) = step.state_sample.as_mut() {
        swap_ecology_sides(sample);
    }
}

fn colony_outcome(outcome: EpisodeOutcome, seat: ControlSeat) -> ViabilityOutcome {
    match (outcome, seat.colony_is_team_zero()) {
        (EpisodeOutcome::Win, true) | (EpisodeOutcome::Loss, false) => {
            ViabilityOutcome::CandidateWin
        }
        (EpisodeOutcome::Loss, true) | (EpisodeOutcome::Win, false) => {
            ViabilityOutcome::CandidateLoss
        }
        (EpisodeOutcome::Timeout, _) => ViabilityOutcome::Timeout,
        (EpisodeOutcome::SafetyAbort, _) => {
            unreachable!("control matrix rejects safety aborts before outcome conversion")
        }
    }
}

fn canonical_controls(
    requested: &[MaintainedMindProfile],
) -> Result<Vec<MaintainedMindProfile>, String> {
    if requested.is_empty() {
        return Err("control matrix requires at least one opponent".into());
    }
    let unique = requested.iter().copied().collect::<HashSet<_>>();
    if unique.len() != requested.len() || unique.iter().any(|profile| profile.is_colony_candidate())
    {
        return Err("control opponents must be unique maintained proxy Minds".into());
    }
    Ok(MaintainedMindProfile::CONTROLS
        .into_iter()
        .filter(|profile| unique.contains(profile))
        .collect())
}

fn summarize_profiles(
    opponents: &[MaintainedMindProfile],
    matchups: &[ControlMatrixMatchup],
) -> Vec<ControlProfileSummary> {
    opponents
        .iter()
        .copied()
        .map(|opponent| {
            let aggregates = matchups
                .iter()
                .filter(|matchup| matchup.report.opponent == opponent)
                .map(|matchup| &matchup.report.aggregate)
                .collect::<Vec<_>>();
            let episodes = aggregates.iter().map(|value| value.episodes).sum();
            let colony_wins = aggregates.iter().map(|value| value.colony_wins).sum();
            let colony_losses = aggregates.iter().map(|value| value.colony_losses).sum();
            let timeouts = aggregates.iter().map(|value| value.timeouts).sum();
            ControlProfileSummary {
                opponent,
                episodes,
                colony_wins,
                colony_losses,
                timeouts,
                colony_win_rate: if episodes == 0 {
                    0.0
                } else {
                    colony_wins as f64 / episodes as f64
                },
            }
        })
        .collect()
}

fn run_matchup(
    env_config: &EnvConfig,
    telemetry_config: &TelemetryConfig,
    candidate: MaintainedMindProfile,
    opponent: MaintainedMindProfile,
    seeds: &[u64],
) -> Result<ControlMatchupReport, String> {
    if !candidate.is_colony_candidate() {
        return Err("control candidate must be a colony policy profile".into());
    }
    if seeds.is_empty() {
        return Err("control evaluation requires at least one seed".into());
    }
    let mut unique = HashSet::with_capacity(seeds.len());
    if seeds.iter().any(|seed| !unique.insert(*seed)) {
        return Err("control evaluation seeds must be unique".into());
    }
    let validation = TrainingConfig {
        env: env_config.clone(),
        telemetry: telemetry_config.clone(),
        ..TrainingConfig::default()
    };
    validation.validate()?;
    if !telemetry_config.enabled {
        return Err("control evaluation requires telemetry".into());
    }
    if env_config.num_teams != 2 {
        return Err("mirrored control evaluation requires exactly two teams".into());
    }

    let scenario = ScenarioProfile::from(env_config);
    let scenario_hash = scenario.semantic_hash()?;
    let semantic_ruleset_hash = env_config.rules.semantic_hash().to_string();
    let expected_colony_cells = env_config.cells_per_team;
    let expected_control_cells = env_config.cells_per_team;
    let mut episodes = Vec::with_capacity(seeds.len().saturating_mul(ControlSeat::MIRRORED.len()));
    let mut telemetry_state: Option<TrainingTelemetryState> = None;
    let mut compiled_ruleset_hash = None;

    for (seed_index, &seed) in seeds.iter().enumerate() {
        for (seat_index, seat) in ControlSeat::MIRRORED.into_iter().enumerate() {
            let team_zero_profile = if seat.colony_is_team_zero() {
                candidate
            } else {
                opponent
            };
            let team_one_profile = if seat.colony_is_team_zero() {
                opponent
            } else {
                candidate
            };
            let opponent_factory: OpponentMindFactory =
                Arc::new(move || Box::new(MaintainedMind::new(team_one_profile)));
            let mut env = BlobEnv::new_with_opponent_factory(
                env_config.clone(),
                zero_reward(),
                seed,
                opponent_factory,
                None,
                None,
            );
            env.enable_telemetry(telemetry_config.clone());
            let mut initial = env
                .telemetry_sample()
                .ok_or("enabled control telemetry omitted its initial sample")?;
            if !seat.colony_is_team_zero() {
                swap_ecology_sides(&mut initial);
            }
            if initial.training_cells != expected_colony_cells
                || initial.opponent_cells != expected_control_cells
            {
                return Err(format!(
                    "engine placed {}/{} colony and {}/{} control cells for seed {seed}",
                    initial.training_cells,
                    expected_colony_cells,
                    initial.opponent_cells,
                    expected_control_cells
                ));
            }
            let episode_id = seed_index
                .saturating_mul(ControlSeat::MIRRORED.len())
                .saturating_add(seat_index) as u64;
            if let Some(state) = telemetry_state.as_mut() {
                state.start_episode(0, episode_id, seed, "control".into(), initial.clone());
            } else {
                telemetry_state = Some(TrainingTelemetryState::new(vec![(
                    seed,
                    "control".into(),
                    initial.clone(),
                )]));
            }
            let compiled = env.compiled_ruleset_hash();
            if compiled_ruleset_hash
                .as_ref()
                .is_some_and(|expected| expected != &compiled)
            {
                return Err("identical control configuration compiled differently".into());
            }
            compiled_ruleset_hash = Some(compiled);
            let mut team_zero_mind = MaintainedMind::new(team_zero_profile);

            loop {
                let mut result = env.step_with_reference_mind(&mut team_zero_mind);
                let mut step_telemetry = result
                    .telemetry
                    .take()
                    .ok_or("enabled control environment omitted step telemetry")?;
                normalize_step_telemetry(&mut step_telemetry, seat);
                telemetry_state
                    .as_mut()
                    .expect("initialized above")
                    .apply_step(
                        0,
                        &step_telemetry,
                        telemetry_config.max_state_samples_per_episode,
                    );
                if !result.done {
                    continue;
                }
                let end_reason = result
                    .end_reason
                    .ok_or("completed control episode omitted its end reason")?;
                if end_reason == EpisodeEndReason::DecisionFrontierSafetyLimit {
                    return Err(format!(
                        "seed {seed} reached the non-scientific decision-frontier safety limit"
                    ));
                }
                let outcome = colony_outcome(
                    result
                        .outcome
                        .ok_or("completed control episode omitted its outcome")?,
                    seat,
                );
                let telemetry_outcome = match outcome {
                    ViabilityOutcome::CandidateWin => TelemetryEpisodeOutcome::Win,
                    ViabilityOutcome::CandidateLoss => TelemetryEpisodeOutcome::Loss,
                    ViabilityOutcome::Timeout => TelemetryEpisodeOutcome::Timeout,
                };
                let mut final_sample = env
                    .telemetry_sample()
                    .ok_or("enabled control telemetry omitted its terminal sample")?;
                if !seat.colony_is_team_zero() {
                    swap_ecology_sides(&mut final_sample);
                }
                telemetry_state
                    .as_mut()
                    .expect("initialized above")
                    .finish_episode(
                        0,
                        telemetry_outcome,
                        result.episode_step,
                        final_sample.clone(),
                        telemetry_config.max_state_samples_per_episode,
                    );
                episodes.push(ControlEpisode {
                    seed,
                    seat,
                    outcome,
                    end_reason,
                    environment_steps: result.episode_step,
                    final_sim_time_quanta: final_sample.sim_time_quanta,
                    colony_cells: if seat.colony_is_team_zero() {
                        result.training_cells
                    } else {
                        result.opponent_cells
                    },
                    control_cells: if seat.colony_is_team_zero() {
                        result.opponent_cells
                    } else {
                        result.training_cells
                    },
                    initial_tracked_mass_energy: initial.tracked_mass_energy,
                    final_tracked_mass_energy: final_sample.tracked_mass_energy,
                    terminal_colony_energy: final_sample.training_energy.clone(),
                    terminal_control_energy: final_sample.opponent_energy.clone(),
                });
                break;
            }
        }
    }

    let compiled_ruleset_hash = compiled_ruleset_hash.expect("seeds are nonempty");
    let mind_abi_version = blob_interface::abi::REFERENCE_MIND_ABI_VERSION;
    let mind_abi_hash = mind_abi_hash();
    let matchup_config_hash = hash_json(&MatchupIdentity {
        mind_abi_version,
        mind_abi_hash: &mind_abi_hash,
        semantic_ruleset_hash: &semantic_ruleset_hash,
        compiled_ruleset_hash: &compiled_ruleset_hash,
        scenario_hash: &scenario_hash,
        candidate,
        opponent,
        seat_policy: ControlSeatPolicy::MirroredTeamZeroAndOne,
        seeds,
        telemetry_config,
    })?;
    let mut telemetry = telemetry_state.expect("seeds are nonempty").summary();
    telemetry.censored_active_episodes = 0;
    Ok(ControlMatchupReport {
        matchup_config_hash,
        semantic_ruleset_hash,
        compiled_ruleset_hash,
        scenario_hash,
        scenario,
        candidate,
        opponent,
        seat_policy: ControlSeatPolicy::MirroredTeamZeroAndOne,
        seeds: seeds.to_vec(),
        telemetry_config: telemetry_config.clone(),
        aggregate: aggregate_episodes(&episodes),
        episodes,
        telemetry,
    })
}

fn validate_matchup(report: &ControlMatchupReport) -> Result<(), String> {
    if !report.candidate.is_colony_candidate()
        || !MaintainedMindProfile::CONTROLS.contains(&report.opponent)
        || report.seeds.is_empty()
        || report.seat_policy != ControlSeatPolicy::MirroredTeamZeroAndOne
        || report.episodes.len()
            != report
                .seeds
                .len()
                .saturating_mul(ControlSeat::MIRRORED.len())
        || report.seeds.iter().enumerate().any(|(seed_index, seed)| {
            ControlSeat::MIRRORED
                .iter()
                .enumerate()
                .any(|(seat_index, seat)| {
                    let episode =
                        &report.episodes[seed_index * ControlSeat::MIRRORED.len() + seat_index];
                    episode.seed != *seed || episode.seat != *seat
                })
        })
    {
        return Err("control matchup profile or seed identity is invalid".into());
    }
    let unique = report.seeds.iter().collect::<HashSet<_>>();
    if unique.len() != report.seeds.len() {
        return Err("control matchup seeds are duplicated".into());
    }
    report.telemetry_config.validate()?;
    if !report.telemetry_config.enabled
        || report.scenario.semantic_hash()? != report.scenario_hash
        || aggregate_episodes(&report.episodes) != report.aggregate
    {
        return Err("control matchup scenario, aggregate, or telemetry is invalid".into());
    }
    if report.telemetry.completed_episodes != report.episodes.len() as u64
        || report.telemetry.censored_active_episodes != 0
        || report.telemetry.wins != report.aggregate.colony_wins as u64
        || report.telemetry.losses != report.aggregate.colony_losses as u64
        || report.telemetry.timeouts != report.aggregate.timeouts as u64
    {
        return Err("control matchup telemetry outcomes are inconsistent".into());
    }
    if report.episodes.iter().any(|episode| match episode.outcome {
        ViabilityOutcome::CandidateWin | ViabilityOutcome::CandidateLoss => {
            episode.end_reason != EpisodeEndReason::Extermination
        }
        ViabilityOutcome::Timeout => {
            episode.end_reason != EpisodeEndReason::SimTimeDeadline
                || episode.final_sim_time_quanta < report.scenario.victory.sim_time_limit_quanta
        }
    }) {
        return Err("control matchup termination reasons are inconsistent".into());
    }
    let expected = hash_json(&MatchupIdentity {
        mind_abi_version: blob_interface::abi::REFERENCE_MIND_ABI_VERSION,
        mind_abi_hash: &mind_abi_hash(),
        semantic_ruleset_hash: &report.semantic_ruleset_hash,
        compiled_ruleset_hash: &report.compiled_ruleset_hash,
        scenario_hash: &report.scenario_hash,
        candidate: report.candidate,
        opponent: report.opponent,
        seat_policy: report.seat_policy,
        seeds: &report.seeds,
        telemetry_config: &report.telemetry_config,
    })?;
    if expected != report.matchup_config_hash {
        return Err("control matchup configuration hash mismatch".into());
    }
    Ok(())
}

pub fn validate_control_matrix_report(report: &ControlMatrixReport) -> Result<(), String> {
    if report.schema_version != CONTROL_MATRIX_SCHEMA_VERSION
        || report.mind_abi_version != blob_interface::abi::REFERENCE_MIND_ABI_VERSION
        || report.mind_abi_hash != mind_abi_hash()
        || report.execution_mode != ControlExecutionMode::NativeMaintainedEntryPoint
        || report.verification_scope != ControlVerificationScope::LocalDeterministic
        || report.server_verified
        || report.replay_committed
        || report.seat_policy != ControlSeatPolicy::MirroredTeamZeroAndOne
        || !report.candidate.is_colony_candidate()
    {
        return Err("control matrix identity or trust scope is invalid".into());
    }
    if canonical_controls(&report.opponents)? != report.opponents
        || report.variants.is_empty()
        || report.seeds.is_empty()
    {
        return Err("control matrix profile or variant grid is invalid".into());
    }
    let variants = report
        .variants
        .iter()
        .map(|variant| (variant.name.as_str(), variant))
        .collect::<HashMap<_, _>>();
    if variants.len() != report.variants.len()
        || report.matchups.len() != report.variants.len() * report.opponents.len()
    {
        return Err("control matrix matchup grid is incomplete".into());
    }
    let mut keys = HashSet::new();
    for matchup in &report.matchups {
        validate_matchup(&matchup.report)
            .map_err(|error| format!("variant {}: {error}", matchup.variant))?;
        let variant = variants
            .get(matchup.variant.as_str())
            .ok_or_else(|| format!("unknown control variant {}", matchup.variant))?;
        if !keys.insert((matchup.variant.as_str(), matchup.report.opponent))
            || !report.opponents.contains(&matchup.report.opponent)
            || matchup.report.candidate != report.candidate
            || matchup.report.seeds != report.seeds
            || matchup.report.semantic_ruleset_hash != variant.semantic_ruleset_hash
            || matchup.report.compiled_ruleset_hash != variant.compiled_ruleset_hash
            || matchup.report.scenario_hash != variant.scenario_hash
        {
            return Err(format!(
                "variant {} matchup identity differs from the matrix",
                matchup.variant
            ));
        }
    }
    if summarize_profiles(&report.opponents, &report.matchups) != report.summaries {
        return Err("control matrix summaries are inconsistent".into());
    }
    let expected_hash = hash_json(&MatrixIdentity {
        mind_abi_version: report.mind_abi_version,
        mind_abi_hash: &report.mind_abi_hash,
        manifest_sha256: &report.manifest_sha256,
        execution_mode: report.execution_mode,
        candidate: report.candidate,
        seat_policy: report.seat_policy,
        opponents: &report.opponents,
    })?;
    if expected_hash != report.matrix_config_hash {
        return Err("control matrix configuration hash mismatch".into());
    }
    Ok(())
}

pub fn run_control_matrix(
    manifest_file: &Path,
    options: &ControlMatrixOptions,
) -> Result<ControlMatrixReport, String> {
    run_control_matrix_internal(manifest_file, options, None)
}

pub fn run_control_matrix_with_progress<F>(
    manifest_file: &Path,
    options: &ControlMatrixOptions,
    observer: F,
) -> Result<ControlMatrixReport, String>
where
    F: Fn(&ControlMatrixProgressReport) -> Result<(), String> + Sync,
{
    run_control_matrix_internal(manifest_file, options, Some(&observer))
}

fn run_control_matrix_internal(
    manifest_file: &Path,
    options: &ControlMatrixOptions,
    observer: Option<&ControlProgressObserver<'_>>,
) -> Result<ControlMatrixReport, String> {
    if options.max_parallel == 0 {
        return Err("control matrix max_parallel must be positive".into());
    }
    if !options.candidate.is_colony_candidate() {
        return Err("control matrix candidate must be a colony policy profile".into());
    }
    let opponents = canonical_controls(&options.opponents)?;
    let sweep = load_validated_sweep(manifest_file)?;
    let mut variant_configs = Vec::with_capacity(sweep.manifest.variants.len());
    let mut variants = Vec::with_capacity(sweep.manifest.variants.len());
    for variant in &sweep.manifest.variants {
        let members = sweep
            .manifest
            .runs
            .iter()
            .zip(&sweep.configs)
            .filter(|(run, _)| run.variant == variant.name)
            .collect::<Vec<_>>();
        let (representative_run, representative_config) = members
            .first()
            .copied()
            .ok_or_else(|| format!("variant {} has no expanded configs", variant.name))?;
        if members.iter().any(|(run, config)| {
            run.semantic_ruleset_hash != representative_run.semantic_ruleset_hash
                || run.compiled_ruleset_hash != representative_run.compiled_ruleset_hash
                || run.scenario_hash != representative_run.scenario_hash
                || config.env != representative_config.env
                || config.telemetry != representative_config.telemetry
        }) {
            return Err(format!(
                "variant {} changes environment or telemetry across seeds",
                variant.name
            ));
        }
        variant_configs.push((
            variant.name.clone(),
            representative_config.clone(),
            representative_run.semantic_ruleset_hash.clone(),
            representative_run.compiled_ruleset_hash.clone(),
            representative_run.scenario_hash.clone(),
        ));
        variants.push(ControlMatrixVariant {
            name: variant.name.clone(),
            description: variant.description.clone(),
            semantic_ruleset_hash: representative_run.semantic_ruleset_hash.clone(),
            compiled_ruleset_hash: representative_run.compiled_ruleset_hash.clone(),
            scenario_hash: representative_run.scenario_hash.clone(),
        });
    }

    let mut jobs = Vec::with_capacity(variants.len() * opponents.len());
    for (variant, config, semantic_ruleset_hash, compiled_ruleset_hash, scenario_hash) in
        variant_configs
    {
        for &opponent in &opponents {
            jobs.push(MatrixJob {
                variant: variant.clone(),
                config: config.clone(),
                semantic_ruleset_hash: semantic_ruleset_hash.clone(),
                compiled_ruleset_hash: compiled_ruleset_hash.clone(),
                scenario_hash: scenario_hash.clone(),
                opponent,
            });
        }
    }
    let mind_abi_version = blob_interface::abi::REFERENCE_MIND_ABI_VERSION;
    let mind_abi_hash = mind_abi_hash();
    let execution_mode = ControlExecutionMode::NativeMaintainedEntryPoint;
    let candidate = options.candidate;
    let seat_policy = ControlSeatPolicy::MirroredTeamZeroAndOne;
    let matrix_config_hash = hash_json(&MatrixIdentity {
        mind_abi_version,
        mind_abi_hash: &mind_abi_hash,
        manifest_sha256: &sweep.manifest_sha256,
        execution_mode,
        candidate,
        seat_policy,
        opponents: &opponents,
    })?;
    let progress_report = |status, error, matchups: Vec<ControlMatrixMatchup>| {
        control_matrix_progress_report(
            status,
            error,
            mind_abi_version,
            &mind_abi_hash,
            &sweep.manifest_sha256,
            &matrix_config_hash,
            execution_mode,
            seat_policy,
            candidate,
            &opponents,
            &sweep.manifest.seeds,
            &variants,
            jobs.len(),
            matchups,
        )
    };
    if let Some(observer) = observer {
        observer(&progress_report(
            ControlMatrixProgressStatus::Running,
            None,
            Vec::new(),
        ))?;
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.max_parallel)
        .build()
        .map_err(|error| format!("failed to create control-matrix worker pool: {error}"))?;
    let completed = Mutex::new(vec![None; jobs.len()]);
    let results = pool.install(|| {
        jobs.par_iter()
            .enumerate()
            .map(|(job_index, job)| {
                let result = run_matchup(
                    &job.config.env,
                    &job.config.telemetry,
                    candidate,
                    job.opponent,
                    &sweep.manifest.seeds,
                )
                .and_then(|report| {
                    if report.semantic_ruleset_hash != job.semantic_ruleset_hash
                        || report.compiled_ruleset_hash != job.compiled_ruleset_hash
                        || report.scenario_hash != job.scenario_hash
                    {
                        return Err("control report differs from its verified variant".into());
                    }
                    Ok(ControlMatrixMatchup {
                        variant: job.variant.clone(),
                        report,
                    })
                })
                .map_err(|error| {
                    format!(
                        "variant {} colony vs {} failed: {error}",
                        job.variant, job.opponent
                    )
                });
                if let (Ok(matchup), Some(observer)) = (&result, observer) {
                    let mut guard = completed
                        .lock()
                        .map_err(|_| "control-matrix progress lock was poisoned".to_string())?;
                    guard[job_index] = Some(matchup.clone());
                    let snapshot = guard.iter().flatten().cloned().collect();
                    observer(&progress_report(
                        ControlMatrixProgressStatus::Running,
                        None,
                        snapshot,
                    ))?;
                }
                result
            })
            .collect::<Vec<_>>()
    });
    let matchups = match results.into_iter().collect::<Result<Vec<_>, _>>() {
        Ok(matchups) => matchups,
        Err(error) => {
            if let Some(observer) = observer {
                let snapshot = completed
                    .into_inner()
                    .map_err(|_| "control-matrix progress lock was poisoned".to_string())?
                    .into_iter()
                    .flatten()
                    .collect();
                let _ = observer(&progress_report(
                    ControlMatrixProgressStatus::Failed,
                    Some(error.clone()),
                    snapshot,
                ));
            }
            return Err(error);
        }
    };
    let report = ControlMatrixReport {
        schema_version: CONTROL_MATRIX_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        mind_abi_version,
        mind_abi_hash: mind_abi_hash.clone(),
        manifest_sha256: sweep.manifest_sha256.clone(),
        matrix_config_hash: matrix_config_hash.clone(),
        execution_mode,
        verification_scope: ControlVerificationScope::LocalDeterministic,
        server_verified: false,
        replay_committed: false,
        seat_policy,
        candidate,
        opponents: opponents.clone(),
        seeds: sweep.manifest.seeds.clone(),
        variants: variants.clone(),
        summaries: summarize_profiles(&opponents, &matchups),
        matchups,
    };
    validate_control_matrix_report(&report)?;
    if let Some(observer) = observer {
        observer(&progress_report(
            ControlMatrixProgressStatus::Complete,
            None,
            report.matchups.clone(),
        ))?;
    }
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn control_matrix_progress_report(
    status: ControlMatrixProgressStatus,
    error: Option<String>,
    mind_abi_version: u16,
    mind_abi_hash: &str,
    manifest_sha256: &str,
    matrix_config_hash: &str,
    execution_mode: ControlExecutionMode,
    seat_policy: ControlSeatPolicy,
    candidate: MaintainedMindProfile,
    opponents: &[MaintainedMindProfile],
    seeds: &[u64],
    variants: &[ControlMatrixVariant],
    total_matchups: usize,
    matchups: Vec<ControlMatrixMatchup>,
) -> ControlMatrixProgressReport {
    let completed_episodes = matchups
        .iter()
        .map(|matchup| matchup.report.aggregate.episodes)
        .sum();
    let total_episodes = total_matchups
        .saturating_mul(seeds.len())
        .saturating_mul(ControlSeat::MIRRORED.len());
    let summaries = summarize_profiles(opponents, &matchups);
    let matchups: Vec<_> = matchups
        .into_iter()
        .map(|matchup| ControlMatrixProgressMatchup {
            variant: matchup.variant,
            opponent: matchup.report.opponent,
            aggregate: matchup.report.aggregate,
            seats: ControlSeat::MIRRORED
                .into_iter()
                .map(|seat| {
                    let episodes = matchup
                        .report
                        .episodes
                        .iter()
                        .filter(|episode| episode.seat == seat)
                        .collect::<Vec<_>>();
                    let colony_wins = episodes
                        .iter()
                        .filter(|episode| episode.outcome == ViabilityOutcome::CandidateWin)
                        .count();
                    let colony_losses = episodes
                        .iter()
                        .filter(|episode| episode.outcome == ViabilityOutcome::CandidateLoss)
                        .count();
                    ControlSeatAggregate {
                        seat,
                        episodes: episodes.len(),
                        colony_wins,
                        colony_losses,
                        timeouts: episodes.len().saturating_sub(colony_wins + colony_losses),
                    }
                })
                .collect(),
        })
        .collect();
    ControlMatrixProgressReport {
        schema_version: CONTROL_MATRIX_PROGRESS_SCHEMA_VERSION,
        report_kind: "blob_control_matrix_progress".into(),
        control_matrix_schema_version: CONTROL_MATRIX_SCHEMA_VERSION,
        status,
        error,
        mind_abi_version,
        mind_abi_hash: mind_abi_hash.into(),
        manifest_sha256: manifest_sha256.into(),
        matrix_config_hash: matrix_config_hash.into(),
        execution_mode,
        verification_scope: ControlVerificationScope::LocalDeterministic,
        server_verified: false,
        replay_committed: false,
        seat_policy,
        candidate,
        opponents: opponents.to_vec(),
        seeds: seeds.to_vec(),
        variants: variants.to_vec(),
        completed_matchups: matchups.len(),
        total_matchups,
        completed_episodes,
        total_episodes,
        summaries,
        matchups,
    }
}

pub fn write_control_matrix_progress(
    output: &Path,
    progress: &ControlMatrixProgressReport,
) -> Result<(), String> {
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let nonce = CONTROL_MATRIX_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "control-matrix progress output needs a UTF-8 name".to_string())?;
    let temporary = parent.join(format!(".{name}.live-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(progress)
        .map_err(|error| format!("failed to encode control-matrix progress: {error}"))?;
    bytes.push(b'\n');
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
        fs::rename(&temporary, output)
            .map_err(|error| format!("failed to replace {}: {error}", output.display()))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", parent.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn load_control_matrix_report(path: &Path) -> Result<ControlMatrixReport, String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > MAX_CONTROL_MATRIX_BYTES {
        return Err(format!(
            "{} is {length} bytes; control-matrix limit is {MAX_CONTROL_MATRIX_BYTES}",
            path.display()
        ));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    decode_control_matrix_report(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))
}

/// Decode and validate the exact bytes used by a downstream hash-bound
/// analysis, avoiding a read/hash/read time-of-check gap.
pub fn decode_control_matrix_report(bytes: &[u8]) -> Result<ControlMatrixReport, String> {
    if bytes.len() as u64 > MAX_CONTROL_MATRIX_BYTES {
        return Err(format!(
            "control matrix is {} bytes; limit is {MAX_CONTROL_MATRIX_BYTES}",
            bytes.len()
        ));
    }
    let report: ControlMatrixReport = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid control matrix JSON: {error}"))?;
    validate_control_matrix_report(&report)?;
    Ok(report)
}

pub fn publish_control_matrix_report(
    output: &Path,
    report: &ControlMatrixReport,
) -> Result<PathBuf, String> {
    validate_control_matrix_report(report)?;
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable control matrix {}",
            output.display()
        ));
    }
    let nonce = CONTROL_MATRIX_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "control-matrix output needs a UTF-8 name".to_string())?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode control matrix: {error}"))?;
    bytes.push(b'\n');
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
        fs::hard_link(&temporary, output)
            .map_err(|error| format!("failed to publish {}: {error}", output.display()))?;
        fs::remove_file(&temporary)
            .map_err(|error| format!("failed to remove {}: {error}", temporary.display()))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", parent.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map(|()| output.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweep::publish_rules_sweep;

    #[test]
    fn small_control_matrix_is_reproducible_complete_and_immutable() {
        let temporary = tempfile::tempdir().unwrap();
        fs::write(
            temporary.path().join("base.toml"),
            r#"
            seed = 1
            num_envs = 1
            rollout_length = 1
            total_timesteps = 1
            eval_interval = 0
            checkpoint_dir = "unused"
            [telemetry]
            enabled = true
            state_sample_interval_steps = 2
            max_state_samples_per_episode = 8
            episode_log_stride = 0
            [env]
            world_size = 8
            cells_per_team = 1
            max_episode_len = 64
            num_teams = 2
            num_scattered_energy = 4
            num_plants = 2
            [env.victory]
            immediate = "extermination"
            deadline = "draw"
            sim_time_limit_quanta = 4096
            "#,
        )
        .unwrap();
        fs::write(
            temporary.path().join("sweep.toml"),
            format!(
                r#"
                base_config = "base.toml"
                output_dir = "{}"
                seeds = [11, 12, 13]
                [[variants]]
                name = "small"
                "#,
                temporary.path().join("sweep").display()
            ),
        )
        .unwrap();
        let manifest = publish_rules_sweep(&temporary.path().join("sweep.toml")).unwrap();
        let manifest_file = Path::new(&manifest.output_directory).join("manifest.json");
        let options = ControlMatrixOptions {
            candidate: MaintainedMindProfile::Colony,
            opponents: vec![
                MaintainedMindProfile::Explorer,
                MaintainedMindProfile::Simple,
            ],
            max_parallel: 2,
        };
        let first = run_control_matrix(&manifest_file, &options).unwrap();
        let progress = Mutex::new(Vec::new());
        let live_file = temporary.path().join("live.json");
        let second = run_control_matrix_with_progress(&manifest_file, &options, |snapshot| {
            write_control_matrix_progress(&live_file, snapshot)?;
            progress.lock().unwrap().push(snapshot.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(first, second);
        let progress = progress.into_inner().unwrap();
        assert_eq!(
            progress
                .iter()
                .map(|snapshot| snapshot.completed_matchups)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 2]
        );
        assert_eq!(
            progress.last().unwrap().status,
            ControlMatrixProgressStatus::Complete
        );
        assert_eq!(
            serde_json::from_slice::<ControlMatrixProgressReport>(&fs::read(live_file).unwrap())
                .unwrap(),
            *progress.last().unwrap()
        );
        assert_eq!(
            first.opponents,
            vec![
                MaintainedMindProfile::Simple,
                MaintainedMindProfile::Explorer
            ]
        );
        assert_eq!(first.matchups.len(), 2);
        assert_eq!(first.summaries.len(), 2);
        for matchup in &first.matchups {
            assert_eq!(matchup.report.episodes.len(), 6);
            for (index, episode) in matchup.report.episodes.iter().enumerate() {
                assert_eq!(
                    episode.seat,
                    if index.is_multiple_of(2) {
                        ControlSeat::ColonyTeamZero
                    } else {
                        ControlSeat::ColonyTeamOne
                    }
                );
                assert_eq!(episode.seed, first.seeds[index / 2]);
            }
        }
        assert!(!first.server_verified);
        assert!(!first.replay_committed);
        validate_control_matrix_report(&first).unwrap();

        let output = temporary.path().join("matrix.json");
        publish_control_matrix_report(&output, &first).unwrap();
        assert_eq!(load_control_matrix_report(&output).unwrap(), first);
        assert!(publish_control_matrix_report(&output, &second).is_err());

        let ablated = run_control_matrix(
            &manifest_file,
            &ControlMatrixOptions {
                candidate: MaintainedMindProfile::ColonySignalDisabled,
                opponents: vec![MaintainedMindProfile::Simple],
                max_parallel: 1,
            },
        )
        .unwrap();
        assert_eq!(
            ablated.candidate,
            MaintainedMindProfile::ColonySignalDisabled
        );
        assert_ne!(ablated.matrix_config_hash, first.matrix_config_hash);
        assert!(ablated
            .matchups
            .iter()
            .all(|matchup| { matchup.report.telemetry.training.signals.emitted_decisions == 0 }));
        validate_control_matrix_report(&ablated).unwrap();

        let mut tampered = first;
        tampered.summaries[0].colony_wins += 1;
        assert!(validate_control_matrix_report(&tampered)
            .unwrap_err()
            .contains("summaries"));
    }

    #[test]
    fn mirrored_seat_inverts_team_zero_outcomes() {
        assert_eq!(
            colony_outcome(EpisodeOutcome::Win, ControlSeat::ColonyTeamZero),
            ViabilityOutcome::CandidateWin
        );
        assert_eq!(
            colony_outcome(EpisodeOutcome::Win, ControlSeat::ColonyTeamOne),
            ViabilityOutcome::CandidateLoss
        );
        assert_eq!(
            colony_outcome(EpisodeOutcome::Loss, ControlSeat::ColonyTeamOne),
            ViabilityOutcome::CandidateWin
        );
        assert_eq!(
            colony_outcome(EpisodeOutcome::Timeout, ControlSeat::ColonyTeamOne),
            ViabilityOutcome::Timeout
        );
    }
}
