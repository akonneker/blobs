//! Cell-level attribution for the adjacent-food retention gate.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blob_interface::types::CellId;
use burn::prelude::Backend;
use serde::{Deserialize, Serialize};

use crate::action::{decompose_policy_action, PolicyActionKind, NUM_POLICY_ACTION_KINDS};
use crate::config::{FeedingCurriculumStage, OpponentProfile, ScenarioProfile, TrainingConfig};
use crate::env::BlobEnv;
use crate::evaluation::greedy_policy_choices;
use crate::model::PolicyValueNet;
use crate::observation::{
    Observation, CURRENT_TILE_PLANT_CAPACITY_FEATURE, HEADER_FEATURES, SLOT_FEATURES,
    SLOT_PLANT_CAPACITY_FEATURE,
};
use crate::sweep::sha256;

pub const FEEDING_TRANSITION_DIAGNOSTIC_SCHEMA_VERSION: u32 = 2;
const MAX_ARTIFACT_BYTES: u64 = 8 * 1024 * 1024;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PolicyRandomnessMode {
    #[default]
    Zero,
    Canonical,
}

impl std::fmt::Display for PolicyRandomnessMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zero => formatter.write_str("zero"),
            Self::Canonical => formatter.write_str("canonical"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdjacentTransitionCounts {
    pub initial_cells: u64,
    pub unique_cells_observed: u64,
    pub terminal_cells: u64,
    pub ready_decisions: u64,
    pub off_plant_visible_plant_decisions: u64,
    pub correct_move_to_plant_decisions: u64,
    pub wrong_target_move_decisions: u64,
    pub non_move_with_visible_plant_decisions: u64,
    pub on_plant_decisions: u64,
    pub consume_on_plant_decisions: u64,
    pub move_off_plant_decisions: u64,
    pub other_on_plant_decisions: u64,
    pub correct_move_success_outcomes: u64,
    pub correct_move_frustrated_outcomes: u64,
    pub correct_move_contested_outcomes: u64,
    pub correct_move_interrupted_outcomes: u64,
    pub correct_move_rejected_outcomes: u64,
    pub correct_move_unobserved_deaths: u64,
    pub correct_move_pending_at_terminal: u64,
    pub successful_moves_to_plant: u64,
    pub successful_consumes_on_plant: u64,
    pub successful_moves_off_plant: u64,
    pub cells_reaching_plant: u64,
    pub cells_successfully_consuming: u64,
    pub cells_leaving_plant: u64,
    pub deaths_before_reaching_plant: u64,
    pub deaths_after_reaching_before_consume: u64,
    pub deaths_after_successful_consume: u64,
    pub off_plant_visible_action_kinds: [u64; NUM_POLICY_ACTION_KINDS],
    pub on_plant_action_kinds: [u64; NUM_POLICY_ACTION_KINDS],
    pub visible_plant_move_efforts: [u64; 3],
}

impl AdjacentTransitionCounts {
    fn add_assign(&mut self, other: &Self) {
        macro_rules! add_fields {
            ($($field:ident),+ $(,)?) => {
                $(self.$field = self.$field.saturating_add(other.$field);)+
            };
        }
        add_fields!(
            initial_cells,
            unique_cells_observed,
            terminal_cells,
            ready_decisions,
            off_plant_visible_plant_decisions,
            correct_move_to_plant_decisions,
            wrong_target_move_decisions,
            non_move_with_visible_plant_decisions,
            on_plant_decisions,
            consume_on_plant_decisions,
            move_off_plant_decisions,
            other_on_plant_decisions,
            correct_move_success_outcomes,
            correct_move_frustrated_outcomes,
            correct_move_contested_outcomes,
            correct_move_interrupted_outcomes,
            correct_move_rejected_outcomes,
            correct_move_unobserved_deaths,
            correct_move_pending_at_terminal,
            successful_moves_to_plant,
            successful_consumes_on_plant,
            successful_moves_off_plant,
            cells_reaching_plant,
            cells_successfully_consuming,
            cells_leaving_plant,
            deaths_before_reaching_plant,
            deaths_after_reaching_before_consume,
            deaths_after_successful_consume,
        );
        for (target, source) in self
            .off_plant_visible_action_kinds
            .iter_mut()
            .zip(other.off_plant_visible_action_kinds)
        {
            *target = target.saturating_add(source);
        }
        for (target, source) in self
            .on_plant_action_kinds
            .iter_mut()
            .zip(other.on_plant_action_kinds)
        {
            *target = target.saturating_add(source);
        }
        for (target, source) in self
            .visible_plant_move_efforts
            .iter_mut()
            .zip(other.visible_plant_move_efforts)
        {
            *target = target.saturating_add(source);
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.off_plant_visible_plant_decisions
            != self
                .correct_move_to_plant_decisions
                .saturating_add(self.wrong_target_move_decisions)
                .saturating_add(self.non_move_with_visible_plant_decisions)
            || self.on_plant_decisions
                != self
                    .consume_on_plant_decisions
                    .saturating_add(self.move_off_plant_decisions)
                    .saturating_add(self.other_on_plant_decisions)
            || self
                .off_plant_visible_action_kinds
                .iter()
                .copied()
                .sum::<u64>()
                != self.off_plant_visible_plant_decisions
            || self.on_plant_action_kinds.iter().copied().sum::<u64>() != self.on_plant_decisions
            || self.visible_plant_move_efforts.iter().copied().sum::<u64>()
                != self
                    .correct_move_to_plant_decisions
                    .saturating_add(self.wrong_target_move_decisions)
        {
            return Err("adjacent transition decision partitions are inconsistent".into());
        }
        let resolved_correct_moves = self
            .correct_move_success_outcomes
            .saturating_add(self.correct_move_frustrated_outcomes)
            .saturating_add(self.correct_move_contested_outcomes)
            .saturating_add(self.correct_move_interrupted_outcomes)
            .saturating_add(self.correct_move_rejected_outcomes)
            .saturating_add(self.correct_move_unobserved_deaths)
            .saturating_add(self.correct_move_pending_at_terminal);
        if resolved_correct_moves > self.correct_move_to_plant_decisions
            || self.successful_moves_to_plant > self.correct_move_success_outcomes
        {
            return Err("adjacent transition move outcomes are inconsistent".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdjacentTransitionSeed {
    pub seed: u64,
    pub counts: AdjacentTransitionCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FeedingTransitionReport {
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub seeds: Vec<u64>,
    #[serde(default, skip_serializing_if = "policy_randomness_is_zero")]
    pub policy_randomness: PolicyRandomnessMode,
    pub episodes: Vec<AdjacentTransitionSeed>,
    pub totals: AdjacentTransitionCounts,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingTransitionArtifact {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub artifact_hash: String,
    pub source_config_sha256: String,
    pub behavior_clone_metadata_sha256: String,
    pub behavior_clone_model_sha256: String,
    pub config: TrainingConfig,
    pub report: FeedingTransitionReport,
}

#[derive(Serialize)]
struct ArtifactIdentity<'a> {
    schema_version: u32,
    package_version: &'a str,
    code_revision: &'a Option<String>,
    source_config_sha256: &'a str,
    behavior_clone_metadata_sha256: &'a str,
    behavior_clone_model_sha256: &'a str,
    config: &'a TrainingConfig,
    report: &'a FeedingTransitionReport,
}

impl FeedingTransitionArtifact {
    pub fn new(
        source_config_sha256: String,
        behavior_clone_metadata_sha256: String,
        behavior_clone_model_sha256: String,
        config: TrainingConfig,
        report: FeedingTransitionReport,
    ) -> Result<Self, String> {
        let mut artifact = Self {
            schema_version: FEEDING_TRANSITION_DIAGNOSTIC_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").into(),
            code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
            artifact_hash: String::new(),
            source_config_sha256,
            behavior_clone_metadata_sha256,
            behavior_clone_model_sha256,
            config,
            report,
        };
        artifact.artifact_hash = artifact.recompute_hash()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn recompute_hash(&self) -> Result<String, String> {
        let bytes = serde_json::to_vec(&ArtifactIdentity {
            schema_version: self.schema_version,
            package_version: &self.package_version,
            code_revision: &self.code_revision,
            source_config_sha256: &self.source_config_sha256,
            behavior_clone_metadata_sha256: &self.behavior_clone_metadata_sha256,
            behavior_clone_model_sha256: &self.behavior_clone_model_sha256,
            config: &self.config,
            report: &self.report,
        })
        .map_err(|error| format!("failed to encode transition identity: {error}"))?;
        Ok(sha256(&bytes))
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(
            self.schema_version,
            1 | FEEDING_TRANSITION_DIAGNOSTIC_SCHEMA_VERSION
        ) || !valid_sha256(&self.source_config_sha256)
            || !valid_sha256(&self.behavior_clone_metadata_sha256)
            || !valid_sha256(&self.behavior_clone_model_sha256)
        {
            return Err("feeding transition artifact identity is invalid".into());
        }
        if self.schema_version == 1 && self.report.policy_randomness != PolicyRandomnessMode::Zero {
            return Err("schema-1 feeding transition artifacts require zero randomness".into());
        }
        self.config.validate()?;
        self.report.validate(&self.config)?;
        if self.artifact_hash != self.recompute_hash()? {
            return Err("feeding transition artifact hash mismatch".into());
        }
        Ok(())
    }
}

impl FeedingTransitionReport {
    fn validate(&self, config: &TrainingConfig) -> Result<(), String> {
        if !valid_sha256(&self.compiled_ruleset_hash)
            || !valid_sha256(&self.scenario_hash)
            || self.seeds.is_empty()
            || self.episodes.len() != self.seeds.len()
            || self
                .episodes
                .iter()
                .map(|episode| episode.seed)
                .ne(self.seeds.iter().copied())
            || self.seeds.iter().copied().collect::<HashSet<_>>().len() != self.seeds.len()
        {
            return Err("feeding transition report envelope is invalid".into());
        }
        let mut effective = config
            .feeding_curriculum
            .environment_for_stage(&config.env, FeedingCurriculumStage::AdjacentFood);
        effective.opponent = OpponentProfile::Wait;
        effective.max_episode_len = config
            .feeding_curriculum
            .promotion
            .evaluation_max_episode_len;
        effective.victory.sim_time_limit_quanta = config
            .feeding_curriculum
            .promotion
            .evaluation_sim_time_limit_quanta;
        if ScenarioProfile::from(&effective).semantic_hash()? != self.scenario_hash
            || BlobEnv::new(effective, config.reward.clone(), self.seeds[0]).compiled_ruleset_hash()
                != self.compiled_ruleset_hash
        {
            return Err("feeding transition scenario binding is invalid".into());
        }
        let mut totals = AdjacentTransitionCounts::default();
        for episode in &self.episodes {
            episode.counts.validate()?;
            totals.add_assign(&episode.counts);
        }
        if totals != self.totals {
            return Err("feeding transition totals do not match episode rows".into());
        }
        self.totals.validate()
    }
}

#[derive(Debug, Clone, Copy)]
enum PendingDecision {
    CorrectMove,
    ConsumeOnPlant,
    MoveOffPlant,
    Other,
}

#[derive(Debug, Default)]
struct CellTrace {
    reached_plant: bool,
    consumed_on_plant: bool,
    left_plant: bool,
    death_recorded: bool,
    pending: Option<PendingDecision>,
}

pub fn evaluate_adjacent_transitions<B: Backend>(
    model: &PolicyValueNet<B>,
    config: &TrainingConfig,
    seeds: &[u64],
    policy_randomness: PolicyRandomnessMode,
    device: &B::Device,
    mut progress: impl FnMut(&AdjacentTransitionSeed),
) -> Result<FeedingTransitionReport, String>
where
    f32: From<B::FloatElem>,
{
    if seeds.is_empty() || seeds.iter().copied().collect::<HashSet<_>>().len() != seeds.len() {
        return Err("feeding transition diagnostics require unique seeds".into());
    }
    let mut effective = config
        .feeding_curriculum
        .environment_for_stage(&config.env, FeedingCurriculumStage::AdjacentFood);
    effective.opponent = OpponentProfile::Wait;
    effective.max_episode_len = config
        .feeding_curriculum
        .promotion
        .evaluation_max_episode_len;
    effective.victory.sim_time_limit_quanta = config
        .feeding_curriculum
        .promotion
        .evaluation_sim_time_limit_quanta;
    let scenario_hash = ScenarioProfile::from(&effective).semantic_hash()?;
    let mut episodes = Vec::with_capacity(seeds.len());
    let mut totals = AdjacentTransitionCounts::default();
    let mut compiled_ruleset_hash = None;

    for &seed in seeds {
        let mut env = BlobEnv::new(effective.clone(), config.reward.clone(), seed);
        let ruleset = env.compiled_ruleset_hash();
        if compiled_ruleset_hash
            .as_ref()
            .is_some_and(|expected| expected != &ruleset)
        {
            return Err("adjacent diagnostic rules compiled inconsistently".into());
        }
        compiled_ruleset_hash = Some(ruleset);
        let mut counts = AdjacentTransitionCounts {
            initial_cells: env.training_cells_alive() as u64,
            ..AdjacentTransitionCounts::default()
        };
        let mut traces = HashMap::<CellId, CellTrace>::new();

        loop {
            let observations = match policy_randomness {
                PolicyRandomnessMode::Canonical => env.get_policy_observations(),
                PolicyRandomnessMode::Zero => env
                    .prepare_training_reference_inputs()?
                    .into_iter()
                    .map(|(cell_id, mut input)| {
                        input.randomness = blob_interface::randomness::PrivateRandom::ZERO;
                        crate::env::PolicyObservation {
                            cell_id,
                            observation: Observation::from_reference(&input),
                            private_memory: input.private_memory,
                        }
                    })
                    .collect(),
            };
            let choices = greedy_policy_choices(model, &observations, device);
            if observations.len() != choices.len() {
                return Err("adjacent diagnostic lost a ready policy row".into());
            }
            for (observation, (choice_cell, choice, _)) in observations.iter().zip(&choices) {
                if observation.cell_id != *choice_cell {
                    return Err("adjacent diagnostic changed ready-cell order".into());
                }
                let trace = traces.entry(observation.cell_id).or_default();
                settle_pending(trace, &observation.observation, &mut counts);
                let on_plant = current_plant(&observation.observation);
                if on_plant && !trace.reached_plant {
                    trace.reached_plant = true;
                    counts.cells_reaching_plant = counts.cells_reaching_plant.saturating_add(1);
                }
                classify_decision(trace, &observation.observation, choice.action, &mut counts)?;
            }
            counts.unique_cells_observed = traces.len() as u64;
            let result = env.step_with_policy_memory(&choices);
            for (cell_id, trace) in &mut traces {
                if !trace.death_recorded && !env.cell_is_alive(*cell_id) {
                    trace.death_recorded = true;
                    if matches!(trace.pending.take(), Some(PendingDecision::CorrectMove)) {
                        counts.correct_move_unobserved_deaths =
                            counts.correct_move_unobserved_deaths.saturating_add(1);
                    }
                    if trace.consumed_on_plant {
                        counts.deaths_after_successful_consume =
                            counts.deaths_after_successful_consume.saturating_add(1);
                    } else if trace.reached_plant {
                        counts.deaths_after_reaching_before_consume = counts
                            .deaths_after_reaching_before_consume
                            .saturating_add(1);
                    } else {
                        counts.deaths_before_reaching_plant =
                            counts.deaths_before_reaching_plant.saturating_add(1);
                    }
                }
            }
            if result.done {
                counts.correct_move_pending_at_terminal =
                    counts.correct_move_pending_at_terminal.saturating_add(
                        traces
                            .values()
                            .filter(|trace| {
                                !trace.death_recorded
                                    && matches!(trace.pending, Some(PendingDecision::CorrectMove))
                            })
                            .count() as u64,
                    );
                counts.terminal_cells = result.training_cells as u64;
                break;
            }
        }
        counts.validate()?;
        let episode = AdjacentTransitionSeed { seed, counts };
        totals.add_assign(&episode.counts);
        progress(&episode);
        episodes.push(episode);
    }

    let report = FeedingTransitionReport {
        compiled_ruleset_hash: compiled_ruleset_hash
            .ok_or_else(|| "adjacent diagnostics produced no ruleset".to_string())?,
        scenario_hash,
        seeds: seeds.to_vec(),
        policy_randomness,
        episodes,
        totals,
    };
    report.validate(config)?;
    Ok(report)
}

fn settle_pending(
    trace: &mut CellTrace,
    observation: &Observation,
    counts: &mut AdjacentTransitionCounts,
) {
    let Some(pending) = trace.pending.take() else {
        return;
    };
    let Some(outcome) = observed_outcome(observation) else {
        return;
    };
    if matches!(pending, PendingDecision::CorrectMove) {
        match outcome {
            ObservedOutcome::Success => {
                counts.correct_move_success_outcomes =
                    counts.correct_move_success_outcomes.saturating_add(1);
            }
            ObservedOutcome::Frustrated => {
                counts.correct_move_frustrated_outcomes =
                    counts.correct_move_frustrated_outcomes.saturating_add(1);
            }
            ObservedOutcome::Contested => {
                counts.correct_move_contested_outcomes =
                    counts.correct_move_contested_outcomes.saturating_add(1);
            }
            ObservedOutcome::Interrupted => {
                counts.correct_move_interrupted_outcomes =
                    counts.correct_move_interrupted_outcomes.saturating_add(1);
            }
            ObservedOutcome::Rejected => {
                counts.correct_move_rejected_outcomes =
                    counts.correct_move_rejected_outcomes.saturating_add(1);
            }
        }
    }
    if outcome != ObservedOutcome::Success {
        return;
    }
    match pending {
        PendingDecision::CorrectMove if current_plant(observation) => {
            counts.successful_moves_to_plant = counts.successful_moves_to_plant.saturating_add(1);
        }
        PendingDecision::ConsumeOnPlant => {
            counts.successful_consumes_on_plant =
                counts.successful_consumes_on_plant.saturating_add(1);
            if !trace.consumed_on_plant {
                trace.consumed_on_plant = true;
                counts.cells_successfully_consuming =
                    counts.cells_successfully_consuming.saturating_add(1);
            }
        }
        PendingDecision::MoveOffPlant if !current_plant(observation) => {
            counts.successful_moves_off_plant = counts.successful_moves_off_plant.saturating_add(1);
            if !trace.left_plant {
                trace.left_plant = true;
                counts.cells_leaving_plant = counts.cells_leaving_plant.saturating_add(1);
            }
        }
        _ => {}
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObservedOutcome {
    Success,
    Frustrated,
    Contested,
    Interrupted,
    Rejected,
}

fn observed_outcome(observation: &Observation) -> Option<ObservedOutcome> {
    if observation.data[6] <= 0.5 {
        return None;
    }
    match (observation.data[7] * 5.0).round() as u8 {
        1 => Some(ObservedOutcome::Success),
        2 => Some(ObservedOutcome::Frustrated),
        3 => Some(ObservedOutcome::Contested),
        4 => Some(ObservedOutcome::Interrupted),
        5 => Some(ObservedOutcome::Rejected),
        _ => None,
    }
}

fn classify_decision(
    trace: &mut CellTrace,
    observation: &Observation,
    action: usize,
    counts: &mut AdjacentTransitionCounts,
) -> Result<(), String> {
    let choice = decompose_policy_action(action)
        .ok_or_else(|| "adjacent diagnostic policy action is outside the catalog".to_string())?;
    let kind = PolicyActionKind::from_index(choice.kind)
        .ok_or_else(|| "adjacent diagnostic action kind is invalid".to_string())?;
    counts.ready_decisions = counts.ready_decisions.saturating_add(1);
    if current_plant(observation) {
        counts.on_plant_decisions = counts.on_plant_decisions.saturating_add(1);
        counts.on_plant_action_kinds[kind.index()] =
            counts.on_plant_action_kinds[kind.index()].saturating_add(1);
        trace.pending = Some(match kind {
            PolicyActionKind::Consume => {
                counts.consume_on_plant_decisions =
                    counts.consume_on_plant_decisions.saturating_add(1);
                PendingDecision::ConsumeOnPlant
            }
            PolicyActionKind::Move => {
                counts.move_off_plant_decisions = counts.move_off_plant_decisions.saturating_add(1);
                PendingDecision::MoveOffPlant
            }
            _ => {
                counts.other_on_plant_decisions = counts.other_on_plant_decisions.saturating_add(1);
                PendingDecision::Other
            }
        });
        return Ok(());
    }

    if visible_plant(observation) {
        counts.off_plant_visible_plant_decisions =
            counts.off_plant_visible_plant_decisions.saturating_add(1);
        counts.off_plant_visible_action_kinds[kind.index()] =
            counts.off_plant_visible_action_kinds[kind.index()].saturating_add(1);
        trace.pending = Some(if kind == PolicyActionKind::Move {
            counts.visible_plant_move_efforts[choice.effort] =
                counts.visible_plant_move_efforts[choice.effort].saturating_add(1);
            if target_has_plant(observation, choice.target) {
                counts.correct_move_to_plant_decisions =
                    counts.correct_move_to_plant_decisions.saturating_add(1);
                PendingDecision::CorrectMove
            } else {
                counts.wrong_target_move_decisions =
                    counts.wrong_target_move_decisions.saturating_add(1);
                PendingDecision::Other
            }
        } else {
            counts.non_move_with_visible_plant_decisions = counts
                .non_move_with_visible_plant_decisions
                .saturating_add(1);
            PendingDecision::Other
        });
    } else {
        trace.pending = Some(PendingDecision::Other);
    }
    Ok(())
}

fn current_plant(observation: &Observation) -> bool {
    observation.data[CURRENT_TILE_PLANT_CAPACITY_FEATURE] > 0.0
}

fn visible_plant(observation: &Observation) -> bool {
    (0..crate::action::NUM_POLICY_TARGETS).any(|target| target_has_plant(observation, target))
}

fn target_has_plant(observation: &Observation, target: usize) -> bool {
    observation
        .data
        .get(HEADER_FEATURES + target * SLOT_FEATURES + SLOT_PLANT_CAPACITY_FEATURE)
        .is_some_and(|capacity| *capacity > 0.0)
}

pub fn publish_feeding_transition(
    output: &Path,
    artifact: &FeedingTransitionArtifact,
) -> Result<PathBuf, String> {
    artifact.validate()?;
    if output.exists() {
        return Err(format!("refusing to replace {}", output.display()));
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "transition output needs a UTF-8 name".to_string())?;
    let staging = parent.join(format!(
        ".{name}.tmp-{}-{}",
        std::process::id(),
        TEMP_NONCE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut bytes = serde_json::to_vec_pretty(artifact)
        .map_err(|error| format!("failed to encode transition artifact: {error}"))?;
    bytes.push(b'\n');
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&staging)
            .map_err(|error| format!("failed to create {}: {error}", staging.display()))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("failed to write {}: {error}", staging.display()))?;
        fs::rename(&staging, output)
            .map_err(|error| format!("failed to publish {}: {error}", output.display()))?;
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", parent.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&staging);
    }
    result.map(|()| output.to_path_buf())
}

pub fn load_feeding_transition(path: &Path) -> Result<FeedingTransitionArtifact, String> {
    if fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len()
        > MAX_ARTIFACT_BYTES
    {
        return Err("feeding transition artifact is too large".into());
    }
    let artifact: FeedingTransitionArtifact = serde_json::from_slice(
        &fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    artifact.validate()?;
    Ok(artifact)
}

pub fn verify_feeding_transition_request(
    artifact: &FeedingTransitionArtifact,
    source_config_sha256: &str,
    behavior_clone_metadata_sha256: &str,
    behavior_clone_model_sha256: &str,
    config: &TrainingConfig,
    seeds: &[u64],
    policy_randomness: PolicyRandomnessMode,
) -> Result<(), String> {
    artifact.validate()?;
    if artifact.source_config_sha256 != source_config_sha256
        || artifact.behavior_clone_metadata_sha256 != behavior_clone_metadata_sha256
        || artifact.behavior_clone_model_sha256 != behavior_clone_model_sha256
        || artifact.config != *config
        || artifact.report.seeds != seeds
        || artifact.report.policy_randomness != policy_randomness
    {
        return Err("feeding transition artifact does not match the requested run".into());
    }
    Ok(())
}

fn policy_randomness_is_zero(mode: &PolicyRandomnessMode) -> bool {
    *mode == PolicyRandomnessMode::Zero
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    use crate::model::PolicyValueNetConfig;

    #[test]
    fn decision_partitions_and_aggregation_are_exact() {
        let mut left = AdjacentTransitionCounts {
            off_plant_visible_plant_decisions: 3,
            correct_move_to_plant_decisions: 1,
            wrong_target_move_decisions: 1,
            non_move_with_visible_plant_decisions: 1,
            on_plant_decisions: 3,
            consume_on_plant_decisions: 1,
            move_off_plant_decisions: 1,
            other_on_plant_decisions: 1,
            off_plant_visible_action_kinds: [1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            on_plant_action_kinds: [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            visible_plant_move_efforts: [1, 1, 0],
            ..AdjacentTransitionCounts::default()
        };
        left.validate().unwrap();
        let right = left.clone();
        left.add_assign(&right);
        assert_eq!(left.off_plant_visible_plant_decisions, 6);
        left.validate().unwrap();
    }

    #[test]
    fn tiny_rollout_is_hash_bound_and_tamper_detecting() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let mut config = TrainingConfig::default();
        config.env.world_size = 8;
        config.env.cells_per_team = 2;
        config.env.num_plants = 2;
        config.env.num_scattered_energy = 0;
        config
            .feeding_curriculum
            .promotion
            .evaluation_max_episode_len = 64;
        config
            .feeding_curriculum
            .promotion
            .evaluation_sim_time_limit_quanta = 1024;
        let device = Default::default();
        let model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<NdArray>(&device);
        let report = evaluate_adjacent_transitions(
            &model,
            &config,
            &[17],
            PolicyRandomnessMode::Zero,
            &device,
            |_| {},
        )
        .expect("tiny diagnostic should run");
        let artifact = FeedingTransitionArtifact::new(
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            config,
            report,
        )
        .unwrap();
        artifact.validate().unwrap();

        let mut tampered = artifact;
        tampered.report.totals.ready_decisions += 1;
        assert!(tampered.validate().is_err());
    }
}
