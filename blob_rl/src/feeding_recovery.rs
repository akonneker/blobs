//! Counterfactual teacher recovery from frozen-policy feeding states.
//!
//! A static correction dataset can show action disagreement but cannot tell us
//! whether the learned policy has already entered an irrecoverable state. This
//! module checkpoints policy-induced decision frontiers, restores the exact
//! simulator and per-cell private state, and lets an observation-legal teacher
//! continue to the original simulation-time deadline.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use burn::prelude::Backend;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{
    EnvConfig, FeedingCurriculumConfig, FeedingCurriculumStage, OpponentProfile, RewardConfig,
    ScenarioProfile, TrainingConfig,
};
use crate::control_matrix::MaintainedMindProfile;
use crate::env::{BlobEnv, BlobEnvCheckpoint, EpisodeOutcome};
use crate::evaluation::greedy_policy_choices;
use crate::model::PolicyValueNet;
use crate::sweep::sha256;
use crate::telemetry::TelemetryConfig;

pub const FEEDING_RECOVERY_SCHEMA_VERSION: u32 = 1;
const MAX_ARTIFACT_BYTES: u64 = 128 * 1024 * 1024;
const MAX_SEEDS: usize = 4096;
const MAX_CHECKPOINTS_PER_SEED: usize = 64;
const MAX_TOTAL_CHECKPOINTS: usize = 65_536;
static RECOVERY_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingRecoveryOptions {
    pub teacher: MaintainedMindProfile,
    pub seeds: Vec<u64>,
    /// After the first policy-induced frontier, space later forks by canonical
    /// simulation time rather than cell count or RL steps.
    pub checkpoint_interval_quanta: u64,
    pub max_checkpoints_per_seed: usize,
    pub min_checkpoints_per_seed: usize,
    /// Do not fork states too close to the original episode deadline for the
    /// teacher to demonstrate recovery.
    pub min_remaining_time_quanta: u64,
    pub min_recovery_rate: f64,
}

impl FeedingRecoveryOptions {
    pub fn validate(&self, curriculum: &FeedingCurriculumConfig) -> Result<(), String> {
        if self.seeds.is_empty() || self.seeds.len() > MAX_SEEDS {
            return Err(format!("feeding recovery requires 1 to {MAX_SEEDS} seeds"));
        }
        let unique = self.seeds.iter().copied().collect::<HashSet<_>>();
        if unique.len() != self.seeds.len() {
            return Err("feeding recovery seeds must be unique".into());
        }
        if self.teacher != MaintainedMindProfile::CollisionAwareForager {
            return Err(
                "feeding recovery schema 1 requires the memoryless collision-aware forager".into(),
            );
        }
        if self.checkpoint_interval_quanta == 0
            || self.max_checkpoints_per_seed == 0
            || self.max_checkpoints_per_seed > MAX_CHECKPOINTS_PER_SEED
            || self.min_checkpoints_per_seed == 0
            || self.min_checkpoints_per_seed > self.max_checkpoints_per_seed
            || self
                .seeds
                .len()
                .saturating_mul(self.max_checkpoints_per_seed)
                .saturating_mul(2)
                > MAX_TOTAL_CHECKPOINTS
        {
            return Err("feeding recovery checkpoint bounds are invalid".into());
        }
        if !self.min_recovery_rate.is_finite() || !(0.0..=1.0).contains(&self.min_recovery_rate) {
            return Err("feeding recovery rate must be finite and in [0, 1]".into());
        }
        if self.min_remaining_time_quanta >= curriculum.promotion.evaluation_sim_time_limit_quanta {
            return Err(
                "feeding recovery minimum remaining time reaches the episode deadline".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryEpisodeOutcome {
    Win,
    Loss,
    Timeout,
    SafetyAbort,
}

impl From<EpisodeOutcome> for RecoveryEpisodeOutcome {
    fn from(value: EpisodeOutcome) -> Self {
        match value {
            EpisodeOutcome::Win => Self::Win,
            EpisodeOutcome::Loss => Self::Loss,
            EpisodeOutcome::Timeout => Self::Timeout,
            EpisodeOutcome::SafetyAbort => Self::SafetyAbort,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FeedingRecoveryEpisode {
    pub stage: FeedingCurriculumStage,
    pub seed: u64,
    pub policy_outcome: RecoveryEpisodeOutcome,
    pub policy_terminal_cells: usize,
    pub policy_movement_successes: u64,
    pub policy_consume_successes: u64,
    pub policy_consumed_energy: u128,
    pub policy_objective_success: bool,
    pub checkpoints: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FeedingRecoveryCheckpoint {
    pub stage: FeedingCurriculumStage,
    pub seed: u64,
    pub checkpoint_ordinal: usize,
    pub policy_frontiers_elapsed: u64,
    pub sim_time_quanta: u64,
    pub remaining_time_quanta: u64,
    pub checkpoint_sha256: String,
    pub cells_at_checkpoint: usize,
    pub prefix_movement_successes: u64,
    pub prefix_consume_successes: u64,
    pub prefix_consumed_energy: u128,
    pub required_movement: bool,
    pub required_consumption: bool,
    pub teacher_outcome: RecoveryEpisodeOutcome,
    pub teacher_terminal_cells: usize,
    pub teacher_decisions: u64,
    pub teacher_movement_successes: u64,
    pub teacher_consume_successes: u64,
    pub teacher_consumed_energy: u128,
    pub recovered: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingRecoveryStageSummary {
    pub stage: FeedingCurriculumStage,
    pub scenario_hash: String,
    pub episodes: usize,
    pub policy_objective_successes: usize,
    pub policy_surviving_episodes: usize,
    pub checkpoints: usize,
    pub recoverable_checkpoints: usize,
    pub recovery_rate: f64,
    pub teacher_safety_aborts: usize,
    pub policy_safety_aborts: usize,
    pub minimum_checkpoints_observed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingRecoveryCheck {
    pub name: String,
    pub observed: f64,
    pub minimum: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingRecoveryReport {
    pub schema_version: u32,
    pub compiled_ruleset_hash: String,
    pub scenario_suite_hash: String,
    pub stages: Vec<FeedingRecoveryStageSummary>,
    pub episodes: Vec<FeedingRecoveryEpisode>,
    pub checkpoints: Vec<FeedingRecoveryCheckpoint>,
    pub checks: Vec<FeedingRecoveryCheck>,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingRecoveryArtifact {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub artifact_hash: String,
    pub source_config_sha256: String,
    pub behavior_clone_metadata_sha256: String,
    pub behavior_clone_model_sha256: String,
    pub config: TrainingConfig,
    pub options: FeedingRecoveryOptions,
    pub report: FeedingRecoveryReport,
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
    options: &'a FeedingRecoveryOptions,
    report: &'a FeedingRecoveryReport,
}

impl FeedingRecoveryArtifact {
    pub fn new(
        source_config_sha256: String,
        behavior_clone_metadata_sha256: String,
        behavior_clone_model_sha256: String,
        config: TrainingConfig,
        options: FeedingRecoveryOptions,
        report: FeedingRecoveryReport,
    ) -> Result<Self, String> {
        let mut artifact = Self {
            schema_version: FEEDING_RECOVERY_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").into(),
            code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
            artifact_hash: String::new(),
            source_config_sha256,
            behavior_clone_metadata_sha256,
            behavior_clone_model_sha256,
            config,
            options,
            report,
        };
        artifact.artifact_hash = artifact.recompute_hash()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn recompute_hash(&self) -> Result<String, String> {
        let identity = ArtifactIdentity {
            schema_version: self.schema_version,
            package_version: &self.package_version,
            code_revision: &self.code_revision,
            source_config_sha256: &self.source_config_sha256,
            behavior_clone_metadata_sha256: &self.behavior_clone_metadata_sha256,
            behavior_clone_model_sha256: &self.behavior_clone_model_sha256,
            config: &self.config,
            options: &self.options,
            report: &self.report,
        };
        let bytes = serde_json::to_vec(&identity)
            .map_err(|error| format!("failed to encode feeding recovery identity: {error}"))?;
        Ok(sha256(&bytes))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != FEEDING_RECOVERY_SCHEMA_VERSION
            || self.package_version.is_empty()
            || !valid_sha256(&self.source_config_sha256)
            || !valid_sha256(&self.behavior_clone_metadata_sha256)
            || !valid_sha256(&self.behavior_clone_model_sha256)
            || !valid_sha256(&self.artifact_hash)
        {
            return Err("feeding recovery artifact identity is invalid".into());
        }
        self.config.validate()?;
        self.options.validate(&self.config.feeding_curriculum)?;
        self.report.validate_against(&self.config, &self.options)?;
        if self.artifact_hash != self.recompute_hash()? {
            return Err("feeding recovery artifact hash mismatch".into());
        }
        Ok(())
    }
}

impl FeedingRecoveryReport {
    pub fn validate_against(
        &self,
        config: &TrainingConfig,
        options: &FeedingRecoveryOptions,
    ) -> Result<(), String> {
        options.validate(&config.feeding_curriculum)?;
        if self.schema_version != FEEDING_RECOVERY_SCHEMA_VERSION
            || !valid_sha256(&self.compiled_ruleset_hash)
            || !valid_sha256(&self.scenario_suite_hash)
            || self.stages.len() != 2
            || self.stages[0].stage != FeedingCurriculumStage::OnFood
            || self.stages[1].stage != FeedingCurriculumStage::AdjacentFood
            || self.episodes.len() != options.seeds.len().saturating_mul(2)
            || self.checkpoints.len() > MAX_TOTAL_CHECKPOINTS
        {
            return Err("feeding recovery report envelope is invalid".into());
        }
        validate_rows(
            self,
            options,
            config
                .feeding_curriculum
                .promotion
                .evaluation_sim_time_limit_quanta,
        )?;
        let expected_stages = summarize(
            config,
            options,
            &self.compiled_ruleset_hash,
            &self.episodes,
            &self.checkpoints,
        )?;
        let expected_checks = recovery_checks(&expected_stages, options);
        let expected_passed = expected_checks.iter().all(|check| check.passed);
        if self.scenario_suite_hash != scenario_suite_hash(&expected_stages)?
            || self.stages != expected_stages
            || self.checks != expected_checks
            || self.passed != expected_passed
        {
            return Err("feeding recovery summaries or verdict were modified".into());
        }
        Ok(())
    }
}

pub fn evaluate_feeding_recovery<B: Backend>(
    model: &PolicyValueNet<B>,
    config: &TrainingConfig,
    options: &FeedingRecoveryOptions,
    device: &B::Device,
) -> Result<FeedingRecoveryReport, String>
where
    f32: From<B::FloatElem>,
{
    config.validate()?;
    options.validate(&config.feeding_curriculum)?;
    let mut episodes = Vec::with_capacity(options.seeds.len() * 2);
    let mut checkpoints = Vec::new();
    let mut compiled_ruleset_hash = None;
    for stage in [
        FeedingCurriculumStage::OnFood,
        FeedingCurriculumStage::AdjacentFood,
    ] {
        let mut env_config = recovery_env_config(config, stage);
        env_config.opponent = OpponentProfile::Wait;
        for &seed in &options.seeds {
            let mut env = BlobEnv::new(env_config.clone(), config.reward.clone(), seed);
            env.enable_telemetry(recovery_telemetry());
            let compiled = env.compiled_ruleset_hash();
            if compiled_ruleset_hash
                .as_ref()
                .is_some_and(|expected| expected != &compiled)
            {
                return Err("feeding recovery environments compiled different rulesets".into());
            }
            compiled_ruleset_hash = Some(compiled);
            let episode = evaluate_policy_episode(
                model,
                &mut env,
                &env_config,
                &config.reward,
                stage,
                seed,
                options,
                device,
                &mut checkpoints,
            )?;
            episodes.push(episode);
        }
    }
    let compiled_ruleset_hash =
        compiled_ruleset_hash.ok_or("feeding recovery produced no environments")?;
    let stages = summarize(
        config,
        options,
        &compiled_ruleset_hash,
        &episodes,
        &checkpoints,
    )?;
    let checks = recovery_checks(&stages, options);
    let scenario_suite_hash = scenario_suite_hash(&stages)?;
    let report = FeedingRecoveryReport {
        schema_version: FEEDING_RECOVERY_SCHEMA_VERSION,
        compiled_ruleset_hash,
        scenario_suite_hash,
        stages,
        episodes,
        checkpoints,
        passed: checks.iter().all(|check| check.passed),
        checks,
    };
    report.validate_against(config, options)?;
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_policy_episode<B: Backend>(
    model: &PolicyValueNet<B>,
    env: &mut BlobEnv,
    env_config: &EnvConfig,
    reward: &RewardConfig,
    stage: FeedingCurriculumStage,
    seed: u64,
    options: &FeedingRecoveryOptions,
    device: &B::Device,
    checkpoints: &mut Vec<FeedingRecoveryCheckpoint>,
) -> Result<FeedingRecoveryEpisode, String>
where
    f32: From<B::FloatElem>,
{
    let deadline = env_config.victory.sim_time_limit_quanta;
    let mut movement_successes = 0_u64;
    let mut consume_successes = 0_u64;
    let mut consumed_energy = 0_u128;
    let mut policy_frontiers = 0_u64;
    let mut checkpoint_count = 0_usize;
    let mut next_checkpoint_time = 0_u64;

    loop {
        let now = env.sim_time_quanta();
        let enough_time = now.saturating_add(options.min_remaining_time_quanta) < deadline;
        let due = policy_frontiers > 0
            && checkpoint_count < options.max_checkpoints_per_seed
            && enough_time
            && (checkpoint_count == 0 || now >= next_checkpoint_time);
        if due {
            let checkpoint = env.checkpoint()?;
            let row = evaluate_checkpoint(
                checkpoint,
                env_config,
                reward,
                stage,
                seed,
                checkpoint_count,
                policy_frontiers,
                now,
                movement_successes,
                consume_successes,
                consumed_energy,
                options.teacher,
            )?;
            checkpoints.push(row);
            checkpoint_count += 1;
            next_checkpoint_time = now.saturating_add(options.checkpoint_interval_quanta);
        }

        let observations = env.get_policy_observations();
        let policy_choices = greedy_policy_choices(model, &observations, device);
        let result = env.step_with_policy_memory(&policy_choices);
        let telemetry = result
            .telemetry
            .as_ref()
            .ok_or("feeding recovery policy telemetry is disabled")?;
        movement_successes =
            movement_successes.saturating_add(telemetry.training.actions.movement.succeeded);
        consume_successes =
            consume_successes.saturating_add(telemetry.training.actions.consume.succeeded);
        consumed_energy =
            consumed_energy.saturating_add(telemetry.training.actions.consume.consumed_energy);
        policy_frontiers = policy_frontiers.saturating_add(1);
        if result.done {
            let outcome = result
                .outcome
                .ok_or("completed feeding recovery policy episode omitted its outcome")?;
            let objective_success = outcome != EpisodeOutcome::SafetyAbort
                && consume_successes > 0
                && consumed_energy > 0
                && (stage == FeedingCurriculumStage::OnFood || movement_successes > 0);
            return Ok(FeedingRecoveryEpisode {
                stage,
                seed,
                policy_outcome: outcome.into(),
                policy_terminal_cells: result.training_cells,
                policy_movement_successes: movement_successes,
                policy_consume_successes: consume_successes,
                policy_consumed_energy: consumed_energy,
                policy_objective_success: objective_success,
                checkpoints: checkpoint_count,
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_checkpoint(
    checkpoint: BlobEnvCheckpoint,
    env_config: &EnvConfig,
    reward: &RewardConfig,
    stage: FeedingCurriculumStage,
    seed: u64,
    checkpoint_ordinal: usize,
    policy_frontiers_elapsed: u64,
    sim_time_quanta: u64,
    prefix_movement_successes: u64,
    prefix_consume_successes: u64,
    prefix_consumed_energy: u128,
    teacher: MaintainedMindProfile,
) -> Result<FeedingRecoveryCheckpoint, String> {
    let checkpoint_sha256 = checkpoint_sha256(&checkpoint)?;
    let cells_at_checkpoint = checkpoint
        .host_cells
        .iter()
        .filter(|cell| cell.team_id.0 == 0)
        .count();
    let deadline = env_config.victory.sim_time_limit_quanta;
    let required_movement =
        stage == FeedingCurriculumStage::AdjacentFood && prefix_movement_successes == 0;
    let required_consumption = prefix_consume_successes == 0 || prefix_consumed_energy == 0;
    let mut branch = BlobEnv::from_checkpoint(env_config.clone(), reward.clone(), checkpoint)?;
    branch.enable_telemetry(recovery_telemetry());
    let mut teacher_decisions = 0_u64;
    let mut teacher_movement_successes = 0_u64;
    let mut teacher_consume_successes = 0_u64;
    let mut teacher_consumed_energy = 0_u128;

    loop {
        let prepared = branch.prepare_training_reference_inputs()?;
        let mut decisions = std::collections::HashMap::with_capacity(prepared.len());
        for (cell_id, input) in prepared {
            let decision = teacher.decide(&input);
            teacher_decisions = teacher_decisions.saturating_add(1);
            decisions.insert(cell_id, decision);
        }
        let result = branch.step_with_training_decisions(decisions);
        let telemetry = result
            .telemetry
            .as_ref()
            .ok_or("feeding recovery teacher telemetry is disabled")?;
        teacher_movement_successes = teacher_movement_successes
            .saturating_add(telemetry.training.actions.movement.succeeded);
        teacher_consume_successes =
            teacher_consume_successes.saturating_add(telemetry.training.actions.consume.succeeded);
        teacher_consumed_energy = teacher_consumed_energy
            .saturating_add(telemetry.training.actions.consume.consumed_energy);
        if result.done {
            let outcome = result
                .outcome
                .ok_or("completed feeding recovery teacher branch omitted its outcome")?;
            let recovered = outcome != EpisodeOutcome::SafetyAbort
                && result.training_cells > 0
                && (!required_movement || teacher_movement_successes > 0)
                && (!required_consumption
                    || (teacher_consume_successes > 0 && teacher_consumed_energy > 0));
            return Ok(FeedingRecoveryCheckpoint {
                stage,
                seed,
                checkpoint_ordinal,
                policy_frontiers_elapsed,
                sim_time_quanta,
                remaining_time_quanta: deadline.saturating_sub(sim_time_quanta),
                checkpoint_sha256,
                cells_at_checkpoint,
                prefix_movement_successes,
                prefix_consume_successes,
                prefix_consumed_energy,
                required_movement,
                required_consumption,
                teacher_outcome: outcome.into(),
                teacher_terminal_cells: result.training_cells,
                teacher_decisions,
                teacher_movement_successes,
                teacher_consume_successes,
                teacher_consumed_energy,
                recovered,
            });
        }
    }
}

fn summarize(
    config: &TrainingConfig,
    options: &FeedingRecoveryOptions,
    compiled_ruleset_hash: &str,
    episodes: &[FeedingRecoveryEpisode],
    checkpoints: &[FeedingRecoveryCheckpoint],
) -> Result<Vec<FeedingRecoveryStageSummary>, String> {
    let mut summaries = Vec::with_capacity(2);
    for stage in [
        FeedingCurriculumStage::OnFood,
        FeedingCurriculumStage::AdjacentFood,
    ] {
        let stage_episodes = episodes
            .iter()
            .filter(|episode| episode.stage == stage)
            .collect::<Vec<_>>();
        let stage_checkpoints = checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.stage == stage)
            .collect::<Vec<_>>();
        let checkpoint_count = stage_checkpoints.len();
        let recoverable = stage_checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.recovered)
            .count();
        let mut stage_env = recovery_env_config(config, stage);
        stage_env.opponent = OpponentProfile::Wait;
        let scenario_hash = ScenarioProfile::from(&stage_env).semantic_hash()?;
        let actual_ruleset = BlobEnv::new(stage_env, config.reward.clone(), options.seeds[0])
            .compiled_ruleset_hash();
        if actual_ruleset != compiled_ruleset_hash {
            return Err("feeding recovery summary ruleset does not match evaluation".into());
        }
        summaries.push(FeedingRecoveryStageSummary {
            stage,
            scenario_hash,
            episodes: stage_episodes.len(),
            policy_objective_successes: stage_episodes
                .iter()
                .filter(|episode| episode.policy_objective_success)
                .count(),
            policy_surviving_episodes: stage_episodes
                .iter()
                .filter(|episode| {
                    episode.policy_terminal_cells > 0
                        && episode.policy_outcome != RecoveryEpisodeOutcome::SafetyAbort
                })
                .count(),
            checkpoints: checkpoint_count,
            recoverable_checkpoints: recoverable,
            recovery_rate: ratio(recoverable, checkpoint_count),
            teacher_safety_aborts: stage_checkpoints
                .iter()
                .filter(|checkpoint| {
                    checkpoint.teacher_outcome == RecoveryEpisodeOutcome::SafetyAbort
                })
                .count(),
            policy_safety_aborts: stage_episodes
                .iter()
                .filter(|episode| episode.policy_outcome == RecoveryEpisodeOutcome::SafetyAbort)
                .count(),
            minimum_checkpoints_observed: stage_episodes
                .iter()
                .map(|episode| episode.checkpoints)
                .min()
                .unwrap_or(0),
        });
    }
    Ok(summaries)
}

fn recovery_checks(
    stages: &[FeedingRecoveryStageSummary],
    options: &FeedingRecoveryOptions,
) -> Vec<FeedingRecoveryCheck> {
    let mut checks = Vec::with_capacity(stages.len() * 4);
    for stage in stages {
        push_check(
            &mut checks,
            format!("{}_minimum_checkpoint_coverage", stage.stage),
            stage.minimum_checkpoints_observed as f64,
            options.min_checkpoints_per_seed as f64,
        );
        push_check(
            &mut checks,
            format!("{}_teacher_recovery_rate", stage.stage),
            stage.recovery_rate,
            options.min_recovery_rate,
        );
        push_check(
            &mut checks,
            format!("{}_no_teacher_safety_aborts", stage.stage),
            f64::from(stage.teacher_safety_aborts == 0),
            1.0,
        );
        push_check(
            &mut checks,
            format!("{}_no_policy_safety_aborts", stage.stage),
            f64::from(stage.policy_safety_aborts == 0),
            1.0,
        );
    }
    checks
}

fn scenario_suite_hash(stages: &[FeedingRecoveryStageSummary]) -> Result<String, String> {
    let identity = stages
        .iter()
        .map(|stage| (stage.stage, stage.scenario_hash.as_str()))
        .collect::<Vec<_>>();
    let bytes = serde_json::to_vec(&identity)
        .map_err(|error| format!("failed to encode feeding recovery scenario suite: {error}"))?;
    Ok(sha256(&bytes))
}

fn push_check(checks: &mut Vec<FeedingRecoveryCheck>, name: String, observed: f64, minimum: f64) {
    checks.push(FeedingRecoveryCheck {
        name,
        observed,
        minimum,
        passed: observed.is_finite() && observed >= minimum,
    });
}

fn validate_rows(
    report: &FeedingRecoveryReport,
    options: &FeedingRecoveryOptions,
    deadline: u64,
) -> Result<(), String> {
    let expected_pairs = [
        FeedingCurriculumStage::OnFood,
        FeedingCurriculumStage::AdjacentFood,
    ]
    .into_iter()
    .flat_map(|stage| {
        options
            .seeds
            .iter()
            .copied()
            .map(move |seed| (stage_key(stage), seed))
    })
    .collect::<HashSet<_>>();
    let actual_pairs = report
        .episodes
        .iter()
        .map(|episode| (stage_key(episode.stage), episode.seed))
        .collect::<HashSet<_>>();
    if actual_pairs != expected_pairs || actual_pairs.len() != report.episodes.len() {
        return Err("feeding recovery episodes do not cover the exact seed suite".into());
    }
    for episode in &report.episodes {
        let objective = episode.policy_outcome != RecoveryEpisodeOutcome::SafetyAbort
            && episode.policy_consume_successes > 0
            && episode.policy_consumed_energy > 0
            && (episode.stage == FeedingCurriculumStage::OnFood
                || episode.policy_movement_successes > 0);
        if episode.policy_objective_success != objective
            || episode.checkpoints > options.max_checkpoints_per_seed
        {
            return Err("feeding recovery episode row is inconsistent".into());
        }
    }
    let mut ordinals = HashSet::new();
    for checkpoint in &report.checkpoints {
        if !expected_pairs.contains(&(stage_key(checkpoint.stage), checkpoint.seed))
            || checkpoint.checkpoint_ordinal >= options.max_checkpoints_per_seed
            || !ordinals.insert((
                stage_key(checkpoint.stage),
                checkpoint.seed,
                checkpoint.checkpoint_ordinal,
            ))
            || !valid_sha256(&checkpoint.checkpoint_sha256)
            || checkpoint.cells_at_checkpoint == 0
            || checkpoint.teacher_decisions == 0
            || checkpoint.policy_frontiers_elapsed == 0
            || checkpoint.remaining_time_quanta
                != deadline.saturating_sub(checkpoint.sim_time_quanta)
            || checkpoint.remaining_time_quanta <= options.min_remaining_time_quanta
            || checkpoint.required_movement
                != (checkpoint.stage == FeedingCurriculumStage::AdjacentFood
                    && checkpoint.prefix_movement_successes == 0)
            || checkpoint.required_consumption
                != (checkpoint.prefix_consume_successes == 0
                    || checkpoint.prefix_consumed_energy == 0)
        {
            return Err("feeding recovery checkpoint identity or counters are invalid".into());
        }
        let recovered = checkpoint.teacher_outcome != RecoveryEpisodeOutcome::SafetyAbort
            && checkpoint.teacher_terminal_cells > 0
            && (!checkpoint.required_movement || checkpoint.teacher_movement_successes > 0)
            && (!checkpoint.required_consumption
                || (checkpoint.teacher_consume_successes > 0
                    && checkpoint.teacher_consumed_energy > 0));
        if checkpoint.recovered != recovered {
            return Err("feeding recovery checkpoint verdict is inconsistent".into());
        }
    }
    for episode in &report.episodes {
        let rows = report
            .checkpoints
            .iter()
            .filter(|checkpoint| {
                checkpoint.stage == episode.stage && checkpoint.seed == episode.seed
            })
            .collect::<Vec<_>>();
        if rows.len() != episode.checkpoints
            || !(0..rows.len()).all(|ordinal| {
                rows.iter()
                    .any(|checkpoint| checkpoint.checkpoint_ordinal == ordinal)
            })
        {
            return Err("feeding recovery episode checkpoint count is inconsistent".into());
        }
    }
    Ok(())
}

fn stage_key(stage: FeedingCurriculumStage) -> u8 {
    match stage {
        FeedingCurriculumStage::OnFood => 0,
        FeedingCurriculumStage::AdjacentFood => 1,
        FeedingCurriculumStage::Contact => 2,
        FeedingCurriculumStage::Skirmish => 3,
        FeedingCurriculumStage::Competitive => 4,
    }
}

fn recovery_env_config(config: &TrainingConfig, stage: FeedingCurriculumStage) -> EnvConfig {
    let mut env = config
        .feeding_curriculum
        .environment_for_stage(&config.env, stage);
    env.max_episode_len = config
        .feeding_curriculum
        .promotion
        .evaluation_max_episode_len;
    env.victory.sim_time_limit_quanta = config
        .feeding_curriculum
        .promotion
        .evaluation_sim_time_limit_quanta;
    env
}

fn recovery_telemetry() -> TelemetryConfig {
    TelemetryConfig {
        enabled: true,
        state_sample_interval_steps: u64::MAX,
        max_state_samples_per_episode: 2,
        episode_log_stride: 0,
    }
}

fn checkpoint_sha256(checkpoint: &BlobEnvCheckpoint) -> Result<String, String> {
    struct HashWriter(Sha256);
    impl Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = HashWriter(Sha256::new());
    serde_json::to_writer(&mut writer, checkpoint)
        .map_err(|error| format!("failed to hash feeding recovery checkpoint: {error}"))?;
    Ok(format!("{:x}", writer.0.finalize()))
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub fn publish_feeding_recovery(
    output: &Path,
    artifact: &FeedingRecoveryArtifact,
) -> Result<PathBuf, String> {
    artifact.validate()?;
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable feeding recovery {}",
            output.display()
        ));
    }
    let nonce = RECOVERY_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("feeding recovery output needs a UTF-8 file name")?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(artifact)
        .map_err(|error| format!("failed to encode feeding recovery: {error}"))?;
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

pub fn load_feeding_recovery(path: &Path) -> Result<FeedingRecoveryArtifact, String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > MAX_ARTIFACT_BYTES {
        return Err(format!(
            "feeding recovery artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
        ));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let artifact: FeedingRecoveryArtifact = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    artifact.validate()?;
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FeedingPromotionGateConfig, ModelConfig};
    use crate::model::PolicyValueNetConfig;

    type TestBackend = burn::backend::NdArray<f32>;

    fn setup() -> (TrainingConfig, FeedingRecoveryOptions) {
        let mut config = TrainingConfig::default();
        config.env.world_size = 8;
        config.env.cells_per_team = 1;
        config.env.num_plants = 2;
        config.env.num_scattered_energy = 2;
        config.env.initial_energy = 500;
        config.env.max_episode_len = 512;
        config.feeding_curriculum.enabled = true;
        config.feeding_curriculum.promotion = FeedingPromotionGateConfig {
            evaluation_max_episode_len: 512,
            evaluation_sim_time_limit_quanta: 65_536,
            ..FeedingPromotionGateConfig::default()
        };
        config.model = ModelConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 4,
        };
        let options = FeedingRecoveryOptions {
            teacher: MaintainedMindProfile::CollisionAwareForager,
            seeds: vec![91],
            checkpoint_interval_quanta: 256,
            max_checkpoints_per_seed: 1,
            min_checkpoints_per_seed: 1,
            min_remaining_time_quanta: 128,
            min_recovery_rate: 0.0,
        };
        (config, options)
    }

    #[test]
    fn counterfactual_recovery_is_deterministic_and_tamper_detecting() {
        let _guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let (config, options) = setup();
        config.validate().unwrap();
        let device = Default::default();
        <TestBackend as Backend>::seed(&device, 42);
        let model = PolicyValueNetConfig {
            hidden1: config.model.hidden1,
            hidden2: config.model.hidden2,
            recurrent_size: config.model.recurrent_size,
        }
        .init::<TestBackend>(&device);
        let left = evaluate_feeding_recovery(&model, &config, &options, &device).unwrap();
        let right = evaluate_feeding_recovery(&model, &config, &options, &device).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.episodes.len(), 2);
        assert_eq!(left.checkpoints.len(), 2);
        assert!(left
            .checkpoints
            .iter()
            .all(|checkpoint| checkpoint.policy_frontiers_elapsed > 0));

        let artifact = FeedingRecoveryArtifact::new(
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            config,
            options,
            left,
        )
        .unwrap();
        artifact.validate().unwrap();
        let mut tampered = artifact;
        tampered.report.checkpoints[0].recovered = !tampered.report.checkpoints[0].recovered;
        assert!(tampered.validate().is_err());
    }

    #[test]
    fn recovery_options_reject_seed_and_time_ambiguity() {
        let (config, mut options) = setup();
        options.seeds.push(options.seeds[0]);
        assert!(options.validate(&config.feeding_curriculum).is_err());
        options.seeds.pop();
        options.min_remaining_time_quanta = config
            .feeding_curriculum
            .promotion
            .evaluation_sim_time_limit_quanta;
        assert!(options.validate(&config.feeding_curriculum).is_err());
    }
}
