//! Bounded, resumable execution and paired aggregation for rules sweeps.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use blob_engine::resolution::ReferenceSimulation;
use serde::{Deserialize, Serialize};

use crate::artifact::{verify_best_pointer, verify_checkpoint_metadata};
use crate::config::{FeedingCurriculumStage, ScenarioProfile, TrainingConfig};
use crate::sweep::{
    experiment_config_hash, sha256, RulesSweepManifest, RulesSweepRun, RULES_SWEEP_SCHEMA_VERSION,
};
use crate::telemetry::{ActionFamilyTelemetry, TrainingTelemetrySummary};
use crate::viability_gate::{verify_viability_gate_requirement, ViabilityGateRequirement};

pub const SWEEP_EXECUTION_SCHEMA_VERSION: u32 = 10;
const MAX_CONTROL_FILE_BYTES: u64 = 16 * 1024 * 1024;
static EXECUTION_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub struct SweepExecutorOptions {
    pub train_program: PathBuf,
    pub train_arguments: Vec<String>,
    /// Optional immutable OCI image identity for remote/container execution.
    pub trainer_container_digest: Option<String>,
    pub max_parallel: usize,
    pub threads_per_run: Option<usize>,
    pub retry_failed: bool,
    /// When set, no trainer may launch until the published decision is freshly
    /// reproduced from its exact matrix and policy and matches this manifest.
    pub required_viability_gate: Option<ViabilityGateRequirement>,
}

/// Immutable execution identity shared by every attempt in a sweep. This is
/// deliberately separate from the scientific plan because the trainer does
/// not exist when a plan is commonly published.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SweepExecutionContract {
    pub schema_version: u32,
    pub manifest_sha256: String,
    pub trainer_sha256: String,
    pub trainer_size_bytes: u64,
    pub trainer_container_digest: Option<String>,
    pub train_arguments: Vec<String>,
    pub threads_per_run: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SweepRunState {
    Running,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SweepRunStatus {
    pub schema_version: u32,
    pub manifest_sha256: String,
    pub execution_contract_sha256: String,
    pub trainer_sha256: String,
    pub variant: String,
    pub replicate: usize,
    pub training_seed: u64,
    pub experiment_config_sha256: String,
    pub state: SweepRunState,
    pub attempts: usize,
    pub started_unix_millis: u64,
    pub finished_unix_millis: Option<u64>,
    pub resume_checkpoint: Option<String>,
    pub exit_code: Option<i32>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingTailMetrics {
    pub update: usize,
    pub actions: u64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub entropy: f64,
    pub approximate_kl: f64,
    pub explained_variance: f64,
    pub episodes: usize,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub win_rate: f64,
    pub average_episode_len: f64,
    pub average_reward: f64,
    pub training_cells_alive: usize,
    pub completed_transitions: usize,
    pub discarded_tails: usize,
    pub mean_elapsed_time: f64,
    pub actions_per_second: f64,
    pub minimum_sim_time_quanta: u64,
    pub maximum_sim_time_quanta: u64,
    pub total_sim_time_quanta: u128,
    pub simulation_quanta_per_second: f64,
    /// Peak host resident set reported by the trainer. This excludes device
    /// memory and is absent on platforms without `getrusage` support.
    pub peak_resident_set_bytes: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvaluationTailMetrics {
    pub update: usize,
    pub actions: u64,
    pub episodes: usize,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub win_rate: f64,
    pub worst_case_win_rate: f64,
    pub average_episode_len: f64,
    pub average_reward: f64,
    pub evaluation_actions: u64,
    pub seed_start: u64,
    pub seed_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SweepRunResult {
    pub schema_version: u32,
    pub manifest_sha256: String,
    pub execution_contract_sha256: String,
    pub trainer_sha256: String,
    pub variant: String,
    pub replicate: usize,
    pub training_seed: u64,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub experiment_config_sha256: String,
    pub config_file_sha256: String,
    pub completed_unix_millis: u64,
    pub training: TrainingTailMetrics,
    pub evaluation: Option<EvaluationTailMetrics>,
    pub competency: Option<CompetencyRunMetrics>,
    pub micro_combat: Option<MicroCombatRunMetrics>,
    pub telemetry: Option<TelemetryRunMetrics>,
}

/// Terminal and learning-speed evidence reconstructed from every complete
/// held-out micro-combat evaluation boundary in a run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MicroCombatRunMetrics {
    pub evaluations: usize,
    pub final_survival_success_rate: f64,
    pub final_elimination_success_rate: f64,
    pub best_survival_success_rate: f64,
    pub best_elimination_success_rate: f64,
    pub final_attack_commitments_per_episode: f64,
    pub best_attack_commitments_per_episode: f64,
    /// Exactly 0.0 or 1.0 so it can share the paired-summary machinery.
    pub qualification_reached: f64,
    /// First qualifying action divided by the terminal action count. Runs
    /// that never qualify are right-censored at 1.0.
    pub normalized_actions_to_qualification_or_budget: f64,
    pub first_qualified_actions: Option<u64>,
}

/// Independently verified best-checkpoint evidence for curricula that require
/// ecology and combat to coexist at one held-out boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompetencyRunMetrics {
    /// Exactly 0.0 or 1.0 in persisted run results. Paired in-memory
    /// differences may be -1.0, 0.0, or 1.0 before aggregation.
    pub joint_qualified: f64,
    pub best_update: Option<usize>,
    pub best_actions: Option<u64>,
    pub on_food_survival_rate: Option<f64>,
    pub adjacent_food_survival_rate: Option<f64>,
    pub skirmish_kills: Option<u64>,
    pub skirmish_damage: Option<u128>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TelemetryRunMetrics {
    pub completed_episodes: f64,
    pub win_rate: f64,
    pub average_episode_steps: f64,
    pub training_births_per_1000_actions: f64,
    pub training_deaths_per_1000_actions: f64,
    pub training_kills_per_1000_actions: f64,
    pub attack_selection_rate: f64,
    pub attack_success_rate: f64,
    pub signal_selection_rate: f64,
    pub signal_energy_per_1000_actions: f64,
    pub signal_decay_per_1000_actions: f64,
    pub terrain_signal_erased_per_1000_actions: f64,
    pub raw_damage_per_successful_attack: f64,
    pub applied_damage_per_successful_attack: f64,
    pub damage_efficiency: f64,
    pub guard_mitigation_rate: f64,
    pub overkill_rate: f64,
    pub elevation_units_lifted_per_1000_actions: f64,
    pub elevation_units_dumped_per_1000_actions: f64,
    pub terrain_mass_lifted_per_1000_actions: f64,
    pub terrain_mass_dumped_per_1000_actions: f64,
    pub action_rejection_rate: f64,
    pub action_frustration_rate: f64,
    pub action_contention_rate: f64,
    pub mean_training_cells: f64,
    pub mean_opponent_cells: f64,
    pub training_plant_occupancy_rate: f64,
    pub opponent_plant_occupancy_rate: f64,
    pub training_major_food_occupancy_rate: f64,
    pub opponent_major_food_occupancy_rate: f64,
    pub mean_training_assimilated_energy: f64,
    pub mean_environment_plant_energy: f64,
    pub mean_environment_loose_energy: f64,
    pub mean_environment_diffuse_energy: f64,
    pub mean_environment_signal_energy: f64,
    pub mean_signal_active_channel_tiles: f64,
    pub mean_signal_observation_total_variation: f64,
    pub mean_encounter_edges: f64,
    pub mean_training_spatial_entropy: f64,
    pub mean_resource_concentration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SweepExecutionSummary {
    pub schema_version: u32,
    pub manifest_sha256: String,
    pub execution_contract_sha256: String,
    pub trainer_sha256: String,
    pub total_runs: usize,
    pub succeeded: usize,
    pub failed: usize,
    pub still_running: usize,
    pub aggregate_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetricSummary {
    pub samples: usize,
    pub mean: f64,
    pub sample_standard_deviation: f64,
    pub standard_error: f64,
    pub confidence_95_half_width: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SweepMetricSummaries {
    pub training_win_rate: MetricSummary,
    pub training_average_reward: MetricSummary,
    pub training_average_episode_len: MetricSummary,
    pub actions_per_second: MetricSummary,
    pub simulation_quanta_per_second: MetricSummary,
    pub evaluation_win_rate: Option<MetricSummary>,
    pub evaluation_worst_case_win_rate: Option<MetricSummary>,
    pub evaluation_average_reward: Option<MetricSummary>,
    pub evaluation_average_episode_len: Option<MetricSummary>,
    /// Mean of exact 0/1 run outcomes, or candidate-minus-control for paired
    /// comparisons.
    pub joint_qualification_rate: Option<MetricSummary>,
    pub peak_resident_set_bytes: Option<MetricSummary>,
    pub micro_combat: Option<MicroCombatSweepMetricSummaries>,
    pub telemetry: Option<TelemetrySweepMetricSummaries>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MicroCombatSweepMetricSummaries {
    pub final_survival_success_rate: MetricSummary,
    pub final_elimination_success_rate: MetricSummary,
    pub best_survival_success_rate: MetricSummary,
    pub best_elimination_success_rate: MetricSummary,
    pub final_attack_commitments_per_episode: MetricSummary,
    pub best_attack_commitments_per_episode: MetricSummary,
    pub qualification_rate: MetricSummary,
    pub normalized_actions_to_qualification_or_budget: MetricSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TelemetrySweepMetricSummaries {
    pub completed_episodes: MetricSummary,
    pub win_rate: MetricSummary,
    pub average_episode_steps: MetricSummary,
    pub training_births_per_1000_actions: MetricSummary,
    pub training_deaths_per_1000_actions: MetricSummary,
    pub training_kills_per_1000_actions: MetricSummary,
    pub attack_selection_rate: MetricSummary,
    pub attack_success_rate: MetricSummary,
    pub signal_selection_rate: MetricSummary,
    pub signal_energy_per_1000_actions: MetricSummary,
    pub signal_decay_per_1000_actions: MetricSummary,
    pub terrain_signal_erased_per_1000_actions: MetricSummary,
    pub raw_damage_per_successful_attack: MetricSummary,
    pub applied_damage_per_successful_attack: MetricSummary,
    pub damage_efficiency: MetricSummary,
    pub guard_mitigation_rate: MetricSummary,
    pub overkill_rate: MetricSummary,
    pub elevation_units_lifted_per_1000_actions: MetricSummary,
    pub elevation_units_dumped_per_1000_actions: MetricSummary,
    pub terrain_mass_lifted_per_1000_actions: MetricSummary,
    pub terrain_mass_dumped_per_1000_actions: MetricSummary,
    pub action_rejection_rate: MetricSummary,
    pub action_frustration_rate: MetricSummary,
    pub action_contention_rate: MetricSummary,
    pub mean_training_cells: MetricSummary,
    pub mean_opponent_cells: MetricSummary,
    pub training_plant_occupancy_rate: MetricSummary,
    pub opponent_plant_occupancy_rate: MetricSummary,
    pub training_major_food_occupancy_rate: MetricSummary,
    pub opponent_major_food_occupancy_rate: MetricSummary,
    pub mean_training_assimilated_energy: MetricSummary,
    pub mean_environment_plant_energy: MetricSummary,
    pub mean_environment_loose_energy: MetricSummary,
    pub mean_environment_diffuse_energy: MetricSummary,
    pub mean_environment_signal_energy: MetricSummary,
    pub mean_signal_active_channel_tiles: MetricSummary,
    pub mean_signal_observation_total_variation: MetricSummary,
    pub mean_encounter_edges: MetricSummary,
    pub mean_training_spatial_entropy: MetricSummary,
    pub mean_resource_concentration: MetricSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VariantAggregate {
    pub variant: String,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub seeds: Vec<u64>,
    pub metrics: SweepMetricSummaries,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PairedVariantComparison {
    pub baseline_variant: String,
    pub candidate_variant: String,
    pub seeds: Vec<u64>,
    /// Candidate minus baseline. Pairing occurs by training seed before any
    /// summary is calculated.
    pub differences: SweepMetricSummaries,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RulesSweepAggregate {
    pub schema_version: u32,
    pub manifest_sha256: String,
    pub execution_contract_sha256: String,
    pub trainer_sha256: String,
    pub baseline_variant: String,
    pub variants: Vec<VariantAggregate>,
    pub paired_comparisons: Vec<PairedVariantComparison>,
}

pub(crate) struct ValidatedSweep {
    pub(crate) manifest: RulesSweepManifest,
    pub(crate) manifest_sha256: String,
    pub(crate) root: PathBuf,
    pub(crate) configs: Vec<TrainingConfig>,
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > MAX_CONTROL_FILE_BYTES {
        return Err(format!(
            "{} is {length} bytes; control-file limit is {MAX_CONTROL_FILE_BYTES}",
            path.display()
        ));
    }
    fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to sync directory {}: {error}", path.display()))
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let nonce = EXECUTION_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} needs a UTF-8 file name", path.display()))?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
    bytes.push(b'\n');
    let result = (|| {
        let mut file = File::create(&temporary)
            .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
        file.write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
        fs::rename(&temporary, path)
            .map_err(|error| format!("failed to publish {}: {error}", path.display()))?;
        sync_directory(parent)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn executable_identity(path: &Path) -> Result<(String, u64), String> {
    let resolved = if path.components().count() == 1 {
        std::env::var_os("PATH")
            .and_then(|paths| {
                std::env::split_paths(&paths)
                    .map(|directory| directory.join(path))
                    .find(|candidate| candidate.is_file())
            })
            .ok_or_else(|| format!("trainer {} was not found on PATH", path.display()))?
    } else {
        path.to_path_buf()
    };
    let bytes = fs::read(&resolved)
        .map_err(|error| format!("failed to read trainer {}: {error}", resolved.display()))?;
    let size = u64::try_from(bytes.len()).map_err(|_| "trainer is too large".to_string())?;
    Ok((sha256(&bytes), size))
}

fn requested_execution_contract(
    sweep: &ValidatedSweep,
    options: &SweepExecutorOptions,
) -> Result<SweepExecutionContract, String> {
    if let Some(digest) = &options.trainer_container_digest {
        let Some(hex) = digest.strip_prefix("sha256:") else {
            return Err("trainer_container_digest must use sha256:<64 lowercase hex>".into());
        };
        if hex.len() != 64
            || !hex
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err("trainer_container_digest must use sha256:<64 lowercase hex>".into());
        }
    }
    let (trainer_sha256, trainer_size_bytes) = executable_identity(&options.train_program)?;
    Ok(SweepExecutionContract {
        schema_version: SWEEP_EXECUTION_SCHEMA_VERSION,
        manifest_sha256: sweep.manifest_sha256.clone(),
        trainer_sha256,
        trainer_size_bytes,
        trainer_container_digest: options.trainer_container_digest.clone(),
        train_arguments: options.train_arguments.clone(),
        threads_per_run: options.threads_per_run,
    })
}

fn load_execution_contract(
    sweep: &ValidatedSweep,
) -> Result<(SweepExecutionContract, String), String> {
    let path = sweep.root.join("execution-contract.json");
    let bytes = read_bounded(&path)?;
    let contract: SweepExecutionContract = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    if contract.schema_version != SWEEP_EXECUTION_SCHEMA_VERSION
        || contract.manifest_sha256 != sweep.manifest_sha256
    {
        return Err(format!(
            "execution contract identity mismatch in {}",
            path.display()
        ));
    }
    Ok((contract, sha256(&bytes)))
}

fn publish_or_verify_execution_contract(
    sweep: &ValidatedSweep,
    requested: &SweepExecutionContract,
) -> Result<(SweepExecutionContract, String), String> {
    let path = sweep.root.join("execution-contract.json");
    if path.exists() {
        let (existing, digest) = load_execution_contract(sweep)?;
        if existing != *requested {
            return Err(format!(
                "trainer or execution settings differ from immutable contract {}",
                path.display()
            ));
        }
        Ok((existing, digest))
    } else {
        write_json_atomic(&path, requested)?;
        load_execution_contract(sweep)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    serde_json::from_slice(&read_bounded(path)?)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))
}

pub(crate) fn load_validated_sweep(manifest_file: &Path) -> Result<ValidatedSweep, String> {
    let manifest_file = fs::canonicalize(manifest_file).map_err(|error| {
        format!(
            "failed to resolve sweep manifest {}: {error}",
            manifest_file.display()
        )
    })?;
    let manifest_bytes = read_bounded(&manifest_file)?;
    let manifest: RulesSweepManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("failed to decode {}: {error}", manifest_file.display()))?;
    if manifest.schema_version != RULES_SWEEP_SCHEMA_VERSION {
        return Err(format!(
            "unsupported rules sweep schema {}; expected {}",
            manifest.schema_version, RULES_SWEEP_SCHEMA_VERSION
        ));
    }
    let root = manifest_file
        .parent()
        .ok_or_else(|| "sweep manifest has no parent".to_string())?
        .to_path_buf();
    if fs::canonicalize(&manifest.output_directory).map_err(|error| {
        format!(
            "failed to resolve manifest output directory {}: {error}",
            manifest.output_directory
        )
    })? != root
    {
        return Err("manifest output_directory does not match its published location".into());
    }
    if manifest.seeds.len() < 3 || manifest.variants.is_empty() {
        return Err("manifest does not describe a replicated rules sweep".into());
    }
    if manifest.runs.len() != manifest.seeds.len() * manifest.variants.len() {
        return Err("manifest run matrix is incomplete".into());
    }

    let variant_names = manifest
        .variants
        .iter()
        .map(|variant| variant.name.as_str())
        .collect::<HashSet<_>>();
    let mut identities = HashSet::new();
    let mut configs = Vec::with_capacity(manifest.runs.len());
    for run in &manifest.runs {
        if !variant_names.contains(run.variant.as_str())
            || run.replicate >= manifest.seeds.len()
            || manifest.seeds[run.replicate] != run.training_seed
            || !identities.insert((run.variant.as_str(), run.training_seed))
        {
            return Err(format!(
                "run {} seed {} is not a unique member of the paired matrix",
                run.variant, run.training_seed
            ));
        }
        let expected_run = root
            .join(&run.variant)
            .join(format!("seed-{}", run.training_seed));
        let actual_run = fs::canonicalize(&run.run_directory).map_err(|error| {
            format!(
                "failed to resolve run directory {}: {error}",
                run.run_directory
            )
        })?;
        if actual_run != expected_run {
            return Err(format!(
                "run {} escapes its manifest directory",
                run.variant
            ));
        }
        let expected_config = expected_run.join("config.toml");
        if fs::canonicalize(&run.config_file)
            .map_err(|error| format!("failed to resolve run config {}: {error}", run.config_file))?
            != expected_config
        {
            return Err(format!("run {} has a substituted config path", run.variant));
        }
        let config_bytes = read_bounded(&expected_config)?;
        if sha256(&config_bytes) != run.config_file_sha256 {
            return Err(format!("run {} config SHA-256 mismatch", run.variant));
        }
        let config = TrainingConfig::from_toml_str(
            std::str::from_utf8(&config_bytes)
                .map_err(|error| format!("run config is not UTF-8: {error}"))?,
        )?;
        config.validate()?;
        if config.seed != run.training_seed
            || Path::new(&config.checkpoint_dir) != expected_run.join("artifacts")
            || config.env.rules.semantic_hash().to_string() != run.semantic_ruleset_hash
            || ScenarioProfile::from(&config.env).semantic_hash()? != run.scenario_hash
            || experiment_config_hash(&config)? != run.experiment_config_sha256
        {
            return Err(format!(
                "run {} config does not match its manifest",
                run.variant
            ));
        }
        let compiled_hash = ReferenceSimulation::new(
            config.env.world_size,
            config.env.world_size,
            config.env.rules.clone(),
        )
        .map_err(|error| format!("run {} rules are invalid: {error}", run.variant))?
        .compiled_ruleset_hash()
        .to_string();
        if compiled_hash != run.compiled_ruleset_hash {
            return Err(format!(
                "run {} compiled ruleset hash mismatch",
                run.variant
            ));
        }
        configs.push(config);
    }

    Ok(ValidatedSweep {
        manifest,
        manifest_sha256: sha256(&manifest_bytes),
        root,
        configs,
    })
}

fn field<'a>(row: &'a HashMap<&str, &str>, name: &str) -> Result<&'a str, String> {
    row.get(name)
        .copied()
        .ok_or_else(|| format!("metrics row is missing {name}"))
}

fn number<T: std::str::FromStr>(row: &HashMap<&str, &str>, name: &str) -> Result<T, String> {
    field(row, name)?
        .parse()
        .map_err(|_| format!("metrics field {name} is invalid"))
}

fn optional_number<T: std::str::FromStr>(
    row: &HashMap<&str, &str>,
    name: &str,
) -> Result<Option<T>, String> {
    match row.get(name).copied().filter(|value| !value.is_empty()) {
        Some(value) => value
            .parse()
            .map(Some)
            .map_err(|_| format!("metrics field {name} is invalid")),
        None => Ok(None),
    }
}

fn csv_rows(content: &str) -> Result<Vec<HashMap<&str, &str>>, String> {
    let mut lines = content.lines();
    let headers = lines
        .next()
        .ok_or_else(|| "metrics file has no header".to_string())?
        .split(',')
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let values = line.split(',').collect::<Vec<_>>();
        if values.len() == headers.len() {
            rows.push(headers.iter().copied().zip(values).collect());
        }
    }
    Ok(rows)
}

fn training_tail(path: &Path) -> Result<TrainingTailMetrics, String> {
    let content = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let rows = csv_rows(&content)?;
    let peak_resident_set_bytes = rows
        .iter()
        .map(|row| optional_number::<f64>(row, "peak_resident_set_bytes"))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .reduce(f64::max);
    let row = rows
        .last()
        .ok_or_else(|| format!("{} has no complete metric rows", path.display()))?;
    Ok(TrainingTailMetrics {
        update: number(row, "update")?,
        actions: number(row, "actions")?,
        policy_loss: number(row, "policy_loss")?,
        value_loss: number(row, "value_loss")?,
        entropy: number(row, "entropy")?,
        approximate_kl: number(row, "approx_kl")?,
        explained_variance: number(row, "explained_variance")?,
        episodes: number(row, "episodes")?,
        wins: number(row, "wins")?,
        losses: number(row, "losses")?,
        timeouts: number(row, "timeouts")?,
        win_rate: number(row, "win_rate")?,
        average_episode_len: number(row, "avg_ep_len")?,
        average_reward: number(row, "avg_reward")?,
        training_cells_alive: number(row, "training_cells_alive")?,
        completed_transitions: number(row, "completed_transitions")?,
        discarded_tails: number(row, "discarded_tails")?,
        mean_elapsed_time: number(row, "mean_elapsed_time")?,
        actions_per_second: number(row, "actions_per_second")?,
        minimum_sim_time_quanta: number(row, "min_sim_time_quanta")?,
        maximum_sim_time_quanta: number(row, "max_sim_time_quanta")?,
        total_sim_time_quanta: number(row, "total_sim_time_quanta")?,
        simulation_quanta_per_second: number(row, "simulation_quanta_per_second")?,
        peak_resident_set_bytes,
    })
}

#[derive(Debug, Clone, Default)]
struct MicroCombatEvaluationPoint {
    scenario_rows: HashMap<String, (String, usize, usize, usize, u64)>,
    survival_episodes: usize,
    survival_successes: usize,
    elimination_episodes: usize,
    elimination_successes: usize,
    safety_aborts: usize,
    attacks_committed: u64,
}

fn micro_combat_curve(
    path: &Path,
    config: &TrainingConfig,
    terminal_actions: u64,
) -> Result<Option<MicroCombatRunMetrics>, String> {
    let micro = &config.combat_curriculum.micro_combat;
    if !micro.enabled {
        return Ok(None);
    }
    let expected_scenario_names = micro
        .suite
        .as_ref()
        .ok_or("enabled micro-combat evaluation has no suite")?
        .scenarios
        .iter()
        .map(|scenario| scenario.name.as_str())
        .collect::<HashSet<_>>();
    let content = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let rows = csv_rows(&content)?;
    let mut points = BTreeMap::<(usize, u64), MicroCombatEvaluationPoint>::new();
    for row in rows {
        let key = (number(&row, "update")?, number(&row, "actions")?);
        let scenario = field(&row, "scenario")?.to_string();
        let objective = field(&row, "objective")?.to_string();
        let episodes: usize = number(&row, "episodes")?;
        let successes: usize = number(&row, "objective_successes")?;
        let safety_aborts: usize = number(&row, "safety_aborts")?;
        let attacks_committed: u64 = number(&row, "attacks_committed")?;
        let point = points.entry(key).or_default();
        let identity = (
            objective.clone(),
            episodes,
            successes,
            safety_aborts,
            attacks_committed,
        );
        if let Some(existing) = point.scenario_rows.get(&scenario) {
            if existing != &identity {
                return Err(format!(
                    "micro-combat evaluation boundary contains inconsistent duplicate scenario {scenario}"
                ));
            }
            continue;
        }
        point.scenario_rows.insert(scenario, identity);
        point.safety_aborts = point.safety_aborts.saturating_add(safety_aborts);
        point.attacks_committed = point.attacks_committed.saturating_add(attacks_committed);
        match objective.as_str() {
            "Survival" => {
                point.survival_episodes = point.survival_episodes.saturating_add(episodes);
                point.survival_successes = point.survival_successes.saturating_add(successes);
            }
            "Elimination" => {
                point.elimination_episodes = point.elimination_episodes.saturating_add(episodes);
                point.elimination_successes = point.elimination_successes.saturating_add(successes);
            }
            objective => return Err(format!("unknown micro-combat objective {objective}")),
        }
    }
    if points.is_empty() {
        return Err("micro-combat evaluation was enabled but no complete rows exist".into());
    }
    if points.values().any(|point| {
        point.scenario_rows.len() != expected_scenario_names.len()
            || point
                .scenario_rows
                .keys()
                .any(|name| !expected_scenario_names.contains(name.as_str()))
            || point.survival_episodes == 0
            || point.elimination_episodes == 0
    }) {
        return Err(
            "micro-combat learning curve contains an incomplete evaluation boundary".into(),
        );
    }
    let rate = |successes: usize, episodes: usize| successes as f64 / episodes as f64;
    let scored = points
        .iter()
        .map(|((_, actions), point)| {
            (
                *actions,
                rate(point.survival_successes, point.survival_episodes),
                rate(point.elimination_successes, point.elimination_episodes),
                point.attacks_committed as f64
                    / (point.survival_episodes + point.elimination_episodes) as f64,
                point.safety_aborts,
            )
        })
        .collect::<Vec<_>>();
    let &(final_actions, final_survival, final_elimination, final_attacks, _) = scored
        .last()
        .expect("nonempty micro-combat curve has a final point");
    if final_actions != terminal_actions {
        return Err(format!(
            "terminal micro-combat evaluation covers {final_actions} actions but training completed at {terminal_actions}"
        ));
    }
    let first_qualified_actions =
        scored
            .iter()
            .find_map(|(actions, survival, elimination, attacks, aborts)| {
                (*aborts == 0
                    && *survival >= micro.min_survival_objective_success_rate
                    && *elimination >= micro.min_elimination_objective_success_rate
                    && *attacks >= micro.min_attack_commitments_per_episode)
                    .then_some(*actions)
            });
    let qualification_reached = if first_qualified_actions.is_some() {
        1.0
    } else {
        0.0
    };
    let normalized_actions_to_qualification_or_budget =
        first_qualified_actions.map_or(1.0, |actions| {
            if terminal_actions == 0 {
                1.0
            } else {
                actions as f64 / terminal_actions as f64
            }
        });
    Ok(Some(MicroCombatRunMetrics {
        evaluations: scored.len(),
        final_survival_success_rate: final_survival,
        final_elimination_success_rate: final_elimination,
        best_survival_success_rate: scored
            .iter()
            .map(|(_, survival, _, _, _)| *survival)
            .fold(0.0, f64::max),
        best_elimination_success_rate: scored
            .iter()
            .map(|(_, _, elimination, _, _)| *elimination)
            .fold(0.0, f64::max),
        final_attack_commitments_per_episode: final_attacks,
        best_attack_commitments_per_episode: scored
            .iter()
            .map(|(_, _, _, attacks, _)| *attacks)
            .fold(0.0, f64::max),
        qualification_reached,
        normalized_actions_to_qualification_or_budget,
        first_qualified_actions,
    }))
}

fn evaluation_tail(path: &Path) -> Result<Option<EvaluationTailMetrics>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let rows = csv_rows(&content)?;
    let Some(row) = rows
        .iter()
        .rev()
        .find(|row| row.get("opponent").is_some_and(|value| *value == "suite"))
    else {
        return Ok(None);
    };
    Ok(Some(EvaluationTailMetrics {
        update: number(row, "update")?,
        actions: number(row, "actions")?,
        episodes: number(row, "episodes")?,
        wins: number(row, "wins")?,
        losses: number(row, "losses")?,
        timeouts: number(row, "timeouts")?,
        win_rate: number(row, "win_rate")?,
        worst_case_win_rate: number(row, "worst_case_win_rate")?,
        average_episode_len: number(row, "avg_ep_len")?,
        average_reward: number(row, "avg_reward")?,
        evaluation_actions: number(row, "evaluation_actions")?,
        seed_start: number(row, "seed_start")?,
        seed_count: number(row, "seed_count")?,
    }))
}

fn total_completed(actions: &ActionFamilyTelemetry) -> u64 {
    actions
        .wait
        .completed
        .saturating_add(actions.movement.completed)
        .saturating_add(actions.attack.completed)
        .saturating_add(actions.guard.completed)
        .saturating_add(actions.consume.completed)
        .saturating_add(actions.split.completed)
        .saturating_add(actions.regurgitate.completed)
        .saturating_add(actions.signal.completed)
        .saturating_add(actions.excavate.completed)
        .saturating_add(actions.deposit_terrain.completed)
}

fn total_status(
    actions: &ActionFamilyTelemetry,
    select: impl Fn(&crate::telemetry::ActionTelemetry) -> u64,
) -> u64 {
    [
        &actions.wait,
        &actions.movement,
        &actions.attack,
        &actions.guard,
        &actions.consume,
        &actions.split,
        &actions.regurgitate,
        &actions.signal,
        &actions.excavate,
        &actions.deposit_terrain,
    ]
    .into_iter()
    .map(select)
    .fold(0_u64, u64::saturating_add)
}

fn rate(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn rate_u128(numerator: u128, denominator: u128) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

impl From<&TrainingTelemetrySummary> for TelemetryRunMetrics {
    fn from(summary: &TrainingTelemetrySummary) -> Self {
        let actions = &summary.training.actions;
        let completed_actions = total_completed(actions);
        let per_1000 = |value: u64| rate(value.saturating_mul(1_000), completed_actions);
        let per_1000_u128 =
            |value: u128| rate_u128(value.saturating_mul(1_000), u128::from(completed_actions));
        let successful_attacks = u128::from(actions.attack.succeeded);
        let damage = &summary.training.damage;
        Self {
            completed_episodes: summary.completed_episodes as f64,
            win_rate: rate(summary.wins, summary.completed_episodes),
            average_episode_steps: summary.average_completed_episode_steps,
            training_births_per_1000_actions: per_1000(summary.training.births),
            training_deaths_per_1000_actions: per_1000(summary.training.deaths),
            training_kills_per_1000_actions: per_1000(summary.training.kills),
            attack_selection_rate: rate(
                actions.attack.committed,
                total_status(actions, |a| a.committed),
            ),
            attack_success_rate: rate(actions.attack.succeeded, actions.attack.completed),
            signal_selection_rate: rate(
                summary.training.signals.emitted_decisions,
                total_status(actions, |a| a.committed),
            ),
            signal_energy_per_1000_actions: per_1000_u128(summary.training.signals.total_energy()),
            signal_decay_per_1000_actions: per_1000_u128(
                summary.signal_field.total_decayed_energy(),
            ),
            terrain_signal_erased_per_1000_actions: per_1000_u128(
                summary.signal_field.total_terrain_erased_energy(),
            ),
            raw_damage_per_successful_attack: rate_u128(damage.raw_dealt, successful_attacks),
            applied_damage_per_successful_attack: rate_u128(
                damage.applied_dealt,
                successful_attacks,
            ),
            damage_efficiency: rate_u128(damage.applied_dealt, damage.raw_dealt),
            guard_mitigation_rate: rate_u128(damage.mitigated_by_target_guard, damage.raw_dealt),
            overkill_rate: rate_u128(damage.overkill_dealt, damage.raw_dealt),
            elevation_units_lifted_per_1000_actions: per_1000(
                summary.training.terrain.elevation_units_lifted,
            ),
            elevation_units_dumped_per_1000_actions: per_1000(
                summary.training.terrain.elevation_units_dumped,
            ),
            terrain_mass_lifted_per_1000_actions: per_1000_u128(
                summary.training.terrain.material_mass_lifted,
            ),
            terrain_mass_dumped_per_1000_actions: per_1000_u128(
                summary.training.terrain.material_mass_dumped,
            ),
            action_rejection_rate: rate(total_status(actions, |a| a.rejected), completed_actions),
            action_frustration_rate: rate(
                total_status(actions, |a| a.frustrated),
                completed_actions,
            ),
            action_contention_rate: rate(total_status(actions, |a| a.contested), completed_actions),
            mean_training_cells: summary.sample_means.training_cells,
            mean_opponent_cells: summary.sample_means.opponent_cells,
            training_plant_occupancy_rate: if summary.sample_means.training_cells == 0.0 {
                0.0
            } else {
                summary.sample_means.training_cells_on_plants / summary.sample_means.training_cells
            },
            opponent_plant_occupancy_rate: if summary.sample_means.opponent_cells == 0.0 {
                0.0
            } else {
                summary.sample_means.opponent_cells_on_plants / summary.sample_means.opponent_cells
            },
            training_major_food_occupancy_rate: if summary.sample_means.training_cells == 0.0 {
                0.0
            } else {
                summary.sample_means.training_cells_on_major_food
                    / summary.sample_means.training_cells
            },
            opponent_major_food_occupancy_rate: if summary.sample_means.opponent_cells == 0.0 {
                0.0
            } else {
                summary.sample_means.opponent_cells_on_major_food
                    / summary.sample_means.opponent_cells
            },
            mean_training_assimilated_energy: summary.sample_means.training_assimilated_energy,
            mean_environment_plant_energy: summary.sample_means.environment_plant_energy,
            mean_environment_loose_energy: summary.sample_means.environment_loose_energy,
            mean_environment_diffuse_energy: summary.sample_means.environment_diffuse_energy,
            mean_environment_signal_energy: summary.sample_means.environment_signal_energy,
            mean_signal_active_channel_tiles: summary.sample_means.signal_active_channel_tiles,
            mean_signal_observation_total_variation: summary
                .sample_means
                .signal_observation_total_variation,
            mean_encounter_edges: summary.sample_means.encounter_edges,
            mean_training_spatial_entropy: summary.sample_means.training_spatial_entropy,
            mean_resource_concentration: summary.sample_means.resource_concentration,
        }
    }
}

fn collect_result(
    manifest_sha256: &str,
    execution_contract_sha256: &str,
    trainer_sha256: &str,
    run: &RulesSweepRun,
    config: &TrainingConfig,
) -> Result<SweepRunResult, String> {
    let artifact_root = Path::new(&config.checkpoint_dir);
    let training = training_tail(&artifact_root.join("metrics.csv"))?;
    verify_training_completion(&training, config)?;
    let evaluation = if config.fixed_evaluation_enabled() {
        let evaluation =
            evaluation_tail(&artifact_root.join("evaluation.csv"))?.ok_or_else(|| {
                "evaluation was enabled but no complete suite row was published".to_string()
            })?;
        if evaluation.actions != training.actions {
            return Err(format!(
                "terminal evaluation covers {} actions but training completed at {}",
                evaluation.actions, training.actions
            ));
        }
        Some(evaluation)
    } else {
        None
    };
    let competency = collect_competency_metrics(artifact_root, config)?;
    let micro_combat = micro_combat_curve(
        &artifact_root.join("micro-combat-evaluation.csv"),
        config,
        training.actions,
    )?;
    let telemetry = if config.telemetry.enabled {
        let summary: TrainingTelemetrySummary =
            read_json(&artifact_root.join("telemetry/summary.json"))?;
        if summary.schema_version != crate::telemetry::TELEMETRY_SCHEMA_VERSION {
            return Err("unsupported training telemetry schema".into());
        }
        let committed = total_status(&summary.training.actions, |action| action.committed);
        if committed != training.actions {
            return Err(format!(
                "telemetry covers {committed} training commitments but metrics cover {} actions",
                training.actions
            ));
        }
        Some(TelemetryRunMetrics::from(&summary))
    } else {
        None
    };
    Ok(SweepRunResult {
        schema_version: SWEEP_EXECUTION_SCHEMA_VERSION,
        manifest_sha256: manifest_sha256.to_string(),
        execution_contract_sha256: execution_contract_sha256.to_string(),
        trainer_sha256: trainer_sha256.to_string(),
        variant: run.variant.clone(),
        replicate: run.replicate,
        training_seed: run.training_seed,
        semantic_ruleset_hash: run.semantic_ruleset_hash.clone(),
        compiled_ruleset_hash: run.compiled_ruleset_hash.clone(),
        scenario_hash: run.scenario_hash.clone(),
        experiment_config_sha256: run.experiment_config_sha256.clone(),
        config_file_sha256: run.config_file_sha256.clone(),
        completed_unix_millis: unix_millis(),
        training,
        evaluation,
        competency,
        micro_combat,
        telemetry,
    })
}

fn collect_competency_metrics(
    artifact_root: &Path,
    config: &TrainingConfig,
) -> Result<Option<CompetencyRunMetrics>, String> {
    if !config.feeding_curriculum.enabled || !config.combat_curriculum.enabled {
        return Ok(None);
    }
    if !artifact_root.join("best.json").exists() {
        return Ok(Some(CompetencyRunMetrics {
            joint_qualified: 0.0,
            best_update: None,
            best_actions: None,
            on_food_survival_rate: None,
            adjacent_food_survival_rate: None,
            skirmish_kills: None,
            skirmish_damage: None,
        }));
    }
    let best = verify_best_pointer(artifact_root)?;
    let feeding = best
        .retention_feeding_evaluation
        .as_ref()
        .or(best.feeding_evaluation.as_ref())
        .ok_or("qualified checkpoint has no feeding report")?;
    let stage = |wanted| {
        feeding
            .stages
            .iter()
            .find(|stage| stage.stage == wanted)
            .ok_or("qualified checkpoint has an incomplete feeding report")
    };
    let on_food = stage(FeedingCurriculumStage::OnFood)?;
    let adjacent_food = stage(FeedingCurriculumStage::AdjacentFood)?;
    let contact = best
        .contact_evaluation
        .as_ref()
        .ok_or("qualified checkpoint has no combat report")?;
    let skirmish_damage = contact
        .variants
        .iter()
        .filter(|variant| variant.stage == FeedingCurriculumStage::Skirmish)
        .map(|variant| variant.damage_dealt)
        .sum();
    Ok(Some(CompetencyRunMetrics {
        joint_qualified: 1.0,
        best_update: Some(best.update),
        best_actions: Some(best.actions),
        on_food_survival_rate: Some(on_food.survival_rate),
        adjacent_food_survival_rate: Some(adjacent_food.survival_rate),
        skirmish_kills: Some(contact.kills_for_stage(FeedingCurriculumStage::Skirmish)),
        skirmish_damage: Some(skirmish_damage),
    }))
}

fn verify_training_completion(
    training: &TrainingTailMetrics,
    config: &TrainingConfig,
) -> Result<(), String> {
    if let Some(target) = config.total_simulation_quanta_per_env {
        if training.minimum_sim_time_quanta < target {
            return Err(format!(
                "training stopped at {} of {} simulation quanta per environment",
                training.minimum_sim_time_quanta, target
            ));
        }
        if training.actions > config.total_timesteps {
            return Err(format!(
                "training exceeded its {}-action safety cap with {} actions",
                config.total_timesteps, training.actions
            ));
        }
    } else if training.actions < config.total_timesteps {
        return Err(format!(
            "training stopped at {} of {} actions",
            training.actions, config.total_timesteps
        ));
    }
    Ok(())
}

fn result_matches_run(
    result: &SweepRunResult,
    manifest_sha256: &str,
    execution_contract_sha256: &str,
    trainer_sha256: &str,
    run: &RulesSweepRun,
) -> bool {
    result.schema_version == SWEEP_EXECUTION_SCHEMA_VERSION
        && result.manifest_sha256 == manifest_sha256
        && result.execution_contract_sha256 == execution_contract_sha256
        && result.trainer_sha256 == trainer_sha256
        && result.variant == run.variant
        && result.replicate == run.replicate
        && result.training_seed == run.training_seed
        && result.semantic_ruleset_hash == run.semantic_ruleset_hash
        && result.compiled_ruleset_hash == run.compiled_ruleset_hash
        && result.scenario_hash == run.scenario_hash
        && result.experiment_config_sha256 == run.experiment_config_sha256
        && result.config_file_sha256 == run.config_file_sha256
        && result
            .training
            .peak_resident_set_bytes
            .is_none_or(|bytes| bytes.is_finite() && bytes >= 0.0)
        && result.micro_combat.as_ref().is_none_or(|metrics| {
            metrics.evaluations > 0
                && [
                    metrics.final_survival_success_rate,
                    metrics.final_elimination_success_rate,
                    metrics.best_survival_success_rate,
                    metrics.best_elimination_success_rate,
                    metrics.qualification_reached,
                    metrics.normalized_actions_to_qualification_or_budget,
                ]
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
                && metrics.final_attack_commitments_per_episode.is_finite()
                && metrics.final_attack_commitments_per_episode >= 0.0
                && metrics.best_attack_commitments_per_episode.is_finite()
                && metrics.best_attack_commitments_per_episode >= 0.0
                && (metrics.qualification_reached == 0.0
                    && metrics.first_qualified_actions.is_none()
                    || metrics.qualification_reached == 1.0
                        && metrics.first_qualified_actions.is_some())
        })
        && result.competency.as_ref().is_none_or(|metrics| {
            (metrics.joint_qualified == 0.0
                && metrics.best_update.is_none()
                && metrics.best_actions.is_none()
                && metrics.on_food_survival_rate.is_none()
                && metrics.adjacent_food_survival_rate.is_none()
                && metrics.skirmish_kills.is_none()
                && metrics.skirmish_damage.is_none())
                || (metrics.joint_qualified == 1.0
                    && metrics.best_update.is_some()
                    && metrics.best_actions.is_some()
                    && metrics
                        .on_food_survival_rate
                        .is_some_and(|value| (0.0..=1.0).contains(&value))
                    && metrics
                        .adjacent_food_survival_rate
                        .is_some_and(|value| (0.0..=1.0).contains(&value))
                    && metrics.skirmish_kills.is_some()
                    && metrics.skirmish_damage.is_some())
        })
}

fn publish_or_verify_result(path: &Path, result: &SweepRunResult) -> Result<(), String> {
    if path.exists() {
        let existing: SweepRunResult = read_json(path)?;
        if !result_matches_run(
            &existing,
            &result.manifest_sha256,
            &result.execution_contract_sha256,
            &result.trainer_sha256,
            &RulesSweepRun {
                variant: result.variant.clone(),
                replicate: result.replicate,
                training_seed: result.training_seed,
                semantic_ruleset_hash: result.semantic_ruleset_hash.clone(),
                compiled_ruleset_hash: result.compiled_ruleset_hash.clone(),
                scenario_hash: result.scenario_hash.clone(),
                experiment_config_sha256: result.experiment_config_sha256.clone(),
                config_file_sha256: result.config_file_sha256.clone(),
                run_directory: String::new(),
                config_file: String::new(),
            },
        ) || existing.training != result.training
            || existing.evaluation != result.evaluation
            || existing.micro_combat != result.micro_combat
            || existing.telemetry != result.telemetry
        {
            return Err(format!(
                "refusing to replace inconsistent result {}",
                path.display()
            ));
        }
        return Ok(());
    }
    write_json_atomic(path, result)
}

fn latest_resume_checkpoint(
    config: &TrainingConfig,
    compiled_ruleset_hash: &str,
) -> Result<Option<PathBuf>, String> {
    let root = Path::new(&config.checkpoint_dir);
    if !root.exists() {
        return Ok(None);
    }
    let mut latest: Option<(u64, PathBuf)> = None;
    for entry in fs::read_dir(root)
        .map_err(|error| format!("failed to inspect {}: {error}", root.display()))?
    {
        let entry =
            entry.map_err(|error| format!("failed to inspect checkpoint entry: {error}"))?;
        if !entry
            .file_type()
            .map_err(|error| error.to_string())?
            .is_dir()
            || !entry
                .file_name()
                .to_string_lossy()
                .starts_with("checkpoint-")
        {
            continue;
        }
        let metadata = verify_checkpoint_metadata(&entry.path())?;
        if metadata.config != *config
            || metadata.training_seed != config.seed
            || metadata.compiled_ruleset_hash != compiled_ruleset_hash
        {
            return Err(format!(
                "checkpoint {} does not belong to this run",
                entry.path().display()
            ));
        }
        if metadata.actions < config.total_timesteps
            && latest
                .as_ref()
                .is_none_or(|(actions, _)| metadata.actions > *actions)
        {
            latest = Some((metadata.actions, entry.path()));
        }
    }
    Ok(latest.map(|(_, path)| path))
}

fn read_status(path: &Path) -> Result<Option<SweepRunStatus>, String> {
    if path.exists() {
        read_json(path).map(Some)
    } else {
        Ok(None)
    }
}

fn run_lease_is_held(path: &Path) -> Result<bool, String> {
    let lease = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .map_err(|error| format!("failed to open run lease {}: {error}", path.display()))?;
    match lease.try_lock() {
        Ok(()) => Ok(false),
        Err(std::fs::TryLockError::WouldBlock) => Ok(true),
        Err(std::fs::TryLockError::Error(error)) => Err(format!(
            "failed to inspect run lease {}: {error}",
            path.display()
        )),
    }
}

fn status_matches_run(
    status: &SweepRunStatus,
    manifest_sha256: &str,
    execution_contract_sha256: &str,
    trainer_sha256: &str,
    run: &RulesSweepRun,
) -> bool {
    status.schema_version == SWEEP_EXECUTION_SCHEMA_VERSION
        && status.manifest_sha256 == manifest_sha256
        && status.execution_contract_sha256 == execution_contract_sha256
        && status.trainer_sha256 == trainer_sha256
        && status.variant == run.variant
        && status.replicate == run.replicate
        && status.training_seed == run.training_seed
        && status.experiment_config_sha256 == run.experiment_config_sha256
}

fn execute_one(
    manifest_sha256: &str,
    execution_contract_sha256: &str,
    contract: &SweepExecutionContract,
    run: &RulesSweepRun,
    config: &TrainingConfig,
    options: &SweepExecutorOptions,
) -> Result<(), String> {
    let run_directory = Path::new(&run.run_directory);
    let status_path = run_directory.join("status.json");
    let result_path = run_directory.join("result.json");
    let lease_path = run_directory.join("trainer.lock");
    let existing_status = read_status(&status_path)?;
    if let Some(status) = &existing_status {
        if !status_matches_run(
            status,
            manifest_sha256,
            execution_contract_sha256,
            &contract.trainer_sha256,
            run,
        ) {
            return Err(format!(
                "status identity mismatch in {}",
                status_path.display()
            ));
        }
        if status.state == SweepRunState::Succeeded {
            let result: SweepRunResult = read_json(&result_path)?;
            if !result_matches_run(
                &result,
                manifest_sha256,
                execution_contract_sha256,
                &contract.trainer_sha256,
                run,
            ) || verify_training_completion(&result.training, config).is_err()
            {
                return Err(format!(
                    "completed result identity mismatch for {}",
                    run.variant
                ));
            }
            return Ok(());
        }
        if status.state == SweepRunState::Failed && !options.retry_failed {
            return Ok(());
        }
        if status.state == SweepRunState::Running && run_lease_is_held(&lease_path)? {
            return Ok(());
        }
    }

    // Recover the narrow crash window after the child completed but before its
    // result/status records were committed. A complete metrics tail is enough
    // to avoid replaying an already-finished immutable run.
    if let Ok(result) = collect_result(
        manifest_sha256,
        execution_contract_sha256,
        &contract.trainer_sha256,
        run,
        config,
    ) {
        publish_or_verify_result(&result_path, &result)?;
        let attempts = existing_status.as_ref().map_or(0, |status| status.attempts);
        write_json_atomic(
            &status_path,
            &SweepRunStatus {
                schema_version: SWEEP_EXECUTION_SCHEMA_VERSION,
                manifest_sha256: manifest_sha256.to_string(),
                execution_contract_sha256: execution_contract_sha256.to_string(),
                trainer_sha256: contract.trainer_sha256.clone(),
                variant: run.variant.clone(),
                replicate: run.replicate,
                training_seed: run.training_seed,
                experiment_config_sha256: run.experiment_config_sha256.clone(),
                state: SweepRunState::Succeeded,
                attempts,
                started_unix_millis: existing_status
                    .as_ref()
                    .map_or_else(unix_millis, |status| status.started_unix_millis),
                finished_unix_millis: Some(unix_millis()),
                resume_checkpoint: existing_status.and_then(|status| status.resume_checkpoint),
                exit_code: Some(0),
                error: None,
            },
        )?;
        return Ok(());
    }

    let resume_checkpoint = latest_resume_checkpoint(config, &run.compiled_ruleset_hash)?;
    let attempts = existing_status
        .as_ref()
        .map_or(1, |status| status.attempts + 1);
    let started = unix_millis();
    let mut status = SweepRunStatus {
        schema_version: SWEEP_EXECUTION_SCHEMA_VERSION,
        manifest_sha256: manifest_sha256.to_string(),
        execution_contract_sha256: execution_contract_sha256.to_string(),
        trainer_sha256: contract.trainer_sha256.clone(),
        variant: run.variant.clone(),
        replicate: run.replicate,
        training_seed: run.training_seed,
        experiment_config_sha256: run.experiment_config_sha256.clone(),
        state: SweepRunState::Running,
        attempts,
        started_unix_millis: started,
        finished_unix_millis: None,
        resume_checkpoint: resume_checkpoint
            .as_ref()
            .map(|path| path.to_string_lossy().into_owned()),
        exit_code: None,
        error: None,
    };
    write_json_atomic(&status_path, &status)?;

    let (current_trainer_sha256, current_trainer_size) =
        executable_identity(&options.train_program)?;
    if current_trainer_sha256 != contract.trainer_sha256
        || current_trainer_size != contract.trainer_size_bytes
    {
        status.state = SweepRunState::Failed;
        status.finished_unix_millis = Some(unix_millis());
        status.error = Some("trainer bytes changed after the execution contract was bound".into());
        write_json_atomic(&status_path, &status)?;
        return Err(status.error.expect("failed status has an error"));
    }

    let stdout_path = run_directory.join(format!("attempt-{attempts:04}.stdout.log"));
    let stderr_path = run_directory.join(format!("attempt-{attempts:04}.stderr.log"));
    let stdout = File::create(&stdout_path)
        .map_err(|error| format!("failed to create {}: {error}", stdout_path.display()))?;
    let stderr = File::create(&stderr_path)
        .map_err(|error| format!("failed to create {}: {error}", stderr_path.display()))?;
    let mut command = Command::new(&options.train_program);
    command
        .args(&options.train_arguments)
        .arg("--config")
        .arg(&run.config_file)
        .arg("--run-lock")
        .arg(&lease_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    if let Some(checkpoint) = &resume_checkpoint {
        command.arg("--resume").arg(checkpoint);
    }
    if let Some(threads) = options.threads_per_run {
        command.env("RAYON_NUM_THREADS", threads.to_string());
    }

    let process_result = command.status().map_err(|error| {
        format!(
            "failed to start {}: {error}",
            options.train_program.display()
        )
    });
    let exit_code = process_result
        .as_ref()
        .ok()
        .and_then(|result| result.code());
    let outcome = match process_result {
        Ok(process_status) if process_status.success() => collect_result(
            manifest_sha256,
            execution_contract_sha256,
            &contract.trainer_sha256,
            run,
            config,
        )
        .and_then(|result| publish_or_verify_result(&result_path, &result)),
        Ok(process_status) => Err(format!("trainer exited with {process_status}")),
        Err(error) => Err(error),
    };
    status.finished_unix_millis = Some(unix_millis());
    status.exit_code = exit_code;
    match &outcome {
        Ok(()) => status.state = SweepRunState::Succeeded,
        Err(error) => {
            status.state = SweepRunState::Failed;
            status.error = Some(error.clone());
        }
    }
    write_json_atomic(&status_path, &status)?;
    outcome
}

/// Execute a validated sweep with at most `max_parallel` child trainers. The
/// manifest lock is an OS advisory lock: it survives as a harmless file, while
/// ownership is automatically released if the executor crashes.
pub fn execute_rules_sweep(
    manifest_file: &Path,
    options: &SweepExecutorOptions,
) -> Result<SweepExecutionSummary, String> {
    if options.max_parallel == 0 {
        return Err("max_parallel must be positive".into());
    }
    if options.threads_per_run == Some(0) {
        return Err("threads_per_run must be positive when set".into());
    }
    let sweep = load_validated_sweep(manifest_file)?;
    if let Some(requirement) = &options.required_viability_gate {
        verify_viability_gate_requirement(requirement, &sweep.manifest_sha256)?;
    }
    let lock_path = sweep.root.join("executor.lock");
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|error| format!("failed to open {}: {error}", lock_path.display()))?;
    lock.try_lock().map_err(|error| {
        format!(
            "another executor holds the sweep lock {}: {error}",
            lock_path.display()
        )
    })?;
    let requested_contract = requested_execution_contract(&sweep, options)?;
    let (execution_contract, execution_contract_sha256) =
        publish_or_verify_execution_contract(&sweep, &requested_contract)?;

    let next = AtomicUsize::new(0);
    let errors = Mutex::new(Vec::new());
    let workers = options.max_parallel.min(sweep.manifest.runs.len());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(run) = sweep.manifest.runs.get(index) else {
                    break;
                };
                if let Err(error) = execute_one(
                    &sweep.manifest_sha256,
                    &execution_contract_sha256,
                    &execution_contract,
                    run,
                    &sweep.configs[index],
                    options,
                ) {
                    errors
                        .lock()
                        .expect("sweep error lock poisoned")
                        .push(format!(
                            "{} seed {}: {error}",
                            run.variant, run.training_seed
                        ));
                }
            });
        }
    });

    let mut succeeded = 0;
    let mut failed = 0;
    let mut still_running = 0;
    for run in &sweep.manifest.runs {
        match read_status(&Path::new(&run.run_directory).join("status.json"))? {
            Some(status) if status.state == SweepRunState::Succeeded => succeeded += 1,
            Some(status) if status.state == SweepRunState::Failed => failed += 1,
            Some(_) => still_running += 1,
            None => failed += 1,
        }
    }
    let aggregate_file = if succeeded == sweep.manifest.runs.len() {
        let path = sweep.root.join("aggregate.json");
        aggregate_validated_sweep(
            &sweep,
            &execution_contract,
            &execution_contract_sha256,
            &path,
        )?;
        Some(path.to_string_lossy().into_owned())
    } else {
        None
    };
    let summary = SweepExecutionSummary {
        schema_version: SWEEP_EXECUTION_SCHEMA_VERSION,
        manifest_sha256: sweep.manifest_sha256,
        execution_contract_sha256,
        trainer_sha256: execution_contract.trainer_sha256,
        total_runs: sweep.manifest.runs.len(),
        succeeded,
        failed,
        still_running,
        aggregate_file,
    };
    write_json_atomic(&sweep.root.join("execution-summary.json"), &summary)?;
    let errors = errors.into_inner().expect("sweep error lock poisoned");
    if errors.is_empty() {
        Ok(summary)
    } else {
        Err(format!(
            "{} run(s) failed:\n{}",
            errors.len(),
            errors.join("\n")
        ))
    }
}

fn metric_summary(values: &[f64]) -> MetricSummary {
    let samples = values.len();
    let mean = values.iter().sum::<f64>() / samples as f64;
    let variance = if samples > 1 {
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / (samples - 1) as f64
    } else {
        0.0
    };
    let sample_standard_deviation = variance.sqrt();
    let standard_error = sample_standard_deviation / (samples as f64).sqrt();
    // Two-sided 95% Student's t critical values for 1..=30 degrees of
    // freedom. Small replicated sweeps must not pretend they have the normal
    // approximation's much tighter interval.
    const T_95: [f64; 30] = [
        12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042,
    ];
    let critical = samples
        .checked_sub(2)
        .and_then(|index| T_95.get(index))
        .copied()
        .unwrap_or(1.96);
    MetricSummary {
        samples,
        mean,
        sample_standard_deviation,
        standard_error,
        confidence_95_half_width: critical * standard_error,
    }
}

fn summaries(results: &[&SweepRunResult]) -> SweepMetricSummaries {
    fn summary_of(
        results: &[&SweepRunResult],
        select: impl Fn(&SweepRunResult) -> f64,
    ) -> MetricSummary {
        metric_summary(
            &results
                .iter()
                .map(|result| select(result))
                .collect::<Vec<_>>(),
        )
    }
    fn evaluation_summary(
        results: &[&SweepRunResult],
        select: impl Fn(&EvaluationTailMetrics) -> f64,
    ) -> Option<MetricSummary> {
        let values = results
            .iter()
            .filter_map(|result| result.evaluation.as_ref().map(&select))
            .collect::<Vec<_>>();
        (!values.is_empty()).then(|| metric_summary(&values))
    }
    fn telemetry_summaries(results: &[&SweepRunResult]) -> Option<TelemetrySweepMetricSummaries> {
        let telemetry = results
            .iter()
            .filter_map(|result| result.telemetry.as_ref())
            .collect::<Vec<_>>();
        if telemetry.len() != results.len() {
            return None;
        }
        let summarize = |select: fn(&TelemetryRunMetrics) -> f64| {
            metric_summary(
                &telemetry
                    .iter()
                    .map(|metrics| select(metrics))
                    .collect::<Vec<_>>(),
            )
        };
        Some(TelemetrySweepMetricSummaries {
            completed_episodes: summarize(|m| m.completed_episodes),
            win_rate: summarize(|m| m.win_rate),
            average_episode_steps: summarize(|m| m.average_episode_steps),
            training_births_per_1000_actions: summarize(|m| m.training_births_per_1000_actions),
            training_deaths_per_1000_actions: summarize(|m| m.training_deaths_per_1000_actions),
            training_kills_per_1000_actions: summarize(|m| m.training_kills_per_1000_actions),
            attack_selection_rate: summarize(|m| m.attack_selection_rate),
            attack_success_rate: summarize(|m| m.attack_success_rate),
            signal_selection_rate: summarize(|m| m.signal_selection_rate),
            signal_energy_per_1000_actions: summarize(|m| m.signal_energy_per_1000_actions),
            signal_decay_per_1000_actions: summarize(|m| m.signal_decay_per_1000_actions),
            terrain_signal_erased_per_1000_actions: summarize(|m| {
                m.terrain_signal_erased_per_1000_actions
            }),
            raw_damage_per_successful_attack: summarize(|m| m.raw_damage_per_successful_attack),
            applied_damage_per_successful_attack: summarize(|m| {
                m.applied_damage_per_successful_attack
            }),
            damage_efficiency: summarize(|m| m.damage_efficiency),
            guard_mitigation_rate: summarize(|m| m.guard_mitigation_rate),
            overkill_rate: summarize(|m| m.overkill_rate),
            elevation_units_lifted_per_1000_actions: summarize(|m| {
                m.elevation_units_lifted_per_1000_actions
            }),
            elevation_units_dumped_per_1000_actions: summarize(|m| {
                m.elevation_units_dumped_per_1000_actions
            }),
            terrain_mass_lifted_per_1000_actions: summarize(|m| {
                m.terrain_mass_lifted_per_1000_actions
            }),
            terrain_mass_dumped_per_1000_actions: summarize(|m| {
                m.terrain_mass_dumped_per_1000_actions
            }),
            action_rejection_rate: summarize(|m| m.action_rejection_rate),
            action_frustration_rate: summarize(|m| m.action_frustration_rate),
            action_contention_rate: summarize(|m| m.action_contention_rate),
            mean_training_cells: summarize(|m| m.mean_training_cells),
            mean_opponent_cells: summarize(|m| m.mean_opponent_cells),
            training_plant_occupancy_rate: summarize(|m| m.training_plant_occupancy_rate),
            opponent_plant_occupancy_rate: summarize(|m| m.opponent_plant_occupancy_rate),
            training_major_food_occupancy_rate: summarize(|m| m.training_major_food_occupancy_rate),
            opponent_major_food_occupancy_rate: summarize(|m| m.opponent_major_food_occupancy_rate),
            mean_training_assimilated_energy: summarize(|m| m.mean_training_assimilated_energy),
            mean_environment_plant_energy: summarize(|m| m.mean_environment_plant_energy),
            mean_environment_loose_energy: summarize(|m| m.mean_environment_loose_energy),
            mean_environment_diffuse_energy: summarize(|m| m.mean_environment_diffuse_energy),
            mean_environment_signal_energy: summarize(|m| m.mean_environment_signal_energy),
            mean_signal_active_channel_tiles: summarize(|m| m.mean_signal_active_channel_tiles),
            mean_signal_observation_total_variation: summarize(|m| {
                m.mean_signal_observation_total_variation
            }),
            mean_encounter_edges: summarize(|m| m.mean_encounter_edges),
            mean_training_spatial_entropy: summarize(|m| m.mean_training_spatial_entropy),
            mean_resource_concentration: summarize(|m| m.mean_resource_concentration),
        })
    }
    fn micro_combat_summaries(
        results: &[&SweepRunResult],
    ) -> Option<MicroCombatSweepMetricSummaries> {
        let micro = results
            .iter()
            .filter_map(|result| result.micro_combat.as_ref())
            .collect::<Vec<_>>();
        if micro.len() != results.len() {
            return None;
        }
        let summarize = |select: fn(&MicroCombatRunMetrics) -> f64| {
            metric_summary(
                &micro
                    .iter()
                    .map(|metrics| select(metrics))
                    .collect::<Vec<_>>(),
            )
        };
        Some(MicroCombatSweepMetricSummaries {
            final_survival_success_rate: summarize(|m| m.final_survival_success_rate),
            final_elimination_success_rate: summarize(|m| m.final_elimination_success_rate),
            best_survival_success_rate: summarize(|m| m.best_survival_success_rate),
            best_elimination_success_rate: summarize(|m| m.best_elimination_success_rate),
            final_attack_commitments_per_episode: summarize(|m| {
                m.final_attack_commitments_per_episode
            }),
            best_attack_commitments_per_episode: summarize(|m| {
                m.best_attack_commitments_per_episode
            }),
            qualification_rate: summarize(|m| m.qualification_reached),
            normalized_actions_to_qualification_or_budget: summarize(|m| {
                m.normalized_actions_to_qualification_or_budget
            }),
        })
    }
    SweepMetricSummaries {
        training_win_rate: summary_of(results, |result| result.training.win_rate),
        training_average_reward: summary_of(results, |result| result.training.average_reward),
        training_average_episode_len: summary_of(results, |result| {
            result.training.average_episode_len
        }),
        actions_per_second: summary_of(results, |result| result.training.actions_per_second),
        simulation_quanta_per_second: summary_of(results, |result| {
            result.training.simulation_quanta_per_second
        }),
        evaluation_win_rate: evaluation_summary(results, |metrics| metrics.win_rate),
        evaluation_worst_case_win_rate: evaluation_summary(results, |metrics| {
            metrics.worst_case_win_rate
        }),
        evaluation_average_reward: evaluation_summary(results, |metrics| metrics.average_reward),
        evaluation_average_episode_len: evaluation_summary(results, |metrics| {
            metrics.average_episode_len
        }),
        joint_qualification_rate: {
            let values = results
                .iter()
                .filter_map(|result| {
                    result
                        .competency
                        .as_ref()
                        .map(|metrics| metrics.joint_qualified)
                })
                .collect::<Vec<_>>();
            (values.len() == results.len()).then(|| metric_summary(&values))
        },
        peak_resident_set_bytes: {
            let values = results
                .iter()
                .filter_map(|result| result.training.peak_resident_set_bytes)
                .collect::<Vec<_>>();
            (values.len() == results.len()).then(|| metric_summary(&values))
        },
        micro_combat: micro_combat_summaries(results),
        telemetry: telemetry_summaries(results),
    }
}

fn telemetry_difference(
    candidate: &TelemetryRunMetrics,
    baseline: &TelemetryRunMetrics,
) -> TelemetryRunMetrics {
    TelemetryRunMetrics {
        completed_episodes: candidate.completed_episodes - baseline.completed_episodes,
        win_rate: candidate.win_rate - baseline.win_rate,
        average_episode_steps: candidate.average_episode_steps - baseline.average_episode_steps,
        training_births_per_1000_actions: candidate.training_births_per_1000_actions
            - baseline.training_births_per_1000_actions,
        training_deaths_per_1000_actions: candidate.training_deaths_per_1000_actions
            - baseline.training_deaths_per_1000_actions,
        training_kills_per_1000_actions: candidate.training_kills_per_1000_actions
            - baseline.training_kills_per_1000_actions,
        attack_selection_rate: candidate.attack_selection_rate - baseline.attack_selection_rate,
        attack_success_rate: candidate.attack_success_rate - baseline.attack_success_rate,
        signal_selection_rate: candidate.signal_selection_rate - baseline.signal_selection_rate,
        signal_energy_per_1000_actions: candidate.signal_energy_per_1000_actions
            - baseline.signal_energy_per_1000_actions,
        signal_decay_per_1000_actions: candidate.signal_decay_per_1000_actions
            - baseline.signal_decay_per_1000_actions,
        terrain_signal_erased_per_1000_actions: candidate.terrain_signal_erased_per_1000_actions
            - baseline.terrain_signal_erased_per_1000_actions,
        raw_damage_per_successful_attack: candidate.raw_damage_per_successful_attack
            - baseline.raw_damage_per_successful_attack,
        applied_damage_per_successful_attack: candidate.applied_damage_per_successful_attack
            - baseline.applied_damage_per_successful_attack,
        damage_efficiency: candidate.damage_efficiency - baseline.damage_efficiency,
        guard_mitigation_rate: candidate.guard_mitigation_rate - baseline.guard_mitigation_rate,
        overkill_rate: candidate.overkill_rate - baseline.overkill_rate,
        elevation_units_lifted_per_1000_actions: candidate.elevation_units_lifted_per_1000_actions
            - baseline.elevation_units_lifted_per_1000_actions,
        elevation_units_dumped_per_1000_actions: candidate.elevation_units_dumped_per_1000_actions
            - baseline.elevation_units_dumped_per_1000_actions,
        terrain_mass_lifted_per_1000_actions: candidate.terrain_mass_lifted_per_1000_actions
            - baseline.terrain_mass_lifted_per_1000_actions,
        terrain_mass_dumped_per_1000_actions: candidate.terrain_mass_dumped_per_1000_actions
            - baseline.terrain_mass_dumped_per_1000_actions,
        action_rejection_rate: candidate.action_rejection_rate - baseline.action_rejection_rate,
        action_frustration_rate: candidate.action_frustration_rate
            - baseline.action_frustration_rate,
        action_contention_rate: candidate.action_contention_rate - baseline.action_contention_rate,
        mean_training_cells: candidate.mean_training_cells - baseline.mean_training_cells,
        mean_opponent_cells: candidate.mean_opponent_cells - baseline.mean_opponent_cells,
        training_plant_occupancy_rate: candidate.training_plant_occupancy_rate
            - baseline.training_plant_occupancy_rate,
        opponent_plant_occupancy_rate: candidate.opponent_plant_occupancy_rate
            - baseline.opponent_plant_occupancy_rate,
        training_major_food_occupancy_rate: candidate.training_major_food_occupancy_rate
            - baseline.training_major_food_occupancy_rate,
        opponent_major_food_occupancy_rate: candidate.opponent_major_food_occupancy_rate
            - baseline.opponent_major_food_occupancy_rate,
        mean_training_assimilated_energy: candidate.mean_training_assimilated_energy
            - baseline.mean_training_assimilated_energy,
        mean_environment_plant_energy: candidate.mean_environment_plant_energy
            - baseline.mean_environment_plant_energy,
        mean_environment_loose_energy: candidate.mean_environment_loose_energy
            - baseline.mean_environment_loose_energy,
        mean_environment_diffuse_energy: candidate.mean_environment_diffuse_energy
            - baseline.mean_environment_diffuse_energy,
        mean_environment_signal_energy: candidate.mean_environment_signal_energy
            - baseline.mean_environment_signal_energy,
        mean_signal_active_channel_tiles: candidate.mean_signal_active_channel_tiles
            - baseline.mean_signal_active_channel_tiles,
        mean_signal_observation_total_variation: candidate.mean_signal_observation_total_variation
            - baseline.mean_signal_observation_total_variation,
        mean_encounter_edges: candidate.mean_encounter_edges - baseline.mean_encounter_edges,
        mean_training_spatial_entropy: candidate.mean_training_spatial_entropy
            - baseline.mean_training_spatial_entropy,
        mean_resource_concentration: candidate.mean_resource_concentration
            - baseline.mean_resource_concentration,
    }
}

fn paired_differences(
    baseline: &[&SweepRunResult],
    candidate: &[&SweepRunResult],
) -> Result<(Vec<u64>, SweepMetricSummaries), String> {
    let baseline = baseline
        .iter()
        .copied()
        .map(|result| (result.training_seed, result))
        .collect::<HashMap<_, _>>();
    let mut seeds = Vec::with_capacity(candidate.len());
    let mut differences = Vec::with_capacity(candidate.len());
    for candidate in candidate {
        let base = baseline
            .get(&candidate.training_seed)
            .ok_or_else(|| format!("baseline is missing seed {}", candidate.training_seed))?;
        seeds.push(candidate.training_seed);
        differences.push(SweepRunResult {
            schema_version: SWEEP_EXECUTION_SCHEMA_VERSION,
            manifest_sha256: String::new(),
            execution_contract_sha256: String::new(),
            trainer_sha256: String::new(),
            variant: String::new(),
            replicate: candidate.replicate,
            training_seed: candidate.training_seed,
            semantic_ruleset_hash: String::new(),
            compiled_ruleset_hash: String::new(),
            scenario_hash: String::new(),
            experiment_config_sha256: String::new(),
            config_file_sha256: String::new(),
            completed_unix_millis: 0,
            training: TrainingTailMetrics {
                update: 0,
                actions: 0,
                policy_loss: 0.0,
                value_loss: 0.0,
                entropy: 0.0,
                approximate_kl: 0.0,
                explained_variance: 0.0,
                episodes: 0,
                wins: 0,
                losses: 0,
                timeouts: 0,
                win_rate: candidate.training.win_rate - base.training.win_rate,
                average_episode_len: candidate.training.average_episode_len
                    - base.training.average_episode_len,
                average_reward: candidate.training.average_reward - base.training.average_reward,
                training_cells_alive: 0,
                completed_transitions: 0,
                discarded_tails: 0,
                mean_elapsed_time: 0.0,
                actions_per_second: candidate.training.actions_per_second
                    - base.training.actions_per_second,
                minimum_sim_time_quanta: 0,
                maximum_sim_time_quanta: 0,
                total_sim_time_quanta: 0,
                simulation_quanta_per_second: candidate.training.simulation_quanta_per_second
                    - base.training.simulation_quanta_per_second,
                peak_resident_set_bytes: match (
                    candidate.training.peak_resident_set_bytes,
                    base.training.peak_resident_set_bytes,
                ) {
                    (Some(candidate), Some(base)) => Some(candidate - base),
                    (None, None) => None,
                    _ => return Err("paired runs disagree on peak-RSS availability".into()),
                },
            },
            evaluation: match (&candidate.evaluation, &base.evaluation) {
                (Some(candidate), Some(base)) => Some(EvaluationTailMetrics {
                    update: 0,
                    actions: 0,
                    episodes: 0,
                    wins: 0,
                    losses: 0,
                    timeouts: 0,
                    win_rate: candidate.win_rate - base.win_rate,
                    worst_case_win_rate: candidate.worst_case_win_rate - base.worst_case_win_rate,
                    average_episode_len: candidate.average_episode_len - base.average_episode_len,
                    average_reward: candidate.average_reward - base.average_reward,
                    evaluation_actions: 0,
                    seed_start: 0,
                    seed_count: 0,
                }),
                (None, None) => None,
                _ => return Err("paired runs disagree on evaluation availability".into()),
            },
            competency: match (&candidate.competency, &base.competency) {
                (Some(candidate), Some(base)) => Some(CompetencyRunMetrics {
                    joint_qualified: candidate.joint_qualified - base.joint_qualified,
                    best_update: None,
                    best_actions: None,
                    on_food_survival_rate: None,
                    adjacent_food_survival_rate: None,
                    skirmish_kills: None,
                    skirmish_damage: None,
                }),
                (None, None) => None,
                _ => return Err("paired runs disagree on competency availability".into()),
            },
            micro_combat: match (&candidate.micro_combat, &base.micro_combat) {
                (Some(candidate), Some(base)) => Some(MicroCombatRunMetrics {
                    evaluations: 0,
                    final_survival_success_rate: candidate.final_survival_success_rate
                        - base.final_survival_success_rate,
                    final_elimination_success_rate: candidate.final_elimination_success_rate
                        - base.final_elimination_success_rate,
                    best_survival_success_rate: candidate.best_survival_success_rate
                        - base.best_survival_success_rate,
                    best_elimination_success_rate: candidate.best_elimination_success_rate
                        - base.best_elimination_success_rate,
                    final_attack_commitments_per_episode: candidate
                        .final_attack_commitments_per_episode
                        - base.final_attack_commitments_per_episode,
                    best_attack_commitments_per_episode: candidate
                        .best_attack_commitments_per_episode
                        - base.best_attack_commitments_per_episode,
                    qualification_reached: candidate.qualification_reached
                        - base.qualification_reached,
                    normalized_actions_to_qualification_or_budget: candidate
                        .normalized_actions_to_qualification_or_budget
                        - base.normalized_actions_to_qualification_or_budget,
                    first_qualified_actions: None,
                }),
                (None, None) => None,
                _ => {
                    return Err("paired runs disagree on micro-combat evidence availability".into())
                }
            },
            telemetry: match (&candidate.telemetry, &base.telemetry) {
                (Some(candidate), Some(base)) => Some(telemetry_difference(candidate, base)),
                (None, None) => None,
                _ => return Err("paired runs disagree on telemetry availability".into()),
            },
        });
    }
    let references = differences.iter().collect::<Vec<_>>();
    Ok((seeds, summaries(&references)))
}

fn aggregate_validated_sweep(
    sweep: &ValidatedSweep,
    contract: &SweepExecutionContract,
    execution_contract_sha256: &str,
    output: &Path,
) -> Result<RulesSweepAggregate, String> {
    let mut results = Vec::with_capacity(sweep.manifest.runs.len());
    for (index, run) in sweep.manifest.runs.iter().enumerate() {
        let status: SweepRunStatus = read_json(&Path::new(&run.run_directory).join("status.json"))?;
        if !status_matches_run(
            &status,
            &sweep.manifest_sha256,
            execution_contract_sha256,
            &contract.trainer_sha256,
            run,
        ) || status.state != SweepRunState::Succeeded
        {
            return Err(format!(
                "run {} seed {} is incomplete",
                run.variant, run.training_seed
            ));
        }
        let result: SweepRunResult = read_json(&Path::new(&run.run_directory).join("result.json"))?;
        if !result_matches_run(
            &result,
            &sweep.manifest_sha256,
            execution_contract_sha256,
            &contract.trainer_sha256,
            run,
        ) {
            return Err(format!(
                "run {} seed {} result mismatch",
                run.variant, run.training_seed
            ));
        }
        verify_training_completion(&result.training, &sweep.configs[index])?;
        let expected_competency = collect_competency_metrics(
            Path::new(&sweep.configs[index].checkpoint_dir),
            &sweep.configs[index],
        )?;
        if result.competency != expected_competency {
            return Err(format!(
                "run {} seed {} competency evidence mismatch",
                run.variant, run.training_seed
            ));
        }
        let expected_micro_combat = micro_combat_curve(
            &Path::new(&sweep.configs[index].checkpoint_dir).join("micro-combat-evaluation.csv"),
            &sweep.configs[index],
            result.training.actions,
        )?;
        if result.micro_combat != expected_micro_combat {
            return Err(format!(
                "run {} seed {} micro-combat evidence mismatch",
                run.variant, run.training_seed
            ));
        }
        results.push(result);
    }

    let baseline_variant = sweep.manifest.variants[0].name.clone();
    let mut variants = Vec::with_capacity(sweep.manifest.variants.len());
    let mut grouped = HashMap::new();
    for variant in &sweep.manifest.variants {
        let members = results
            .iter()
            .filter(|result| result.variant == variant.name)
            .collect::<Vec<_>>();
        if members.len() != sweep.manifest.seeds.len() {
            return Err(format!(
                "variant {} is missing paired results",
                variant.name
            ));
        }
        variants.push(VariantAggregate {
            variant: variant.name.clone(),
            semantic_ruleset_hash: members[0].semantic_ruleset_hash.clone(),
            compiled_ruleset_hash: members[0].compiled_ruleset_hash.clone(),
            scenario_hash: members[0].scenario_hash.clone(),
            seeds: members.iter().map(|result| result.training_seed).collect(),
            metrics: summaries(&members),
        });
        grouped.insert(variant.name.as_str(), members);
    }
    let baseline = grouped
        .get(baseline_variant.as_str())
        .expect("first manifest variant was grouped");
    let mut paired_comparisons = Vec::new();
    for variant in sweep.manifest.variants.iter().skip(1) {
        let candidate = grouped
            .get(variant.name.as_str())
            .expect("manifest variant was grouped");
        let (seeds, differences) = paired_differences(baseline, candidate)?;
        paired_comparisons.push(PairedVariantComparison {
            baseline_variant: baseline_variant.clone(),
            candidate_variant: variant.name.clone(),
            seeds,
            differences,
        });
    }
    let aggregate = RulesSweepAggregate {
        schema_version: SWEEP_EXECUTION_SCHEMA_VERSION,
        manifest_sha256: sweep.manifest_sha256.clone(),
        execution_contract_sha256: execution_contract_sha256.to_string(),
        trainer_sha256: contract.trainer_sha256.clone(),
        baseline_variant,
        variants,
        paired_comparisons,
    };
    if output.exists() {
        let existing: RulesSweepAggregate = read_json(output)?;
        if existing != aggregate {
            return Err(format!(
                "refusing to replace inconsistent aggregate {}",
                output.display()
            ));
        }
    } else {
        write_json_atomic(output, &aggregate)?;
    }
    Ok(aggregate)
}

/// Rebuild the immutable paired aggregate from a fully completed sweep. This
/// never accepts a partial matrix or a status/result whose hashes do not match
/// the plan.
pub fn aggregate_rules_sweep(manifest_file: &Path) -> Result<RulesSweepAggregate, String> {
    let sweep = load_validated_sweep(manifest_file)?;
    let (contract, execution_contract_sha256) = load_execution_contract(&sweep)?;
    let output = sweep.root.join("aggregate.json");
    aggregate_validated_sweep(&sweep, &contract, &execution_contract_sha256, &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweep::publish_rules_sweep;
    use crate::viability_gate::{
        evaluate_viability_gates, publish_viability_gate_report, ViabilityGateRequirement,
    };
    use crate::viability_matrix::{
        publish_viability_matrix_report, run_viability_matrix, ViabilityMatrixOptions,
    };

    fn write_base(path: &Path) {
        fs::write(
            path,
            r#"
            seed = 1
            num_envs = 1
            rollout_length = 4
            total_timesteps = 8
            eval_interval = 1
            eval_episodes = 1
            evaluation_opponents = ["wait"]
            checkpoint_interval = 1

            [telemetry]
            enabled = false

            [model]
            hidden1 = 8
            hidden2 = 8

            [env]
            world_size = 8
            cells_per_team = 1
            max_episode_len = 64
            num_scattered_energy = 2
            num_plants = 1

            [env.victory]
            sim_time_limit_quanta = 8192

            [self_play]
            max_opponent_pool = 0
            "#,
        )
        .unwrap();
    }

    fn planned_sweep_with_telemetry(temporary: &Path, telemetry_enabled: bool) -> PathBuf {
        write_base(&temporary.join("base.toml"));
        if telemetry_enabled {
            let base = temporary.join("base.toml");
            let content = fs::read_to_string(&base)
                .unwrap()
                .replace("enabled = false", "enabled = true");
            fs::write(base, content).unwrap();
        }
        let spec = temporary.join("sweep.toml");
        fs::write(
            &spec,
            r#"
            base_config = "base.toml"
            output_dir = "planned"
            seeds = [11, 12, 13]

            [[variants]]
            name = "baseline"

            [[variants]]
            name = "candidate"
            [variants.rules]
            digestion_rate_numerator = 2
            "#,
        )
        .unwrap();
        publish_rules_sweep(&spec).unwrap();
        temporary.join("planned/manifest.json")
    }

    fn planned_sweep(temporary: &Path) -> PathBuf {
        planned_sweep_with_telemetry(temporary, false)
    }

    fn write_complete_metrics(config: &TrainingConfig, offset: f64) {
        let root = Path::new(&config.checkpoint_dir);
        fs::create_dir_all(root).unwrap();
        fs::write(
            root.join("metrics.csv"),
            format!(
                "update,actions,policy_loss,value_loss,entropy,approx_kl,clip_fraction,explained_variance,ppo_optimizer_steps,ppo_epochs_completed,kl_early_stop,episodes,wins,losses,timeouts,win_rate,avg_ep_len,avg_reward,training_cells_alive,completed_transitions,discarded_tails,mean_elapsed_time,actions_per_second,min_sim_time_quanta,max_sim_time_quanta,total_sim_time_quanta,simulation_quanta_per_second\n1,8,0.1,0.2,0.3,0.01,0.0,0.4,1,1,false,2,1,1,0,{},{},{},1,8,0,1.0,100.0,1024,1040,2064,10000.0\n",
                0.5 + offset,
                7.0 + offset,
                2.0 + offset
            ),
        )
        .unwrap();
        fs::write(
            root.join("evaluation.csv"),
            format!(
                "update,actions,opponent,episodes,wins,losses,timeouts,win_rate,worst_case_win_rate,avg_ep_len,avg_reward,evaluation_actions,seed_start,seed_count\n1,8,suite,1,1,0,0,{},{},{},{},8,1000,1\n",
                0.6 + offset,
                0.5 + offset,
                6.0 + offset,
                3.0 + offset
            ),
        )
        .unwrap();
    }

    fn write_complete_telemetry(config: &TrainingConfig, offset: f64) {
        let mut summary = TrainingTelemetrySummary {
            schema_version: crate::telemetry::TELEMETRY_SCHEMA_VERSION,
            completed_episodes: 10,
            censored_active_episodes: 1,
            wins: 6,
            losses: 3,
            timeouts: 1,
            average_completed_episode_steps: 7.0 + offset,
            state_samples: 20,
            ..TrainingTelemetrySummary::default()
        };
        summary.training.actions.wait.committed = 6;
        summary.training.actions.wait.completed = 6;
        summary.training.actions.wait.succeeded = 6;
        summary.training.actions.attack.committed = 2;
        summary.training.actions.attack.completed = 2;
        summary.training.actions.attack.succeeded = 1;
        summary.training.births = 2;
        summary.training.deaths = 1;
        summary.training.kills = 3;
        summary.sample_means.training_cells = 4.0 + offset;
        summary.sample_means.opponent_cells = 3.0 - offset;
        summary.sample_means.environment_plant_energy = 100.0 + offset;
        summary.sample_means.encounter_edges = 2.0 + offset;
        summary.sample_means.resource_concentration = 0.2 + offset;
        crate::telemetry::publish_training_summary(Path::new(&config.checkpoint_dir), &summary)
            .unwrap();
    }

    fn gate_requirement(
        temporary: &Path,
        manifest_file: &Path,
        label: &str,
        policy: &str,
    ) -> ViabilityGateRequirement {
        let matrix = run_viability_matrix(
            manifest_file,
            &ViabilityMatrixOptions {
                candidates: vec![crate::config::OpponentProfile::Random],
                opponents: vec![crate::config::OpponentProfile::Wait],
                baseline_variant: None,
                max_parallel: 2,
                max_micro_actions: 1_000,
            },
        )
        .unwrap();
        let matrix_file = temporary.join(format!("{label}-matrix.json"));
        publish_viability_matrix_report(&matrix_file, &matrix).unwrap();
        let gate_spec_file = temporary.join(format!("{label}-gates.toml"));
        fs::write(&gate_spec_file, policy).unwrap();
        let decision = evaluate_viability_gates(&matrix_file, &gate_spec_file).unwrap();
        let decision_file = temporary.join(format!("{label}-decision.json"));
        publish_viability_gate_report(&decision_file, &decision).unwrap();
        ViabilityGateRequirement {
            decision_file,
            matrix_file,
            gate_spec_file,
        }
    }

    #[test]
    fn required_viability_gate_fails_before_launch_and_accepts_exact_passing_artifacts() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest_file = planned_sweep_with_telemetry(temporary.path(), true);
        let failing = gate_requirement(
            temporary.path(),
            &manifest_file,
            "failing",
            r#"
            schema_version = 2
            [absolute]
            min_applied_damage_per_episode_for_combat_profiles = 1.0e300
            "#,
        );
        let error = execute_rules_sweep(
            &manifest_file,
            &SweepExecutorOptions {
                train_program: PathBuf::from("false"),
                train_arguments: Vec::new(),
                trainer_container_digest: None,
                max_parallel: 2,
                threads_per_run: Some(1),
                retry_failed: false,
                required_viability_gate: Some(failing),
            },
        )
        .unwrap_err();
        assert!(error.contains("viability gate failed"));
        assert!(!temporary.path().join("planned/executor.lock").exists());
        let sweep = load_validated_sweep(&manifest_file).unwrap();
        assert!(sweep
            .manifest
            .runs
            .iter()
            .all(|run| { !Path::new(&run.run_directory).join("status.json").exists() }));

        let passing = gate_requirement(
            temporary.path(),
            &manifest_file,
            "passing",
            r#"
            schema_version = 2
            [absolute]
            max_timeout_rate = 1.0
            min_candidate_survival_rate = 0.0
            max_commit_rejection_rate = 1.0
            max_frustrated_resolution_rate = 1.0
            [paired]
            max_regressed_outcome_rate = 1.0
            max_candidate_win_rate_drop = 1.0
            max_final_candidate_cell_mean_drop = 100.0
            max_timeout_rate_increase = 1.0
            "#,
        );
        for (run, config) in sweep.manifest.runs.iter().zip(&sweep.configs) {
            let offset = if run.variant == "candidate" { 0.1 } else { 0.0 };
            write_complete_metrics(config, offset);
            write_complete_telemetry(config, offset);
        }
        let summary = execute_rules_sweep(
            &manifest_file,
            &SweepExecutorOptions {
                train_program: PathBuf::from("false"),
                train_arguments: Vec::new(),
                trainer_container_digest: None,
                max_parallel: 2,
                threads_per_run: Some(1),
                retry_failed: false,
                required_viability_gate: Some(passing),
            },
        )
        .unwrap();
        assert_eq!(summary.succeeded, sweep.manifest.runs.len());
    }

    #[test]
    fn recovers_complete_runs_and_publishes_paired_differences() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest_file = planned_sweep(temporary.path());
        let sweep = load_validated_sweep(&manifest_file).unwrap();
        let options = SweepExecutorOptions {
            train_program: PathBuf::from("false"),
            train_arguments: Vec::new(),
            trainer_container_digest: None,
            max_parallel: 2,
            threads_per_run: Some(1),
            retry_failed: false,
            required_viability_gate: None,
        };
        for (run, config) in sweep.manifest.runs.iter().zip(&sweep.configs) {
            let offset = if run.variant == "candidate" { 0.1 } else { 0.0 };
            write_complete_metrics(config, offset);
        }
        let summary = execute_rules_sweep(&manifest_file, &options).unwrap();
        assert_eq!(summary.succeeded, 6);
        let aggregate: RulesSweepAggregate =
            read_json(&temporary.path().join("planned/aggregate.json")).unwrap();
        assert_eq!(aggregate.paired_comparisons.len(), 1);
        let difference = &aggregate.paired_comparisons[0].differences;
        assert!((difference.training_average_reward.mean - 0.1).abs() < 1.0e-12);
        assert!((difference.evaluation_win_rate.as_ref().unwrap().mean - 0.1).abs() < 1.0e-12);
        assert_eq!(difference.training_average_reward.samples, 3);
    }

    #[test]
    fn world_time_only_evaluation_is_required_in_sweep_results() {
        let temporary = tempfile::tempdir().unwrap();
        let mut config = TrainingConfig::from_toml_str(include_str!(
            "../config/competitive_transfer_large_skirmish_stage_evaluation.toml"
        ))
        .unwrap();
        config.total_timesteps = 8;
        config.total_simulation_quanta_per_env = None;
        config.telemetry.enabled = false;
        config.checkpoint_dir = temporary.path().join("artifacts").display().to_string();
        assert_eq!(config.eval_interval, 0);
        assert!(config.fixed_evaluation_enabled());
        write_complete_metrics(&config, 0.0);

        let run = RulesSweepRun {
            variant: "world-time".into(),
            replicate: 0,
            training_seed: config.seed,
            semantic_ruleset_hash: "rules".into(),
            compiled_ruleset_hash: "compiled".into(),
            scenario_hash: "scenario".into(),
            experiment_config_sha256: "experiment".into(),
            config_file_sha256: "config".into(),
            run_directory: temporary.path().display().to_string(),
            config_file: temporary.path().join("config.toml").display().to_string(),
        };
        let result = collect_result("manifest", "contract", "trainer", &run, &config).unwrap();
        assert!(result.evaluation.is_some());
        assert_eq!(result.evaluation.unwrap().actions, 8);
    }

    #[test]
    fn sweep_aggregate_includes_paired_ecology_and_action_metrics() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest_file = planned_sweep_with_telemetry(temporary.path(), true);
        let sweep = load_validated_sweep(&manifest_file).unwrap();
        for (run, config) in sweep.manifest.runs.iter().zip(&sweep.configs) {
            let offset = if run.variant == "candidate" { 0.1 } else { 0.0 };
            write_complete_metrics(config, offset);
            write_complete_telemetry(config, offset);
        }
        let summary = execute_rules_sweep(
            &manifest_file,
            &SweepExecutorOptions {
                train_program: PathBuf::from("false"),
                train_arguments: Vec::new(),
                trainer_container_digest: None,
                max_parallel: 2,
                threads_per_run: Some(1),
                retry_failed: false,
                required_viability_gate: None,
            },
        )
        .unwrap();
        assert_eq!(summary.succeeded, 6);
        let aggregate: RulesSweepAggregate =
            read_json(&temporary.path().join("planned/aggregate.json")).unwrap();
        let telemetry = aggregate.paired_comparisons[0]
            .differences
            .telemetry
            .as_ref()
            .unwrap();
        assert!((telemetry.mean_training_cells.mean - 0.1).abs() < 1.0e-12);
        assert!((telemetry.mean_encounter_edges.mean - 0.1).abs() < 1.0e-12);
        assert_eq!(telemetry.attack_success_rate.mean, 0.0);
        assert_eq!(telemetry.mean_training_cells.samples, 3);
    }

    #[test]
    fn rejects_a_modified_planned_config_before_launch() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest_file = planned_sweep(temporary.path());
        let manifest: RulesSweepManifest = read_json(&manifest_file).unwrap();
        fs::write(&manifest.runs[0].config_file, "seed = 99\n").unwrap();
        let error = execute_rules_sweep(
            &manifest_file,
            &SweepExecutorOptions {
                train_program: PathBuf::from("false"),
                train_arguments: Vec::new(),
                trainer_container_digest: None,
                max_parallel: 1,
                threads_per_run: None,
                retry_failed: false,
                required_viability_gate: None,
            },
        )
        .unwrap_err();
        assert!(error.contains("config SHA-256 mismatch"));
    }

    #[test]
    fn confidence_summary_uses_sample_variance() {
        let summary = metric_summary(&[1.0, 2.0, 3.0]);
        assert_eq!(summary.samples, 3);
        assert_eq!(summary.mean, 2.0);
        assert_eq!(summary.sample_standard_deviation, 1.0);
        assert!((summary.standard_error - 1.0 / 3.0_f64.sqrt()).abs() < 1.0e-12);
        assert!((summary.confidence_95_half_width - 4.303 / 3.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn micro_combat_curve_reports_terminal_skill_and_first_gate_crossing() {
        let temporary = tempfile::tempdir().unwrap();
        let mut config = TrainingConfig::from_toml_str(include_str!(
            "../config/micro_combat_curriculum_256.toml"
        ))
        .unwrap();
        config
            .combat_curriculum
            .micro_combat
            .min_attack_commitments_per_episode = 0.25;
        let suite = config
            .combat_curriculum
            .micro_combat
            .suite
            .as_ref()
            .unwrap();
        let mut csv = String::from(
            "update,actions,scenario,objective,episodes,wins,losses,timeouts,safety_aborts,alive_at_end,objective_successes,objective_success_rate,scientific_survival_rate,survival_time_quanta,attacks_committed,damage_dealt,damage_received,kills,seed_count\n",
        );
        for (update, actions, elimination_successes, attacks) in [(1, 100, 0, 0), (2, 200, 2, 2)] {
            for scenario in &suite.scenarios {
                let successes = match scenario.objective {
                    crate::micro_combat::MicroCombatObjective::Survival => 3,
                    crate::micro_combat::MicroCombatObjective::Elimination => elimination_successes,
                };
                csv.push_str(&format!(
                    "{update},{actions},{},{:?},4,0,0,4,0,3,{successes},0,0,0,{attacks},0,0,0,4\n",
                    scenario.name, scenario.objective
                ));
            }
        }
        let path = temporary.path().join("micro-combat-evaluation.csv");
        fs::write(&path, csv).unwrap();
        let metrics = micro_combat_curve(&path, &config, 200).unwrap().unwrap();
        assert_eq!(metrics.evaluations, 2);
        assert_eq!(metrics.final_survival_success_rate, 0.75);
        assert_eq!(metrics.final_elimination_success_rate, 0.5);
        assert_eq!(metrics.final_attack_commitments_per_episode, 0.5);
        assert_eq!(metrics.best_attack_commitments_per_episode, 0.5);
        assert_eq!(metrics.qualification_reached, 1.0);
        assert_eq!(metrics.first_qualified_actions, Some(200));
        assert_eq!(metrics.normalized_actions_to_qualification_or_budget, 1.0);
        assert!(micro_combat_curve(&path, &config, 201)
            .unwrap_err()
            .contains("terminal micro-combat evaluation"));
    }

    #[test]
    fn paired_competency_summary_uses_exact_binary_run_outcomes() {
        fn result(seed: u64, qualified: bool) -> SweepRunResult {
            SweepRunResult {
                schema_version: SWEEP_EXECUTION_SCHEMA_VERSION,
                manifest_sha256: String::new(),
                execution_contract_sha256: String::new(),
                trainer_sha256: String::new(),
                variant: String::new(),
                replicate: 0,
                training_seed: seed,
                semantic_ruleset_hash: String::new(),
                compiled_ruleset_hash: String::new(),
                scenario_hash: String::new(),
                experiment_config_sha256: String::new(),
                config_file_sha256: String::new(),
                completed_unix_millis: 0,
                training: TrainingTailMetrics {
                    update: 0,
                    actions: 0,
                    policy_loss: 0.0,
                    value_loss: 0.0,
                    entropy: 0.0,
                    approximate_kl: 0.0,
                    explained_variance: 0.0,
                    episodes: 0,
                    wins: 0,
                    losses: 0,
                    timeouts: 0,
                    win_rate: 0.0,
                    average_episode_len: 0.0,
                    average_reward: 0.0,
                    training_cells_alive: 0,
                    completed_transitions: 0,
                    discarded_tails: 0,
                    mean_elapsed_time: 0.0,
                    actions_per_second: 0.0,
                    minimum_sim_time_quanta: 0,
                    maximum_sim_time_quanta: 0,
                    total_sim_time_quanta: 0,
                    simulation_quanta_per_second: 0.0,
                    peak_resident_set_bytes: None,
                },
                evaluation: None,
                competency: Some(CompetencyRunMetrics {
                    joint_qualified: if qualified { 1.0 } else { 0.0 },
                    best_update: qualified.then_some(1),
                    best_actions: qualified.then_some(1),
                    on_food_survival_rate: qualified.then_some(1.0),
                    adjacent_food_survival_rate: qualified.then_some(1.0),
                    skirmish_kills: qualified.then_some(1),
                    skirmish_damage: qualified.then_some(1),
                }),
                micro_combat: Some(MicroCombatRunMetrics {
                    evaluations: 2,
                    final_survival_success_rate: 1.0,
                    final_elimination_success_rate: if qualified { 1.0 } else { 0.0 },
                    best_survival_success_rate: 1.0,
                    best_elimination_success_rate: if qualified { 1.0 } else { 0.0 },
                    final_attack_commitments_per_episode: if qualified { 1.0 } else { 0.0 },
                    best_attack_commitments_per_episode: if qualified { 1.0 } else { 0.0 },
                    qualification_reached: if qualified { 1.0 } else { 0.0 },
                    normalized_actions_to_qualification_or_budget: if qualified {
                        0.5
                    } else {
                        1.0
                    },
                    first_qualified_actions: qualified.then_some(1),
                }),
                telemetry: None,
            }
        }

        let controls = [result(42, true), result(43, true), result(44, false)];
        let candidates = [result(42, true), result(43, true), result(44, true)];
        let control_refs = controls.iter().collect::<Vec<_>>();
        let candidate_refs = candidates.iter().collect::<Vec<_>>();
        assert_eq!(
            summaries(&control_refs)
                .joint_qualification_rate
                .unwrap()
                .mean,
            2.0 / 3.0
        );
        let (_, differences) = paired_differences(&control_refs, &candidate_refs).unwrap();
        assert_eq!(
            differences.joint_qualification_rate.unwrap().mean,
            1.0 / 3.0
        );
        assert_eq!(
            differences.micro_combat.unwrap().qualification_rate.mean,
            1.0 / 3.0
        );
    }

    #[test]
    fn telemetry_run_metrics_include_damage_and_terrain_effects() {
        let mut summary = TrainingTelemetrySummary::default();
        summary.training.actions.wait.completed = 8;
        summary.training.actions.wait.committed = 8;
        summary.training.actions.attack.completed = 2;
        summary.training.actions.attack.committed = 2;
        summary.training.actions.attack.succeeded = 2;
        summary.training.signals.emitted_decisions = 2;
        summary.training.signals.channel_energy = [5, 5, 5, 5];
        summary.signal_field.decayed_energy = [2, 1, 1, 1];
        summary.signal_field.terrain_erased_energy = [0, 2, 0, 0];
        summary.training.damage.raw_dealt = 40;
        summary.training.damage.mitigated_by_target_guard = 10;
        summary.training.damage.applied_dealt = 20;
        summary.training.damage.overkill_dealt = 10;
        summary.training.terrain.elevation_units_lifted = 2;
        summary.training.terrain.elevation_units_dumped = 1;
        summary.training.terrain.material_mass_lifted = 20;
        summary.training.terrain.material_mass_dumped = 10;
        summary.sample_means.environment_signal_energy = 7.0;
        summary.sample_means.signal_active_channel_tiles = 3.0;
        summary.sample_means.signal_observation_total_variation = 12.0;
        summary.sample_means.training_cells = 10.0;
        summary.sample_means.opponent_cells = 8.0;
        summary.sample_means.training_cells_on_plants = 4.0;
        summary.sample_means.opponent_cells_on_plants = 2.0;
        summary.sample_means.training_cells_on_major_food = 5.0;
        summary.sample_means.opponent_cells_on_major_food = 4.0;

        let metrics = TelemetryRunMetrics::from(&summary);
        assert_eq!(metrics.raw_damage_per_successful_attack, 20.0);
        assert_eq!(metrics.applied_damage_per_successful_attack, 10.0);
        assert_eq!(metrics.damage_efficiency, 0.5);
        assert_eq!(metrics.guard_mitigation_rate, 0.25);
        assert_eq!(metrics.overkill_rate, 0.25);
        assert_eq!(metrics.signal_selection_rate, 0.2);
        assert_eq!(metrics.signal_energy_per_1000_actions, 2_000.0);
        assert_eq!(metrics.signal_decay_per_1000_actions, 500.0);
        assert_eq!(metrics.terrain_signal_erased_per_1000_actions, 200.0);
        assert_eq!(metrics.elevation_units_lifted_per_1000_actions, 200.0);
        assert_eq!(metrics.elevation_units_dumped_per_1000_actions, 100.0);
        assert_eq!(metrics.terrain_mass_lifted_per_1000_actions, 2_000.0);
        assert_eq!(metrics.terrain_mass_dumped_per_1000_actions, 1_000.0);
        assert_eq!(metrics.mean_environment_signal_energy, 7.0);
        assert_eq!(metrics.mean_signal_active_channel_tiles, 3.0);
        assert_eq!(metrics.mean_signal_observation_total_variation, 12.0);
        assert_eq!(metrics.training_plant_occupancy_rate, 0.4);
        assert_eq!(metrics.opponent_plant_occupancy_rate, 0.25);
        assert_eq!(metrics.training_major_food_occupancy_rate, 0.5);
        assert_eq!(metrics.opponent_major_food_occupancy_rate, 0.5);
    }

    #[test]
    fn failed_runs_are_durable_and_require_explicit_retry() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest_file = planned_sweep(temporary.path());
        let options = SweepExecutorOptions {
            train_program: PathBuf::from("false"),
            train_arguments: Vec::new(),
            trainer_container_digest: None,
            max_parallel: 2,
            threads_per_run: Some(1),
            retry_failed: false,
            required_viability_gate: None,
        };
        assert!(execute_rules_sweep(&manifest_file, &options).is_err());
        let manifest: RulesSweepManifest = read_json(&manifest_file).unwrap();
        let status_path = Path::new(&manifest.runs[0].run_directory).join("status.json");
        let first: SweepRunStatus = read_json(&status_path).unwrap();
        assert_eq!(first.state, SweepRunState::Failed);
        assert_eq!(first.attempts, 1);

        let summary = execute_rules_sweep(&manifest_file, &options).unwrap();
        assert_eq!(summary.failed, manifest.runs.len());
        let skipped: SweepRunStatus = read_json(&status_path).unwrap();
        assert_eq!(skipped.attempts, 1);

        let retry = SweepExecutorOptions {
            retry_failed: true,
            ..options
        };
        assert!(execute_rules_sweep(&manifest_file, &retry).is_err());
        let retried: SweepRunStatus = read_json(&status_path).unwrap();
        assert_eq!(retried.attempts, 2);
    }

    #[test]
    fn execution_contract_rejects_trainer_or_container_substitution() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest_file = planned_sweep(temporary.path());
        let first = SweepExecutorOptions {
            train_program: PathBuf::from("false"),
            train_arguments: Vec::new(),
            trainer_container_digest: Some(format!("sha256:{}", "a".repeat(64))),
            max_parallel: 1,
            threads_per_run: Some(1),
            retry_failed: false,
            required_viability_gate: None,
        };
        assert!(execute_rules_sweep(&manifest_file, &first).is_err());

        let substituted_binary = SweepExecutorOptions {
            train_program: PathBuf::from("true"),
            ..first.clone()
        };
        let error = execute_rules_sweep(&manifest_file, &substituted_binary).unwrap_err();
        assert!(error.contains("immutable contract"));

        let substituted_container = SweepExecutorOptions {
            trainer_container_digest: Some(format!("sha256:{}", "b".repeat(64))),
            ..first
        };
        let error = execute_rules_sweep(&manifest_file, &substituted_container).unwrap_err();
        assert!(error.contains("immutable contract"));
    }

    #[test]
    fn live_orphan_lease_prevents_duplicate_launch() {
        let temporary = tempfile::tempdir().unwrap();
        let manifest_file = planned_sweep(temporary.path());
        let sweep = load_validated_sweep(&manifest_file).unwrap();
        let options = SweepExecutorOptions {
            train_program: PathBuf::from("false"),
            train_arguments: Vec::new(),
            trainer_container_digest: None,
            max_parallel: 2,
            threads_per_run: Some(1),
            retry_failed: false,
            required_viability_gate: None,
        };
        let requested_contract = requested_execution_contract(&sweep, &options).unwrap();
        let (contract, execution_contract_sha256) =
            publish_or_verify_execution_contract(&sweep, &requested_contract).unwrap();
        for (run, config) in sweep.manifest.runs.iter().zip(&sweep.configs).skip(1) {
            let offset = if run.variant == "candidate" { 0.1 } else { 0.0 };
            write_complete_metrics(config, offset);
        }
        let run = &sweep.manifest.runs[0];
        let run_directory = Path::new(&run.run_directory);
        let lease_path = run_directory.join("trainer.lock");
        let lease = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(&lease_path)
            .unwrap();
        lease.try_lock().unwrap();
        write_json_atomic(
            &run_directory.join("status.json"),
            &SweepRunStatus {
                schema_version: SWEEP_EXECUTION_SCHEMA_VERSION,
                manifest_sha256: sweep.manifest_sha256.clone(),
                execution_contract_sha256,
                trainer_sha256: contract.trainer_sha256,
                variant: run.variant.clone(),
                replicate: run.replicate,
                training_seed: run.training_seed,
                experiment_config_sha256: run.experiment_config_sha256.clone(),
                state: SweepRunState::Running,
                attempts: 1,
                started_unix_millis: unix_millis(),
                finished_unix_millis: None,
                resume_checkpoint: None,
                exit_code: None,
                error: None,
            },
        )
        .unwrap();

        let summary = execute_rules_sweep(&manifest_file, &options).unwrap();
        assert_eq!(summary.succeeded, 5);
        assert_eq!(summary.still_running, 1);
        let status: SweepRunStatus = read_json(&run_directory.join("status.json")).unwrap();
        assert_eq!(status.state, SweepRunState::Running);
        assert_eq!(status.attempts, 1);
    }
}
