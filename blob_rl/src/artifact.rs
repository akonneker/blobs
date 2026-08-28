//! Immutable training artifacts with atomic directory publication.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use burn::optim::Optimizer;
use burn::prelude::*;
use burn::record::{DefaultRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::competency_frontier::{CompetencyFrontier, SpecialistTeacherSelection};
use crate::config::{
    OpponentProfile, SelfPlayConfig, SpecialistDistillationConfig, TrainingConfig,
};
use crate::contact_evaluation::ContactEvaluationReport;
use crate::env::BlobEnvCheckpoint;
use crate::evaluation::EvaluationMetrics;
use crate::feeding_curriculum::FeedingPromotionReport;
use crate::micro_combat::{MicroCombatEvaluationReport, MicroCombatRotationState};
use crate::model::{PolicyValueNet, PolicyValueNetConfig};
use crate::telemetry::TrainingTelemetryState;

pub const TRAINING_ARTIFACT_SCHEMA_VERSION: u32 = 38;
const MAX_METADATA_BYTES: u64 = 1024 * 1024;
const MAX_RESUME_STATE_BYTES: u64 = 512 * 1024 * 1024;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

/// Metadata stored beside an immutable model and optimizer record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    /// Numerical execution stack used for training and exact continuation.
    /// Policy-only imports remain backend-portable, but exact resume does not.
    pub training_backend: String,
    pub update: usize,
    pub actions: u64,
    /// Exact canonical world-time exposure at this update boundary. These
    /// counters make competency timing inspectable without decoding the much
    /// larger exact-resume sidecar.
    pub minimum_sim_time_quanta_per_env: u64,
    pub maximum_sim_time_quanta_per_env: u64,
    pub total_sim_time_quanta: u128,
    pub training_seed: u64,
    pub compiled_ruleset_hash: String,
    pub model_file: String,
    pub model_sha256: String,
    pub optimizer_file: String,
    pub optimizer_sha256: String,
    pub resume_file: String,
    pub resume_sha256: String,
    pub record_precision: String,
    pub resume_scope: String,
    pub config: TrainingConfig,
    pub evaluation: Option<EvaluationMetrics>,
    /// Complete independently recomputable feeding competency report from the
    /// same held-out evaluation boundary.
    pub feeding_evaluation: Option<FeedingPromotionReport>,
    /// Independent replay of the initial clone's qualification seed suite.
    /// Omitted when it is identical to the configured evaluation suite.
    pub retention_feeding_evaluation: Option<FeedingPromotionReport>,
    /// Held-out direct-contact and multi-cell skirmish evidence from the same
    /// primary seed suite.
    pub contact_evaluation: Option<ContactEvaluationReport>,
    /// Independent held-out survival and elimination evidence for asymmetric
    /// micro-combat scenarios.
    pub micro_combat_evaluation: Option<MicroCombatEvaluationReport>,
    /// This checkpoint passed the rollout-pool gates. The resume record stores
    /// the pool immediately before this member is appended, avoiding a
    /// self-referential artifact hash.
    pub rollout_pool_promotion: Option<LeaguePromotion>,
}

/// Stable execution-stack identity embedded by release/container builds.
/// Local builds fall back to the selected Cargo backend.
pub fn training_backend_id<B: Backend>() -> &'static str {
    option_env!("BLOB_TRAINING_BACKEND").unwrap_or(std::any::type_name::<B>())
}

/// Immutable identity and location of one promoted checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RolloutSnapshotDescriptor {
    pub directory: String,
    pub checkpoint: String,
    pub update: usize,
    pub actions: u64,
    pub model_sha256: String,
}

/// Rating assigned to a newly promoted checkpoint. Its descriptor is derived
/// only after publication, avoiding a self-referential model hash.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LeaguePromotion {
    pub rating: f64,
    pub evaluation_games: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RolloutLeagueMember {
    pub snapshot: RolloutSnapshotDescriptor,
    pub rating: f64,
    pub evaluation_games: u64,
    pub rollout_selections: u64,
}

/// Opponent currently assigned to an environment. Snapshot assignments are
/// hash-bound rather than indexed so bounded eviction cannot retarget them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RolloutOpponentAssignment {
    Baseline { profile: OpponentProfile },
    Snapshot { model_sha256: String },
}

/// Every mutable value needed to continue from the boundary after a PPO
/// update. Metrics files are append-only outputs, not inputs to the trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResumeState {
    pub schema_version: u32,
    pub total_timesteps: u64,
    /// Cumulative authoritative world-clock exposure for each environment.
    /// This persists across episode resets and is part of exact continuation.
    pub cumulative_sim_time_quanta: Vec<u64>,
    pub update_count: usize,
    pub action_rng: ChaCha12Rng,
    pub optimizer_rng: ChaCha12Rng,
    pub opponent_rng: ChaCha12Rng,
    pub env_episode_ids: Vec<u64>,
    pub env_episode_rewards: Vec<f32>,
    pub environments: Vec<BlobEnvCheckpoint>,
    pub best_evaluation: Option<EvaluationMetrics>,
    pub best_contact_evaluation: Option<ContactEvaluationReport>,
    pub best_micro_combat_evaluation: Option<MicroCombatEvaluationReport>,
    /// Bounded host-only ecology/combat Pareto evidence. Entries point only to
    /// immutable evaluated checkpoints and do not affect Mind inputs.
    pub competency_frontier: CompetencyFrontier,
    /// Exact stage-local distillation teachers for the following update.
    /// Selection identities must be members of `competency_frontier`.
    pub specialist_teachers: SpecialistTeacherSelection,
    pub rollout_pool: Vec<RolloutLeagueMember>,
    /// Evicted pool members still needed by episodes already in progress.
    pub active_retired_snapshots: Vec<RolloutSnapshotDescriptor>,
    pub environment_opponents: Vec<RolloutOpponentAssignment>,
    /// Curriculum scenario actually active in each in-flight environment.
    pub environment_curriculum_stages: Vec<crate::config::FeedingCurriculumStage>,
    /// Stage-local balanced rotation cursors and the scenario assigned to each
    /// in-flight environment. Both are required for exact continuation.
    pub micro_combat_rotation: MicroCombatRotationState,
    pub environment_micro_combat_scenarios: Vec<Option<usize>>,
    /// Exact bounded host-telemetry continuation. This is scientifically
    /// useful output state, never an input to policy or physics.
    pub telemetry: Option<TrainingTelemetryState>,
}

fn validate_specialist_teacher_selection(
    state: &TrainingResumeState,
    config: &SpecialistDistillationConfig,
) -> Result<(), String> {
    let expected = if config.enabled {
        state
            .competency_frontier
            .specialist_teachers(config.combat_precursor_min_skirmish_damage)
    } else {
        SpecialistTeacherSelection::default()
    };
    if state.specialist_teachers != expected {
        return Err(
            "specialist teachers do not match deterministic configured frontier selection".into(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BestCheckpointPointer {
    pub schema_version: u32,
    pub checkpoint: String,
    pub update: usize,
    pub actions: u64,
    pub evaluation: EvaluationMetrics,
    pub feeding_evaluation: Option<FeedingPromotionReport>,
    pub retention_feeding_evaluation: Option<FeedingPromotionReport>,
    pub contact_evaluation: Option<ContactEvaluationReport>,
    pub micro_combat_evaluation: Option<MicroCombatEvaluationReport>,
}

/// Integrity-checked immutable policy ready for stateless opponent inference.
pub struct PolicySnapshot<B: Backend> {
    pub model: PolicyValueNet<B>,
    pub directory: String,
    pub checkpoint: String,
    pub update: usize,
    pub actions: u64,
    pub model_sha256: String,
}

impl<B: Backend> PolicySnapshot<B> {
    pub fn descriptor(&self) -> RolloutSnapshotDescriptor {
        RolloutSnapshotDescriptor {
            directory: self.directory.clone(),
            checkpoint: self.checkpoint.clone(),
            update: self.update,
            actions: self.actions,
            model_sha256: self.model_sha256.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RolloutPoolManifest {
    pub schema_version: u32,
    pub update: usize,
    pub actions: u64,
    pub compiled_ruleset_hash: String,
    pub self_play: SelfPlayConfig,
    /// Gate report for the checkpoint promoted by this manifest update.
    pub promotion_feeding_evaluation: Option<FeedingPromotionReport>,
    pub promotion_retention_feeding_evaluation: Option<FeedingPromotionReport>,
    pub promotion_contact_evaluation: Option<ContactEvaluationReport>,
    pub promotion_micro_combat_evaluation: Option<MicroCombatEvaluationReport>,
    pub members: Vec<RolloutLeagueMember>,
}

struct TempDirectory(Option<PathBuf>);

impl Drop for TempDirectory {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_dir_all(path);
        }
    }
}

fn unique_temp_directory(root: &Path, stem: &str) -> Result<TempDirectory, String> {
    for _ in 0..100 {
        let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(".{stem}.tmp-{}-{nonce}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(TempDirectory(Some(path))),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "failed to create artifact staging directory: {error}"
                ));
            }
        }
    }
    Err("failed to allocate a unique artifact staging directory".into())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("failed to open {} for hashing: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn verified_metadata(directory: &Path) -> Result<CheckpointMetadata, String> {
    let metadata_path = directory.join("metadata.json");
    let metadata: CheckpointMetadata =
        serde_json::from_slice(&read_bounded(&metadata_path, MAX_METADATA_BYTES)?)
            .map_err(|error| format!("failed to decode {}: {error}", metadata_path.display()))?;
    if metadata.schema_version != TRAINING_ARTIFACT_SCHEMA_VERSION {
        return Err(format!(
            "unsupported RL artifact schema {}; expected {}",
            metadata.schema_version, TRAINING_ARTIFACT_SCHEMA_VERSION
        ));
    }
    metadata
        .config
        .validate()
        .map_err(|error| format!("checkpoint contains an invalid training config: {error}"))?;
    if metadata.minimum_sim_time_quanta_per_env > metadata.maximum_sim_time_quanta_per_env
        || metadata.total_sim_time_quanta < u128::from(metadata.maximum_sim_time_quanta_per_env)
    {
        return Err("checkpoint metadata contains invalid simulation-time exposure".into());
    }
    if metadata.model_file != "model.mpk"
        || metadata.optimizer_file != "optimizer.mpk"
        || metadata.resume_file != "resume.mpk"
        || metadata.record_precision != "full"
        || metadata.resume_scope != "exact-update-boundary"
        || metadata.training_backend.trim().is_empty()
    {
        return Err("checkpoint metadata declares an unsupported artifact layout".into());
    }
    let model_path = directory.join(&metadata.model_file);
    let optimizer_path = directory.join(&metadata.optimizer_file);
    let resume_path = directory.join(&metadata.resume_file);
    if sha256_file(&model_path)? != metadata.model_sha256
        || sha256_file(&optimizer_path)? != metadata.optimizer_sha256
        || sha256_file(&resume_path)? != metadata.resume_sha256
    {
        return Err("checkpoint artifact SHA-256 mismatch".into());
    }
    validate_evaluation_artifact_binding(&metadata)?;
    Ok(metadata)
}

fn validate_evaluation_artifact_binding(metadata: &CheckpointMetadata) -> Result<(), String> {
    match (
        metadata.config.feeding_curriculum.enabled,
        metadata.evaluation.as_ref(),
        metadata.feeding_evaluation.as_ref(),
    ) {
        (true, Some(evaluation), Some(feeding)) => feeding.validate_against(
            &metadata.compiled_ruleset_hash,
            &evaluation.seeds,
            &metadata.config.feeding_curriculum.promotion,
        )?,
        (true, Some(_), None) => {
            return Err("evaluated curriculum checkpoint is missing its feeding gate report".into())
        }
        (true, None, Some(_)) | (false, _, Some(_)) => {
            return Err("checkpoint has an unexpected feeding gate report".into())
        }
        (true, None, None) | (false, _, None) => {}
    }
    if let Some(retention) = &metadata.retention_feeding_evaluation {
        let qualification = metadata
            .config
            .initial_policy
            .as_ref()
            .and_then(|initial| initial.qualification.as_ref())
            .ok_or("retention feeding evaluation has no initial qualification binding")?;
        if metadata.feeding_evaluation.is_none()
            || metadata
                .evaluation
                .as_ref()
                .is_some_and(|evaluation| evaluation.seeds == retention.seeds)
            || qualification.artifact_hash.trim().is_empty()
        {
            return Err(
                "retention feeding evaluation is missing or duplicates its primary suite".into(),
            );
        }
        retention.validate_against(
            &metadata.compiled_ruleset_hash,
            &retention.seeds,
            &metadata.config.feeding_curriculum.promotion,
        )?;
    }
    match (
        metadata.config.combat_curriculum.enabled,
        metadata.evaluation.as_ref(),
        metadata.contact_evaluation.as_ref(),
    ) {
        (true, Some(evaluation), Some(contact)) => contact.validate_against(
            &metadata.compiled_ruleset_hash,
            &evaluation.seeds,
            &metadata.config.combat_curriculum,
        )?,
        (true, Some(_), None) => {
            return Err("evaluated combat checkpoint is missing its contact report".into())
        }
        (true, None, Some(_)) | (false, _, Some(_)) => {
            return Err("checkpoint has an unexpected contact report".into())
        }
        (true, None, None) | (false, _, None) => {}
    }
    let micro_config = &metadata.config.combat_curriculum.micro_combat;
    match (
        micro_config.enabled,
        metadata.evaluation.as_ref(),
        metadata.micro_combat_evaluation.as_ref(),
    ) {
        (true, Some(evaluation), Some(micro)) => micro.validate_against(
            &metadata.compiled_ruleset_hash,
            micro_config
                .suite
                .as_ref()
                .ok_or("micro-combat evaluation has no configured suite")?,
            &evaluation.seeds,
        )?,
        (true, Some(_), None) => {
            return Err("evaluated checkpoint is missing its micro-combat report".into())
        }
        (true, None, Some(_)) | (false, _, Some(_)) => {
            return Err("checkpoint has an unexpected micro-combat report".into())
        }
        (true, None, None) | (false, _, None) => {}
    }
    if metadata.rollout_pool_promotion.is_some() {
        if metadata.evaluation.is_none() {
            return Err("rollout-pool promotion is missing held-out evaluation".into());
        }
        if metadata.config.feeding_curriculum.enabled
            && (!metadata
                .feeding_evaluation
                .as_ref()
                .is_some_and(|report| report.passed)
                || metadata
                    .retention_feeding_evaluation
                    .as_ref()
                    .is_some_and(|report| !report.passed))
        {
            return Err("rollout-pool promotion did not pass the feeding gate".into());
        }
        if metadata.config.combat_curriculum.enabled
            && !metadata.contact_evaluation.as_ref().is_some_and(|report| {
                report.meets_promotion_thresholds(&metadata.config.combat_curriculum)
            })
        {
            return Err("rollout-pool promotion has no active contact evidence".into());
        }
        if micro_config.enabled
            && !metadata
                .micro_combat_evaluation
                .as_ref()
                .is_some_and(|report| report.meets_promotion_thresholds(micro_config))
        {
            return Err("rollout-pool promotion failed a micro-combat gate".into());
        }
    }
    Ok(())
}

/// Read a checkpoint's metadata only after verifying its complete immutable
/// file set. Sweep resumption uses this to avoid selecting a directory by name
/// alone or resuming from a partially published/tampered checkpoint.
pub fn verify_checkpoint_metadata(directory: &Path) -> Result<CheckpointMetadata, String> {
    verified_metadata(directory)
}

fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>, String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > max_bytes {
        return Err(format!(
            "{} is {length} bytes; limit is {max_bytes}",
            path.display()
        ));
    }
    fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))
}

fn write_json_file(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    serde_json::to_writer_pretty(&mut file, value)
        .map_err(|error| format!("failed to serialize {}: {error}", path.display()))?;
    file.write_all(b"\n")
        .map_err(|error| format!("failed to finish {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync {}: {error}", path.display()))
}

fn sync_file(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| format!("failed to sync {}: {error}", path.display()))
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to sync directory {}: {error}", path.display()))
}

/// Publish a model, optimizer, and metadata as one immutable directory. The
/// final directory appears only after every staged file has been written,
/// hashed, and synced.
#[allow(clippy::too_many_arguments)]
pub fn publish_checkpoint<B, O>(
    root: &Path,
    model: &PolicyValueNet<B>,
    optimizer: &O,
    config: &TrainingConfig,
    update: usize,
    actions: u64,
    compiled_ruleset_hash: &str,
    evaluation: Option<&EvaluationMetrics>,
    feeding_evaluation: Option<&FeedingPromotionReport>,
    retention_feeding_evaluation: Option<&FeedingPromotionReport>,
    contact_evaluation: Option<&ContactEvaluationReport>,
    micro_combat_evaluation: Option<&MicroCombatEvaluationReport>,
    rollout_pool_promotion: Option<&LeaguePromotion>,
    resume_state: &TrainingResumeState,
) -> Result<PathBuf, String>
where
    B: AutodiffBackend,
    O: Optimizer<PolicyValueNet<B>, B>,
{
    validate_specialist_teacher_selection(resume_state, &config.specialist_distillation)?;
    let minimum_sim_time_quanta_per_env = resume_state
        .cumulative_sim_time_quanta
        .iter()
        .copied()
        .min()
        .unwrap_or(0);
    let maximum_sim_time_quanta_per_env = resume_state
        .cumulative_sim_time_quanta
        .iter()
        .copied()
        .max()
        .unwrap_or(0);
    let total_sim_time_quanta = resume_state
        .cumulative_sim_time_quanta
        .iter()
        .map(|quanta| u128::from(*quanta))
        .sum();
    fs::create_dir_all(root)
        .map_err(|error| format!("failed to create artifact root {}: {error}", root.display()))?;
    let directory_name = format!("checkpoint-{update:08}");
    let final_directory = root.join(&directory_name);
    if final_directory.exists() {
        return Err(format!(
            "refusing to replace immutable checkpoint {}",
            final_directory.display()
        ));
    }

    let mut staging = unique_temp_directory(root, &directory_name)?;
    let staging_path = staging
        .0
        .as_ref()
        .expect("a live staging guard always owns its path");
    // Training artifacts use full precision. Compact half-precision records
    // remain appropriate for deployment export, but would make continuation
    // lossy before optimizer/environment state is restored.
    let recorder = DefaultRecorder::new();
    model
        .clone()
        .save_file(staging_path.join("model"), &recorder)
        .map_err(|error| format!("failed to save model record: {error}"))?;
    recorder
        .record(optimizer.to_record(), staging_path.join("optimizer"))
        .map_err(|error| format!("failed to save optimizer record: {error}"))?;

    let resume_path = staging_path.join("resume.mpk");
    let resume_bytes = rmp_serde::to_vec_named(resume_state)
        .map_err(|error| format!("failed to serialize exact resume state: {error}"))?;
    let mut resume_file = File::create(&resume_path)
        .map_err(|error| format!("failed to create {}: {error}", resume_path.display()))?;
    resume_file
        .write_all(&resume_bytes)
        .and_then(|()| resume_file.sync_all())
        .map_err(|error| format!("failed to write {}: {error}", resume_path.display()))?;

    let model_path = staging_path.join("model.mpk");
    let optimizer_path = staging_path.join("optimizer.mpk");
    sync_file(&model_path)?;
    sync_file(&optimizer_path)?;
    let metadata = CheckpointMetadata {
        schema_version: TRAINING_ARTIFACT_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        training_backend: training_backend_id::<B>().to_string(),
        update,
        actions,
        minimum_sim_time_quanta_per_env,
        maximum_sim_time_quanta_per_env,
        total_sim_time_quanta,
        training_seed: config.seed,
        compiled_ruleset_hash: compiled_ruleset_hash.to_string(),
        model_file: "model.mpk".into(),
        model_sha256: sha256_file(&model_path)?,
        optimizer_file: "optimizer.mpk".into(),
        optimizer_sha256: sha256_file(&optimizer_path)?,
        resume_file: "resume.mpk".into(),
        resume_sha256: sha256_file(&resume_path)?,
        record_precision: "full".into(),
        resume_scope: "exact-update-boundary".into(),
        config: config.clone(),
        evaluation: evaluation.cloned(),
        feeding_evaluation: feeding_evaluation.cloned(),
        retention_feeding_evaluation: retention_feeding_evaluation.cloned(),
        contact_evaluation: contact_evaluation.cloned(),
        micro_combat_evaluation: micro_combat_evaluation.cloned(),
        rollout_pool_promotion: rollout_pool_promotion.cloned(),
    };
    validate_evaluation_artifact_binding(&metadata)?;
    write_json_file(&staging_path.join("metadata.json"), &metadata)?;
    sync_directory(staging_path)?;
    fs::rename(staging_path, &final_directory).map_err(|error| {
        format!(
            "failed to publish checkpoint {}: {error}",
            final_directory.display()
        )
    })?;
    staging.0 = None;
    sync_directory(root)?;
    Ok(final_directory)
}

/// Load and integrity-check an exact update-boundary checkpoint.
pub fn load_checkpoint<B, O>(
    directory: &Path,
    model_config: &PolicyValueNetConfig,
    optimizer: O,
    device: &B::Device,
) -> Result<
    (
        PolicyValueNet<B>,
        O,
        TrainingResumeState,
        CheckpointMetadata,
    ),
    String,
>
where
    B: AutodiffBackend,
    O: Optimizer<PolicyValueNet<B>, B>,
{
    let metadata = verified_metadata(directory)?;
    let current_backend = training_backend_id::<B>();
    if metadata.training_backend != current_backend {
        return Err(format!(
            "checkpoint training backend {} cannot be resumed exactly with {}",
            metadata.training_backend, current_backend
        ));
    }
    let resume_path = directory.join(&metadata.resume_file);

    let recorder = DefaultRecorder::new();
    let model = model_config
        .init(device)
        .load_file(directory.join("model"), &recorder, device)
        .map_err(|error| format!("failed to load model record: {error}"))?;
    let optimizer_record = recorder
        .load(directory.join("optimizer"), device)
        .map_err(|error| format!("failed to load optimizer record: {error}"))?;
    let optimizer = optimizer.load_record(optimizer_record);
    let resume_state: TrainingResumeState =
        rmp_serde::from_slice(&read_bounded(&resume_path, MAX_RESUME_STATE_BYTES)?)
            .map_err(|error| format!("failed to decode exact resume state: {error}"))?;
    validate_specialist_teacher_selection(&resume_state, &metadata.config.specialist_distillation)?;
    if resume_state.schema_version != TRAINING_ARTIFACT_SCHEMA_VERSION
        || resume_state.total_timesteps != metadata.actions
        || resume_state.update_count != metadata.update
        || resume_state
            .cumulative_sim_time_quanta
            .iter()
            .copied()
            .min()
            .unwrap_or(0)
            != metadata.minimum_sim_time_quanta_per_env
        || resume_state
            .cumulative_sim_time_quanta
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            != metadata.maximum_sim_time_quanta_per_env
        || resume_state
            .cumulative_sim_time_quanta
            .iter()
            .map(|quanta| u128::from(*quanta))
            .sum::<u128>()
            != metadata.total_sim_time_quanta
    {
        return Err("resume state counters do not match checkpoint metadata".into());
    }
    Ok((model, optimizer, resume_state, metadata))
}

/// Load a full-precision policy for inference after verifying the complete
/// immutable artifact, including optimizer and exact-resume sidecars.
pub fn load_policy_snapshot<B: Backend>(
    directory: &Path,
    device: &B::Device,
) -> Result<PolicySnapshot<B>, String> {
    let metadata = verified_metadata(directory)?;
    let model_config = PolicyValueNetConfig {
        hidden1: metadata.config.model.hidden1,
        hidden2: metadata.config.model.hidden2,
        recurrent_size: metadata.config.model.recurrent_size,
    };
    let model = model_config
        .init(device)
        .load_file(directory.join("model"), &DefaultRecorder::new(), device)
        .map_err(|error| format!("failed to load snapshot model record: {error}"))?;
    let checkpoint = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "snapshot path has no UTF-8 checkpoint name".to_string())?
        .to_string();
    Ok(PolicySnapshot {
        model,
        directory: directory.to_string_lossy().into_owned(),
        checkpoint,
        update: metadata.update,
        actions: metadata.actions,
        model_sha256: metadata.model_sha256,
    })
}

/// Load a previously promoted checkpoint and reject path substitution even if
/// the replacement is itself a valid artifact.
pub fn load_rollout_snapshot<B: Backend>(
    descriptor: &RolloutSnapshotDescriptor,
    device: &B::Device,
) -> Result<PolicySnapshot<B>, String> {
    let snapshot = load_policy_snapshot(Path::new(&descriptor.directory), device)?;
    if snapshot.descriptor() != *descriptor {
        return Err("rollout snapshot descriptor does not match its artifact".into());
    }
    Ok(snapshot)
}

/// Publish an immutable, human-inspectable description of the active pool.
#[allow(clippy::too_many_arguments)]
pub fn publish_rollout_pool_manifest(
    root: &Path,
    update: usize,
    actions: u64,
    compiled_ruleset_hash: &str,
    self_play: &SelfPlayConfig,
    promotion_feeding_evaluation: Option<&FeedingPromotionReport>,
    promotion_retention_feeding_evaluation: Option<&FeedingPromotionReport>,
    promotion_contact_evaluation: Option<&ContactEvaluationReport>,
    promotion_micro_combat_evaluation: Option<&MicroCombatEvaluationReport>,
    members: &[RolloutLeagueMember],
) -> Result<PathBuf, String> {
    if promotion_feeding_evaluation
        .is_some_and(|report| !report.passed || report.ruleset_hash != compiled_ruleset_hash)
        || promotion_retention_feeding_evaluation
            .is_some_and(|report| !report.passed || report.ruleset_hash != compiled_ruleset_hash)
    {
        return Err("rollout-pool manifest promotion has an invalid feeding gate".into());
    }
    if promotion_contact_evaluation
        .is_some_and(|report| !report.is_active() || report.ruleset_hash != compiled_ruleset_hash)
    {
        return Err("rollout-pool manifest promotion has invalid contact evidence".into());
    }
    if promotion_micro_combat_evaluation
        .is_some_and(|report| report.ruleset_hash != compiled_ruleset_hash)
    {
        return Err("rollout-pool manifest promotion has invalid micro-combat evidence".into());
    }
    let pool_root = root.join("opponent-pool");
    fs::create_dir_all(&pool_root).map_err(|error| {
        format!(
            "failed to create rollout-pool directory {}: {error}",
            pool_root.display()
        )
    })?;
    let final_path = pool_root.join(format!("manifest-{update:08}.json"));
    if final_path.exists() {
        return Err(format!(
            "refusing to replace immutable rollout-pool manifest {}",
            final_path.display()
        ));
    }
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = pool_root.join(format!(
        ".manifest-{update:08}.tmp-{}-{nonce}",
        std::process::id()
    ));
    let manifest = RolloutPoolManifest {
        schema_version: TRAINING_ARTIFACT_SCHEMA_VERSION,
        update,
        actions,
        compiled_ruleset_hash: compiled_ruleset_hash.to_string(),
        self_play: self_play.clone(),
        promotion_feeding_evaluation: promotion_feeding_evaluation.cloned(),
        promotion_retention_feeding_evaluation: promotion_retention_feeding_evaluation.cloned(),
        promotion_contact_evaluation: promotion_contact_evaluation.cloned(),
        promotion_micro_combat_evaluation: promotion_micro_combat_evaluation.cloned(),
        members: members.to_vec(),
    };
    validate_rollout_pool_manifest(root, &manifest)?;
    write_json_file(&temporary, &manifest)?;
    fs::rename(&temporary, &final_path)
        .map_err(|error| format!("failed to publish rollout-pool manifest: {error}"))?;
    sync_directory(&pool_root)?;
    Ok(final_path)
}

/// Verify a pool manifest against the promoted checkpoint and every active
/// member artifact. This is intentionally an offline/audit path: it hashes the
/// complete checkpoint file sets rather than trusting descriptor strings.
pub fn verify_rollout_pool_manifest(path: &Path) -> Result<RolloutPoolManifest, String> {
    let manifest: RolloutPoolManifest =
        serde_json::from_slice(&read_bounded(path, MAX_METADATA_BYTES)?)
            .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    let expected_name = format!("manifest-{:08}.json", manifest.update);
    if manifest.schema_version != TRAINING_ARTIFACT_SCHEMA_VERSION
        || path.file_name().and_then(|name| name.to_str()) != Some(expected_name.as_str())
    {
        return Err("rollout-pool manifest has an unsupported schema or filename".into());
    }
    let root = path
        .parent()
        .and_then(Path::parent)
        .ok_or("rollout-pool manifest is not inside an artifact root")?;
    validate_rollout_pool_manifest(root, &manifest)?;
    Ok(manifest)
}

fn validate_rollout_pool_manifest(
    root: &Path,
    manifest: &RolloutPoolManifest,
) -> Result<(), String> {
    let promotion_checkpoint = root.join(format!("checkpoint-{:08}", manifest.update));
    let promotion = verified_metadata(&promotion_checkpoint)?;
    if promotion.actions != manifest.actions
        || promotion.compiled_ruleset_hash != manifest.compiled_ruleset_hash
        || promotion.config.self_play != manifest.self_play
        || promotion.rollout_pool_promotion.is_none()
        || promotion.feeding_evaluation != manifest.promotion_feeding_evaluation
        || promotion.retention_feeding_evaluation != manifest.promotion_retention_feeding_evaluation
        || promotion.contact_evaluation != manifest.promotion_contact_evaluation
        || promotion.micro_combat_evaluation != manifest.promotion_micro_combat_evaluation
    {
        return Err("rollout-pool manifest does not match its promotion checkpoint".into());
    }
    if promotion.config.combat_curriculum.enabled
        && !manifest
            .promotion_contact_evaluation
            .as_ref()
            .is_some_and(|report| {
                report.meets_promotion_thresholds(&promotion.config.combat_curriculum)
            })
    {
        return Err("rollout-pool manifest does not meet combat promotion thresholds".into());
    }
    if promotion.config.combat_curriculum.micro_combat.enabled
        && !manifest
            .promotion_micro_combat_evaluation
            .as_ref()
            .is_some_and(|report| {
                report.meets_promotion_thresholds(&promotion.config.combat_curriculum.micro_combat)
            })
    {
        return Err("rollout-pool manifest does not meet micro-combat thresholds".into());
    }
    for member in &manifest.members {
        let directory = Path::new(&member.snapshot.directory);
        let metadata = verified_metadata(directory)?;
        if directory.file_name().and_then(|name| name.to_str())
            != Some(member.snapshot.checkpoint.as_str())
            || metadata.update != member.snapshot.update
            || metadata.actions != member.snapshot.actions
            || metadata.model_sha256 != member.snapshot.model_sha256
            || metadata.compiled_ruleset_hash != manifest.compiled_ruleset_hash
            || metadata.rollout_pool_promotion.is_none()
        {
            return Err("rollout-pool member does not match its promoted checkpoint".into());
        }
    }
    Ok(())
}

/// Atomically point `best.json` at an already-published immutable checkpoint.
pub fn publish_best_pointer(root: &Path, checkpoint: &Path) -> Result<(), String> {
    let metadata = verified_metadata(checkpoint)?;
    let evaluation = metadata
        .evaluation
        .clone()
        .ok_or("best checkpoint has no held-out evaluation")?;
    if metadata.config.feeding_curriculum.enabled
        && (!metadata
            .feeding_evaluation
            .as_ref()
            .is_some_and(|report| report.passed)
            || metadata
                .retention_feeding_evaluation
                .as_ref()
                .is_some_and(|report| !report.passed))
    {
        return Err("best checkpoint did not pass the feeding gate".into());
    }
    if metadata.config.combat_curriculum.enabled
        && !metadata.contact_evaluation.as_ref().is_some_and(|report| {
            report.meets_promotion_thresholds(&metadata.config.combat_curriculum)
        })
    {
        return Err("best checkpoint has no active contact evidence".into());
    }
    if metadata.config.combat_curriculum.micro_combat.enabled
        && !metadata
            .micro_combat_evaluation
            .as_ref()
            .is_some_and(|report| {
                report.meets_promotion_thresholds(&metadata.config.combat_curriculum.micro_combat)
            })
    {
        return Err("best checkpoint failed a micro-combat gate".into());
    }
    let checkpoint_name = checkpoint
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "checkpoint path has no UTF-8 directory name".to_string())?;
    let pointer = BestCheckpointPointer {
        schema_version: TRAINING_ARTIFACT_SCHEMA_VERSION,
        checkpoint: checkpoint_name.to_string(),
        update: metadata.update,
        actions: metadata.actions,
        evaluation,
        feeding_evaluation: metadata.feeding_evaluation,
        retention_feeding_evaluation: metadata.retention_feeding_evaluation,
        contact_evaluation: metadata.contact_evaluation,
        micro_combat_evaluation: metadata.micro_combat_evaluation,
    };
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = root.join(format!(".best.json.tmp-{}-{nonce}", std::process::id()));
    write_json_file(&temporary, &pointer)?;
    fs::rename(&temporary, root.join("best.json"))
        .map_err(|error| format!("failed to publish best checkpoint pointer: {error}"))?;
    sync_directory(root)
}

/// Verify that the mutable best pointer exactly mirrors the immutable
/// checkpoint it names, including the independently recomputable feeding gate.
pub fn verify_best_pointer(root: &Path) -> Result<BestCheckpointPointer, String> {
    let path = root.join("best.json");
    let pointer: BestCheckpointPointer =
        serde_json::from_slice(&read_bounded(&path, MAX_METADATA_BYTES)?)
            .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    let checkpoint_component = Path::new(&pointer.checkpoint);
    if pointer.schema_version != TRAINING_ARTIFACT_SCHEMA_VERSION
        || checkpoint_component.components().count() != 1
        || checkpoint_component
            .file_name()
            .and_then(|name| name.to_str())
            != Some(pointer.checkpoint.as_str())
    {
        return Err("best pointer has an unsupported schema or checkpoint path".into());
    }
    let metadata = verified_metadata(&root.join(&pointer.checkpoint))?;
    if metadata.update != pointer.update
        || metadata.actions != pointer.actions
        || metadata.evaluation.as_ref() != Some(&pointer.evaluation)
        || metadata.feeding_evaluation != pointer.feeding_evaluation
        || metadata.retention_feeding_evaluation != pointer.retention_feeding_evaluation
        || metadata.contact_evaluation != pointer.contact_evaluation
        || metadata.micro_combat_evaluation != pointer.micro_combat_evaluation
    {
        return Err("best pointer does not match its immutable checkpoint".into());
    }
    if metadata.config.feeding_curriculum.enabled
        && (!pointer
            .feeding_evaluation
            .as_ref()
            .is_some_and(|report| report.passed)
            || pointer
                .retention_feeding_evaluation
                .as_ref()
                .is_some_and(|report| !report.passed))
    {
        return Err("best pointer does not contain a passing feeding gate".into());
    }
    if metadata.config.combat_curriculum.enabled
        && !pointer.contact_evaluation.as_ref().is_some_and(|report| {
            report.meets_promotion_thresholds(&metadata.config.combat_curriculum)
        })
    {
        return Err("best pointer contains no active contact evidence".into());
    }
    if metadata.config.combat_curriculum.micro_combat.enabled
        && !pointer
            .micro_combat_evaluation
            .as_ref()
            .is_some_and(|report| {
                report.meets_promotion_thresholds(&metadata.config.combat_curriculum.micro_combat)
            })
    {
        return Err("best pointer failed a micro-combat gate".into());
    }
    Ok(pointer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::optim::AdamWConfig;
    use rand::SeedableRng;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn checkpoint_directory_is_complete_immutable_and_hash_bound() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let device = Default::default();
        TestBackend::seed(&device, 11);
        let model: PolicyValueNet<TestBackend> =
            crate::model::PolicyValueNetConfig::new().init(&device);
        let optimizer = AdamWConfig::new().init();
        let config = TrainingConfig::default();
        let resume_state = TrainingResumeState {
            schema_version: TRAINING_ARTIFACT_SCHEMA_VERSION,
            total_timesteps: 1234,
            cumulative_sim_time_quanta: vec![111, 222],
            update_count: 7,
            action_rng: ChaCha12Rng::seed_from_u64(1),
            optimizer_rng: ChaCha12Rng::seed_from_u64(2),
            opponent_rng: ChaCha12Rng::seed_from_u64(3),
            env_episode_ids: Vec::new(),
            env_episode_rewards: Vec::new(),
            environments: Vec::new(),
            best_evaluation: None,
            best_contact_evaluation: None,
            best_micro_combat_evaluation: None,
            competency_frontier: CompetencyFrontier::default(),
            specialist_teachers: SpecialistTeacherSelection::default(),
            rollout_pool: Vec::new(),
            active_retired_snapshots: Vec::new(),
            environment_opponents: Vec::new(),
            environment_curriculum_stages: Vec::new(),
            micro_combat_rotation: MicroCombatRotationState::default(),
            environment_micro_combat_scenarios: Vec::new(),
            telemetry: None,
        };
        let mut invalid_teacher_state = resume_state.clone();
        invalid_teacher_state.specialist_teachers.ecology =
            Some(crate::competency_frontier::CompetencyCheckpointIdentity {
                checkpoint_directory: temporary.path().display().to_string(),
                checkpoint: "checkpoint-00000007".into(),
                update: 7,
                actions: 1234,
            });
        let specialist_config = SpecialistDistillationConfig {
            enabled: true,
            ..SpecialistDistillationConfig::default()
        };
        assert!(
            validate_specialist_teacher_selection(&invalid_teacher_state, &specialist_config)
                .is_err()
        );

        let checkpoint = publish_checkpoint(
            temporary.path(),
            &model,
            &optimizer,
            &config,
            7,
            1234,
            "ruleset-hash",
            None,
            None,
            None,
            None,
            None,
            None,
            &resume_state,
        )
        .unwrap();
        assert_eq!(checkpoint.file_name().unwrap(), "checkpoint-00000007");
        assert!(checkpoint.join("model.mpk").is_file());
        assert!(checkpoint.join("optimizer.mpk").is_file());
        assert!(checkpoint.join("resume.mpk").is_file());
        let metadata: CheckpointMetadata =
            serde_json::from_slice(&fs::read(checkpoint.join("metadata.json")).unwrap()).unwrap();
        assert_eq!(metadata.update, 7);
        assert_eq!(metadata.actions, 1234);
        assert_eq!(metadata.minimum_sim_time_quanta_per_env, 111);
        assert_eq!(metadata.maximum_sim_time_quanta_per_env, 222);
        assert_eq!(metadata.total_sim_time_quanta, 333);
        assert_eq!(
            metadata.training_backend,
            training_backend_id::<TestBackend>()
        );
        assert_eq!(
            metadata.model_sha256,
            sha256_file(&checkpoint.join("model.mpk")).unwrap()
        );
        assert_eq!(
            metadata.optimizer_sha256,
            sha256_file(&checkpoint.join("optimizer.mpk")).unwrap()
        );
        assert_eq!(
            metadata.resume_sha256,
            sha256_file(&checkpoint.join("resume.mpk")).unwrap()
        );
        let (_, _, loaded_state, loaded_metadata) = load_checkpoint::<TestBackend, _>(
            &checkpoint,
            &crate::model::PolicyValueNetConfig::new(),
            AdamWConfig::new().init(),
            &device,
        )
        .unwrap();
        assert_eq!(loaded_state.update_count, 7);
        assert_eq!(loaded_state.total_timesteps, 1234);
        assert_eq!(loaded_state.cumulative_sim_time_quanta, vec![111, 222]);
        assert_eq!(loaded_metadata.resume_scope, "exact-update-boundary");
        let metadata_path = checkpoint.join("metadata.json");
        let mut incompatible_metadata = metadata.clone();
        incompatible_metadata.training_backend = "different-backend".into();
        write_json_file(&metadata_path, &incompatible_metadata).unwrap();
        let backend_error = load_checkpoint::<TestBackend, _>(
            &checkpoint,
            &crate::model::PolicyValueNetConfig::new(),
            AdamWConfig::new().init(),
            &device,
        )
        .err()
        .expect("an exact cross-backend resume must be rejected");
        assert!(backend_error.contains("cannot be resumed exactly"));
        write_json_file(&metadata_path, &metadata).unwrap();
        let snapshot = load_policy_snapshot::<NdArray<f32>>(&checkpoint, &device).unwrap();
        assert_eq!(snapshot.checkpoint, "checkpoint-00000007");
        assert_eq!(snapshot.update, 7);
        assert_eq!(snapshot.actions, 1234);
        assert_eq!(snapshot.model_sha256, metadata.model_sha256);
        let descriptor = snapshot.descriptor();
        assert_eq!(
            load_rollout_snapshot::<NdArray<f32>>(&descriptor, &device)
                .unwrap()
                .descriptor(),
            descriptor
        );
        let mut substituted = descriptor.clone();
        substituted.update += 1;
        assert!(load_rollout_snapshot::<NdArray<f32>>(&substituted, &device).is_err());
        assert!(publish_checkpoint(
            temporary.path(),
            &model,
            &optimizer,
            &config,
            7,
            1234,
            "ruleset-hash",
            None,
            None,
            None,
            None,
            None,
            None,
            &resume_state,
        )
        .is_err());

        let model_path = checkpoint.join("model.mpk");
        let mut tampered = fs::read(&model_path).unwrap();
        tampered[0] ^= 1;
        fs::write(model_path, tampered).unwrap();
        let error = load_policy_snapshot::<NdArray<f32>>(&checkpoint, &device)
            .err()
            .expect("a tampered snapshot must be rejected");
        assert!(error.contains("SHA-256 mismatch"));
    }

    #[test]
    fn best_pointer_references_an_immutable_checkpoint() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let device = Default::default();
        let model: PolicyValueNet<TestBackend> =
            crate::model::PolicyValueNetConfig::new().init(&device);
        let optimizer = AdamWConfig::new().init();
        let mut config = TrainingConfig::default();
        config.feeding_curriculum.enabled = true;
        config.initial_policy = Some(crate::config::InitialPolicyConfig {
            directory: "qualified-clone".into(),
            artifact_sha256: "a".repeat(64),
            qualification: Some(crate::config::InitialPolicyQualificationConfig {
                path: "qualification.json".into(),
                artifact_hash: "b".repeat(64),
            }),
        });
        let evaluation = EvaluationMetrics {
            seeds: vec![100],
            opponents: Vec::new(),
            episodes: 1,
            wins: 1,
            losses: 0,
            timeouts: 0,
            worst_case_win_rate: 1.0,
            win_rate: 1.0,
            average_episode_len: 4.0,
            average_reward: 2.5,
            actions: 8,
        };
        let stage = |stage| crate::feeding_curriculum::FeedingStageMetrics {
            stage,
            episodes: 1,
            successful_episodes: 1,
            initial_cells: 1,
            surviving_cells: 1,
            movement_successes: 1,
            consume_successes: 1,
            consumed_energy: 2,
            safety_aborts: 0,
            episode_success_rate: 1.0,
            survival_rate: 1.0,
            consumed_energy_per_initial_cell: 2.0,
        };
        let feeding = crate::feeding_curriculum::report_from_metrics(
            "ruleset-hash".into(),
            evaluation.seeds.clone(),
            [
                stage(crate::config::FeedingCurriculumStage::OnFood),
                stage(crate::config::FeedingCurriculumStage::AdjacentFood),
            ],
            &config.feeding_curriculum.promotion,
        );
        let retention_feeding = crate::feeding_curriculum::report_from_metrics(
            "ruleset-hash".into(),
            vec![200],
            [
                stage(crate::config::FeedingCurriculumStage::OnFood),
                stage(crate::config::FeedingCurriculumStage::AdjacentFood),
            ],
            &config.feeding_curriculum.promotion,
        );
        let failed_stage = |stage| crate::feeding_curriculum::FeedingStageMetrics {
            stage,
            episodes: 1,
            successful_episodes: 0,
            initial_cells: 1,
            surviving_cells: 0,
            movement_successes: 0,
            consume_successes: 0,
            consumed_energy: 0,
            safety_aborts: 0,
            episode_success_rate: 0.0,
            survival_rate: 0.0,
            consumed_energy_per_initial_cell: 0.0,
        };
        let failed_retention = crate::feeding_curriculum::report_from_metrics(
            "ruleset-hash".into(),
            vec![200],
            [
                failed_stage(crate::config::FeedingCurriculumStage::OnFood),
                failed_stage(crate::config::FeedingCurriculumStage::AdjacentFood),
            ],
            &config.feeding_curriculum.promotion,
        );
        let resume_state = TrainingResumeState {
            schema_version: TRAINING_ARTIFACT_SCHEMA_VERSION,
            total_timesteps: 90,
            cumulative_sim_time_quanta: Vec::new(),
            update_count: 3,
            action_rng: ChaCha12Rng::seed_from_u64(1),
            optimizer_rng: ChaCha12Rng::seed_from_u64(2),
            opponent_rng: ChaCha12Rng::seed_from_u64(3),
            env_episode_ids: Vec::new(),
            env_episode_rewards: Vec::new(),
            environments: Vec::new(),
            best_evaluation: None,
            best_contact_evaluation: None,
            best_micro_combat_evaluation: None,
            competency_frontier: CompetencyFrontier::default(),
            specialist_teachers: SpecialistTeacherSelection::default(),
            rollout_pool: Vec::new(),
            active_retired_snapshots: Vec::new(),
            environment_opponents: Vec::new(),
            environment_curriculum_stages: Vec::new(),
            micro_combat_rotation: MicroCombatRotationState::default(),
            environment_micro_combat_scenarios: Vec::new(),
            telemetry: None,
        };
        let promotion = LeaguePromotion {
            rating: 1_000.0,
            evaluation_games: 1,
        };
        let mut rejected_state = resume_state.clone();
        rejected_state.total_timesteps = 80;
        rejected_state.update_count = 2;
        let rejected = publish_checkpoint(
            temporary.path(),
            &model,
            &optimizer,
            &config,
            2,
            80,
            "ruleset-hash",
            Some(&evaluation),
            Some(&feeding),
            Some(&failed_retention),
            None,
            None,
            None,
            &rejected_state,
        )
        .unwrap();
        assert!(verify_checkpoint_metadata(&rejected).is_ok());
        assert!(publish_best_pointer(temporary.path(), &rejected).is_err());
        let checkpoint = publish_checkpoint(
            temporary.path(),
            &model,
            &optimizer,
            &config,
            3,
            90,
            "ruleset-hash",
            Some(&evaluation),
            Some(&feeding),
            Some(&retention_feeding),
            None,
            None,
            Some(&promotion),
            &resume_state,
        )
        .unwrap();
        publish_best_pointer(temporary.path(), &checkpoint).unwrap();
        let pointer = verify_best_pointer(temporary.path()).unwrap();
        assert_eq!(pointer.checkpoint, "checkpoint-00000003");
        assert_eq!(pointer.evaluation, evaluation);
        assert_eq!(pointer.feeding_evaluation, Some(feeding.clone()));
        assert_eq!(
            pointer.retention_feeding_evaluation,
            Some(retention_feeding.clone())
        );

        let snapshot = load_policy_snapshot::<NdArray<f32>>(&checkpoint, &device).unwrap();
        let member = RolloutLeagueMember {
            snapshot: snapshot.descriptor(),
            rating: promotion.rating,
            evaluation_games: promotion.evaluation_games,
            rollout_selections: 0,
        };
        assert!(publish_rollout_pool_manifest(
            temporary.path(),
            2,
            80,
            "ruleset-hash",
            &config.self_play,
            Some(&feeding),
            Some(&failed_retention),
            None,
            None,
            std::slice::from_ref(&member),
        )
        .is_err());
        let manifest_path = publish_rollout_pool_manifest(
            temporary.path(),
            3,
            90,
            "ruleset-hash",
            &config.self_play,
            Some(&feeding),
            Some(&retention_feeding),
            None,
            None,
            &[member],
        )
        .unwrap();
        let manifest = verify_rollout_pool_manifest(&manifest_path).unwrap();
        assert_eq!(manifest.promotion_feeding_evaluation, Some(feeding.clone()));
        assert_eq!(
            manifest.promotion_retention_feeding_evaluation,
            Some(retention_feeding)
        );
        let mut tampered_manifest = manifest;
        tampered_manifest
            .promotion_feeding_evaluation
            .as_mut()
            .unwrap()
            .passed = false;
        write_json_file(&manifest_path, &tampered_manifest).unwrap();
        assert!(verify_rollout_pool_manifest(&manifest_path).is_err());

        let mut tampered_pointer = pointer;
        tampered_pointer.feeding_evaluation.as_mut().unwrap().checks[0].passed = false;
        write_json_file(&temporary.path().join("best.json"), &tampered_pointer).unwrap();
        assert!(verify_best_pointer(temporary.path()).is_err());

        let metadata_path = checkpoint.join("metadata.json");
        let mut metadata: CheckpointMetadata =
            serde_json::from_slice(&fs::read(&metadata_path).unwrap()).unwrap();
        metadata.feeding_evaluation.as_mut().unwrap().passed = false;
        write_json_file(&metadata_path, &metadata).unwrap();
        assert!(verify_checkpoint_metadata(&checkpoint).is_err());
    }
}
