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

use crate::config::{OpponentProfile, SelfPlayConfig, TrainingConfig};
use crate::env::BlobEnvCheckpoint;
use crate::evaluation::EvaluationMetrics;
use crate::model::{PolicyValueNet, PolicyValueNetConfig};
use crate::telemetry::TrainingTelemetryState;

pub const TRAINING_ARTIFACT_SCHEMA_VERSION: u32 = 16;
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
    pub rollout_pool: Vec<RolloutLeagueMember>,
    /// Evicted pool members still needed by episodes already in progress.
    pub active_retired_snapshots: Vec<RolloutSnapshotDescriptor>,
    pub environment_opponents: Vec<RolloutOpponentAssignment>,
    /// Exact bounded host-telemetry continuation. This is scientifically
    /// useful output state, never an input to policy or physics.
    pub telemetry: Option<TrainingTelemetryState>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BestCheckpointPointer {
    pub schema_version: u32,
    pub checkpoint: String,
    pub update: usize,
    pub actions: u64,
    pub evaluation: EvaluationMetrics,
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
    Ok(metadata)
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
    rollout_pool_promotion: Option<&LeaguePromotion>,
    resume_state: &TrainingResumeState,
) -> Result<PathBuf, String>
where
    B: AutodiffBackend,
    O: Optimizer<PolicyValueNet<B>, B>,
{
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
        rollout_pool_promotion: rollout_pool_promotion.cloned(),
    };
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
    if resume_state.schema_version != TRAINING_ARTIFACT_SCHEMA_VERSION
        || resume_state.total_timesteps != metadata.actions
        || resume_state.update_count != metadata.update
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
pub fn publish_rollout_pool_manifest(
    root: &Path,
    update: usize,
    actions: u64,
    compiled_ruleset_hash: &str,
    self_play: &SelfPlayConfig,
    members: &[RolloutLeagueMember],
) -> Result<PathBuf, String> {
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
    write_json_file(
        &temporary,
        &RolloutPoolManifest {
            schema_version: TRAINING_ARTIFACT_SCHEMA_VERSION,
            update,
            actions,
            compiled_ruleset_hash: compiled_ruleset_hash.to_string(),
            self_play: self_play.clone(),
            members: members.to_vec(),
        },
    )?;
    fs::rename(&temporary, &final_path)
        .map_err(|error| format!("failed to publish rollout-pool manifest: {error}"))?;
    sync_directory(&pool_root)?;
    Ok(final_path)
}

/// Atomically point `best.json` at an already-published immutable checkpoint.
pub fn publish_best_pointer(
    root: &Path,
    checkpoint: &Path,
    update: usize,
    actions: u64,
    evaluation: &EvaluationMetrics,
) -> Result<(), String> {
    let checkpoint_name = checkpoint
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "checkpoint path has no UTF-8 directory name".to_string())?;
    let pointer = BestCheckpointPointer {
        schema_version: TRAINING_ARTIFACT_SCHEMA_VERSION,
        checkpoint: checkpoint_name.to_string(),
        update,
        actions,
        evaluation: evaluation.clone(),
    };
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = root.join(format!(".best.json.tmp-{}-{nonce}", std::process::id()));
    write_json_file(&temporary, &pointer)?;
    fs::rename(&temporary, root.join("best.json"))
        .map_err(|error| format!("failed to publish best checkpoint pointer: {error}"))?;
    sync_directory(root)
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
            rollout_pool: Vec::new(),
            active_retired_snapshots: Vec::new(),
            environment_opponents: Vec::new(),
            telemetry: None,
        };

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
        let member = RolloutLeagueMember {
            snapshot: descriptor.clone(),
            rating: 1_024.0,
            evaluation_games: 8,
            rollout_selections: 3,
        };
        let manifest = publish_rollout_pool_manifest(
            temporary.path(),
            7,
            1234,
            "ruleset-hash",
            &config.self_play,
            std::slice::from_ref(&member),
        )
        .unwrap();
        let decoded: RolloutPoolManifest =
            serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
        assert_eq!(decoded.members, vec![member.clone()]);
        assert!(publish_rollout_pool_manifest(
            temporary.path(),
            7,
            1234,
            "ruleset-hash",
            &config.self_play,
            &[member],
        )
        .is_err());
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
        let temporary = tempfile::tempdir().unwrap();
        let checkpoint = temporary.path().join("checkpoint-00000003");
        fs::create_dir(&checkpoint).unwrap();
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
        publish_best_pointer(temporary.path(), &checkpoint, 3, 90, &evaluation).unwrap();
        let pointer: BestCheckpointPointer =
            serde_json::from_slice(&fs::read(temporary.path().join("best.json")).unwrap()).unwrap();
        assert_eq!(pointer.checkpoint, "checkpoint-00000003");
        assert_eq!(pointer.evaluation, evaluation);
    }
}
