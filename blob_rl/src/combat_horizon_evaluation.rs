//! Immutable combat-horizon sensitivity evidence for one frozen checkpoint.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::artifact::verify_checkpoint_metadata;
use crate::checkpoint_evaluation::CombatEvaluationHorizons;
use crate::config::{FeedingCurriculumStage, TrainingConfig};
use crate::contact_evaluation::ContactEvaluationReport;
use crate::env::BlobEnv;
use crate::sweep::sha256;

pub const COMBAT_HORIZON_EVALUATION_SCHEMA_VERSION: u32 = 1;
const MAX_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CombatHorizonResult {
    pub horizons: CombatEvaluationHorizons,
    pub report: ContactEvaluationReport,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CombatHorizonEvaluationArtifact {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub artifact_hash: String,
    pub checkpoint_metadata_sha256: String,
    pub checkpoint_model_sha256: String,
    pub checkpoint_update: usize,
    pub checkpoint_actions: u64,
    /// Exact immutable training configuration. Per-result horizon overrides
    /// are deliberately stored separately and may not alter any other field.
    pub config: TrainingConfig,
    pub seeds: Vec<u64>,
    pub results: Vec<CombatHorizonResult>,
}

#[derive(Serialize)]
struct ArtifactIdentity<'a> {
    schema_version: u32,
    package_version: &'a str,
    code_revision: &'a Option<String>,
    checkpoint_metadata_sha256: &'a str,
    checkpoint_model_sha256: &'a str,
    checkpoint_update: usize,
    checkpoint_actions: u64,
    config: &'a TrainingConfig,
    seeds: &'a [u64],
    results: &'a [CombatHorizonResult],
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

impl CombatHorizonEvaluationArtifact {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        checkpoint_metadata_sha256: String,
        checkpoint_model_sha256: String,
        checkpoint_update: usize,
        checkpoint_actions: u64,
        config: TrainingConfig,
        seeds: Vec<u64>,
        results: Vec<CombatHorizonResult>,
    ) -> Result<Self, String> {
        let mut artifact = Self {
            schema_version: COMBAT_HORIZON_EVALUATION_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").into(),
            code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
            artifact_hash: String::new(),
            checkpoint_metadata_sha256,
            checkpoint_model_sha256,
            checkpoint_update,
            checkpoint_actions,
            config,
            seeds,
            results,
        };
        artifact.artifact_hash = artifact.recompute_hash()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn recompute_hash(&self) -> Result<String, String> {
        serde_json::to_vec(&ArtifactIdentity {
            schema_version: self.schema_version,
            package_version: &self.package_version,
            code_revision: &self.code_revision,
            checkpoint_metadata_sha256: &self.checkpoint_metadata_sha256,
            checkpoint_model_sha256: &self.checkpoint_model_sha256,
            checkpoint_update: self.checkpoint_update,
            checkpoint_actions: self.checkpoint_actions,
            config: &self.config,
            seeds: &self.seeds,
            results: &self.results,
        })
        .map(|bytes| sha256(&bytes))
        .map_err(|error| format!("failed to encode combat-horizon identity: {error}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != COMBAT_HORIZON_EVALUATION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported combat-horizon schema {}; expected {}",
                self.schema_version, COMBAT_HORIZON_EVALUATION_SCHEMA_VERSION
            ));
        }
        self.config.validate()?;
        if !valid_sha256(&self.checkpoint_metadata_sha256)
            || !valid_sha256(&self.checkpoint_model_sha256)
            || self.seeds.is_empty()
            || self.results.is_empty()
        {
            return Err("combat-horizon hashes, seeds, or results are invalid".into());
        }

        let mut seen = std::collections::HashSet::new();
        for result in &self.results {
            if !seen.insert(result.horizons) {
                return Err("combat-horizon results contain a duplicate horizon pair".into());
            }
            let mut evaluation_config = self.config.clone();
            result.horizons.apply_to(&mut evaluation_config)?;
            let contact_start = evaluation_config
                .combat_curriculum
                .on_food_sim_time_quanta_per_cycle
                .saturating_add(
                    evaluation_config
                        .combat_curriculum
                        .adjacent_food_sim_time_quanta_per_cycle,
                );
            let contact_env = evaluation_config
                .combat_curriculum
                .evaluation_environment_for_stage(
                    &evaluation_config.feeding_curriculum,
                    &evaluation_config.env,
                    FeedingCurriculumStage::Contact,
                    contact_start,
                );
            let ruleset =
                BlobEnv::new(contact_env, evaluation_config.reward.clone(), self.seeds[0])
                    .compiled_ruleset_hash();
            result.report.validate_against(
                &ruleset,
                &self.seeds,
                &evaluation_config.combat_curriculum,
            )?;
            if result.passed
                != result
                    .report
                    .meets_promotion_thresholds(&evaluation_config.combat_curriculum)
            {
                return Err("combat-horizon result has an inconsistent pass decision".into());
            }
        }
        if self.artifact_hash != self.recompute_hash()? {
            return Err("combat-horizon artifact hash mismatch".into());
        }
        Ok(())
    }
}

pub fn publish_combat_horizon_evaluation(
    output: &Path,
    artifact: &CombatHorizonEvaluationArtifact,
) -> Result<PathBuf, String> {
    artifact.validate()?;
    if output.exists() {
        return Err(format!(
            "refusing to replace combat-horizon evaluation {}",
            output.display()
        ));
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "combat-horizon output needs a UTF-8 name".to_string())?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let staging = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(artifact)
        .map_err(|error| format!("failed to encode combat-horizon evaluation: {error}"))?;
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

pub fn load_combat_horizon_evaluation(
    path: &Path,
) -> Result<CombatHorizonEvaluationArtifact, String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > MAX_ARTIFACT_BYTES {
        return Err(format!(
            "combat-horizon artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
        ));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let artifact: CombatHorizonEvaluationArtifact = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    artifact.validate()?;
    Ok(artifact)
}

pub fn verify_combat_horizon_evaluation(
    path: &Path,
    expected_artifact_hash: &str,
    checkpoint: &Path,
) -> Result<CombatHorizonEvaluationArtifact, String> {
    let artifact = load_combat_horizon_evaluation(path)?;
    if artifact.artifact_hash != expected_artifact_hash {
        return Err(format!(
            "combat-horizon artifact hash mismatch: expected {expected_artifact_hash}, found {}",
            artifact.artifact_hash
        ));
    }
    let metadata = verify_checkpoint_metadata(checkpoint)?;
    let metadata_bytes = fs::read(checkpoint.join("metadata.json"))
        .map_err(|error| format!("failed to read checkpoint metadata: {error}"))?;
    if artifact.checkpoint_metadata_sha256 != sha256(&metadata_bytes)
        || artifact.checkpoint_model_sha256 != metadata.model_sha256
        || artifact.checkpoint_update != metadata.update
        || artifact.checkpoint_actions != metadata.actions
        || artifact.config != metadata.config
    {
        return Err("combat-horizon evaluation names a different checkpoint".into());
    }
    Ok(artifact)
}
