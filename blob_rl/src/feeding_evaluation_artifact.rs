//! Immutable feeding-competency evidence for a behavior-cloned warm start.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::config::{FeedingCurriculumStage, TrainingConfig};
use crate::env::BlobEnv;
use crate::feeding_curriculum::FeedingPromotionReport;
use crate::sweep::sha256;

pub const FEEDING_EVALUATION_ARTIFACT_SCHEMA_VERSION: u32 = 3;
const MAX_ARTIFACT_BYTES: u64 = 4 * 1024 * 1024;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingEvaluationArtifact {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub artifact_hash: String,
    pub source_config_sha256: String,
    pub behavior_clone_metadata_sha256: String,
    pub behavior_clone_model_sha256: String,
    pub config: TrainingConfig,
    pub report: FeedingPromotionReport,
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
    report: &'a FeedingPromotionReport,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

impl FeedingEvaluationArtifact {
    pub fn new(
        source_config_sha256: String,
        behavior_clone_metadata_sha256: String,
        behavior_clone_model_sha256: String,
        config: TrainingConfig,
        report: FeedingPromotionReport,
    ) -> Result<Self, String> {
        let mut artifact = Self {
            schema_version: FEEDING_EVALUATION_ARTIFACT_SCHEMA_VERSION,
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
        let identity = ArtifactIdentity {
            schema_version: self.schema_version,
            package_version: &self.package_version,
            code_revision: &self.code_revision,
            source_config_sha256: &self.source_config_sha256,
            behavior_clone_metadata_sha256: &self.behavior_clone_metadata_sha256,
            behavior_clone_model_sha256: &self.behavior_clone_model_sha256,
            config: &self.config,
            report: &self.report,
        };
        let bytes = serde_json::to_vec(&identity)
            .map_err(|error| format!("failed to encode feeding-evaluation identity: {error}"))?;
        Ok(sha256(&bytes))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != FEEDING_EVALUATION_ARTIFACT_SCHEMA_VERSION {
            return Err(format!(
                "unsupported feeding-evaluation artifact schema {}; expected {}",
                self.schema_version, FEEDING_EVALUATION_ARTIFACT_SCHEMA_VERSION
            ));
        }
        self.config.validate()?;
        if !valid_sha256(&self.source_config_sha256)
            || !valid_sha256(&self.behavior_clone_metadata_sha256)
            || !valid_sha256(&self.behavior_clone_model_sha256)
            || self.report.seeds.is_empty()
        {
            return Err("feeding-evaluation artifact hashes or seed suite are invalid".into());
        }
        let staged = self
            .config
            .feeding_curriculum
            .environment_for_stage(&self.config.env, FeedingCurriculumStage::OnFood);
        let expected_ruleset =
            BlobEnv::new(staged, self.config.reward.clone(), self.report.seeds[0])
                .compiled_ruleset_hash();
        self.report.validate_against(
            &expected_ruleset,
            &self.report.seeds,
            &self.config.feeding_curriculum.promotion,
        )?;
        if self.artifact_hash != self.recompute_hash()? {
            return Err("feeding-evaluation artifact hash mismatch".into());
        }
        Ok(())
    }
}

pub fn publish_feeding_evaluation(
    output: &Path,
    artifact: &FeedingEvaluationArtifact,
) -> Result<PathBuf, String> {
    artifact.validate()?;
    if output.exists() {
        return Err(format!(
            "refusing to replace feeding evaluation {}",
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
        .ok_or_else(|| "feeding-evaluation output needs a UTF-8 name".to_string())?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let staging = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(artifact)
        .map_err(|error| format!("failed to encode feeding evaluation: {error}"))?;
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

pub fn load_feeding_evaluation(path: &Path) -> Result<FeedingEvaluationArtifact, String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > MAX_ARTIFACT_BYTES {
        return Err(format!(
            "feeding-evaluation artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
        ));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let artifact: FeedingEvaluationArtifact = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    artifact.validate()?;
    Ok(artifact)
}

/// Verify that a fresh PPO run is initialized from the exact behavior clone
/// qualified by a passing report under the same model, environment, rules,
/// and promotion thresholds. PPO-only controls and rollout-stage scheduling
/// may differ without invalidating the prerequisite evidence.
pub fn verify_feeding_initial_policy(
    path: &Path,
    expected_artifact_hash: &str,
    behavior_clone_metadata_sha256: &str,
    config: &TrainingConfig,
) -> Result<FeedingEvaluationArtifact, String> {
    let artifact = load_feeding_evaluation(path)?;
    if artifact.artifact_hash != expected_artifact_hash {
        return Err(format!(
            "feeding qualification artifact hash mismatch: expected {expected_artifact_hash}, found {}",
            artifact.artifact_hash
        ));
    }
    if artifact.behavior_clone_metadata_sha256 != behavior_clone_metadata_sha256 {
        return Err("feeding qualification names a different behavior clone".into());
    }
    if !artifact.report.passed {
        return Err("feeding qualification did not pass".into());
    }
    if artifact.config.model != config.model || artifact.config.env != config.env {
        return Err("feeding qualification model or environment mismatch".into());
    }
    let staged = config
        .feeding_curriculum
        .environment_for_stage(&config.env, FeedingCurriculumStage::OnFood);
    let expected_ruleset = BlobEnv::new(staged, config.reward.clone(), artifact.report.seeds[0])
        .compiled_ruleset_hash();
    artifact.report.validate_against(
        &expected_ruleset,
        &artifact.report.seeds,
        &config.feeding_curriculum.promotion,
    )?;
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feeding_curriculum::{report_from_metrics, FeedingStageMetrics};

    fn test_artifact() -> FeedingEvaluationArtifact {
        let config = TrainingConfig::default();
        let seed = 71;
        let ruleset = BlobEnv::new(
            config
                .feeding_curriculum
                .environment_for_stage(&config.env, FeedingCurriculumStage::OnFood),
            config.reward.clone(),
            seed,
        )
        .compiled_ruleset_hash();
        let stage = |stage, movement| FeedingStageMetrics {
            stage,
            episodes: 1,
            successful_episodes: 1,
            initial_cells: 1,
            surviving_cells: 1,
            movement_successes: movement,
            consume_successes: 1,
            consumed_energy: 2,
            safety_aborts: 0,
            episode_success_rate: 1.0,
            survival_rate: 1.0,
            consumed_energy_per_initial_cell: 2.0,
        };
        let report = report_from_metrics(
            ruleset,
            vec![seed],
            [
                stage(FeedingCurriculumStage::OnFood, 0),
                stage(FeedingCurriculumStage::AdjacentFood, 1),
            ],
            &config.feeding_curriculum.promotion,
        );
        FeedingEvaluationArtifact::new(
            "a".repeat(64),
            "b".repeat(64),
            "c".repeat(64),
            config,
            report,
        )
        .unwrap()
    }

    #[test]
    fn artifact_recomputes_identity_and_rejects_tampering() {
        let mut artifact = test_artifact();
        artifact.validate().unwrap();
        artifact.report.stages[0].consumed_energy += 1;
        assert!(artifact.validate().is_err());
    }

    #[test]
    fn initial_policy_qualification_binds_clone_environment_and_passing_gate() {
        let artifact = test_artifact();
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("feeding.json");
        publish_feeding_evaluation(&path, &artifact).unwrap();
        verify_feeding_initial_policy(
            &path,
            &artifact.artifact_hash,
            &artifact.behavior_clone_metadata_sha256,
            &artifact.config,
        )
        .unwrap();

        let mut changed = artifact.config.clone();
        changed.env.initial_energy += 1;
        assert!(verify_feeding_initial_policy(
            &path,
            &artifact.artifact_hash,
            &artifact.behavior_clone_metadata_sha256,
            &changed,
        )
        .unwrap_err()
        .contains("environment mismatch"));
        assert!(verify_feeding_initial_policy(
            &path,
            &"0".repeat(64),
            &artifact.behavior_clone_metadata_sha256,
            &artifact.config,
        )
        .unwrap_err()
        .contains("artifact hash mismatch"));
    }

    #[test]
    fn publication_round_trips_and_refuses_replacement() {
        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("feeding.json");
        let artifact = test_artifact();
        publish_feeding_evaluation(&output, &artifact).unwrap();
        assert_eq!(load_feeding_evaluation(&output).unwrap(), artifact);
        assert!(publish_feeding_evaluation(&output, &artifact).is_err());
    }
}
