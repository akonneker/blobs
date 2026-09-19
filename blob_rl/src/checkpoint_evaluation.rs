//! Immutable retention/contact evidence bound to one training checkpoint.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::artifact::verify_checkpoint_metadata;
use crate::config::{FeedingCurriculumStage, TrainingConfig};
use crate::contact_evaluation::ContactEvaluationReport;
use crate::env::BlobEnv;
use crate::feeding_curriculum::FeedingPromotionReport;
use crate::micro_combat::MicroCombatEvaluationReport;
use crate::sweep::sha256;

pub const CHECKPOINT_EVALUATION_SCHEMA_VERSION: u32 = 7;
const MAX_ARTIFACT_BYTES: u64 = 4 * 1024 * 1024;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointEvaluationArtifact {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub artifact_hash: String,
    pub checkpoint_metadata_sha256: String,
    pub checkpoint_model_sha256: String,
    pub checkpoint_update: usize,
    pub checkpoint_actions: u64,
    /// Held-out combat horizons used for this evaluation. These may differ
    /// from the horizons bound into the immutable training configuration.
    pub combat_evaluation_horizons: CombatEvaluationHorizons,
    pub config: TrainingConfig,
    pub feeding: FeedingPromotionReport,
    pub contact: ContactEvaluationReport,
    pub micro_combat: Option<MicroCombatEvaluationReport>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct CombatEvaluationHorizons {
    pub contact_sim_time_limit_quanta: u64,
    pub skirmish_sim_time_limit_quanta: u64,
}

impl CombatEvaluationHorizons {
    pub fn from_config(config: &TrainingConfig) -> Self {
        Self {
            contact_sim_time_limit_quanta: config
                .combat_curriculum
                .contact_evaluation_sim_time_limit_quanta,
            skirmish_sim_time_limit_quanta: config
                .combat_curriculum
                .skirmish_evaluation_sim_time_limit_quanta,
        }
    }

    pub fn apply_to(self, config: &mut TrainingConfig) -> Result<(), String> {
        if self.contact_sim_time_limit_quanta == 0 || self.skirmish_sim_time_limit_quanta == 0 {
            return Err("combat evaluation horizons must be positive".into());
        }
        config
            .combat_curriculum
            .contact_evaluation_sim_time_limit_quanta = self.contact_sim_time_limit_quanta;
        config
            .combat_curriculum
            .skirmish_evaluation_sim_time_limit_quanta = self.skirmish_sim_time_limit_quanta;
        config.validate()
    }
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
    combat_evaluation_horizons: &'a CombatEvaluationHorizons,
    config: &'a TrainingConfig,
    feeding: &'a FeedingPromotionReport,
    contact: &'a ContactEvaluationReport,
    micro_combat: &'a Option<MicroCombatEvaluationReport>,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

impl CheckpointEvaluationArtifact {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        checkpoint_metadata_sha256: String,
        checkpoint_model_sha256: String,
        checkpoint_update: usize,
        checkpoint_actions: u64,
        combat_evaluation_horizons: CombatEvaluationHorizons,
        config: TrainingConfig,
        feeding: FeedingPromotionReport,
        contact: ContactEvaluationReport,
        micro_combat: Option<MicroCombatEvaluationReport>,
    ) -> Result<Self, String> {
        let mut artifact = Self {
            schema_version: CHECKPOINT_EVALUATION_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").into(),
            code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
            artifact_hash: String::new(),
            checkpoint_metadata_sha256,
            checkpoint_model_sha256,
            checkpoint_update,
            checkpoint_actions,
            combat_evaluation_horizons,
            config,
            feeding,
            contact,
            micro_combat,
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
            checkpoint_metadata_sha256: &self.checkpoint_metadata_sha256,
            checkpoint_model_sha256: &self.checkpoint_model_sha256,
            checkpoint_update: self.checkpoint_update,
            checkpoint_actions: self.checkpoint_actions,
            combat_evaluation_horizons: &self.combat_evaluation_horizons,
            config: &self.config,
            feeding: &self.feeding,
            contact: &self.contact,
            micro_combat: &self.micro_combat,
        };
        serde_json::to_vec(&identity)
            .map(|bytes| sha256(&bytes))
            .map_err(|error| format!("failed to encode checkpoint-evaluation identity: {error}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CHECKPOINT_EVALUATION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported checkpoint-evaluation schema {}; expected {}",
                self.schema_version, CHECKPOINT_EVALUATION_SCHEMA_VERSION
            ));
        }
        self.config.validate()?;
        let mut evaluation_config = self.config.clone();
        self.combat_evaluation_horizons
            .apply_to(&mut evaluation_config)?;
        if !valid_sha256(&self.checkpoint_metadata_sha256)
            || !valid_sha256(&self.checkpoint_model_sha256)
            || self.feeding.seeds.is_empty()
            || self.feeding.seeds != self.contact.seeds
        {
            return Err("checkpoint-evaluation hashes or seed suites are invalid".into());
        }
        let feeding_env = self
            .config
            .feeding_curriculum
            .environment_for_stage(&self.config.env, FeedingCurriculumStage::OnFood);
        let feeding_ruleset = BlobEnv::new(
            feeding_env,
            self.config.reward.clone(),
            self.feeding.seeds[0],
        )
        .compiled_ruleset_hash();
        self.feeding.validate_against(
            &feeding_ruleset,
            &self.feeding.seeds,
            &self.config.feeding_curriculum.promotion,
        )?;
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
        let contact_ruleset = BlobEnv::new(
            contact_env,
            self.config.reward.clone(),
            self.contact.seeds[0],
        )
        .compiled_ruleset_hash();
        self.contact.validate_against(
            &contact_ruleset,
            &self.contact.seeds,
            &evaluation_config.combat_curriculum,
        )?;
        match (
            self.config.combat_curriculum.micro_combat.enabled,
            self.micro_combat.as_ref(),
        ) {
            (true, Some(micro)) => {
                if micro.seeds != self.feeding.seeds {
                    return Err("micro-combat evaluation uses a different seed suite".into());
                }
                micro.validate_against(
                    &contact_ruleset,
                    self.config
                        .combat_curriculum
                        .micro_combat
                        .suite
                        .as_ref()
                        .ok_or("micro-combat evaluation has no configured suite")?,
                    &self.feeding.seeds,
                )?;
            }
            (true, None) => {
                return Err("checkpoint evaluation is missing micro-combat evidence".into())
            }
            (false, Some(_)) => {
                return Err("checkpoint evaluation has unexpected micro-combat evidence".into())
            }
            (false, None) => {}
        }
        if self.artifact_hash != self.recompute_hash()? {
            return Err("checkpoint-evaluation artifact hash mismatch".into());
        }
        Ok(())
    }

    pub fn passed(&self) -> bool {
        let micro_passed = if self.config.combat_curriculum.micro_combat.enabled {
            self.micro_combat.as_ref().is_some_and(|report| {
                report.meets_promotion_thresholds(&self.config.combat_curriculum.micro_combat)
            })
        } else {
            self.micro_combat.is_none()
        };
        self.feeding.passed
            && self
                .contact
                .meets_promotion_thresholds(&self.config.combat_curriculum)
            && micro_passed
    }
}

pub fn publish_checkpoint_evaluation(
    output: &Path,
    artifact: &CheckpointEvaluationArtifact,
) -> Result<PathBuf, String> {
    artifact.validate()?;
    if output.exists() {
        return Err(format!(
            "refusing to replace checkpoint evaluation {}",
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
        .ok_or_else(|| "checkpoint-evaluation output needs a UTF-8 name".to_string())?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let staging = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(artifact)
        .map_err(|error| format!("failed to encode checkpoint evaluation: {error}"))?;
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

pub fn load_checkpoint_evaluation(path: &Path) -> Result<CheckpointEvaluationArtifact, String> {
    let bytes = crate::artifact_io::read_bounded(path, MAX_ARTIFACT_BYTES)?;
    let artifact: CheckpointEvaluationArtifact = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    artifact.validate()?;
    Ok(artifact)
}

/// Verify both sides of the evidence binding. This re-hashes and integrity
/// checks the complete checkpoint before accepting a separately published
/// evaluation artifact, preventing path substitution or edited evidence.
pub fn verify_checkpoint_evaluation(
    path: &Path,
    expected_artifact_hash: &str,
    checkpoint: &Path,
) -> Result<CheckpointEvaluationArtifact, String> {
    let artifact = load_checkpoint_evaluation(path)?;
    if artifact.artifact_hash != expected_artifact_hash {
        return Err(format!(
            "checkpoint-evaluation artifact hash mismatch: expected {expected_artifact_hash}, found {}",
            artifact.artifact_hash
        ));
    }
    let metadata = verify_checkpoint_metadata(checkpoint)?;
    let metadata_bytes =
        crate::artifact_io::read_bounded(&checkpoint.join("metadata.json"), 1024 * 1024)?;
    if artifact.checkpoint_metadata_sha256 != sha256(&metadata_bytes)
        || artifact.checkpoint_model_sha256 != metadata.model_sha256
        || artifact.checkpoint_update != metadata.update
        || artifact.checkpoint_actions != metadata.actions
        || artifact.config != metadata.config
    {
        return Err("checkpoint evaluation names a different checkpoint".into());
    }
    Ok(artifact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combat_evaluation_horizons_change_only_the_gate_windows() {
        let source = TrainingConfig::default();
        let mut evaluation = source.clone();
        let horizons = CombatEvaluationHorizons {
            contact_sim_time_limit_quanta: 12_345,
            skirmish_sim_time_limit_quanta: 67_890,
        };
        horizons.apply_to(&mut evaluation).unwrap();

        assert_eq!(
            evaluation
                .combat_curriculum
                .contact_evaluation_sim_time_limit_quanta,
            12_345
        );
        assert_eq!(
            evaluation
                .combat_curriculum
                .skirmish_evaluation_sim_time_limit_quanta,
            67_890
        );
        evaluation
            .combat_curriculum
            .contact_evaluation_sim_time_limit_quanta = source
            .combat_curriculum
            .contact_evaluation_sim_time_limit_quanta;
        evaluation
            .combat_curriculum
            .skirmish_evaluation_sim_time_limit_quanta = source
            .combat_curriculum
            .skirmish_evaluation_sim_time_limit_quanta;
        assert_eq!(evaluation, source);
    }

    #[test]
    fn combat_evaluation_horizons_reject_zero() {
        let mut config = TrainingConfig::default();
        let error = CombatEvaluationHorizons {
            contact_sim_time_limit_quanta: 0,
            skirmish_sim_time_limit_quanta: 1,
        }
        .apply_to(&mut config)
        .unwrap_err();
        assert!(error.contains("positive"));
    }
}
