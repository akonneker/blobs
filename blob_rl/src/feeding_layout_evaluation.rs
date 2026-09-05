//! Cross-layout feeding qualification for one immutable learned policy.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use blob_engine::engine::StartingCellLayout;
use serde::{Deserialize, Serialize};

use crate::config::{FeedingCurriculumStage, TrainingConfig};
use crate::control_matrix::MaintainedMindProfile;
use crate::env::BlobEnv;
use crate::feeding_curriculum::FeedingPromotionReport;
use crate::sweep::sha256;

pub const FEEDING_LAYOUT_EVALUATION_SCHEMA_VERSION: u32 = 2;
const MAX_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

/// Ecology layouts in which every founder can have a vacant adjacent-food
/// target. Dense block is deliberately absent because it encloses cells.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum FeedingQualificationLayout {
    Line,
    Checkerboard,
    Ring,
    LooseRandom,
    Random,
}

impl FeedingQualificationLayout {
    pub const REQUIRED: [Self; 5] = [
        Self::Line,
        Self::Checkerboard,
        Self::Ring,
        Self::LooseRandom,
        Self::Random,
    ];

    pub const fn starting_layout(self) -> StartingCellLayout {
        match self {
            Self::Line => StartingCellLayout::Line,
            Self::Checkerboard => StartingCellLayout::Checkerboard,
            Self::Ring => StartingCellLayout::Ring,
            Self::LooseRandom => StartingCellLayout::LooseRandom,
            Self::Random => StartingCellLayout::Random,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingLayoutTrial {
    pub layout: FeedingQualificationLayout,
    pub seed: u64,
    pub report: FeedingPromotionReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeedingLayoutPolicyIdentity {
    BehaviorClone {
        metadata_sha256: String,
        model_sha256: String,
    },
    MaintainedTeacher {
        profile: MaintainedMindProfile,
        mind_abi_sha256: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingLayoutEvaluationArtifact {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub artifact_hash: String,
    pub source_config_sha256: String,
    pub policy: FeedingLayoutPolicyIdentity,
    pub config: TrainingConfig,
    pub layouts: Vec<FeedingQualificationLayout>,
    pub seeds: Vec<u64>,
    pub trials: Vec<FeedingLayoutTrial>,
    pub passed: bool,
}

#[derive(Serialize)]
struct ArtifactIdentity<'a> {
    schema_version: u32,
    package_version: &'a str,
    code_revision: &'a Option<String>,
    source_config_sha256: &'a str,
    policy: &'a FeedingLayoutPolicyIdentity,
    config: &'a TrainingConfig,
    layouts: &'a [FeedingQualificationLayout],
    seeds: &'a [u64],
    trials: &'a [FeedingLayoutTrial],
    passed: bool,
}

impl FeedingLayoutEvaluationArtifact {
    pub fn new(
        source_config_sha256: String,
        policy: FeedingLayoutPolicyIdentity,
        config: TrainingConfig,
        layouts: Vec<FeedingQualificationLayout>,
        seeds: Vec<u64>,
        trials: Vec<FeedingLayoutTrial>,
    ) -> Result<Self, String> {
        let passed = !trials.is_empty() && trials.iter().all(|trial| trial.report.passed);
        let mut artifact = Self {
            schema_version: FEEDING_LAYOUT_EVALUATION_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").into(),
            code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_owned),
            artifact_hash: String::new(),
            source_config_sha256,
            policy,
            config,
            layouts,
            seeds,
            trials,
            passed,
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
            policy: &self.policy,
            config: &self.config,
            layouts: &self.layouts,
            seeds: &self.seeds,
            trials: &self.trials,
            passed: self.passed,
        };
        serde_json::to_vec(&identity)
            .map(|bytes| sha256(&bytes))
            .map_err(|error| format!("failed to encode feeding-layout identity: {error}"))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != FEEDING_LAYOUT_EVALUATION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported feeding-layout schema {}; expected {}",
                self.schema_version, FEEDING_LAYOUT_EVALUATION_SCHEMA_VERSION
            ));
        }
        self.config.validate()?;
        if !valid_sha256(&self.source_config_sha256) || !self.policy.has_valid_hashes() {
            return Err("feeding-layout artifact hashes are invalid".into());
        }
        if self.layouts.is_empty() || self.seeds.is_empty() {
            return Err("feeding-layout evaluation requires layouts and seeds".into());
        }
        if !all_unique(&self.layouts) || !all_unique(&self.seeds) {
            return Err("feeding-layout layouts and seeds must be unique".into());
        }
        let expected_trials = self
            .layouts
            .len()
            .checked_mul(self.seeds.len())
            .ok_or("feeding-layout trial count overflow")?;
        if self.trials.len() != expected_trials {
            return Err("feeding-layout trial matrix is incomplete".into());
        }

        for ((layout, seed), trial) in self
            .layouts
            .iter()
            .flat_map(|layout| self.seeds.iter().map(move |seed| (*layout, *seed)))
            .zip(&self.trials)
        {
            if trial.layout != layout || trial.seed != seed || trial.report.seeds != [seed] {
                return Err("feeding-layout trial matrix is misordered or misidentified".into());
            }
            let mut effective = self.config.clone();
            effective.env.starting_cell_layout = layout.starting_layout();
            effective.validate()?;
            let staged = effective
                .feeding_curriculum
                .environment_for_stage(&effective.env, FeedingCurriculumStage::OnFood);
            let ruleset =
                BlobEnv::new(staged, effective.reward.clone(), seed).compiled_ruleset_hash();
            trial.report.validate_against(
                &ruleset,
                &[seed],
                &effective.feeding_curriculum.promotion,
            )?;
        }
        let expected_passed = self.trials.iter().all(|trial| trial.report.passed);
        if self.passed != expected_passed || self.artifact_hash != self.recompute_hash()? {
            return Err("feeding-layout verdict or artifact hash mismatch".into());
        }
        Ok(())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

impl FeedingLayoutPolicyIdentity {
    fn has_valid_hashes(&self) -> bool {
        match self {
            Self::BehaviorClone {
                metadata_sha256,
                model_sha256,
            } => valid_sha256(metadata_sha256) && valid_sha256(model_sha256),
            Self::MaintainedTeacher {
                mind_abi_sha256, ..
            } => valid_sha256(mind_abi_sha256),
        }
    }
}

fn all_unique<T: Copy + Eq + std::hash::Hash>(values: &[T]) -> bool {
    values.iter().copied().collect::<HashSet<_>>().len() == values.len()
}

pub fn publish_feeding_layout_evaluation(
    output: &Path,
    artifact: &FeedingLayoutEvaluationArtifact,
) -> Result<PathBuf, String> {
    artifact.validate()?;
    if output.exists() {
        return Err(format!(
            "refusing to replace feeding-layout evaluation {}",
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
        .ok_or_else(|| "feeding-layout output needs a UTF-8 name".to_string())?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let staging = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(artifact)
        .map_err(|error| format!("failed to encode feeding-layout evaluation: {error}"))?;
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
    result.map(|()| output.to_owned())
}

pub fn load_feeding_layout_evaluation(
    path: &Path,
) -> Result<FeedingLayoutEvaluationArtifact, String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > MAX_ARTIFACT_BYTES {
        return Err(format!(
            "feeding-layout artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
        ));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let artifact: FeedingLayoutEvaluationArtifact = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    artifact.validate()?;
    Ok(artifact)
}

/// Verify that an immutable artifact is exactly the requested evaluation.
/// Schema 2 binds the canonical private-randomness policy path; schema-1
/// layout evidence deliberately fails during `artifact.validate()`.
pub fn verify_feeding_layout_evaluation_request(
    artifact: &FeedingLayoutEvaluationArtifact,
    source_config_sha256: &str,
    policy: &FeedingLayoutPolicyIdentity,
    config: &TrainingConfig,
    layouts: &[FeedingQualificationLayout],
    seeds: &[u64],
) -> Result<(), String> {
    artifact.validate()?;
    if artifact.source_config_sha256 != source_config_sha256
        || &artifact.policy != policy
        || &artifact.config != config
        || artifact.layouts != layouts
        || artifact.seeds != seeds
    {
        return Err("feeding-layout resume request does not match immutable artifact".into());
    }
    Ok(())
}

/// Merge independently published layout/seed shards into one complete matrix.
/// Layouts and seeds retain first-seen order; every Cartesian pair must appear
/// exactly once and all policy/config identities must match.
pub fn merge_feeding_layout_evaluations(
    artifacts: &[FeedingLayoutEvaluationArtifact],
) -> Result<FeedingLayoutEvaluationArtifact, String> {
    let first = artifacts
        .first()
        .ok_or_else(|| "feeding-layout merge requires at least one artifact".to_string())?;
    first.validate()?;
    let mut layouts = Vec::new();
    let mut seeds = Vec::new();
    let mut trials = HashMap::new();

    for artifact in artifacts {
        artifact.validate()?;
        if artifact.source_config_sha256 != first.source_config_sha256
            || artifact.policy != first.policy
            || artifact.config != first.config
        {
            return Err("feeding-layout shards name different configs or models".into());
        }
        for layout in &artifact.layouts {
            if !layouts.contains(layout) {
                layouts.push(*layout);
            }
        }
        for seed in &artifact.seeds {
            if !seeds.contains(seed) {
                seeds.push(*seed);
            }
        }
        for trial in &artifact.trials {
            if trials
                .insert((trial.layout, trial.seed), trial.clone())
                .is_some()
            {
                return Err(format!(
                    "duplicate feeding-layout trial {:?}/{}",
                    trial.layout, trial.seed
                ));
            }
        }
    }

    let mut ordered = Vec::with_capacity(layouts.len().saturating_mul(seeds.len()));
    for layout in &layouts {
        for seed in &seeds {
            ordered.push(
                trials
                    .remove(&(*layout, *seed))
                    .ok_or_else(|| format!("missing feeding-layout trial {layout:?}/{seed}"))?,
            );
        }
    }
    if !trials.is_empty() {
        return Err("feeding-layout merge retained trials outside its matrix".into());
    }
    FeedingLayoutEvaluationArtifact::new(
        first.source_config_sha256.clone(),
        first.policy.clone(),
        first.config.clone(),
        layouts,
        seeds,
        ordered,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feeding_curriculum::{report_from_metrics, FeedingStageMetrics};

    fn report(
        config: &TrainingConfig,
        layout: FeedingQualificationLayout,
        seed: u64,
    ) -> FeedingPromotionReport {
        let mut effective = config.clone();
        effective.env.starting_cell_layout = layout.starting_layout();
        let staged = effective
            .feeding_curriculum
            .environment_for_stage(&effective.env, FeedingCurriculumStage::OnFood);
        let ruleset = BlobEnv::new(staged, effective.reward.clone(), seed).compiled_ruleset_hash();
        let metrics = |stage, movement| FeedingStageMetrics {
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
        report_from_metrics(
            ruleset,
            vec![seed],
            [
                metrics(FeedingCurriculumStage::OnFood, 0),
                metrics(FeedingCurriculumStage::AdjacentFood, 1),
            ],
            &effective.feeding_curriculum.promotion,
        )
    }

    fn artifact() -> FeedingLayoutEvaluationArtifact {
        let mut config = TrainingConfig::default();
        config.env.starting_cell_layout = StartingCellLayout::Checkerboard;
        let layouts = vec![
            FeedingQualificationLayout::Line,
            FeedingQualificationLayout::Checkerboard,
        ];
        let seeds = vec![71, 72];
        let trials = layouts
            .iter()
            .flat_map(|layout| {
                seeds.iter().map(|seed| FeedingLayoutTrial {
                    layout: *layout,
                    seed: *seed,
                    report: report(&config, *layout, *seed),
                })
            })
            .collect();
        FeedingLayoutEvaluationArtifact::new(
            "a".repeat(64),
            FeedingLayoutPolicyIdentity::BehaviorClone {
                metadata_sha256: "b".repeat(64),
                model_sha256: "c".repeat(64),
            },
            config,
            layouts,
            seeds,
            trials,
        )
        .unwrap()
    }

    #[test]
    fn resume_request_requires_exact_config_policy_layout_and_seed_order() {
        let artifact = artifact();
        verify_feeding_layout_evaluation_request(
            &artifact,
            &artifact.source_config_sha256,
            &artifact.policy,
            &artifact.config,
            &artifact.layouts,
            &artifact.seeds,
        )
        .unwrap();

        let mut reversed = artifact.seeds.clone();
        reversed.reverse();
        assert!(verify_feeding_layout_evaluation_request(
            &artifact,
            &artifact.source_config_sha256,
            &artifact.policy,
            &artifact.config,
            &artifact.layouts,
            &reversed,
        )
        .is_err());

        let mut wrong_policy = artifact.policy.clone();
        let FeedingLayoutPolicyIdentity::BehaviorClone { model_sha256, .. } = &mut wrong_policy
        else {
            unreachable!()
        };
        *model_sha256 = "f".repeat(64);
        assert!(verify_feeding_layout_evaluation_request(
            &artifact,
            &artifact.source_config_sha256,
            &wrong_policy,
            &artifact.config,
            &artifact.layouts,
            &artifact.seeds,
        )
        .is_err());
    }

    #[test]
    fn complete_layout_matrix_validates_and_detects_tampering() {
        let artifact = artifact();
        artifact.validate().unwrap();

        let mut missing = artifact.clone();
        missing.trials.pop();
        assert!(missing.validate().unwrap_err().contains("incomplete"));

        let mut duplicate = artifact.clone();
        duplicate.seeds[1] = duplicate.seeds[0];
        assert!(duplicate.validate().unwrap_err().contains("unique"));

        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("layout.json");
        publish_feeding_layout_evaluation(&path, &artifact).unwrap();
        assert_eq!(load_feeding_layout_evaluation(&path).unwrap(), artifact);
        assert!(publish_feeding_layout_evaluation(&path, &artifact).is_err());
    }

    #[test]
    fn merge_requires_a_unique_complete_cartesian_matrix() {
        let complete = artifact();
        let shards = complete
            .trials
            .iter()
            .map(|trial| {
                FeedingLayoutEvaluationArtifact::new(
                    complete.source_config_sha256.clone(),
                    complete.policy.clone(),
                    complete.config.clone(),
                    vec![trial.layout],
                    vec![trial.seed],
                    vec![trial.clone()],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(merge_feeding_layout_evaluations(&shards).unwrap(), complete);
        assert!(
            merge_feeding_layout_evaluations(&[shards[0].clone(), shards[0].clone()])
                .unwrap_err()
                .contains("duplicate")
        );
        assert!(
            merge_feeding_layout_evaluations(&[shards[0].clone(), shards[3].clone()])
                .unwrap_err()
                .contains("missing")
        );
    }
}
