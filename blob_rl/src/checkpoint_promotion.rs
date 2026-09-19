//! Policy-driven checkpoint promotion decisions over immutable evidence.
//!
//! Promotion is deliberately separate from checkpoint validity. A rejected
//! decision never makes the checkpoint unloadable, and a recorded override
//! can accept behavioral risk without weakening artifact integrity checks.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::artifact::verify_checkpoint_metadata;
use crate::config::ScenarioProfile;
use crate::scale_qualification::{
    load_scale_qualification, ScaleProfileQualification, ScaleQualificationReport,
};
use crate::sweep::sha256;
use crate::viability::hash_json;

pub const CHECKPOINT_PROMOTION_SCHEMA_VERSION: u32 = 1;
const MAX_POLICY_BYTES: u64 = 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 1024 * 1024;
const MAX_QUALIFICATION_BYTES: u64 = 256 * 1024 * 1024;
const MAX_DECISION_BYTES: u64 = 4 * 1024 * 1024;
static PROMOTION_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QualificationEnforcement {
    Required,
    Advisory,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointPromotionPolicy {
    pub schema_version: u32,
    pub qualification: QualificationEnforcement,
    pub require_held_out_evaluation: bool,
    pub require_feeding_competency: bool,
    pub require_rollout_pool_acceptance: bool,
}

impl CheckpointPromotionPolicy {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CHECKPOINT_PROMOTION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported checkpoint promotion policy schema {}; expected {}",
                self.schema_version, CHECKPOINT_PROMOTION_SCHEMA_VERSION
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointPromotionVerdict {
    Approved,
    ApprovedWithWarnings,
    ApprovedWithOverride,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PromotionEvidenceCheck {
    pub name: String,
    pub passed: bool,
    pub blocking: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CheckpointPromotionDecision {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub decision_hash: String,
    pub checkpoint_directory: String,
    pub checkpoint_metadata_sha256: String,
    pub checkpoint_update: usize,
    pub checkpoint_actions: u64,
    pub checkpoint_model_sha256: String,
    pub checkpoint_world_size: usize,
    pub source_profile: Option<String>,
    pub qualification_file: String,
    pub qualification_file_sha256: String,
    pub qualification_hash: String,
    pub target_profile: String,
    pub target_world_size: usize,
    pub target_scenario_hash: String,
    pub policy_file: String,
    pub policy_file_sha256: String,
    pub policy: CheckpointPromotionPolicy,
    pub checks: Vec<PromotionEvidenceCheck>,
    pub blocking_failures: usize,
    pub warnings: usize,
    pub override_reason: Option<String>,
    pub verdict: CheckpointPromotionVerdict,
}

#[derive(Serialize)]
struct DecisionIdentity<'a> {
    checkpoint_directory: &'a str,
    checkpoint_metadata_sha256: &'a str,
    checkpoint_update: usize,
    checkpoint_actions: u64,
    checkpoint_model_sha256: &'a str,
    checkpoint_world_size: usize,
    source_profile: &'a Option<String>,
    qualification_file: &'a str,
    qualification_file_sha256: &'a str,
    qualification_hash: &'a str,
    target_profile: &'a str,
    target_world_size: usize,
    target_scenario_hash: &'a str,
    policy_file: &'a str,
    policy_file_sha256: &'a str,
    policy: &'a CheckpointPromotionPolicy,
    checks: &'a [PromotionEvidenceCheck],
    blocking_failures: usize,
    warnings: usize,
    override_reason: &'a Option<String>,
    verdict: CheckpointPromotionVerdict,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>, String> {
    crate::artifact_io::read_bounded(path, maximum).map_err(|error| format!("{label}: {error}"))
}

fn normalized_override(reason: Option<&str>) -> Result<Option<String>, String> {
    let reason = reason.map(str::trim).filter(|reason| !reason.is_empty());
    if reason.is_some_and(|reason| reason.len() > 1024) {
        return Err("checkpoint promotion override reason is too long".into());
    }
    Ok(reason.map(str::to_string))
}

fn verdict(
    checks: &[PromotionEvidenceCheck],
    override_reason: &Option<String>,
) -> Result<(usize, usize, CheckpointPromotionVerdict), String> {
    let blocking_failures = checks
        .iter()
        .filter(|check| !check.passed && check.blocking)
        .count();
    let warnings = checks
        .iter()
        .filter(|check| !check.passed && !check.blocking)
        .count();
    let verdict = match (blocking_failures, warnings, override_reason.is_some()) {
        (0, 0, false) => CheckpointPromotionVerdict::Approved,
        (0, _, false) => CheckpointPromotionVerdict::ApprovedWithWarnings,
        (0, _, true) => {
            return Err("checkpoint promotion override has no blocking failure to override".into())
        }
        (_, _, true) => CheckpointPromotionVerdict::ApprovedWithOverride,
        (_, _, false) => CheckpointPromotionVerdict::Rejected,
    };
    Ok((blocking_failures, warnings, verdict))
}

fn decision_hash(report: &CheckpointPromotionDecision) -> Result<String, String> {
    hash_json(&DecisionIdentity {
        checkpoint_directory: &report.checkpoint_directory,
        checkpoint_metadata_sha256: &report.checkpoint_metadata_sha256,
        checkpoint_update: report.checkpoint_update,
        checkpoint_actions: report.checkpoint_actions,
        checkpoint_model_sha256: &report.checkpoint_model_sha256,
        checkpoint_world_size: report.checkpoint_world_size,
        source_profile: &report.source_profile,
        qualification_file: &report.qualification_file,
        qualification_file_sha256: &report.qualification_file_sha256,
        qualification_hash: &report.qualification_hash,
        target_profile: &report.target_profile,
        target_world_size: report.target_world_size,
        target_scenario_hash: &report.target_scenario_hash,
        policy_file: &report.policy_file,
        policy_file_sha256: &report.policy_file_sha256,
        policy: &report.policy,
        checks: &report.checks,
        blocking_failures: report.blocking_failures,
        warnings: report.warnings,
        override_reason: &report.override_reason,
        verdict: report.verdict,
    })
}

pub fn validate_checkpoint_promotion_decision(
    report: &CheckpointPromotionDecision,
) -> Result<(), String> {
    if report.schema_version != CHECKPOINT_PROMOTION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported checkpoint promotion schema {}; expected {}",
            report.schema_version, CHECKPOINT_PROMOTION_SCHEMA_VERSION
        ));
    }
    report.policy.validate()?;
    for value in [
        &report.checkpoint_metadata_sha256,
        &report.checkpoint_model_sha256,
        &report.qualification_file_sha256,
        &report.qualification_hash,
        &report.target_scenario_hash,
        &report.policy_file_sha256,
        &report.decision_hash,
    ] {
        if !valid_sha256(value) {
            return Err("checkpoint promotion contains an invalid SHA-256 identity".into());
        }
    }
    if report.target_profile.is_empty()
        || report.target_world_size <= report.checkpoint_world_size
        || report.checks.is_empty()
    {
        return Err("checkpoint promotion scale transition is invalid".into());
    }
    if report
        .override_reason
        .as_deref()
        .is_some_and(|reason| reason.trim() != reason || reason.is_empty() || reason.len() > 1024)
    {
        return Err("checkpoint promotion override reason is invalid".into());
    }
    let (blocking_failures, warnings, expected_verdict) =
        verdict(&report.checks, &report.override_reason)?;
    if report.blocking_failures != blocking_failures
        || report.warnings != warnings
        || report.verdict != expected_verdict
    {
        return Err("checkpoint promotion verdict is inconsistent".into());
    }
    if decision_hash(report)? != report.decision_hash {
        return Err("checkpoint promotion decision hash mismatch".into());
    }
    Ok(())
}

fn target_profile<'a>(
    qualification: &'a ScaleQualificationReport,
    name: &str,
) -> Result<(usize, &'a ScaleProfileQualification), String> {
    qualification
        .profiles
        .iter()
        .enumerate()
        .find(|(_, profile)| profile.name == name)
        .ok_or_else(|| format!("scale qualification has no target profile {name}"))
}

fn qualification_check(
    qualification: &ScaleQualificationReport,
    target_index: usize,
    blocking: bool,
) -> PromotionEvidenceCheck {
    let included = qualification.profiles[..=target_index]
        .iter()
        .map(|profile| profile.name.as_str())
        .collect::<HashSet<_>>();
    let relevant = qualification
        .checks
        .iter()
        .filter(|check| {
            check
                .profile
                .as_deref()
                .is_none_or(|profile| included.contains(profile))
        })
        .collect::<Vec<_>>();
    let failures = relevant.iter().filter(|check| !check.passed).count();
    PromotionEvidenceCheck {
        name: "target_scale_ecology".into(),
        passed: failures == 0,
        blocking,
        detail: format!(
            "{} relevant qualification checks evaluated; {failures} failed",
            relevant.len()
        ),
    }
}

pub fn evaluate_checkpoint_promotion(
    checkpoint_directory: &Path,
    qualification_file: &Path,
    policy_file: &Path,
    target_name: &str,
    override_reason: Option<&str>,
) -> Result<CheckpointPromotionDecision, String> {
    let checkpoint_directory = fs::canonicalize(checkpoint_directory).map_err(|error| {
        format!(
            "failed to resolve checkpoint {}: {error}",
            checkpoint_directory.display()
        )
    })?;
    let metadata = verify_checkpoint_metadata(&checkpoint_directory)?;
    let metadata_bytes = read_bounded(
        &checkpoint_directory.join("metadata.json"),
        MAX_METADATA_BYTES,
        "checkpoint metadata",
    )?;
    let qualification_file = fs::canonicalize(qualification_file).map_err(|error| {
        format!(
            "failed to resolve qualification {}: {error}",
            qualification_file.display()
        )
    })?;
    let qualification_bytes = read_bounded(
        &qualification_file,
        MAX_QUALIFICATION_BYTES,
        "scale qualification",
    )?;
    let qualification = load_scale_qualification(&qualification_file)?;
    let policy_file = fs::canonicalize(policy_file).map_err(|error| {
        format!(
            "failed to resolve policy {}: {error}",
            policy_file.display()
        )
    })?;
    let policy_bytes = read_bounded(&policy_file, MAX_POLICY_BYTES, "promotion policy")?;
    let policy: CheckpointPromotionPolicy = toml::from_str(
        std::str::from_utf8(&policy_bytes)
            .map_err(|error| format!("promotion policy is not UTF-8: {error}"))?,
    )
    .map_err(|error| format!("failed to parse promotion policy: {error}"))?;
    policy.validate()?;
    let (target_index, target) = target_profile(&qualification, target_name)?;
    let checkpoint_scenario = ScenarioProfile::from(&metadata.config.env);
    let checkpoint_scenario_hash = checkpoint_scenario.semantic_hash()?;
    let semantic_ruleset_hash = metadata.config.env.rules.semantic_hash().to_string();
    if semantic_ruleset_hash != target.characterization.semantic_ruleset_hash {
        return Err("checkpoint and target qualification use different semantic physics".into());
    }
    if metadata.config.env.world_size >= target.metrics.world_size {
        return Err("promotion target must be larger than the checkpoint world".into());
    }
    let source_profile = qualification
        .profiles
        .iter()
        .find(|profile| profile.characterization.scenario_hash == checkpoint_scenario_hash)
        .map(|profile| profile.name.clone());
    if source_profile.is_none()
        && metadata.config.env.world_size >= qualification.profiles[0].metrics.world_size
    {
        return Err(
            "checkpoint does not match a declared source profile; use a matching qualification"
                .into(),
        );
    }
    if let Some(source) = source_profile.as_deref() {
        let source_index = qualification
            .profiles
            .iter()
            .position(|profile| profile.name == source)
            .expect("source profile was selected above");
        if source_index >= target_index {
            return Err("promotion target must follow the checkpoint source profile".into());
        }
    }
    let qualification_blocking = policy.qualification == QualificationEnforcement::Required;
    let mut checks = vec![qualification_check(
        &qualification,
        target_index,
        qualification_blocking,
    )];
    if policy.require_held_out_evaluation {
        checks.push(PromotionEvidenceCheck {
            name: "held_out_evaluation".into(),
            passed: metadata.evaluation.is_some(),
            blocking: true,
            detail: "checkpoint carries an integrity-checked held-out evaluation".into(),
        });
    }
    if policy.require_feeding_competency {
        let passed = metadata
            .feeding_evaluation
            .as_ref()
            .is_some_and(|report| report.passed)
            && metadata
                .retention_feeding_evaluation
                .as_ref()
                .is_none_or(|report| report.passed);
        let suites = usize::from(metadata.feeding_evaluation.is_some())
            + usize::from(metadata.retention_feeding_evaluation.is_some());
        checks.push(PromotionEvidenceCheck {
            name: "feeding_competency".into(),
            passed,
            blocking: true,
            detail: if !metadata.config.feeding_curriculum.enabled {
                "promotion policy requires feeding evidence, but the checkpoint disabled that evaluation"
                    .into()
            } else {
                format!("enabled feeding curriculum passed all {suites} competency suites")
            },
        });
    }
    if policy.require_rollout_pool_acceptance {
        checks.push(PromotionEvidenceCheck {
            name: "rollout_pool_acceptance".into(),
            passed: metadata.rollout_pool_promotion.is_some(),
            blocking: true,
            detail: "checkpoint passed its source-scale self-play pool gates".into(),
        });
    }
    let override_reason = normalized_override(override_reason)?;
    let (blocking_failures, warnings, report_verdict) = verdict(&checks, &override_reason)?;
    let mut report = CheckpointPromotionDecision {
        schema_version: CHECKPOINT_PROMOTION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        decision_hash: String::new(),
        checkpoint_directory: checkpoint_directory.to_string_lossy().into_owned(),
        checkpoint_metadata_sha256: sha256(&metadata_bytes),
        checkpoint_update: metadata.update,
        checkpoint_actions: metadata.actions,
        checkpoint_model_sha256: metadata.model_sha256,
        checkpoint_world_size: metadata.config.env.world_size,
        source_profile,
        qualification_file: qualification_file.to_string_lossy().into_owned(),
        qualification_file_sha256: sha256(&qualification_bytes),
        qualification_hash: qualification.qualification_hash.clone(),
        target_profile: target.name.clone(),
        target_world_size: target.metrics.world_size,
        target_scenario_hash: target.characterization.scenario_hash.clone(),
        policy_file: policy_file.to_string_lossy().into_owned(),
        policy_file_sha256: sha256(&policy_bytes),
        policy,
        checks,
        blocking_failures,
        warnings,
        override_reason,
        verdict: report_verdict,
    };
    report.decision_hash = decision_hash(&report)?;
    validate_checkpoint_promotion_decision(&report)?;
    Ok(report)
}

pub fn verify_checkpoint_promotion_evidence(
    report: &CheckpointPromotionDecision,
) -> Result<(), String> {
    verify_checkpoint_promotion_evidence_at(
        report,
        Path::new(&report.checkpoint_directory),
        Path::new(&report.qualification_file),
        Path::new(&report.policy_file),
    )
}

pub fn verify_checkpoint_promotion_evidence_at(
    report: &CheckpointPromotionDecision,
    checkpoint_directory: &Path,
    qualification_file: &Path,
    policy_file: &Path,
) -> Result<(), String> {
    validate_checkpoint_promotion_decision(report)?;
    let mut reproduced = evaluate_checkpoint_promotion(
        checkpoint_directory,
        qualification_file,
        policy_file,
        &report.target_profile,
        report.override_reason.as_deref(),
    )?;
    // Locations are provenance, not content identity. Auditors may supply
    // relocated copies; the bound file hashes still have to match exactly.
    reproduced
        .checkpoint_directory
        .clone_from(&report.checkpoint_directory);
    reproduced
        .qualification_file
        .clone_from(&report.qualification_file);
    reproduced.policy_file.clone_from(&report.policy_file);
    reproduced.decision_hash = decision_hash(&reproduced)?;
    if reproduced.decision_hash != report.decision_hash {
        return Err("checkpoint promotion no longer matches its source evidence".into());
    }
    Ok(())
}

pub fn load_checkpoint_promotion(path: &Path) -> Result<CheckpointPromotionDecision, String> {
    let bytes = read_bounded(path, MAX_DECISION_BYTES, "checkpoint promotion")?;
    let report: CheckpointPromotionDecision = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    validate_checkpoint_promotion_decision(&report)?;
    Ok(report)
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to sync directory {}: {error}", path.display()))
}

pub fn publish_checkpoint_promotion(
    output: &Path,
    report: &CheckpointPromotionDecision,
) -> Result<PathBuf, String> {
    verify_checkpoint_promotion_evidence(report)?;
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode checkpoint promotion: {error}"))?;
    if bytes.len() as u64 > MAX_DECISION_BYTES {
        return Err("checkpoint promotion decision is too large".into());
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable checkpoint promotion {}",
            output.display()
        ));
    }
    let nonce = PROMOTION_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "checkpoint promotion output needs a UTF-8 name".to_string())?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
    let result = file
        .write_all(&bytes)
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.sync_all());
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(format!("failed to write {}: {error}", temporary.display()));
    }
    fs::rename(&temporary, output).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        format!("failed to publish {}: {error}", output.display())
    })?;
    sync_directory(parent)?;
    Ok(output.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(blocking_failure: bool, warning: bool) -> Vec<PromotionEvidenceCheck> {
        vec![
            PromotionEvidenceCheck {
                name: "required".into(),
                passed: !blocking_failure,
                blocking: true,
                detail: "required evidence".into(),
            },
            PromotionEvidenceCheck {
                name: "advisory".into(),
                passed: !warning,
                blocking: false,
                detail: "advisory evidence".into(),
            },
        ]
    }

    fn synthetic_report(checks: Vec<PromotionEvidenceCheck>) -> CheckpointPromotionDecision {
        let (blocking_failures, warnings, report_verdict) = verdict(&checks, &None).unwrap();
        let hash = "a".repeat(64);
        let mut report = CheckpointPromotionDecision {
            schema_version: CHECKPOINT_PROMOTION_SCHEMA_VERSION,
            package_version: "test".into(),
            code_revision: None,
            decision_hash: String::new(),
            checkpoint_directory: "/tmp/checkpoint".into(),
            checkpoint_metadata_sha256: hash.clone(),
            checkpoint_update: 7,
            checkpoint_actions: 11,
            checkpoint_model_sha256: hash.clone(),
            checkpoint_world_size: 32,
            source_profile: None,
            qualification_file: "/tmp/qualification.json".into(),
            qualification_file_sha256: hash.clone(),
            qualification_hash: hash.clone(),
            target_profile: "selection-256".into(),
            target_world_size: 256,
            target_scenario_hash: hash.clone(),
            policy_file: "/tmp/policy.toml".into(),
            policy_file_sha256: hash,
            policy: CheckpointPromotionPolicy {
                schema_version: CHECKPOINT_PROMOTION_SCHEMA_VERSION,
                qualification: QualificationEnforcement::Required,
                require_held_out_evaluation: true,
                require_feeding_competency: true,
                require_rollout_pool_acceptance: false,
            },
            checks,
            blocking_failures,
            warnings,
            override_reason: None,
            verdict: report_verdict,
        };
        report.decision_hash = decision_hash(&report).unwrap();
        report
    }

    #[test]
    fn policy_failures_reject_without_invalidating_the_candidate() {
        assert_eq!(
            verdict(&evidence(true, false), &None).unwrap(),
            (1, 0, CheckpointPromotionVerdict::Rejected)
        );
        assert_eq!(
            verdict(&evidence(false, true), &None).unwrap(),
            (0, 1, CheckpointPromotionVerdict::ApprovedWithWarnings)
        );
    }

    #[test]
    fn explicit_reason_overrides_only_blocking_failures() {
        let reason = Some("intentional sparse-ecology experiment".to_string());
        assert_eq!(
            verdict(&evidence(true, true), &reason).unwrap(),
            (1, 1, CheckpointPromotionVerdict::ApprovedWithOverride)
        );
        assert!(verdict(&evidence(false, true), &reason)
            .unwrap_err()
            .contains("no blocking failure"));
    }

    #[test]
    fn override_reasons_are_trimmed_bounded_and_nonempty() {
        assert_eq!(
            normalized_override(Some("  deliberate treatment  ")).unwrap(),
            Some("deliberate treatment".into())
        );
        assert_eq!(normalized_override(Some("  ")).unwrap(), None);
        assert!(normalized_override(Some(&"x".repeat(1025))).is_err());
    }

    #[test]
    fn decision_validation_recomputes_verdict_and_hash() {
        let report = synthetic_report(evidence(false, true));
        validate_checkpoint_promotion_decision(&report).unwrap();

        let mut tampered = report.clone();
        tampered.checks[1].passed = true;
        assert!(validate_checkpoint_promotion_decision(&tampered).is_err());

        let mut inconsistent = report;
        inconsistent.warnings = 0;
        inconsistent.decision_hash = decision_hash(&inconsistent).unwrap();
        assert!(validate_checkpoint_promotion_decision(&inconsistent)
            .unwrap_err()
            .contains("verdict is inconsistent"));
    }
}
