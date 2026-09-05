//! Hash-bound evidence ladder for separating physics, interface, and training failures.
//!
//! This first schema binds explicit evidence claims to immutable artifacts and
//! derives the earliest failed or missing stage. Artifact-specific readers can
//! subsequently replace claims with independently decoded decisions without
//! changing the causal ordering defined here.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::demonstration::{
    load_demonstrations, DemonstrationCollection, DEMONSTRATION_SCHEMA_VERSION,
};
use crate::feeding_recovery::load_feeding_recovery;

pub const SCENARIO_DIAGNOSIS_SCHEMA_VERSION: u32 = 1;
const MAX_SPEC_BYTES: u64 = 1024 * 1024;
const MAX_REPORT_BYTES: u64 = 4 * 1024 * 1024;
const MAX_EVIDENCE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_SCENARIO_NAME_CHARS: usize = 256;
const MAX_ARTIFACT_KIND_CHARS: usize = 128;
const MAX_ARTIFACT_REFERENCE_CHARS: usize = 4096;
const MAX_SUMMARY_CHARS: usize = 2048;
static DIAGNOSIS_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticStage {
    AnalyticalBounds,
    FullStateFeasibility,
    MindFeasibility,
    MaintainedTeacher,
    OnPolicyTeacherRecovery,
    LearnedOnPolicyAgreement,
    HeldOutRollout,
}

impl DiagnosticStage {
    pub const ALL: [Self; 7] = [
        Self::AnalyticalBounds,
        Self::FullStateFeasibility,
        Self::MindFeasibility,
        Self::MaintainedTeacher,
        Self::OnPolicyTeacherRecovery,
        Self::LearnedOnPolicyAgreement,
        Self::HeldOutRollout,
    ];

    fn permits_not_applicable(self) -> bool {
        matches!(self, Self::AnalyticalBounds | Self::FullStateFeasibility)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceOutcome {
    Passed,
    Failed,
    NotRun,
    NotApplicable,
}

impl EvidenceOutcome {
    fn is_assessed(self) -> bool {
        matches!(self, Self::Passed | Self::Failed)
    }

    fn blocks_later_stages(self) -> bool {
        matches!(self, Self::Failed | Self::NotRun)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticEvidenceSpec {
    pub stage: DiagnosticStage,
    pub outcome: EvidenceOutcome,
    pub artifact_kind: String,
    /// Relative paths are resolved from the specification's directory.
    pub artifact: Option<PathBuf>,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioDiagnosisSpec {
    pub schema_version: u32,
    pub scenario_name: String,
    pub ruleset_hash: String,
    pub scenario_hash: String,
    /// Exactly one entry per stage, in [`DiagnosticStage::ALL`] order.
    pub evidence: Vec<DiagnosticEvidenceSpec>,
}

impl ScenarioDiagnosisSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SCENARIO_DIAGNOSIS_SCHEMA_VERSION {
            return Err(format!(
                "unsupported scenario diagnosis schema {}; expected {}",
                self.schema_version, SCENARIO_DIAGNOSIS_SCHEMA_VERSION
            ));
        }
        validate_bounded_text(
            &self.scenario_name,
            MAX_SCENARIO_NAME_CHARS,
            "scenario name",
        )?;
        if !valid_sha256(&self.ruleset_hash) || !valid_sha256(&self.scenario_hash) {
            return Err(
                "scenario diagnosis requires lowercase ruleset and scenario SHA-256 identities"
                    .into(),
            );
        }
        if self.evidence.len() != DiagnosticStage::ALL.len() {
            return Err(format!(
                "scenario diagnosis requires exactly {} evidence stages",
                DiagnosticStage::ALL.len()
            ));
        }

        let mut blocked = false;
        for (expected, evidence) in DiagnosticStage::ALL.iter().zip(&self.evidence) {
            if evidence.stage != *expected {
                return Err(format!(
                    "scenario diagnosis evidence is not in canonical order: expected {expected:?}, found {:?}",
                    evidence.stage
                ));
            }
            validate_bounded_text(
                &evidence.artifact_kind,
                MAX_ARTIFACT_KIND_CHARS,
                "artifact kind",
            )?;
            validate_bounded_text(&evidence.summary, MAX_SUMMARY_CHARS, "evidence summary")?;
            if evidence.outcome == EvidenceOutcome::NotApplicable
                && !evidence.stage.permits_not_applicable()
            {
                return Err(format!(
                    "{:?} is a required stage and cannot be not_applicable",
                    evidence.stage
                ));
            }
            if evidence.outcome.is_assessed() != evidence.artifact.is_some() {
                return Err(format!(
                    "{:?} must cite one artifact exactly when its outcome is passed or failed",
                    evidence.stage
                ));
            }
            if let Some(path) = &evidence.artifact {
                let reference = path
                    .to_str()
                    .ok_or("diagnostic artifact paths must be UTF-8")?;
                validate_bounded_text(
                    reference,
                    MAX_ARTIFACT_REFERENCE_CHARS,
                    "artifact reference",
                )?;
            }
            if blocked && evidence.outcome != EvidenceOutcome::NotRun {
                return Err(format!(
                    "{:?} must be not_run because an earlier stage blocked the ladder",
                    evidence.stage
                ));
            }
            blocked |= evidence.outcome.blocks_later_stages();
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BoundDiagnosticEvidence {
    pub stage: DiagnosticStage,
    pub outcome: EvidenceOutcome,
    pub artifact_kind: String,
    pub artifact_reference: Option<String>,
    pub artifact_file_sha256: Option<String>,
    pub artifact_size_bytes: Option<u64>,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticAttribution {
    PhysicsOrObjective,
    MindInterfaceOrActionSemantics,
    MaintainedPolicyGap,
    OnPolicyDistributionShift,
    ArchitectureOrTraining,
    RolloutCompetence,
    Competent,
    Inconclusive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioDiagnosisReport {
    pub schema_version: u32,
    pub package_version: String,
    pub scenario_name: String,
    pub ruleset_hash: String,
    pub scenario_hash: String,
    pub spec_file_sha256: String,
    pub diagnosis_hash: String,
    pub attribution: DiagnosticAttribution,
    pub first_blocking_stage: Option<DiagnosticStage>,
    pub conclusive: bool,
    pub competent: bool,
    pub evidence: Vec<BoundDiagnosticEvidence>,
}

#[derive(Serialize)]
struct DiagnosisIdentity<'a> {
    schema_version: u32,
    package_version: &'a str,
    scenario_name: &'a str,
    ruleset_hash: &'a str,
    scenario_hash: &'a str,
    spec_file_sha256: &'a str,
    attribution: DiagnosticAttribution,
    first_blocking_stage: Option<DiagnosticStage>,
    evidence: &'a [BoundDiagnosticEvidence],
}

pub fn evaluate_scenario_diagnosis_files(
    spec_path: &Path,
) -> Result<ScenarioDiagnosisReport, String> {
    let spec_bytes = read_bounded(spec_path, MAX_SPEC_BYTES, "scenario diagnosis spec")?;
    let spec_text = std::str::from_utf8(&spec_bytes)
        .map_err(|error| format!("scenario diagnosis spec is not UTF-8: {error}"))?;
    let spec: ScenarioDiagnosisSpec = toml::from_str(spec_text)
        .map_err(|error| format!("failed to parse scenario diagnosis spec: {error}"))?;
    let base = spec_path.parent().unwrap_or_else(|| Path::new("."));
    evaluate_scenario_diagnosis(&spec, sha256(&spec_bytes), base)
}

pub fn evaluate_scenario_diagnosis(
    spec: &ScenarioDiagnosisSpec,
    spec_file_sha256: String,
    artifact_base: &Path,
) -> Result<ScenarioDiagnosisReport, String> {
    spec.validate()?;
    if !valid_sha256(&spec_file_sha256) {
        return Err("scenario diagnosis spec identity is not a lowercase SHA-256".into());
    }

    let evidence = spec
        .evidence
        .iter()
        .map(|claim| {
            bind_evidence(
                claim,
                artifact_base,
                &spec.ruleset_hash,
                &spec.scenario_hash,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (attribution, first_blocking_stage) = classify(&evidence);
    let conclusive = attribution != DiagnosticAttribution::Inconclusive;
    let competent = attribution == DiagnosticAttribution::Competent;
    let package_version = env!("CARGO_PKG_VERSION").to_owned();
    let identity = DiagnosisIdentity {
        schema_version: SCENARIO_DIAGNOSIS_SCHEMA_VERSION,
        package_version: &package_version,
        scenario_name: &spec.scenario_name,
        ruleset_hash: &spec.ruleset_hash,
        scenario_hash: &spec.scenario_hash,
        spec_file_sha256: &spec_file_sha256,
        attribution,
        first_blocking_stage,
        evidence: &evidence,
    };
    let diagnosis_hash = sha256(
        &serde_json::to_vec(&identity)
            .map_err(|error| format!("failed to encode scenario diagnosis identity: {error}"))?,
    );
    let report = ScenarioDiagnosisReport {
        schema_version: SCENARIO_DIAGNOSIS_SCHEMA_VERSION,
        package_version,
        scenario_name: spec.scenario_name.clone(),
        ruleset_hash: spec.ruleset_hash.clone(),
        scenario_hash: spec.scenario_hash.clone(),
        spec_file_sha256,
        diagnosis_hash,
        attribution,
        first_blocking_stage,
        conclusive,
        competent,
        evidence,
    };
    validate_scenario_diagnosis_report(&report)?;
    Ok(report)
}

pub fn validate_scenario_diagnosis_report(report: &ScenarioDiagnosisReport) -> Result<(), String> {
    if report.schema_version != SCENARIO_DIAGNOSIS_SCHEMA_VERSION
        || report.package_version.is_empty()
        || !valid_sha256(&report.ruleset_hash)
        || !valid_sha256(&report.scenario_hash)
        || !valid_sha256(&report.spec_file_sha256)
        || !valid_sha256(&report.diagnosis_hash)
    {
        return Err("scenario diagnosis report identity is invalid".into());
    }
    validate_bounded_text(
        &report.scenario_name,
        MAX_SCENARIO_NAME_CHARS,
        "scenario name",
    )?;
    validate_bound_evidence(&report.evidence)?;
    let (attribution, first_blocking_stage) = classify(&report.evidence);
    if report.attribution != attribution
        || report.first_blocking_stage != first_blocking_stage
        || report.conclusive != (attribution != DiagnosticAttribution::Inconclusive)
        || report.competent != (attribution == DiagnosticAttribution::Competent)
    {
        return Err("scenario diagnosis attribution is inconsistent with its evidence".into());
    }
    let identity = DiagnosisIdentity {
        schema_version: report.schema_version,
        package_version: &report.package_version,
        scenario_name: &report.scenario_name,
        ruleset_hash: &report.ruleset_hash,
        scenario_hash: &report.scenario_hash,
        spec_file_sha256: &report.spec_file_sha256,
        attribution,
        first_blocking_stage,
        evidence: &report.evidence,
    };
    let expected_hash = sha256(
        &serde_json::to_vec(&identity)
            .map_err(|error| format!("failed to encode scenario diagnosis identity: {error}"))?,
    );
    if report.diagnosis_hash != expected_hash {
        return Err("scenario diagnosis hash does not match its contents".into());
    }
    Ok(())
}

pub fn verify_scenario_diagnosis_files(
    report_path: &Path,
    spec_path: &Path,
) -> Result<ScenarioDiagnosisReport, String> {
    let report_bytes = read_bounded(report_path, MAX_REPORT_BYTES, "scenario diagnosis report")?;
    let report: ScenarioDiagnosisReport = serde_json::from_slice(&report_bytes)
        .map_err(|error| format!("failed to decode scenario diagnosis report: {error}"))?;
    validate_scenario_diagnosis_report(&report)?;
    let expected = evaluate_scenario_diagnosis_files(spec_path)?;
    if report != expected {
        return Err("scenario diagnosis report does not match its bound inputs".into());
    }
    Ok(report)
}

pub fn publish_scenario_diagnosis(
    output: &Path,
    report: &ScenarioDiagnosisReport,
) -> Result<PathBuf, String> {
    validate_scenario_diagnosis_report(report)?;
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable scenario diagnosis {}",
            output.display()
        ));
    }
    let nonce = DIAGNOSIS_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("scenario diagnosis output needs a UTF-8 file name")?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode scenario diagnosis report: {error}"))?;
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

fn bind_evidence(
    claim: &DiagnosticEvidenceSpec,
    artifact_base: &Path,
    ruleset_hash: &str,
    scenario_hash: &str,
) -> Result<BoundDiagnosticEvidence, String> {
    let (artifact_reference, artifact_file_sha256, artifact_size_bytes) =
        if let Some(reference) = &claim.artifact {
            let path = if reference.is_absolute() {
                reference.clone()
            } else {
                artifact_base.join(reference)
            };
            let (digest, size) = hash_bounded_file(&path, MAX_EVIDENCE_BYTES, "evidence artifact")?;
            validate_typed_evidence(claim, &path, ruleset_hash, scenario_hash)?;
            (
                Some(
                    reference
                        .to_str()
                        .ok_or("diagnostic artifact paths must be UTF-8")?
                        .to_owned(),
                ),
                Some(digest),
                Some(size),
            )
        } else {
            (None, None, None)
        };
    Ok(BoundDiagnosticEvidence {
        stage: claim.stage,
        outcome: claim.outcome,
        artifact_kind: claim.artifact_kind.clone(),
        artifact_reference,
        artifact_file_sha256,
        artifact_size_bytes,
        summary: claim.summary.clone(),
    })
}

fn validate_typed_evidence(
    claim: &DiagnosticEvidenceSpec,
    path: &Path,
    ruleset_hash: &str,
    scenario_hash: &str,
) -> Result<(), String> {
    let expected = match claim.artifact_kind.as_str() {
        "feeding_recovery_artifact" => {
            if claim.stage != DiagnosticStage::OnPolicyTeacherRecovery {
                return Err(
                    "feeding recovery evidence belongs only to on_policy_teacher_recovery".into(),
                );
            }
            let artifact = load_feeding_recovery(path)?;
            if artifact.report.compiled_ruleset_hash != ruleset_hash
                || artifact.report.scenario_suite_hash != scenario_hash
            {
                return Err(
                    "feeding recovery evidence names a different ruleset or scenario suite".into(),
                );
            }
            artifact.report.passed
        }
        "policy_correction_dataset" => {
            if claim.stage != DiagnosticStage::LearnedOnPolicyAgreement {
                return Err(
                    "policy correction evidence belongs only to learned_on_policy_agreement".into(),
                );
            }
            if path.file_name().and_then(|name| name.to_str()) != Some("manifest.json") {
                return Err("policy correction evidence must reference its manifest.json".into());
            }
            let directory = path
                .parent()
                .ok_or("policy correction manifest has no dataset directory")?;
            let loaded = load_demonstrations(directory)?;
            if loaded.manifest.schema_version != DEMONSTRATION_SCHEMA_VERSION
                || loaded.manifest.compiled_ruleset_hash != ruleset_hash
                || loaded.manifest.scenario_hash != scenario_hash
            {
                return Err(
                    "policy correction evidence names another schema, ruleset, or scenario".into(),
                );
            }
            let DemonstrationCollection::GreedyPolicyCorrection {
                agreement_passed, ..
            } = loaded.manifest.collection
            else {
                return Err("agreement evidence is not a policy-correction dataset".into());
            };
            agreement_passed
        }
        _ => return Ok(()),
    };
    let expected = if expected {
        EvidenceOutcome::Passed
    } else {
        EvidenceOutcome::Failed
    };
    if claim.outcome != expected {
        return Err(format!(
            "{} evidence is {expected:?}, not {:?}",
            claim.artifact_kind, claim.outcome
        ));
    }
    Ok(())
}

fn validate_bound_evidence(evidence: &[BoundDiagnosticEvidence]) -> Result<(), String> {
    if evidence.len() != DiagnosticStage::ALL.len() {
        return Err("scenario diagnosis report has the wrong number of stages".into());
    }
    let mut blocked = false;
    for (expected, item) in DiagnosticStage::ALL.iter().zip(evidence) {
        if item.stage != *expected {
            return Err("scenario diagnosis report stages are not in canonical order".into());
        }
        validate_bounded_text(
            &item.artifact_kind,
            MAX_ARTIFACT_KIND_CHARS,
            "artifact kind",
        )?;
        validate_bounded_text(&item.summary, MAX_SUMMARY_CHARS, "evidence summary")?;
        if item.outcome == EvidenceOutcome::NotApplicable && !item.stage.permits_not_applicable() {
            return Err("a required diagnostic stage is marked not_applicable".into());
        }
        let bound = item.artifact_reference.is_some()
            && item
                .artifact_file_sha256
                .as_deref()
                .is_some_and(valid_sha256)
            && item
                .artifact_size_bytes
                .is_some_and(|size| size <= MAX_EVIDENCE_BYTES);
        let empty = item.artifact_reference.is_none()
            && item.artifact_file_sha256.is_none()
            && item.artifact_size_bytes.is_none();
        if (item.outcome.is_assessed() && !bound) || (!item.outcome.is_assessed() && !empty) {
            return Err("scenario diagnosis evidence binding is inconsistent".into());
        }
        if let Some(reference) = &item.artifact_reference {
            validate_bounded_text(
                reference,
                MAX_ARTIFACT_REFERENCE_CHARS,
                "artifact reference",
            )?;
        }
        if blocked && item.outcome != EvidenceOutcome::NotRun {
            return Err("scenario diagnosis evaluates a stage after an earlier blocker".into());
        }
        blocked |= item.outcome.blocks_later_stages();
    }
    Ok(())
}

fn classify(
    evidence: &[BoundDiagnosticEvidence],
) -> (DiagnosticAttribution, Option<DiagnosticStage>) {
    for item in evidence {
        match item.outcome {
            EvidenceOutcome::Passed | EvidenceOutcome::NotApplicable => {}
            EvidenceOutcome::NotRun => {
                return (DiagnosticAttribution::Inconclusive, Some(item.stage));
            }
            EvidenceOutcome::Failed => {
                let attribution = match item.stage {
                    DiagnosticStage::AnalyticalBounds | DiagnosticStage::FullStateFeasibility => {
                        DiagnosticAttribution::PhysicsOrObjective
                    }
                    DiagnosticStage::MindFeasibility => {
                        DiagnosticAttribution::MindInterfaceOrActionSemantics
                    }
                    DiagnosticStage::MaintainedTeacher => {
                        DiagnosticAttribution::MaintainedPolicyGap
                    }
                    DiagnosticStage::OnPolicyTeacherRecovery => {
                        DiagnosticAttribution::OnPolicyDistributionShift
                    }
                    DiagnosticStage::LearnedOnPolicyAgreement => {
                        DiagnosticAttribution::ArchitectureOrTraining
                    }
                    DiagnosticStage::HeldOutRollout => DiagnosticAttribution::RolloutCompetence,
                };
                return (attribution, Some(item.stage));
            }
        }
    }
    (DiagnosticAttribution::Competent, None)
}

fn validate_bounded_text(value: &str, max_chars: usize, label: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.chars().count() > max_chars {
        return Err(format!("{label} must contain 1 to {max_chars} characters"));
    }
    Ok(())
}

fn hash_bounded_file(path: &Path, limit: u64, label: &str) -> Result<(String, u64), String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {label} {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("{label} {} is not a regular file", path.display()));
    }
    if metadata.len() > limit {
        return Err(format!(
            "{label} {} is {} bytes; limit is {limit}",
            path.display(),
            metadata.len()
        ));
    }
    let mut file = File::open(path)
        .map_err(|error| format!("failed to open {label} {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {label} {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        total = total
            .checked_add(read as u64)
            .ok_or_else(|| format!("{label} {} size overflow", path.display()))?;
        if total > limit {
            return Err(format!(
                "{label} {} changed beyond its size limit",
                path.display()
            ));
        }
        hasher.update(&buffer[..read]);
    }
    Ok((hex_digest(hasher.finalize()), total))
}

fn sha256(bytes: &[u8]) -> String {
    hex_digest(Sha256::digest(bytes))
}

fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_bounded(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>, String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {label} {}: {error}", path.display()))?
        .len();
    if length > limit {
        return Err(format!(
            "{label} {} is {length} bytes; limit is {limit}",
            path.display()
        ));
    }
    fs::read(path).map_err(|error| format!("failed to read {label} {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        FeedingCurriculumStage, FeedingPromotionGateConfig, ModelConfig, TrainingConfig,
    };
    use crate::control_matrix::MaintainedMindProfile;
    use crate::demonstration::{
        generate_policy_correction_demonstrations, publish_demonstrations, PolicyCorrectionOptions,
    };
    use crate::feeding_recovery::{
        evaluate_feeding_recovery, publish_feeding_recovery, FeedingRecoveryArtifact,
        FeedingRecoveryOptions,
    };
    use crate::model::PolicyValueNetConfig;

    type TestBackend = burn::backend::NdArray<f32>;

    fn evidence(outcomes: [EvidenceOutcome; 7]) -> Vec<DiagnosticEvidenceSpec> {
        DiagnosticStage::ALL
            .into_iter()
            .zip(outcomes)
            .map(|(stage, outcome)| DiagnosticEvidenceSpec {
                stage,
                outcome,
                artifact_kind: format!("{stage:?}_report"),
                artifact: outcome
                    .is_assessed()
                    .then(|| PathBuf::from(format!("{stage:?}.json"))),
                summary: format!("evidence for {stage:?}"),
            })
            .collect()
    }

    fn spec(outcomes: [EvidenceOutcome; 7]) -> ScenarioDiagnosisSpec {
        ScenarioDiagnosisSpec {
            schema_version: SCENARIO_DIAGNOSIS_SCHEMA_VERSION,
            scenario_name: "feeding retention".into(),
            ruleset_hash: "a".repeat(64),
            scenario_hash: "b".repeat(64),
            evidence: evidence(outcomes),
        }
    }

    fn materialize_artifacts(directory: &Path, spec: &ScenarioDiagnosisSpec) {
        for item in &spec.evidence {
            if let Some(path) = &item.artifact {
                fs::write(directory.join(path), format!("{:?}", item.stage)).unwrap();
            }
        }
    }

    #[test]
    fn classifies_each_failure_boundary() {
        let temporary = tempfile::tempdir().unwrap();
        let expected = [
            DiagnosticAttribution::PhysicsOrObjective,
            DiagnosticAttribution::PhysicsOrObjective,
            DiagnosticAttribution::MindInterfaceOrActionSemantics,
            DiagnosticAttribution::MaintainedPolicyGap,
            DiagnosticAttribution::OnPolicyDistributionShift,
            DiagnosticAttribution::ArchitectureOrTraining,
            DiagnosticAttribution::RolloutCompetence,
        ];
        for (failed_index, expected_attribution) in expected.into_iter().enumerate() {
            let mut outcomes = [EvidenceOutcome::Passed; 7];
            outcomes[failed_index] = EvidenceOutcome::Failed;
            for outcome in &mut outcomes[failed_index + 1..] {
                *outcome = EvidenceOutcome::NotRun;
            }
            let spec = spec(outcomes);
            materialize_artifacts(temporary.path(), &spec);
            let report =
                evaluate_scenario_diagnosis(&spec, "c".repeat(64), temporary.path()).unwrap();
            assert_eq!(report.attribution, expected_attribution);
            assert_eq!(
                report.first_blocking_stage,
                Some(DiagnosticStage::ALL[failed_index])
            );
            assert!(report.conclusive);
            assert!(!report.competent);
        }
    }

    #[test]
    fn complete_ladder_is_competent_and_missing_stage_is_inconclusive() {
        let temporary = tempfile::tempdir().unwrap();
        let complete = spec([EvidenceOutcome::Passed; 7]);
        materialize_artifacts(temporary.path(), &complete);
        let report =
            evaluate_scenario_diagnosis(&complete, "c".repeat(64), temporary.path()).unwrap();
        assert_eq!(report.attribution, DiagnosticAttribution::Competent);
        assert!(report.competent);

        let mut outcomes = [EvidenceOutcome::Passed; 7];
        outcomes[4..].fill(EvidenceOutcome::NotRun);
        let incomplete = spec(outcomes);
        materialize_artifacts(temporary.path(), &incomplete);
        let report =
            evaluate_scenario_diagnosis(&incomplete, "d".repeat(64), temporary.path()).unwrap();
        assert_eq!(report.attribution, DiagnosticAttribution::Inconclusive);
        assert_eq!(
            report.first_blocking_stage,
            Some(DiagnosticStage::OnPolicyTeacherRecovery)
        );
        assert!(!report.conclusive);
    }

    #[test]
    fn validation_rejects_skips_and_unbound_claims() {
        let mut skipped = spec([
            EvidenceOutcome::Passed,
            EvidenceOutcome::NotRun,
            EvidenceOutcome::Passed,
            EvidenceOutcome::Passed,
            EvidenceOutcome::Passed,
            EvidenceOutcome::Passed,
            EvidenceOutcome::Passed,
        ]);
        assert!(skipped.validate().is_err());

        skipped.evidence[1].outcome = EvidenceOutcome::Passed;
        skipped.evidence[1].artifact = None;
        assert!(skipped.validate().is_err());

        let mut invalid_na = spec([EvidenceOutcome::Passed; 7]);
        invalid_na.evidence[2].outcome = EvidenceOutcome::NotApplicable;
        invalid_na.evidence[2].artifact = None;
        assert!(invalid_na.validate().is_err());
    }

    #[test]
    fn report_is_hash_bound_verified_and_immutable() {
        let temporary = tempfile::tempdir().unwrap();
        let spec = spec([EvidenceOutcome::Passed; 7]);
        materialize_artifacts(temporary.path(), &spec);
        let spec_path = temporary.path().join("diagnosis.toml");
        fs::write(&spec_path, toml::to_string_pretty(&spec).unwrap()).unwrap();
        let report = evaluate_scenario_diagnosis_files(&spec_path).unwrap();
        let report_path = temporary.path().join("diagnosis.json");
        publish_scenario_diagnosis(&report_path, &report).unwrap();
        assert_eq!(
            verify_scenario_diagnosis_files(&report_path, &spec_path).unwrap(),
            report
        );
        assert!(publish_scenario_diagnosis(&report_path, &report).is_err());

        fs::write(
            temporary.path().join("HeldOutRollout.json"),
            "changed evidence",
        )
        .unwrap();
        assert!(verify_scenario_diagnosis_files(&report_path, &spec_path).is_err());

        let mut tampered = report;
        tampered.attribution = DiagnosticAttribution::PhysicsOrObjective;
        assert!(validate_scenario_diagnosis_report(&tampered).is_err());
    }

    #[test]
    fn typed_feeding_recovery_recomputes_stage_five_outcome() {
        let _guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut config = TrainingConfig::default();
        config.env.world_size = 8;
        config.env.cells_per_team = 1;
        config.env.initial_energy = 500;
        config.env.num_plants = 2;
        config.env.num_scattered_energy = 2;
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
            seeds: vec![119],
            checkpoint_interval_quanta: 256,
            max_checkpoints_per_seed: 1,
            min_checkpoints_per_seed: 1,
            min_remaining_time_quanta: 128,
            min_recovery_rate: 0.0,
        };
        let device = Default::default();
        let model = PolicyValueNetConfig {
            hidden1: config.model.hidden1,
            hidden2: config.model.hidden2,
            recurrent_size: config.model.recurrent_size,
        }
        .init::<TestBackend>(&device);
        let report = evaluate_feeding_recovery(&model, &config, &options, &device).unwrap();
        assert!(report.passed);
        let ruleset_hash = report.compiled_ruleset_hash.clone();
        let scenario_hash = report.scenario_suite_hash.clone();
        let artifact = FeedingRecoveryArtifact::new(
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            config,
            options,
            report,
        )
        .unwrap();
        let recovery_path = temporary.path().join("recovery.json");
        publish_feeding_recovery(&recovery_path, &artifact).unwrap();

        let mut claims = evidence([EvidenceOutcome::Passed; 7]);
        for index in [0, 1] {
            claims[index].outcome = EvidenceOutcome::NotApplicable;
            claims[index].artifact = None;
        }
        claims[4].artifact_kind = "feeding_recovery_artifact".into();
        claims[4].artifact = Some(PathBuf::from("recovery.json"));
        for claim in claims.iter().filter(|claim| {
            claim.outcome == EvidenceOutcome::Passed
                && claim.artifact_kind != "feeding_recovery_artifact"
        }) {
            fs::write(
                temporary.path().join(claim.artifact.as_ref().unwrap()),
                format!("{:?}", claim.stage),
            )
            .unwrap();
        }
        let spec = ScenarioDiagnosisSpec {
            schema_version: SCENARIO_DIAGNOSIS_SCHEMA_VERSION,
            scenario_name: "typed feeding recovery".into(),
            ruleset_hash,
            scenario_hash,
            evidence: claims.clone(),
        };
        let diagnosis =
            evaluate_scenario_diagnosis(&spec, "4".repeat(64), temporary.path()).unwrap();
        assert_eq!(diagnosis.attribution, DiagnosticAttribution::Competent);

        claims[4].outcome = EvidenceOutcome::Failed;
        for claim in &mut claims[5..] {
            claim.outcome = EvidenceOutcome::NotRun;
            claim.artifact = None;
        }
        let false_claim = ScenarioDiagnosisSpec {
            evidence: claims,
            ..spec
        };
        assert!(
            evaluate_scenario_diagnosis(&false_claim, "5".repeat(64), temporary.path()).is_err()
        );
    }

    #[test]
    fn typed_policy_correction_recomputes_stage_six_outcome() {
        let _guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut config = TrainingConfig::default();
        config.env.world_size = 8;
        config.env.cells_per_team = 1;
        config.env.initial_energy = 500;
        config.env.num_plants = 2;
        config.env.num_scattered_energy = 2;
        config.feeding_curriculum.enabled = true;
        config.env = config
            .feeding_curriculum
            .environment_for_stage(&config.env, FeedingCurriculumStage::OnFood);
        config.model = ModelConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 4,
        };
        let device = Default::default();
        let model = PolicyValueNetConfig {
            hidden1: config.model.hidden1,
            hidden2: config.model.hidden2,
            recurrent_size: config.model.recurrent_size,
        }
        .init::<TestBackend>(&device);
        let options = PolicyCorrectionOptions {
            teacher: MaintainedMindProfile::CollisionAwareForager,
            seeds: vec![131],
            max_samples: 16,
            behavior_clone_metadata_sha256: "1".repeat(64),
            behavior_clone_model_sha256: "2".repeat(64),
            minimum_policy_agreement_rate: 1.0,
            label_mode: crate::demonstration::PolicyCorrectionLabelMode::FullTeacher,
        };
        let (manifest, payload) = generate_policy_correction_demonstrations(
            &config,
            "3".repeat(64),
            &options,
            &model,
            &device,
        )
        .unwrap();
        let agreement_passed = match &manifest.collection {
            DemonstrationCollection::GreedyPolicyCorrection {
                agreement_passed, ..
            } => *agreement_passed,
            _ => unreachable!(),
        };
        let dataset = temporary.path().join("corrections");
        publish_demonstrations(&dataset, &manifest, &payload).unwrap();
        let manifest_path = dataset.join("manifest.json");
        let claim = DiagnosticEvidenceSpec {
            stage: DiagnosticStage::LearnedOnPolicyAgreement,
            outcome: if agreement_passed {
                EvidenceOutcome::Passed
            } else {
                EvidenceOutcome::Failed
            },
            artifact_kind: "policy_correction_dataset".into(),
            artifact: Some(manifest_path.clone()),
            summary: "typed exact-input agreement".into(),
        };

        validate_typed_evidence(
            &claim,
            &manifest_path,
            &manifest.compiled_ruleset_hash,
            &manifest.scenario_hash,
        )
        .unwrap();

        let false_claim = DiagnosticEvidenceSpec {
            outcome: if agreement_passed {
                EvidenceOutcome::Failed
            } else {
                EvidenceOutcome::Passed
            },
            ..claim
        };
        assert!(validate_typed_evidence(
            &false_claim,
            &manifest_path,
            &manifest.compiled_ruleset_hash,
            &manifest.scenario_hash,
        )
        .is_err());
    }
}
