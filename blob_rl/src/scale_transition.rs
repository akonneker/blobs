//! Portable, immutable scale-transition jobs and bounded trainer execution.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::artifact::verify_checkpoint_metadata;
use crate::checkpoint_promotion::{
    load_checkpoint_promotion, verify_checkpoint_promotion_evidence_at, CheckpointPromotionVerdict,
};
use crate::config::{ScenarioProfile, TrainingConfig};
use crate::sweep::{experiment_config_hash, sha256};
use crate::viability::hash_json;

pub const SCALE_TRANSITION_SCHEMA_VERSION: u32 = 1;
const MAX_CONTROL_BYTES: u64 = 16 * 1024 * 1024;
const MAX_EVIDENCE_BYTES: u64 = 256 * 1024 * 1024;
static TRANSITION_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScaleTransitionMode {
    Promoted,
    Experimental,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScaleTransitionContract {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub contract_hash: String,
    pub mode: ScaleTransitionMode,
    pub source_checkpoint: String,
    pub source_metadata_sha256: String,
    pub source_model_sha256: String,
    pub source_world_size: usize,
    pub target_source_config_sha256: String,
    pub target_config_file: String,
    pub target_experiment_config_sha256: String,
    pub target_scenario_hash: String,
    pub target_world_size: usize,
    pub semantic_ruleset_hash: String,
    pub model_shape: [usize; 3],
    pub promotion_decision_file: Option<String>,
    pub promotion_decision_sha256: Option<String>,
    pub promotion_decision_hash: Option<String>,
    pub promotion_verdict: Option<CheckpointPromotionVerdict>,
    pub qualification_file: Option<String>,
    pub policy_file: Option<String>,
    pub experimental_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ScaleTransitionPlanOptions {
    pub source_checkpoint: PathBuf,
    pub target_config: PathBuf,
    pub promotion_decision: Option<PathBuf>,
    pub experimental_reason: Option<String>,
    pub output_directory: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScaleTransitionExecutionContract {
    pub schema_version: u32,
    pub transition_contract_hash: String,
    pub trainer_sha256: String,
    pub trainer_size_bytes: u64,
    pub trainer_container_digest: Option<String>,
    pub threads: Option<usize>,
    pub arguments: Vec<String>,
    pub recovery_strategy: String,
    pub execution_hash: String,
}

#[derive(Debug, Clone)]
pub struct ScaleTransitionRunOptions {
    pub train_program: PathBuf,
    pub source_checkpoint: Option<PathBuf>,
    pub trainer_container_digest: Option<String>,
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScaleTransitionExecutionResult {
    pub schema_version: u32,
    pub transition_contract_hash: String,
    pub execution_hash: String,
    pub source_model_sha256: String,
    pub target_experiment_config_sha256: String,
    pub resumed_from: Option<String>,
    pub exit_code: i32,
    pub succeeded: bool,
    pub result_hash: String,
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

fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let mut file = File::open(path)
        .map_err(|error| format!("failed to open {} for hashing: {error}", path.display()))?;
    let size = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
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
    Ok((format!("{:x}", hasher.finalize()), size))
}

fn normalized_reason(reason: Option<&str>) -> Result<Option<String>, String> {
    let reason = reason.map(str::trim).filter(|reason| !reason.is_empty());
    if reason.is_some_and(|reason| reason.len() > 2048) {
        return Err("scale-transition experimental reason is too long".into());
    }
    Ok(reason.map(str::to_string))
}

fn contract_hash(contract: &ScaleTransitionContract) -> Result<String, String> {
    let mut identity = contract.clone();
    identity.contract_hash.clear();
    hash_json(&identity)
}

fn execution_hash(contract: &ScaleTransitionExecutionContract) -> Result<String, String> {
    let mut identity = contract.clone();
    identity.execution_hash.clear();
    hash_json(&identity)
}

fn result_hash(result: &ScaleTransitionExecutionResult) -> Result<String, String> {
    let mut identity = result.clone();
    identity.result_hash.clear();
    hash_json(&identity)
}

pub fn validate_scale_transition_contract(
    contract: &ScaleTransitionContract,
) -> Result<(), String> {
    if contract.schema_version != SCALE_TRANSITION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported scale-transition schema {}; expected {}",
            contract.schema_version, SCALE_TRANSITION_SCHEMA_VERSION
        ));
    }
    for value in [
        &contract.contract_hash,
        &contract.source_metadata_sha256,
        &contract.source_model_sha256,
        &contract.target_source_config_sha256,
        &contract.target_experiment_config_sha256,
        &contract.target_scenario_hash,
        &contract.semantic_ruleset_hash,
    ] {
        if !valid_sha256(value) {
            return Err("scale-transition contract contains an invalid SHA-256 identity".into());
        }
    }
    if contract.target_config_file != "target-config.toml"
        || contract.source_world_size >= contract.target_world_size
        || contract.model_shape.contains(&0)
    {
        return Err("scale-transition contract has an invalid scale or model shape".into());
    }
    match contract.mode {
        ScaleTransitionMode::Promoted => {
            if contract.experimental_reason.is_some()
                || contract.promotion_decision_file.as_deref() != Some("promotion-decision.json")
                || contract.qualification_file.as_deref() != Some("qualification.json")
                || contract.policy_file.as_deref() != Some("promotion-policy.toml")
                || contract.promotion_verdict == Some(CheckpointPromotionVerdict::Rejected)
            {
                return Err("promoted transition lacks approved evidence".into());
            }
        }
        ScaleTransitionMode::Experimental => {
            if contract
                .experimental_reason
                .as_deref()
                .is_none_or(|reason| {
                    reason.trim() != reason || reason.is_empty() || reason.len() > 2048
                })
            {
                return Err("experimental transition requires a recorded reason".into());
            }
        }
    }
    let evidence_fields = [
        contract.promotion_decision_sha256.as_deref(),
        contract.promotion_decision_hash.as_deref(),
    ];
    if evidence_fields
        .iter()
        .flatten()
        .any(|value| !valid_sha256(value))
        || evidence_fields[0].is_some() != evidence_fields[1].is_some()
        || contract.promotion_decision_file.is_some() != evidence_fields[0].is_some()
        || contract.promotion_decision_file.is_some() != contract.promotion_verdict.is_some()
        || contract.promotion_decision_file.is_some() != contract.qualification_file.is_some()
        || contract.promotion_decision_file.is_some() != contract.policy_file.is_some()
    {
        return Err("scale-transition promotion evidence is inconsistent".into());
    }
    if contract_hash(contract)? != contract.contract_hash {
        return Err("scale-transition contract hash mismatch".into());
    }
    Ok(())
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
    bytes.push(b'\n');
    write_synced(path, &bytes)
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to sync directory {}: {error}", path.display()))
}

pub fn publish_scale_transition_job(
    options: &ScaleTransitionPlanOptions,
) -> Result<PathBuf, String> {
    let source_checkpoint = fs::canonicalize(&options.source_checkpoint).map_err(|error| {
        format!(
            "failed to resolve source checkpoint {}: {error}",
            options.source_checkpoint.display()
        )
    })?;
    let source_metadata = verify_checkpoint_metadata(&source_checkpoint)?;
    let source_metadata_bytes = read_bounded(
        &source_checkpoint.join("metadata.json"),
        MAX_CONTROL_BYTES,
        "checkpoint metadata",
    )?;
    let target_config_file = fs::canonicalize(&options.target_config).map_err(|error| {
        format!(
            "failed to resolve target config {}: {error}",
            options.target_config.display()
        )
    })?;
    let target_source_bytes = read_bounded(
        &target_config_file,
        MAX_CONTROL_BYTES,
        "target training config",
    )?;
    let mut target_config = TrainingConfig::from_toml_str(
        std::str::from_utf8(&target_source_bytes)
            .map_err(|error| format!("target config is not UTF-8: {error}"))?,
    )?;
    target_config.validate()?;
    if target_config.initial_policy.is_some() {
        return Err("target config cannot specify another initial policy".into());
    }
    if source_metadata.config.model != target_config.model {
        return Err("source checkpoint and target config model shapes differ".into());
    }
    if source_metadata.config.env.rules.semantic_hash() != target_config.env.rules.semantic_hash() {
        return Err("scale transition cannot change semantic physics".into());
    }
    if source_metadata.config.env.world_size >= target_config.env.world_size {
        return Err("scale transition target must have a larger world".into());
    }
    let output = &options.output_directory;
    if output.exists() {
        return Err(format!(
            "refusing to replace scale-transition job {}",
            output.display()
        ));
    }
    let output_parent = output
        .parent()
        .ok_or_else(|| "scale-transition output has no parent".to_string())?;
    fs::create_dir_all(output_parent).map_err(|error| {
        format!(
            "failed to create scale-transition parent {}: {error}",
            output_parent.display()
        )
    })?;
    let output_parent = fs::canonicalize(output_parent).map_err(|error| {
        format!(
            "failed to resolve scale-transition parent {}: {error}",
            output_parent.display()
        )
    })?;
    let output_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "scale-transition output needs a UTF-8 name".to_string())?;
    let final_output = output_parent.join(output_name);
    let nonce = TRANSITION_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let staging = output_parent.join(format!(".{output_name}.tmp-{}-{nonce}", std::process::id()));
    fs::create_dir(&staging).map_err(|error| {
        format!(
            "failed to create scale-transition staging {}: {error}",
            staging.display()
        )
    })?;
    let result = (|| {
        target_config.checkpoint_dir = "artifacts".into();
        let target_toml = toml::to_string_pretty(&target_config)
            .map_err(|error| format!("failed to encode target config: {error}"))?;
        write_synced(&staging.join("target-config.toml"), target_toml.as_bytes())?;
        let reason = normalized_reason(options.experimental_reason.as_deref())?;
        let mode = if reason.is_some() {
            ScaleTransitionMode::Experimental
        } else {
            ScaleTransitionMode::Promoted
        };
        let mut promotion_decision_sha256 = None;
        let mut promotion_decision_hash = None;
        let mut promotion_verdict = None;
        let mut promotion_decision_file = None;
        let mut qualification_file = None;
        let mut policy_file = None;
        if let Some(decision_path) = &options.promotion_decision {
            let decision_path = fs::canonicalize(decision_path).map_err(|error| {
                format!(
                    "failed to resolve promotion decision {}: {error}",
                    decision_path.display()
                )
            })?;
            let decision_bytes =
                read_bounded(&decision_path, MAX_CONTROL_BYTES, "promotion decision")?;
            let decision = load_checkpoint_promotion(&decision_path)?;
            let source_qualification = Path::new(&decision.qualification_file);
            let source_policy = Path::new(&decision.policy_file);
            verify_checkpoint_promotion_evidence_at(
                &decision,
                &source_checkpoint,
                source_qualification,
                source_policy,
            )?;
            if decision.checkpoint_metadata_sha256 != sha256(&source_metadata_bytes)
                || decision.target_world_size != target_config.env.world_size
                || decision.target_scenario_hash
                    != ScenarioProfile::from(&target_config.env).semantic_hash()?
            {
                return Err(
                    "promotion decision does not bind this source and target transition".into(),
                );
            }
            if mode == ScaleTransitionMode::Promoted
                && decision.verdict == CheckpointPromotionVerdict::Rejected
            {
                return Err("rejected promotion requires experimental mode and a reason".into());
            }
            let qualification_bytes = read_bounded(
                source_qualification,
                MAX_EVIDENCE_BYTES,
                "qualification evidence",
            )?;
            let policy_bytes = read_bounded(source_policy, MAX_CONTROL_BYTES, "promotion policy")?;
            write_synced(&staging.join("promotion-decision.json"), &decision_bytes)?;
            write_synced(&staging.join("qualification.json"), &qualification_bytes)?;
            write_synced(&staging.join("promotion-policy.toml"), &policy_bytes)?;
            promotion_decision_sha256 = Some(sha256(&decision_bytes));
            promotion_decision_hash = Some(decision.decision_hash);
            promotion_verdict = Some(decision.verdict);
            promotion_decision_file = Some("promotion-decision.json".into());
            qualification_file = Some("qualification.json".into());
            policy_file = Some("promotion-policy.toml".into());
        } else if mode == ScaleTransitionMode::Promoted {
            return Err("promoted transition requires a promotion decision".into());
        }
        let scenario_hash = ScenarioProfile::from(&target_config.env).semantic_hash()?;
        let semantic_ruleset_hash = target_config.env.rules.semantic_hash().to_string();
        let mut contract = ScaleTransitionContract {
            schema_version: SCALE_TRANSITION_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").to_string(),
            code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
            contract_hash: String::new(),
            mode,
            source_checkpoint: source_checkpoint.to_string_lossy().into_owned(),
            source_metadata_sha256: sha256(&source_metadata_bytes),
            source_model_sha256: source_metadata.model_sha256,
            source_world_size: source_metadata.config.env.world_size,
            target_source_config_sha256: sha256(&target_source_bytes),
            target_config_file: "target-config.toml".into(),
            target_experiment_config_sha256: experiment_config_hash(&target_config)?,
            target_scenario_hash: scenario_hash,
            target_world_size: target_config.env.world_size,
            semantic_ruleset_hash,
            model_shape: [
                target_config.model.hidden1,
                target_config.model.recurrent_size,
                target_config.model.hidden2,
            ],
            promotion_decision_file,
            promotion_decision_sha256,
            promotion_decision_hash,
            promotion_verdict,
            qualification_file,
            policy_file,
            experimental_reason: reason,
        };
        contract.contract_hash = contract_hash(&contract)?;
        validate_scale_transition_contract(&contract)?;
        write_json(&staging.join("contract.json"), &contract)?;
        sync_directory(&staging)?;
        Ok(contract)
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    fs::rename(&staging, &final_output).map_err(|error| {
        let _ = fs::remove_dir_all(&staging);
        format!("failed to publish {}: {error}", final_output.display())
    })?;
    sync_directory(&output_parent)?;
    verify_scale_transition_job(&final_output, None)?;
    Ok(final_output)
}

pub fn load_scale_transition_contract(job: &Path) -> Result<ScaleTransitionContract, String> {
    let path = job.join("contract.json");
    let contract: ScaleTransitionContract = serde_json::from_slice(&read_bounded(
        &path,
        MAX_CONTROL_BYTES,
        "scale-transition contract",
    )?)
    .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    validate_scale_transition_contract(&contract)?;
    Ok(contract)
}

pub fn verify_scale_transition_job(
    job: &Path,
    source_override: Option<&Path>,
) -> Result<ScaleTransitionContract, String> {
    let job = fs::canonicalize(job).map_err(|error| {
        format!(
            "failed to resolve transition job {}: {error}",
            job.display()
        )
    })?;
    let contract = load_scale_transition_contract(&job)?;
    let source = source_override
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(&contract.source_checkpoint));
    let source = fs::canonicalize(&source).map_err(|error| {
        format!(
            "failed to resolve transition source {}: {error}",
            source.display()
        )
    })?;
    let metadata = verify_checkpoint_metadata(&source)?;
    let metadata_bytes = read_bounded(
        &source.join("metadata.json"),
        MAX_CONTROL_BYTES,
        "checkpoint metadata",
    )?;
    if sha256(&metadata_bytes) != contract.source_metadata_sha256
        || metadata.model_sha256 != contract.source_model_sha256
        || metadata.config.env.world_size != contract.source_world_size
    {
        return Err("transition source checkpoint identity mismatch".into());
    }
    let target_path = job.join(&contract.target_config_file);
    let target_bytes = read_bounded(&target_path, MAX_CONTROL_BYTES, "target config")?;
    let target = TrainingConfig::from_toml_str(
        std::str::from_utf8(&target_bytes)
            .map_err(|error| format!("target config is not UTF-8: {error}"))?,
    )?;
    target.validate()?;
    if target.checkpoint_dir != "artifacts"
        || experiment_config_hash(&target)? != contract.target_experiment_config_sha256
        || ScenarioProfile::from(&target.env).semantic_hash()? != contract.target_scenario_hash
        || target.env.rules.semantic_hash().to_string() != contract.semantic_ruleset_hash
        || [
            target.model.hidden1,
            target.model.recurrent_size,
            target.model.hidden2,
        ] != contract.model_shape
    {
        return Err("transition target configuration identity mismatch".into());
    }
    if let Some(decision_file) = &contract.promotion_decision_file {
        let decision_path = job.join(decision_file);
        let decision_bytes = read_bounded(&decision_path, MAX_CONTROL_BYTES, "promotion decision")?;
        if Some(sha256(&decision_bytes)) != contract.promotion_decision_sha256 {
            return Err("transition promotion decision file hash mismatch".into());
        }
        let decision = load_checkpoint_promotion(&decision_path)?;
        verify_checkpoint_promotion_evidence_at(
            &decision,
            &source,
            &job.join(
                contract
                    .qualification_file
                    .as_deref()
                    .expect("validated evidence trio"),
            ),
            &job.join(
                contract
                    .policy_file
                    .as_deref()
                    .expect("validated evidence trio"),
            ),
        )?;
        if Some(decision.decision_hash) != contract.promotion_decision_hash
            || Some(decision.verdict) != contract.promotion_verdict
        {
            return Err("transition promotion evidence identity mismatch".into());
        }
    }
    let execution_path = job.join("execution-contract.json");
    let execution = if execution_path.exists() {
        let execution: ScaleTransitionExecutionContract = serde_json::from_slice(&read_bounded(
            &execution_path,
            MAX_CONTROL_BYTES,
            "transition execution contract",
        )?)
        .map_err(|error| format!("failed to decode execution contract: {error}"))?;
        validate_execution_contract(&execution)?;
        if execution.transition_contract_hash != contract.contract_hash {
            return Err("transition execution contract names another job".into());
        }
        Some(execution)
    } else {
        None
    };
    let result_path = job.join("execution-result.json");
    if result_path.exists() {
        let result: ScaleTransitionExecutionResult = serde_json::from_slice(&read_bounded(
            &result_path,
            MAX_CONTROL_BYTES,
            "transition execution result",
        )?)
        .map_err(|error| format!("failed to decode execution result: {error}"))?;
        validate_execution_result(&result)?;
        let execution = execution
            .as_ref()
            .ok_or("transition result has no execution contract")?;
        if result.transition_contract_hash != contract.contract_hash
            || result.execution_hash != execution.execution_hash
            || result.source_model_sha256 != contract.source_model_sha256
            || result.target_experiment_config_sha256 != contract.target_experiment_config_sha256
        {
            return Err("transition execution result identity mismatch".into());
        }
        if let Some(resumed_from) = &result.resumed_from {
            verify_checkpoint_metadata(&job.join(resumed_from)).map_err(|error| {
                format!("transition resume checkpoint no longer verifies: {error}")
            })?;
        }
    }
    Ok(contract)
}

fn validate_execution_contract(contract: &ScaleTransitionExecutionContract) -> Result<(), String> {
    if contract.schema_version != SCALE_TRANSITION_SCHEMA_VERSION
        || !valid_sha256(&contract.transition_contract_hash)
        || !valid_sha256(&contract.trainer_sha256)
        || !valid_sha256(&contract.execution_hash)
        || contract.trainer_size_bytes == 0
        || contract.threads == Some(0)
        || contract.arguments
            != [
                "--config",
                "target-config.toml",
                "--load-model",
                "model.mpk",
            ]
        || contract.recovery_strategy != "newest-verified-update-boundary"
        || contract
            .trainer_container_digest
            .as_ref()
            .is_some_and(|digest| {
                digest
                    .strip_prefix("sha256:")
                    .is_none_or(|hash| !valid_sha256(hash))
            })
    {
        return Err("invalid scale-transition execution contract".into());
    }
    if execution_hash(contract)? != contract.execution_hash {
        return Err("scale-transition execution hash mismatch".into());
    }
    Ok(())
}

fn validate_execution_result(result: &ScaleTransitionExecutionResult) -> Result<(), String> {
    if result.schema_version != SCALE_TRANSITION_SCHEMA_VERSION
        || !valid_sha256(&result.transition_contract_hash)
        || !valid_sha256(&result.execution_hash)
        || !valid_sha256(&result.source_model_sha256)
        || !valid_sha256(&result.target_experiment_config_sha256)
        || !valid_sha256(&result.result_hash)
        || result.succeeded != (result.exit_code == 0)
        || result.resumed_from.as_deref().is_some_and(|path| {
            let mut components = Path::new(path).components();
            !matches!(
                (components.next(), components.next(), components.next()),
                (
                    Some(std::path::Component::Normal(root)),
                    Some(std::path::Component::Normal(checkpoint)),
                    None
                ) if root == "artifacts"
                    && checkpoint
                        .to_str()
                        .and_then(|name| name.strip_prefix("checkpoint-"))
                        .is_some_and(|update| update.parse::<usize>().is_ok())
            )
        })
    {
        return Err("invalid scale-transition execution result".into());
    }
    if result_hash(result)? != result.result_hash {
        return Err("scale-transition result hash mismatch".into());
    }
    Ok(())
}

fn newest_verified_checkpoint(
    artifacts: &Path,
    expected_config: &TrainingConfig,
) -> Result<Option<PathBuf>, String> {
    if !artifacts.exists() {
        return Ok(None);
    }
    let mut checkpoints = Vec::new();
    for entry in fs::read_dir(artifacts)
        .map_err(|error| format!("failed to inspect {}: {error}", artifacts.display()))?
    {
        let entry = entry.map_err(|error| format!("failed to inspect artifact entry: {error}"))?;
        if !entry
            .file_type()
            .map_err(|error| format!("failed to inspect artifact type: {error}"))?
            .is_dir()
        {
            continue;
        }
        let name = entry.file_name();
        let Some(update) = name
            .to_str()
            .and_then(|name| name.strip_prefix("checkpoint-"))
            .and_then(|value| value.parse::<usize>().ok())
        else {
            continue;
        };
        let metadata = verify_checkpoint_metadata(&entry.path()).map_err(|error| {
            format!(
                "published transition checkpoint {} is invalid: {error}",
                entry.path().display()
            )
        })?;
        if metadata.config != *expected_config {
            return Err(format!(
                "published transition checkpoint {} belongs to another target configuration",
                entry.path().display()
            ));
        }
        checkpoints.push((update, entry.path()));
    }
    checkpoints.sort_unstable_by_key(|(update, _)| *update);
    Ok(checkpoints.pop().map(|(_, path)| path))
}

pub fn execute_scale_transition_job(
    job: &Path,
    options: &ScaleTransitionRunOptions,
) -> Result<ScaleTransitionExecutionResult, String> {
    let job = fs::canonicalize(job).map_err(|error| {
        format!(
            "failed to resolve transition job {}: {error}",
            job.display()
        )
    })?;
    let contract = verify_scale_transition_job(&job, options.source_checkpoint.as_deref())?;
    let source = options
        .source_checkpoint
        .clone()
        .unwrap_or_else(|| PathBuf::from(&contract.source_checkpoint));
    let source = fs::canonicalize(&source).map_err(|error| {
        format!(
            "failed to resolve transition source {}: {error}",
            source.display()
        )
    })?;
    let train_program = fs::canonicalize(&options.train_program).map_err(|error| {
        format!(
            "failed to resolve trainer {}: {error}",
            options.train_program.display()
        )
    })?;
    if !train_program.is_file() {
        return Err("scale-transition trainer is not a file".into());
    }
    let (trainer_sha256, trainer_size_bytes) = sha256_file(&train_program)?;
    let mut execution = ScaleTransitionExecutionContract {
        schema_version: SCALE_TRANSITION_SCHEMA_VERSION,
        transition_contract_hash: contract.contract_hash.clone(),
        trainer_sha256,
        trainer_size_bytes,
        trainer_container_digest: options.trainer_container_digest.clone(),
        threads: options.threads,
        arguments: vec![
            "--config".into(),
            "target-config.toml".into(),
            "--load-model".into(),
            "model.mpk".into(),
        ],
        recovery_strategy: "newest-verified-update-boundary".into(),
        execution_hash: String::new(),
    };
    execution.execution_hash = execution_hash(&execution)?;
    validate_execution_contract(&execution)?;
    let execution_path = job.join("execution-contract.json");
    if execution_path.exists() {
        let existing: ScaleTransitionExecutionContract = serde_json::from_slice(&read_bounded(
            &execution_path,
            MAX_CONTROL_BYTES,
            "transition execution contract",
        )?)
        .map_err(|error| format!("failed to decode execution contract: {error}"))?;
        validate_execution_contract(&existing)?;
        if existing != execution {
            return Err("transition job is already bound to another trainer execution".into());
        }
    } else {
        write_json(&execution_path, &execution)?;
        sync_directory(&job)?;
    }
    let result_path = job.join("execution-result.json");
    if result_path.exists() {
        let existing: ScaleTransitionExecutionResult = serde_json::from_slice(&read_bounded(
            &result_path,
            MAX_CONTROL_BYTES,
            "transition execution result",
        )?)
        .map_err(|error| format!("failed to decode execution result: {error}"))?;
        validate_execution_result(&existing)?;
        return Err(format!(
            "scale-transition job already finished with exit code {}",
            existing.exit_code
        ));
    }
    let model_path = source.join("model.mpk");
    let target_config =
        TrainingConfig::from_file(&job.join("target-config.toml").to_string_lossy())?;
    let resume_checkpoint = newest_verified_checkpoint(&job.join("artifacts"), &target_config)?;
    let mut command = Command::new(&train_program);
    command
        .current_dir(&job)
        .arg("--config")
        .arg("target-config.toml");
    if let Some(resume) = &resume_checkpoint {
        command.arg("--resume").arg(resume);
    } else {
        command.arg("--load-model").arg(&model_path);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    if let Some(threads) = options.threads {
        command.env("RAYON_NUM_THREADS", threads.to_string());
    }
    let status = command
        .status()
        .map_err(|error| format!("failed to launch scale-transition trainer: {error}"))?;
    let exit_code = status.code().unwrap_or(-1);
    let mut result = ScaleTransitionExecutionResult {
        schema_version: SCALE_TRANSITION_SCHEMA_VERSION,
        transition_contract_hash: contract.contract_hash,
        execution_hash: execution.execution_hash,
        source_model_sha256: contract.source_model_sha256,
        target_experiment_config_sha256: contract.target_experiment_config_sha256,
        resumed_from: resume_checkpoint.map(|path| {
            path.strip_prefix(&job)
                .expect("transition checkpoint is inside its job")
                .to_string_lossy()
                .into_owned()
        }),
        exit_code,
        succeeded: status.success(),
        result_hash: String::new(),
    };
    result.result_hash = result_hash(&result)?;
    validate_execution_result(&result)?;
    write_json(&result_path, &result)?;
    sync_directory(&job)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_contract(mode: ScaleTransitionMode) -> ScaleTransitionContract {
        let hash = "a".repeat(64);
        let has_evidence = mode == ScaleTransitionMode::Promoted;
        let mut contract = ScaleTransitionContract {
            schema_version: SCALE_TRANSITION_SCHEMA_VERSION,
            package_version: "test".into(),
            code_revision: None,
            contract_hash: String::new(),
            mode,
            source_checkpoint: "/source".into(),
            source_metadata_sha256: hash.clone(),
            source_model_sha256: hash.clone(),
            source_world_size: 32,
            target_source_config_sha256: hash.clone(),
            target_config_file: "target-config.toml".into(),
            target_experiment_config_sha256: hash.clone(),
            target_scenario_hash: hash.clone(),
            target_world_size: 256,
            semantic_ruleset_hash: hash.clone(),
            model_shape: [128, 64, 64],
            promotion_decision_file: has_evidence.then(|| "promotion-decision.json".into()),
            promotion_decision_sha256: has_evidence.then(|| hash.clone()),
            promotion_decision_hash: has_evidence.then(|| hash.clone()),
            promotion_verdict: has_evidence.then_some(CheckpointPromotionVerdict::Approved),
            qualification_file: has_evidence.then(|| "qualification.json".into()),
            policy_file: has_evidence.then(|| "promotion-policy.toml".into()),
            experimental_reason: (mode == ScaleTransitionMode::Experimental)
                .then(|| "deliberate treatment".into()),
        };
        contract.contract_hash = contract_hash(&contract).unwrap();
        contract
    }

    #[test]
    fn promoted_and_experimental_contracts_are_distinct_and_valid() {
        validate_scale_transition_contract(&synthetic_contract(ScaleTransitionMode::Promoted))
            .unwrap();
        validate_scale_transition_contract(&synthetic_contract(ScaleTransitionMode::Experimental))
            .unwrap();
    }

    #[test]
    fn experimental_contract_requires_reason_and_promoted_requires_evidence() {
        let mut experimental = synthetic_contract(ScaleTransitionMode::Experimental);
        experimental.experimental_reason = None;
        experimental.contract_hash = contract_hash(&experimental).unwrap();
        assert!(validate_scale_transition_contract(&experimental).is_err());

        let mut promoted = synthetic_contract(ScaleTransitionMode::Promoted);
        promoted.promotion_verdict = Some(CheckpointPromotionVerdict::Rejected);
        promoted.contract_hash = contract_hash(&promoted).unwrap();
        assert!(validate_scale_transition_contract(&promoted).is_err());
    }

    #[test]
    fn contract_hash_detects_tampering() {
        let mut contract = synthetic_contract(ScaleTransitionMode::Promoted);
        contract.target_world_size = 512;
        assert!(validate_scale_transition_contract(&contract)
            .unwrap_err()
            .contains("hash mismatch"));
    }

    #[test]
    fn execution_result_is_hash_bound() {
        let hash = "b".repeat(64);
        let mut result = ScaleTransitionExecutionResult {
            schema_version: SCALE_TRANSITION_SCHEMA_VERSION,
            transition_contract_hash: hash.clone(),
            execution_hash: hash.clone(),
            source_model_sha256: hash.clone(),
            target_experiment_config_sha256: hash,
            resumed_from: Some("artifacts/checkpoint-00000007".into()),
            exit_code: 0,
            succeeded: true,
            result_hash: String::new(),
        };
        result.result_hash = result_hash(&result).unwrap();
        validate_execution_result(&result).unwrap();
        result.exit_code = 2;
        assert!(validate_execution_result(&result).is_err());
    }
}
