//! Restart-safe orchestration from sweep planning through optional training.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::OpponentProfile;
use crate::sweep::{publish_rules_sweep, sha256, RulesSweepSpec};
use crate::sweep_execution::{
    execute_rules_sweep, load_validated_sweep, SweepExecutionSummary, SweepExecutorOptions,
};
use crate::viability_gate::{
    evaluate_viability_gates, publish_viability_gate_report, reproduce_viability_gate_requirement,
    ViabilityGateDecision, ViabilityGateRequirement,
};
use crate::viability_matrix::{
    load_viability_matrix_report, publish_viability_matrix_report, run_viability_matrix,
    ViabilityMatrixOptions,
};

pub const VIABILITY_PREFLIGHT_SCHEMA_VERSION: u32 = 2;
const MAX_PREFLIGHT_CONFIG_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct ViabilityPreflightOptions {
    pub candidates: Vec<OpponentProfile>,
    pub opponents: Vec<OpponentProfile>,
    pub baseline_variant: Option<String>,
    pub matrix_max_parallel: usize,
    pub matrix_max_micro_actions: usize,
    pub gate_spec_file: PathBuf,
    pub execute_training: Option<SweepExecutorOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityPreflightReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub manifest_file: String,
    pub manifest_sha256: String,
    pub plan_reused: bool,
    pub matrix_file: String,
    pub matrix_config_hash: String,
    pub matrix_reused: bool,
    pub decision_file: String,
    pub gate_config_hash: String,
    pub decision_reused: bool,
    pub gate_decision: ViabilityGateDecision,
    pub failed_checks: usize,
    pub training: Option<SweepExecutionSummary>,
}

fn requested_manifest_path(spec_file: &Path, spec: &RulesSweepSpec) -> Result<PathBuf, String> {
    if spec.output_dir.trim().is_empty() {
        return Err("sweep output directory must not be empty".into());
    }
    let spec_directory = spec_file
        .parent()
        .ok_or_else(|| "sweep specification has no parent directory".to_string())?;
    let requested = Path::new(&spec.output_dir);
    let requested = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        spec_directory.join(requested)
    };
    let name = requested
        .file_name()
        .ok_or_else(|| "sweep output directory needs a name".to_string())?;
    let parent = requested
        .parent()
        .ok_or_else(|| "sweep output directory has no parent".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("failed to resolve {}: {error}", parent.display()))?;
    Ok(parent.join(name).join("manifest.json"))
}

fn canonical_profiles(
    label: &str,
    profiles: &[OpponentProfile],
) -> Result<Vec<OpponentProfile>, String> {
    if profiles.is_empty() {
        return Err(format!("preflight requires at least one {label}"));
    }
    let unique = profiles.iter().copied().collect::<HashSet<_>>();
    if unique.len() != profiles.len() {
        return Err(format!("preflight {label} must be unique"));
    }
    Ok(OpponentProfile::ALL
        .into_iter()
        .filter(|profile| unique.contains(profile))
        .collect())
}

/// Run or resume every immutable pre-training stage. Existing artifacts are
/// fully checked before reuse and are never replaced when inconsistent.
pub fn run_viability_preflight(
    spec_file: &Path,
    options: ViabilityPreflightOptions,
) -> Result<ViabilityPreflightReport, String> {
    if options.matrix_max_parallel == 0 {
        return Err("matrix_max_parallel must be positive".into());
    }
    if options.matrix_max_micro_actions == 0 {
        return Err("matrix_max_micro_actions must be positive".into());
    }
    let candidates = canonical_profiles("candidate profile", &options.candidates)?;
    let opponents = canonical_profiles("opponent profile", &options.opponents)?;
    let spec_file = fs::canonicalize(spec_file).map_err(|error| {
        format!(
            "failed to resolve sweep specification {}: {error}",
            spec_file.display()
        )
    })?;
    let spec_bytes = crate::artifact_io::read_bounded(&spec_file, MAX_PREFLIGHT_CONFIG_BYTES)?;
    let spec: RulesSweepSpec = toml::from_str(
        std::str::from_utf8(&spec_bytes)
            .map_err(|error| format!("sweep specification is not UTF-8: {error}"))?,
    )
    .map_err(|error| format!("failed to parse sweep specification: {error}"))?;
    let manifest_file = requested_manifest_path(&spec_file, &spec)?;
    let plan_reused = manifest_file.is_file();
    if !plan_reused {
        match publish_rules_sweep(&spec_file) {
            Ok(_) => {}
            Err(error) if manifest_file.is_file() => {
                // A concurrent planner may have won publication. The complete
                // plan is verified below before it can be reused.
                let _ = error;
            }
            Err(error) => return Err(error),
        }
    }
    let sweep = load_validated_sweep(&manifest_file)?;
    if Path::new(&sweep.manifest.spec_file) != spec_file
        || sweep.manifest.spec_sha256 != sha256(&spec_bytes)
    {
        return Err("existing sweep plan does not match the requested specification".into());
    }
    let base_config_file = Path::new(&sweep.manifest.base_config_file);
    let base_bytes =
        crate::artifact_io::read_bounded(base_config_file, MAX_PREFLIGHT_CONFIG_BYTES)?;
    if sha256(&base_bytes) != sweep.manifest.base_config_sha256 {
        return Err("existing sweep plan base configuration has changed".into());
    }

    let preflight_directory = sweep.root.join("preflight");
    fs::create_dir_all(&preflight_directory).map_err(|error| {
        format!(
            "failed to create preflight directory {}: {error}",
            preflight_directory.display()
        )
    })?;
    let lock_path = preflight_directory.join("preflight.lock");
    let lock = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|error| format!("failed to open {}: {error}", lock_path.display()))?;
    lock.try_lock().map_err(|error| {
        format!(
            "another preflight holds the sweep lock {}: {error}",
            lock_path.display()
        )
    })?;

    let baseline_variant = options
        .baseline_variant
        .clone()
        .unwrap_or_else(|| sweep.manifest.variants[0].name.clone());
    let matrix_file = preflight_directory.join("viability-matrix.json");
    let matrix_reused = matrix_file.is_file();
    let matrix = if matrix_reused {
        let matrix = load_viability_matrix_report(&matrix_file)?;
        if matrix.manifest_sha256 != sweep.manifest_sha256
            || matrix.baseline_variant != baseline_variant
            || matrix.candidates != candidates
            || matrix.opponents != opponents
            || matrix.ecological_options.max_micro_actions != options.matrix_max_micro_actions
        {
            return Err("existing viability matrix does not match requested preflight".into());
        }
        matrix
    } else {
        let matrix = run_viability_matrix(
            &manifest_file,
            &ViabilityMatrixOptions {
                candidates: candidates.clone(),
                opponents: opponents.clone(),
                baseline_variant: Some(baseline_variant),
                max_parallel: options.matrix_max_parallel,
                max_micro_actions: options.matrix_max_micro_actions,
            },
        )?;
        publish_viability_matrix_report(&matrix_file, &matrix)?;
        matrix
    };
    if matrix.package_version != env!("CARGO_PKG_VERSION")
        || matrix.code_revision.as_deref() != option_env!("BLOB_CODE_REVISION")
    {
        return Err("viability matrix was produced by a different engine build".into());
    }

    let decision_file = preflight_directory.join("viability-decision.json");
    let requirement = ViabilityGateRequirement {
        decision_file: decision_file.clone(),
        matrix_file: matrix_file.clone(),
        gate_spec_file: options.gate_spec_file,
    };
    let decision_reused = decision_file.is_file();
    if !decision_reused {
        let decision =
            evaluate_viability_gates(&requirement.matrix_file, &requirement.gate_spec_file)?;
        publish_viability_gate_report(&decision_file, &decision)?;
    }
    let decision = reproduce_viability_gate_requirement(&requirement, &sweep.manifest_sha256)?;

    let training = if decision.decision == ViabilityGateDecision::Passed {
        if let Some(mut executor) = options.execute_training {
            if executor.required_viability_gate.is_some() {
                return Err("preflight owns the executor viability-gate requirement".into());
            }
            executor.required_viability_gate = Some(requirement);
            Some(execute_rules_sweep(&manifest_file, &executor)?)
        } else {
            None
        }
    } else {
        None
    };

    Ok(ViabilityPreflightReport {
        schema_version: VIABILITY_PREFLIGHT_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        manifest_file: manifest_file.to_string_lossy().into_owned(),
        manifest_sha256: sweep.manifest_sha256,
        plan_reused,
        matrix_file: matrix_file.to_string_lossy().into_owned(),
        matrix_config_hash: matrix.matrix_config_hash,
        matrix_reused,
        decision_file: decision_file.to_string_lossy().into_owned(),
        gate_config_hash: decision.gate_config_hash,
        decision_reused,
        gate_decision: decision.decision,
        failed_checks: decision.failed_checks,
        training,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_inputs(root: &Path, policy: &str) -> (PathBuf, PathBuf) {
        fs::write(
            root.join("base.toml"),
            r#"
            seed = 1
            num_envs = 1
            rollout_length = 4
            total_timesteps = 8
            eval_interval = 0
            eval_episodes = 1
            checkpoint_interval = 0
            evaluation_opponents = ["wait"]

            [telemetry]
            enabled = true
            state_sample_interval_steps = 2
            max_state_samples_per_episode = 8
            episode_log_stride = 0

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
            sim_time_limit_quanta = 6144

            [self_play]
            max_opponent_pool = 0
            "#,
        )
        .unwrap();
        let spec = root.join("sweep.toml");
        fs::write(
            &spec,
            r#"
            base_config = "base.toml"
            output_dir = "planned"
            seeds = [71, 72, 73]

            [[variants]]
            name = "baseline"

            [[variants]]
            name = "candidate"
            [variants.rules]
            digestion_rate_numerator = 2
            "#,
        )
        .unwrap();
        let gates = root.join("gates.toml");
        fs::write(&gates, policy).unwrap();
        (spec, gates)
    }

    fn options(gates: PathBuf) -> ViabilityPreflightOptions {
        ViabilityPreflightOptions {
            candidates: vec![OpponentProfile::Random],
            opponents: vec![OpponentProfile::Wait],
            baseline_variant: None,
            matrix_max_parallel: 2,
            matrix_max_micro_actions: 1_000,
            gate_spec_file: gates,
            execute_training: None,
        }
    }

    const PASSING_POLICY: &str = r#"
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
    "#;

    #[test]
    fn preflight_reuses_verified_stages_and_rejects_changed_inputs() {
        let temporary = tempfile::tempdir().unwrap();
        let (spec, gates) = write_inputs(temporary.path(), PASSING_POLICY);
        let first = run_viability_preflight(&spec, options(gates.clone())).unwrap();
        assert_eq!(first.gate_decision, ViabilityGateDecision::Passed);
        assert!(!first.plan_reused);
        assert!(!first.matrix_reused);
        assert!(!first.decision_reused);

        let second = run_viability_preflight(&spec, options(gates.clone())).unwrap();
        assert!(second.plan_reused);
        assert!(second.matrix_reused);
        assert!(second.decision_reused);
        assert_eq!(first.manifest_sha256, second.manifest_sha256);
        assert_eq!(first.matrix_config_hash, second.matrix_config_hash);
        assert_eq!(first.gate_config_hash, second.gate_config_hash);

        let original_spec = fs::read(&spec).unwrap();
        let mut changed_spec = original_spec.clone();
        changed_spec.extend_from_slice(b"\n# changed after planning\n");
        fs::write(&spec, changed_spec).unwrap();
        assert!(run_viability_preflight(&spec, options(gates.clone()))
            .unwrap_err()
            .contains("does not match the requested specification"));
        fs::write(&spec, original_spec).unwrap();

        let base_file = temporary.path().join("base.toml");
        let original_base = fs::read(&base_file).unwrap();
        let mut changed_base = original_base.clone();
        changed_base.extend_from_slice(b"\n# changed after planning\n");
        fs::write(&base_file, changed_base).unwrap();
        assert!(run_viability_preflight(&spec, options(gates.clone()))
            .unwrap_err()
            .contains("base configuration has changed"));
        fs::write(&base_file, original_base).unwrap();

        let matrix_file = PathBuf::from(second.matrix_file);
        let mut matrix = load_viability_matrix_report(&matrix_file).unwrap();
        matrix.matchups[0].report.aggregate.timeouts += 1;
        fs::write(&matrix_file, serde_json::to_vec_pretty(&matrix).unwrap()).unwrap();
        assert!(run_viability_preflight(&spec, options(gates))
            .unwrap_err()
            .contains("aggregate"));
    }

    #[test]
    fn failed_preflight_publishes_diagnostics_but_never_launches_training() {
        let temporary = tempfile::tempdir().unwrap();
        let (spec, gates) = write_inputs(
            temporary.path(),
            r#"
            schema_version = 2
            [absolute]
            min_applied_damage_per_episode_for_combat_profiles = 1.0e300
            "#,
        );
        let report = run_viability_preflight(&spec, options(gates)).unwrap();
        assert_eq!(report.gate_decision, ViabilityGateDecision::Failed);
        assert!(report.failed_checks > 0);
        assert!(report.training.is_none());
        assert!(Path::new(&report.decision_file).is_file());
        assert!(!temporary.path().join("planned/executor.lock").exists());
        let sweep = load_validated_sweep(Path::new(&report.manifest_file)).unwrap();
        assert!(sweep
            .manifest
            .runs
            .iter()
            .all(|run| { !Path::new(&run.run_directory).join("status.json").exists() }));
    }
}
