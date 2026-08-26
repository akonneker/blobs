//! Bounded, deterministic execution of baseline viability matrices.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::{OpponentProfile, TrainingConfig};
use crate::sweep_execution::load_validated_sweep;
use crate::viability::{
    run_baseline_viability, validate_viability_report, ViabilityOutcome, ViabilityReport,
};

pub const VIABILITY_MATRIX_SCHEMA_VERSION: u32 = 2;
const MAX_VIABILITY_MATRIX_BYTES: u64 = 64 * 1024 * 1024;
static MATRIX_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViabilityMatrixOptions {
    pub candidates: Vec<OpponentProfile>,
    pub opponents: Vec<OpponentProfile>,
    pub baseline_variant: Option<String>,
    pub max_parallel: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityMatrixVariant {
    pub name: String,
    pub description: Option<String>,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityMatrixMatchup {
    pub variant: String,
    pub report: ViabilityReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityDifferenceSummary {
    pub samples: usize,
    pub mean: f64,
    pub sample_standard_deviation: f64,
    pub standard_error: f64,
    pub confidence_95_half_width: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PairedViabilityComparison {
    pub baseline_variant: String,
    pub candidate_variant: String,
    pub candidate_profile: OpponentProfile,
    pub opponent_profile: OpponentProfile,
    pub seeds: Vec<u64>,
    /// Outcome order is loss < timeout < win.
    pub improved_outcomes: usize,
    pub unchanged_outcomes: usize,
    pub regressed_outcomes: usize,
    /// Difference of paired win indicators, candidate variant minus baseline.
    pub candidate_win_difference: ViabilityDifferenceSummary,
    pub final_candidate_cell_difference: ViabilityDifferenceSummary,
    pub final_opponent_cell_difference: ViabilityDifferenceSummary,
    pub environment_step_difference: ViabilityDifferenceSummary,
    pub final_sim_time_quanta_difference: ViabilityDifferenceSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityMatrixReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub mind_abi_version: u16,
    pub mind_abi_hash: String,
    pub manifest_sha256: String,
    pub matrix_config_hash: String,
    pub baseline_variant: String,
    pub candidates: Vec<OpponentProfile>,
    pub opponents: Vec<OpponentProfile>,
    pub seeds: Vec<u64>,
    pub variants: Vec<ViabilityMatrixVariant>,
    pub matchups: Vec<ViabilityMatrixMatchup>,
    pub paired_comparisons: Vec<PairedViabilityComparison>,
}

#[derive(Serialize)]
struct MatrixIdentity<'a> {
    mind_abi_version: u16,
    mind_abi_hash: &'a str,
    manifest_sha256: &'a str,
    baseline_variant: &'a str,
    candidates: &'a [OpponentProfile],
    opponents: &'a [OpponentProfile],
}

#[derive(Clone)]
struct MatrixJob {
    variant: String,
    config: TrainingConfig,
    semantic_ruleset_hash: String,
    compiled_ruleset_hash: String,
    scenario_hash: String,
    candidate: OpponentProfile,
    opponent: OpponentProfile,
}

fn canonical_profiles(
    label: &str,
    requested: &[OpponentProfile],
) -> Result<Vec<OpponentProfile>, String> {
    if requested.is_empty() {
        return Err(format!("viability matrix requires at least one {label}"));
    }
    let unique = requested.iter().copied().collect::<HashSet<_>>();
    if unique.len() != requested.len() {
        return Err(format!("viability matrix {label} must be unique"));
    }
    Ok(OpponentProfile::ALL
        .into_iter()
        .filter(|profile| unique.contains(profile))
        .collect())
}

fn summarize(values: &[f64]) -> ViabilityDifferenceSummary {
    let samples = values.len();
    let mean = values.iter().sum::<f64>() / samples as f64;
    let sample_standard_deviation = if samples > 1 {
        let squared_error = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        (squared_error / (samples - 1) as f64).sqrt()
    } else {
        0.0
    };
    let standard_error = sample_standard_deviation / (samples as f64).sqrt();
    ViabilityDifferenceSummary {
        samples,
        mean,
        sample_standard_deviation,
        standard_error,
        confidence_95_half_width: standard_error * 1.96,
    }
}

fn outcome_rank(outcome: ViabilityOutcome) -> u8 {
    match outcome {
        ViabilityOutcome::CandidateLoss => 0,
        ViabilityOutcome::Timeout => 1,
        ViabilityOutcome::CandidateWin => 2,
    }
}

fn paired_comparison(
    baseline_variant: &str,
    candidate_variant: &str,
    baseline: &ViabilityReport,
    candidate: &ViabilityReport,
) -> Result<PairedViabilityComparison, String> {
    if baseline.candidate != candidate.candidate
        || baseline.opponent != candidate.opponent
        || baseline.seeds != candidate.seeds
        || baseline.episodes.len() != candidate.episodes.len()
    {
        return Err(format!(
            "cannot pair {candidate_variant} against {baseline_variant}: matchup or seeds differ"
        ));
    }
    let mut improved_outcomes = 0;
    let mut unchanged_outcomes = 0;
    let mut regressed_outcomes = 0;
    let mut win_differences = Vec::with_capacity(candidate.episodes.len());
    let mut candidate_cell_differences = Vec::with_capacity(candidate.episodes.len());
    let mut opponent_cell_differences = Vec::with_capacity(candidate.episodes.len());
    let mut step_differences = Vec::with_capacity(candidate.episodes.len());
    let mut quanta_differences = Vec::with_capacity(candidate.episodes.len());

    for (baseline_episode, candidate_episode) in baseline.episodes.iter().zip(&candidate.episodes) {
        if baseline_episode.seed != candidate_episode.seed {
            return Err(format!(
                "cannot pair {candidate_variant} against {baseline_variant}: episode order differs"
            ));
        }
        match outcome_rank(candidate_episode.outcome).cmp(&outcome_rank(baseline_episode.outcome)) {
            std::cmp::Ordering::Greater => improved_outcomes += 1,
            std::cmp::Ordering::Equal => unchanged_outcomes += 1,
            std::cmp::Ordering::Less => regressed_outcomes += 1,
        }
        win_differences.push(
            f64::from(candidate_episode.outcome == ViabilityOutcome::CandidateWin)
                - f64::from(baseline_episode.outcome == ViabilityOutcome::CandidateWin),
        );
        candidate_cell_differences.push(
            candidate_episode.candidate_cells as f64 - baseline_episode.candidate_cells as f64,
        );
        opponent_cell_differences
            .push(candidate_episode.opponent_cells as f64 - baseline_episode.opponent_cells as f64);
        step_differences.push(
            candidate_episode.environment_steps as f64 - baseline_episode.environment_steps as f64,
        );
        quanta_differences.push(
            candidate_episode.final_sim_time_quanta as f64
                - baseline_episode.final_sim_time_quanta as f64,
        );
    }

    Ok(PairedViabilityComparison {
        baseline_variant: baseline_variant.to_string(),
        candidate_variant: candidate_variant.to_string(),
        candidate_profile: candidate.candidate,
        opponent_profile: candidate.opponent,
        seeds: candidate.seeds.clone(),
        improved_outcomes,
        unchanged_outcomes,
        regressed_outcomes,
        candidate_win_difference: summarize(&win_differences),
        final_candidate_cell_difference: summarize(&candidate_cell_differences),
        final_opponent_cell_difference: summarize(&opponent_cell_differences),
        environment_step_difference: summarize(&step_differences),
        final_sim_time_quanta_difference: summarize(&quanta_differences),
    })
}

/// Validate matrix completeness, nested report identities, and every derived
/// paired comparison before the report is used by an automated gate.
pub fn validate_viability_matrix_report(report: &ViabilityMatrixReport) -> Result<(), String> {
    if report.schema_version != VIABILITY_MATRIX_SCHEMA_VERSION {
        return Err(format!(
            "unsupported viability matrix schema {}; expected {}",
            report.schema_version, VIABILITY_MATRIX_SCHEMA_VERSION
        ));
    }
    if report.mind_abi_version != blob_interface::abi::REFERENCE_MIND_ABI_VERSION
        || report.mind_abi_hash != crate::viability::mind_abi_hash()
    {
        return Err("viability matrix Mind ABI identity mismatch".into());
    }
    if canonical_profiles("candidate profile", &report.candidates)? != report.candidates
        || canonical_profiles("opponent profile", &report.opponents)? != report.opponents
    {
        return Err("viability matrix profiles are not in canonical order".into());
    }
    if report.variants.is_empty() || report.seeds.is_empty() {
        return Err("viability matrix requires variants and seeds".into());
    }
    let mut variant_names = HashSet::with_capacity(report.variants.len());
    if report
        .variants
        .iter()
        .any(|variant| !variant_names.insert(variant.name.as_str()))
        || !variant_names.contains(report.baseline_variant.as_str())
    {
        return Err("viability matrix variants are duplicated or omit the baseline".into());
    }
    let expected_matchups =
        report.variants.len() * report.candidates.len() * report.opponents.len();
    if report.matchups.len() != expected_matchups {
        return Err("viability matrix matchup grid is incomplete".into());
    }
    let variants = report
        .variants
        .iter()
        .map(|variant| (variant.name.as_str(), variant))
        .collect::<HashMap<_, _>>();
    let mut matchup_keys = HashSet::with_capacity(report.matchups.len());
    let mut by_matchup = HashMap::with_capacity(report.matchups.len());
    for matchup in &report.matchups {
        validate_viability_report(&matchup.report)
            .map_err(|error| format!("variant {}: {error}", matchup.variant))?;
        let variant = variants
            .get(matchup.variant.as_str())
            .ok_or_else(|| format!("unknown viability matrix variant {}", matchup.variant))?;
        let key = (
            matchup.variant.as_str(),
            matchup.report.candidate,
            matchup.report.opponent,
        );
        if !matchup_keys.insert(key)
            || !report.candidates.contains(&matchup.report.candidate)
            || !report.opponents.contains(&matchup.report.opponent)
            || matchup.report.seeds != report.seeds
            || matchup.report.semantic_ruleset_hash != variant.semantic_ruleset_hash
            || matchup.report.compiled_ruleset_hash != variant.compiled_ruleset_hash
            || matchup.report.scenario_hash != variant.scenario_hash
        {
            return Err(format!(
                "variant {} matchup identity does not match the matrix",
                matchup.variant
            ));
        }
        by_matchup.insert(key, &matchup.report);
    }

    let mut expected_comparisons = Vec::with_capacity(
        (report.variants.len() - 1) * report.candidates.len() * report.opponents.len(),
    );
    for variant in report
        .variants
        .iter()
        .filter(|variant| variant.name != report.baseline_variant)
    {
        for &candidate in &report.candidates {
            for &opponent in &report.opponents {
                let baseline = by_matchup
                    .get(&(report.baseline_variant.as_str(), candidate, opponent))
                    .ok_or("viability matrix is missing a baseline matchup")?;
                let candidate_report = by_matchup
                    .get(&(variant.name.as_str(), candidate, opponent))
                    .ok_or("viability matrix is missing a candidate matchup")?;
                expected_comparisons.push(paired_comparison(
                    &report.baseline_variant,
                    &variant.name,
                    baseline,
                    candidate_report,
                )?);
            }
        }
    }
    if expected_comparisons != report.paired_comparisons {
        return Err("viability matrix paired comparisons are inconsistent".into());
    }
    let expected_hash = crate::sweep::sha256(
        &serde_json::to_vec(&MatrixIdentity {
            mind_abi_version: report.mind_abi_version,
            mind_abi_hash: &report.mind_abi_hash,
            manifest_sha256: &report.manifest_sha256,
            baseline_variant: &report.baseline_variant,
            candidates: &report.candidates,
            opponents: &report.opponents,
        })
        .map_err(|error| format!("failed to encode viability matrix identity: {error}"))?,
    );
    if expected_hash != report.matrix_config_hash {
        return Err("viability matrix configuration hash mismatch".into());
    }
    Ok(())
}

/// Load and fully validate a bounded matrix report for restart-safe reuse.
pub fn load_viability_matrix_report(path: &Path) -> Result<ViabilityMatrixReport, String> {
    let length = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?
        .len();
    if length > MAX_VIABILITY_MATRIX_BYTES {
        return Err(format!(
            "{} is {length} bytes; viability matrix limit is {MAX_VIABILITY_MATRIX_BYTES}",
            path.display()
        ));
    }
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let report: ViabilityMatrixReport = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    validate_viability_matrix_report(&report)?;
    Ok(report)
}

/// Execute every variant/profile matchup using the manifest's paired seed
/// suite. Results are restored to canonical order after bounded parallel work.
pub fn run_viability_matrix(
    manifest_file: &Path,
    options: &ViabilityMatrixOptions,
) -> Result<ViabilityMatrixReport, String> {
    if options.max_parallel == 0 {
        return Err("viability matrix max_parallel must be positive".into());
    }
    let candidates = canonical_profiles("candidate profile", &options.candidates)?;
    let opponents = canonical_profiles("opponent profile", &options.opponents)?;
    let sweep = load_validated_sweep(manifest_file)?;
    let baseline_variant = options
        .baseline_variant
        .clone()
        .unwrap_or_else(|| sweep.manifest.variants[0].name.clone());
    if !sweep
        .manifest
        .variants
        .iter()
        .any(|variant| variant.name == baseline_variant)
    {
        return Err(format!(
            "baseline variant {baseline_variant} is not present in the sweep manifest"
        ));
    }

    let mut variant_configs = Vec::with_capacity(sweep.manifest.variants.len());
    let mut variants = Vec::with_capacity(sweep.manifest.variants.len());
    for variant in &sweep.manifest.variants {
        let members = sweep
            .manifest
            .runs
            .iter()
            .zip(&sweep.configs)
            .filter(|(run, _)| run.variant == variant.name)
            .collect::<Vec<_>>();
        let (representative_run, representative_config) = members
            .first()
            .copied()
            .ok_or_else(|| format!("variant {} has no expanded configs", variant.name))?;
        if members.iter().any(|(run, config)| {
            run.semantic_ruleset_hash != representative_run.semantic_ruleset_hash
                || run.compiled_ruleset_hash != representative_run.compiled_ruleset_hash
                || run.scenario_hash != representative_run.scenario_hash
                || config.env != representative_config.env
                || config.telemetry != representative_config.telemetry
        }) {
            return Err(format!(
                "variant {} changes environment or telemetry across paired seeds",
                variant.name
            ));
        }
        variant_configs.push((
            variant.name.clone(),
            representative_config.clone(),
            representative_run.semantic_ruleset_hash.clone(),
            representative_run.compiled_ruleset_hash.clone(),
            representative_run.scenario_hash.clone(),
        ));
        variants.push(ViabilityMatrixVariant {
            name: variant.name.clone(),
            description: variant.description.clone(),
            semantic_ruleset_hash: representative_run.semantic_ruleset_hash.clone(),
            compiled_ruleset_hash: representative_run.compiled_ruleset_hash.clone(),
            scenario_hash: representative_run.scenario_hash.clone(),
        });
    }

    let mut jobs = Vec::with_capacity(variants.len() * candidates.len() * opponents.len());
    for (variant, config, semantic_ruleset_hash, compiled_ruleset_hash, scenario_hash) in
        variant_configs
    {
        for &candidate in &candidates {
            for &opponent in &opponents {
                jobs.push(MatrixJob {
                    variant: variant.clone(),
                    config: config.clone(),
                    semantic_ruleset_hash: semantic_ruleset_hash.clone(),
                    compiled_ruleset_hash: compiled_ruleset_hash.clone(),
                    scenario_hash: scenario_hash.clone(),
                    candidate,
                    opponent,
                });
            }
        }
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.max_parallel)
        .build()
        .map_err(|error| format!("failed to create viability worker pool: {error}"))?;
    let results = pool.install(|| {
        jobs.par_iter()
            .map(|job| {
                run_baseline_viability(
                    &job.config.env,
                    &job.config.telemetry,
                    job.candidate,
                    job.opponent,
                    &sweep.manifest.seeds,
                )
                .and_then(|report| {
                    if report.semantic_ruleset_hash != job.semantic_ruleset_hash
                        || report.compiled_ruleset_hash != job.compiled_ruleset_hash
                        || report.scenario_hash != job.scenario_hash
                    {
                        return Err(
                            "viability report identity differs from its verified variant"
                                .to_string(),
                        );
                    }
                    Ok(ViabilityMatrixMatchup {
                        variant: job.variant.clone(),
                        report,
                    })
                })
                .map_err(|error| {
                    format!(
                        "variant {} matchup {} vs {} failed: {error}",
                        job.variant, job.candidate, job.opponent
                    )
                })
            })
            .collect::<Vec<_>>()
    });
    let matchups = results.into_iter().collect::<Result<Vec<_>, _>>()?;

    let by_matchup = matchups
        .iter()
        .map(|matchup| {
            (
                (
                    matchup.variant.as_str(),
                    matchup.report.candidate,
                    matchup.report.opponent,
                ),
                &matchup.report,
            )
        })
        .collect::<HashMap<_, _>>();
    let mut paired_comparisons =
        Vec::with_capacity((variants.len() - 1) * candidates.len() * opponents.len());
    for variant in variants
        .iter()
        .filter(|variant| variant.name != baseline_variant)
    {
        for &candidate in &candidates {
            for &opponent in &opponents {
                let baseline = by_matchup
                    .get(&(baseline_variant.as_str(), candidate, opponent))
                    .expect("complete canonical job matrix contains baseline");
                let report = by_matchup
                    .get(&(variant.name.as_str(), candidate, opponent))
                    .expect("complete canonical job matrix contains candidate");
                paired_comparisons.push(paired_comparison(
                    &baseline_variant,
                    &variant.name,
                    baseline,
                    report,
                )?);
            }
        }
    }

    let mind_abi_version = blob_interface::abi::REFERENCE_MIND_ABI_VERSION;
    let mind_abi_hash = crate::viability::mind_abi_hash();
    let matrix_config_hash = crate::sweep::sha256(
        &serde_json::to_vec(&MatrixIdentity {
            mind_abi_version,
            mind_abi_hash: &mind_abi_hash,
            manifest_sha256: &sweep.manifest_sha256,
            baseline_variant: &baseline_variant,
            candidates: &candidates,
            opponents: &opponents,
        })
        .map_err(|error| format!("failed to encode viability matrix identity: {error}"))?,
    );
    Ok(ViabilityMatrixReport {
        schema_version: VIABILITY_MATRIX_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        mind_abi_version,
        mind_abi_hash,
        manifest_sha256: sweep.manifest_sha256,
        matrix_config_hash,
        baseline_variant,
        candidates,
        opponents,
        seeds: sweep.manifest.seeds,
        variants,
        matchups,
        paired_comparisons,
    })
}

/// Atomically publish one matrix report while refusing replacement.
pub fn publish_viability_matrix_report(
    output: &Path,
    report: &ViabilityMatrixReport,
) -> Result<PathBuf, String> {
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable viability matrix {}",
            output.display()
        ));
    }
    let nonce = MATRIX_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "viability matrix output needs a UTF-8 file name".to_string())?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode viability matrix: {error}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweep::publish_rules_sweep;

    fn planned_sweep() -> (tempfile::TempDir, PathBuf) {
        let temporary = tempfile::tempdir().unwrap();
        fs::write(
            temporary.path().join("base.toml"),
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
            sim_time_limit_quanta = 8192

            [self_play]
            max_opponent_pool = 0
            "#,
        )
        .unwrap();
        let spec = temporary.path().join("sweep.toml");
        fs::write(
            &spec,
            r#"
            base_config = "base.toml"
            output_dir = "planned"
            seeds = [11, 12, 13]

            [[variants]]
            name = "baseline"

            [[variants]]
            name = "faster-digestion"
            [variants.rules]
            digestion_rate_numerator = 2
            "#,
        )
        .unwrap();
        let manifest = publish_rules_sweep(&spec).unwrap();
        let manifest_file = PathBuf::from(manifest.output_directory).join("manifest.json");
        (temporary, manifest_file)
    }

    fn options(max_parallel: usize) -> ViabilityMatrixOptions {
        ViabilityMatrixOptions {
            candidates: vec![OpponentProfile::Forager, OpponentProfile::Random],
            opponents: vec![OpponentProfile::Wait],
            baseline_variant: None,
            max_parallel,
        }
    }

    #[test]
    fn matrix_is_reproducible_across_parallelism_and_fully_paired() {
        let (_temporary, manifest) = planned_sweep();
        let serial = run_viability_matrix(&manifest, &options(1)).unwrap();
        let parallel = run_viability_matrix(&manifest, &options(2)).unwrap();
        let mut reordered_options = options(3);
        reordered_options.candidates.reverse();
        let reordered = run_viability_matrix(&manifest, &reordered_options).unwrap();
        assert_eq!(serial, parallel);
        assert_eq!(serial, reordered);
        assert_eq!(serial.baseline_variant, "baseline");
        assert_eq!(serial.matchups.len(), 4);
        assert_eq!(serial.paired_comparisons.len(), 2);
        assert!(serial.paired_comparisons.iter().all(|comparison| {
            comparison.seeds == vec![11, 12, 13]
                && comparison.improved_outcomes
                    + comparison.unchanged_outcomes
                    + comparison.regressed_outcomes
                    == 3
                && comparison.candidate_win_difference.samples == 3
        }));
        validate_viability_matrix_report(&serial).unwrap();
        let mut tampered = serial;
        tampered.paired_comparisons[0].regressed_outcomes += 1;
        assert!(validate_viability_matrix_report(&tampered)
            .unwrap_err()
            .contains("paired comparisons"));
    }

    #[test]
    fn matrix_validates_options_and_publishes_immutably() {
        let (temporary, manifest) = planned_sweep();
        let mut invalid = options(0);
        assert!(run_viability_matrix(&manifest, &invalid)
            .unwrap_err()
            .contains("max_parallel"));
        invalid.max_parallel = 1;
        invalid.baseline_variant = Some("missing".into());
        assert!(run_viability_matrix(&manifest, &invalid)
            .unwrap_err()
            .contains("not present"));
        invalid.baseline_variant = None;
        invalid.candidates.push(OpponentProfile::Forager);
        assert!(run_viability_matrix(&manifest, &invalid)
            .unwrap_err()
            .contains("must be unique"));

        let report = run_viability_matrix(&manifest, &options(2)).unwrap();
        let output = temporary.path().join("matrix.json");
        publish_viability_matrix_report(&output, &report).unwrap();
        let bytes = fs::read(&output).unwrap();
        assert!(publish_viability_matrix_report(&output, &report).is_err());
        assert_eq!(fs::read(&output).unwrap(), bytes);
        let decoded: ViabilityMatrixReport = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded.matrix_config_hash, report.matrix_config_hash);
    }
}
