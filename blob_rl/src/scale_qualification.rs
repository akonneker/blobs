//! Immutable, bounded qualification of ecological behavior across world scales.
//!
//! Large worlds should preserve the intended ecological regime without making
//! every pre-training check execute a complete match. This module runs the
//! deterministic ecological characterization at each declared scale and gates
//! the scale relationships that are meant to remain invariant.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::TrainingConfig;
use crate::ecological_characterization::{
    characterize_ecology, validate_ecological_characterization, EcologicalCharacterizationOptions,
    EcologicalCharacterizationReport,
};
use crate::sweep::sha256;
use crate::viability::hash_json;

pub const SCALE_QUALIFICATION_SCHEMA_VERSION: u32 = 1;
const MAX_SPEC_BYTES: u64 = 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 4 * 1024 * 1024;
const MAX_REPORT_BYTES: u64 = 256 * 1024 * 1024;
static QUALIFICATION_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ScaleProfileSpec {
    pub name: String,
    pub config: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScaleQualificationSpec {
    pub seeds: Vec<u64>,
    pub max_micro_actions: usize,
    #[serde(default = "default_max_parallel_profiles")]
    pub max_parallel_profiles: usize,
    pub minimum_world_size: usize,
    pub maximum_population_density_relative_deviation: f64,
    pub maximum_plant_density_relative_deviation: f64,
    pub maximum_loose_energy_density_relative_deviation: f64,
    pub maximum_deadline_per_width_relative_deviation: f64,
    #[serde(default = "default_true")]
    pub require_identical_rules: bool,
    #[serde(default = "default_true")]
    pub require_all_starting_cells_reach_plants: bool,
    #[serde(default = "default_true")]
    pub require_team_p90_plant_routes_within_travel_endurance: bool,
    pub profiles: Vec<ScaleProfileSpec>,
}

const fn default_max_parallel_profiles() -> usize {
    1
}

const fn default_true() -> bool {
    true
}

impl ScaleQualificationSpec {
    pub fn validate(&self) -> Result<(), String> {
        EcologicalCharacterizationOptions {
            seeds: self.seeds.clone(),
            max_micro_actions: self.max_micro_actions,
        }
        .validate()?;
        if self.max_parallel_profiles == 0 || self.minimum_world_size == 0 {
            return Err("scale qualification limits must be positive".into());
        }
        for value in [
            self.maximum_population_density_relative_deviation,
            self.maximum_plant_density_relative_deviation,
            self.maximum_loose_energy_density_relative_deviation,
            self.maximum_deadline_per_width_relative_deviation,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(
                    "scale qualification relative deviations must be finite and nonnegative".into(),
                );
            }
        }
        if self.profiles.is_empty() {
            return Err("scale qualification requires at least one profile".into());
        }
        let mut names = HashSet::with_capacity(self.profiles.len());
        for profile in &self.profiles {
            if !valid_name(&profile.name) || profile.config.trim().is_empty() {
                return Err("scale qualification profile names or paths are invalid".into());
            }
            if !names.insert(&profile.name) {
                return Err("scale qualification profile names must be unique".into());
            }
        }
        Ok(())
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().enumerate().all(|(index, byte)| match byte {
            b'a'..=b'z' | b'0'..=b'9' => true,
            b'-' | b'_' => index > 0,
            _ => false,
        })
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScaleQualificationComparator {
    AtLeast,
    AtMost,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScaleQualificationDecision {
    Passed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScaleProfileMetrics {
    pub world_size: usize,
    pub cells_per_team: usize,
    pub population_density_per_team: f64,
    pub plant_density: f64,
    pub loose_energy_source_density: f64,
    pub deadline_quanta_per_world_width: f64,
    pub reachable_starting_cell_fraction: f64,
    pub best_travel_endurance_units: f64,
    pub minimum_team_p90_plant_endurance_margin_units: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScaleProfileQualification {
    pub name: String,
    pub config_file: String,
    pub config_sha256: String,
    pub metrics: ScaleProfileMetrics,
    pub characterization: EcologicalCharacterizationReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScaleQualificationCheck {
    pub profile: Option<String>,
    pub metric: String,
    pub observed: Option<f64>,
    pub comparator: ScaleQualificationComparator,
    pub threshold: f64,
    pub passed: bool,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScaleQualificationReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub spec_file_sha256: String,
    pub qualification_hash: String,
    pub spec: ScaleQualificationSpec,
    pub profiles: Vec<ScaleProfileQualification>,
    pub checks: Vec<ScaleQualificationCheck>,
    pub failed_checks: usize,
    pub decision: ScaleQualificationDecision,
}

#[derive(Serialize)]
struct QualificationIdentity<'a> {
    spec_file_sha256: &'a str,
    spec: &'a ScaleQualificationSpec,
    profiles: Vec<ProfileIdentity<'a>>,
}

#[derive(Serialize)]
struct ProfileIdentity<'a> {
    name: &'a str,
    config_sha256: &'a str,
    characterization_hash: &'a str,
    characterization_results_hash: &'a str,
}

fn relative_deviation(value: f64, reference: f64) -> f64 {
    if reference == 0.0 {
        if value == 0.0 {
            0.0
        } else {
            f64::MAX
        }
    } else {
        (value - reference).abs() / reference.abs()
    }
}

fn profile_metrics(report: &EcologicalCharacterizationReport) -> ScaleProfileMetrics {
    let scenario = &report.scenario;
    let area = (scenario.world_size as f64).powi(2);
    let total_starting = report
        .resource_access
        .teams
        .iter()
        .map(|team| team.starting_cell_samples)
        .sum::<usize>();
    let total_unreachable = report
        .resource_access
        .teams
        .iter()
        .map(|team| {
            team.distance_to_plants
                .weighted_distance_units
                .unreachable_samples
        })
        .sum::<usize>();
    let reachable_starting_cell_fraction = if total_starting == 0 {
        0.0
    } else {
        (total_starting - total_unreachable) as f64 / total_starting as f64
    };
    let best_travel_endurance_units = report
        .travel
        .iter()
        .filter(|travel| !travel.censored)
        .map(|travel| travel.completed_distance_q10 as f64 / 1024.0)
        .fold(0.0_f64, f64::max);
    let minimum_team_p90_plant_endurance_margin_units = report
        .resource_access
        .teams
        .iter()
        .map(|team| team.distance_to_plants.weighted_distance_units.p90)
        .collect::<Option<Vec<_>>>()
        .and_then(|distances| {
            distances
                .into_iter()
                .map(|distance| best_travel_endurance_units - distance)
                .reduce(f64::min)
        });
    ScaleProfileMetrics {
        world_size: scenario.world_size,
        cells_per_team: scenario.cells_per_team,
        population_density_per_team: scenario.cells_per_team as f64 / area,
        plant_density: scenario.num_plants as f64 / area,
        loose_energy_source_density: scenario.num_scattered_energy as f64 / area,
        deadline_quanta_per_world_width: scenario.victory.sim_time_limit_quanta as f64
            / scenario.world_size as f64,
        reachable_starting_cell_fraction,
        best_travel_endurance_units,
        minimum_team_p90_plant_endurance_margin_units,
    }
}

fn check(
    profile: Option<&str>,
    metric: &str,
    observed: Option<f64>,
    comparator: ScaleQualificationComparator,
    threshold: f64,
    rationale: &str,
) -> ScaleQualificationCheck {
    let passed = observed.is_some_and(|observed| {
        observed.is_finite()
            && match comparator {
                ScaleQualificationComparator::AtLeast => observed >= threshold,
                ScaleQualificationComparator::AtMost => observed <= threshold,
            }
    });
    ScaleQualificationCheck {
        profile: profile.map(str::to_string),
        metric: metric.to_string(),
        observed,
        comparator,
        threshold,
        passed,
        rationale: rationale.to_string(),
    }
}

fn build_checks(
    spec: &ScaleQualificationSpec,
    profiles: &[ScaleProfileQualification],
) -> Vec<ScaleQualificationCheck> {
    let reference = &profiles[0];
    let mut checks = Vec::new();
    for profile in profiles {
        let name = Some(profile.name.as_str());
        checks.push(check(
            name,
            "minimum_world_size",
            Some(profile.metrics.world_size as f64),
            ScaleQualificationComparator::AtLeast,
            spec.minimum_world_size as f64,
            "Every promoted learning profile must meet the declared board-size floor.",
        ));
        checks.push(check(
            name,
            "population_density_relative_deviation",
            Some(relative_deviation(
                profile.metrics.population_density_per_team,
                reference.metrics.population_density_per_team,
            )),
            ScaleQualificationComparator::AtMost,
            spec.maximum_population_density_relative_deviation,
            "Population density must not change merely because the board is larger.",
        ));
        checks.push(check(
            name,
            "plant_density_relative_deviation",
            Some(relative_deviation(
                profile.metrics.plant_density,
                reference.metrics.plant_density,
            )),
            ScaleQualificationComparator::AtMost,
            spec.maximum_plant_density_relative_deviation,
            "Plant encounter frequency should remain comparable across scales.",
        ));
        checks.push(check(
            name,
            "loose_energy_density_relative_deviation",
            Some(relative_deviation(
                profile.metrics.loose_energy_source_density,
                reference.metrics.loose_energy_source_density,
            )),
            ScaleQualificationComparator::AtMost,
            spec.maximum_loose_energy_density_relative_deviation,
            "Loose-energy abundance should remain comparable across scales.",
        ));
        checks.push(check(
            name,
            "deadline_per_width_relative_deviation",
            Some(relative_deviation(
                profile.metrics.deadline_quanta_per_world_width,
                reference.metrics.deadline_quanta_per_world_width,
            )),
            ScaleQualificationComparator::AtMost,
            spec.maximum_deadline_per_width_relative_deviation,
            "Canonical match time should scale with traversal distance, not cell count.",
        ));
        if spec.require_all_starting_cells_reach_plants {
            checks.push(check(
                name,
                "reachable_starting_cell_fraction",
                Some(profile.metrics.reachable_starting_cell_fraction),
                ScaleQualificationComparator::AtLeast,
                1.0,
                "Every starting cell must have a movement-valid route to a plant.",
            ));
        }
        if spec.require_team_p90_plant_routes_within_travel_endurance {
            checks.push(check(
                name,
                "minimum_team_p90_plant_endurance_margin_units",
                profile
                    .metrics
                    .minimum_team_p90_plant_endurance_margin_units,
                ScaleQualificationComparator::AtLeast,
                0.0,
                "Each team's p90 plant route must fit within canonical no-food endurance.",
            ));
        }
        if spec.require_identical_rules {
            checks.push(check(
                name,
                "rules_match_reference_profile",
                Some(f64::from(
                    profile.characterization.semantic_ruleset_hash
                        == reference.characterization.semantic_ruleset_hash,
                )),
                ScaleQualificationComparator::AtLeast,
                1.0,
                "A scale qualification must vary world scale rather than semantic physics; compiled hashes remain dimension-specific.",
            ));
        }
    }
    checks
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn qualification_hash(
    spec_file_sha256: &str,
    spec: &ScaleQualificationSpec,
    profiles: &[ScaleProfileQualification],
) -> Result<String, String> {
    hash_json(&QualificationIdentity {
        spec_file_sha256,
        spec,
        profiles: profiles
            .iter()
            .map(|profile| ProfileIdentity {
                name: &profile.name,
                config_sha256: &profile.config_sha256,
                characterization_hash: &profile.characterization.characterization_hash,
                characterization_results_hash: &profile.characterization.results_hash,
            })
            .collect(),
    })
}

pub fn validate_scale_qualification(report: &ScaleQualificationReport) -> Result<(), String> {
    if report.schema_version != SCALE_QUALIFICATION_SCHEMA_VERSION {
        return Err(format!(
            "unsupported scale qualification schema {}; expected {}",
            report.schema_version, SCALE_QUALIFICATION_SCHEMA_VERSION
        ));
    }
    report.spec.validate()?;
    if !valid_sha256(&report.spec_file_sha256) {
        return Err("scale qualification spec SHA-256 is invalid".into());
    }
    if report.profiles.len() != report.spec.profiles.len() {
        return Err("scale qualification profile count mismatch".into());
    }
    let mut previous_world_size = None;
    for (profile, declared) in report.profiles.iter().zip(&report.spec.profiles) {
        if profile.name != declared.name || !valid_sha256(&profile.config_sha256) {
            return Err("scale qualification profile identity mismatch".into());
        }
        validate_ecological_characterization(&profile.characterization)?;
        if profile.metrics != profile_metrics(&profile.characterization) {
            return Err(format!(
                "scale qualification metrics mismatch for {}",
                profile.name
            ));
        }
        if previous_world_size.is_some_and(|previous| previous >= profile.metrics.world_size) {
            return Err("scale qualification profiles must have increasing world sizes".into());
        }
        previous_world_size = Some(profile.metrics.world_size);
    }
    let expected_checks = build_checks(&report.spec, &report.profiles);
    if report.checks != expected_checks {
        return Err("scale qualification checks are inconsistent".into());
    }
    let failed_checks = report.checks.iter().filter(|check| !check.passed).count();
    let decision = if failed_checks == 0 {
        ScaleQualificationDecision::Passed
    } else {
        ScaleQualificationDecision::Failed
    };
    if report.failed_checks != failed_checks || report.decision != decision {
        return Err("scale qualification decision is inconsistent".into());
    }
    if qualification_hash(&report.spec_file_sha256, &report.spec, &report.profiles)?
        != report.qualification_hash
    {
        return Err("scale qualification hash mismatch".into());
    }
    Ok(())
}

pub fn load_scale_qualification(path: &Path) -> Result<ScaleQualificationReport, String> {
    let bytes = read_bounded(path, MAX_REPORT_BYTES, "scale qualification")?;
    let report: ScaleQualificationReport = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    validate_scale_qualification(&report)?;
    Ok(report)
}

#[derive(Clone)]
struct QualificationJob {
    profile: ScaleProfileSpec,
    config_file: PathBuf,
    config_sha256: String,
    config: TrainingConfig,
}

fn read_bounded(path: &Path, maximum: u64, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {label} {}: {error}", path.display()))?;
    if metadata.len() > maximum {
        return Err(format!("{label} {} is too large", path.display()));
    }
    fs::read(path).map_err(|error| format!("failed to read {label} {}: {error}", path.display()))
}

pub fn run_scale_qualification(spec_file: &Path) -> Result<ScaleQualificationReport, String> {
    let spec_file = fs::canonicalize(spec_file).map_err(|error| {
        format!(
            "failed to resolve scale qualification spec {}: {error}",
            spec_file.display()
        )
    })?;
    let spec_bytes = read_bounded(&spec_file, MAX_SPEC_BYTES, "qualification spec")?;
    let spec: ScaleQualificationSpec = toml::from_str(
        std::str::from_utf8(&spec_bytes)
            .map_err(|error| format!("qualification spec is not UTF-8: {error}"))?,
    )
    .map_err(|error| format!("failed to parse qualification spec: {error}"))?;
    spec.validate()?;
    let spec_directory = spec_file.parent().unwrap_or_else(|| Path::new("."));
    let mut jobs = Vec::with_capacity(spec.profiles.len());
    for profile in &spec.profiles {
        let requested = Path::new(&profile.config);
        let config_file = fs::canonicalize(if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            spec_directory.join(requested)
        })
        .map_err(|error| {
            format!(
                "failed to resolve scale profile {} config: {error}",
                profile.name
            )
        })?;
        let config_bytes = read_bounded(&config_file, MAX_CONFIG_BYTES, "training config")?;
        let config = TrainingConfig::from_toml_str(
            std::str::from_utf8(&config_bytes)
                .map_err(|error| format!("training config is not UTF-8: {error}"))?,
        )?;
        config.validate()?;
        jobs.push(QualificationJob {
            profile: profile.clone(),
            config_file,
            config_sha256: sha256(&config_bytes),
            config,
        });
    }
    for pair in jobs.windows(2) {
        if pair[0].config.env.world_size >= pair[1].config.env.world_size {
            return Err("scale qualification profiles must have increasing world sizes".into());
        }
    }
    let ecological_options = EcologicalCharacterizationOptions {
        seeds: spec.seeds.clone(),
        max_micro_actions: spec.max_micro_actions,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(spec.max_parallel_profiles.min(jobs.len()))
        .build()
        .map_err(|error| format!("failed to create qualification worker pool: {error}"))?;
    let characterized = pool.install(|| {
        jobs.par_iter()
            .map(|job| {
                let characterization =
                    characterize_ecology(&job.config.env, &ecological_options)
                        .map_err(|error| format!("profile {} failed: {error}", job.profile.name))?;
                let metrics = profile_metrics(&characterization);
                Ok(ScaleProfileQualification {
                    name: job.profile.name.clone(),
                    config_file: job.config_file.to_string_lossy().into_owned(),
                    config_sha256: job.config_sha256.clone(),
                    metrics,
                    characterization,
                })
            })
            .collect::<Result<Vec<_>, String>>()
    })?;
    let checks = build_checks(&spec, &characterized);
    let failed_checks = checks.iter().filter(|check| !check.passed).count();
    let decision = if failed_checks == 0 {
        ScaleQualificationDecision::Passed
    } else {
        ScaleQualificationDecision::Failed
    };
    let spec_file_sha256 = sha256(&spec_bytes);
    let qualification_hash = qualification_hash(&spec_file_sha256, &spec, &characterized)?;
    let report = ScaleQualificationReport {
        schema_version: SCALE_QUALIFICATION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        spec_file_sha256,
        qualification_hash,
        spec,
        profiles: characterized,
        checks,
        failed_checks,
        decision,
    };
    validate_scale_qualification(&report)?;
    Ok(report)
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to sync directory {}: {error}", path.display()))
}

pub fn publish_scale_qualification(
    output: &Path,
    report: &ScaleQualificationReport,
) -> Result<PathBuf, String> {
    validate_scale_qualification(report)?;
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode scale qualification: {error}"))?;
    if bytes.len() as u64 > MAX_REPORT_BYTES {
        return Err("scale qualification report is too large".into());
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable scale qualification {}",
            output.display()
        ));
    }
    let nonce = QUALIFICATION_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "scale qualification output needs a UTF-8 name".to_string())?;
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
    use tempfile::tempdir;

    fn write_profile(path: &Path, world_size: usize, scale: usize) {
        let mut config = TrainingConfig::default();
        config.env.world_size = world_size;
        config.env.cells_per_team = scale;
        config.env.num_plants = scale;
        config.env.num_scattered_energy = scale * 2;
        config.env.victory.sim_time_limit_quanta = world_size as u64 * 1024;
        config.env.max_episode_len = 1_000;
        config.validate().unwrap();
        fs::write(path, toml::to_string_pretty(&config).unwrap()).unwrap();
    }

    fn write_spec(directory: &Path, density_tolerance: f64) -> PathBuf {
        let path = directory.join("scale.toml");
        fs::write(
            &path,
            format!(
                r#"seeds = [11]
max_micro_actions = 1000
max_parallel_profiles = 2
minimum_world_size = 8
maximum_population_density_relative_deviation = {density_tolerance}
maximum_plant_density_relative_deviation = {density_tolerance}
maximum_loose_energy_density_relative_deviation = {density_tolerance}
maximum_deadline_per_width_relative_deviation = 0.0
require_identical_rules = true
require_all_starting_cells_reach_plants = true
require_team_p90_plant_routes_within_travel_endurance = true

[[profiles]]
name = "small"
config = "small.toml"

[[profiles]]
name = "large"
config = "large.toml"
"#
            ),
        )
        .unwrap();
        path
    }

    #[test]
    fn qualification_is_deterministic_and_self_validating() {
        let directory = tempdir().unwrap();
        write_profile(&directory.path().join("small.toml"), 8, 1);
        write_profile(&directory.path().join("large.toml"), 16, 4);
        let spec = write_spec(directory.path(), 0.0);
        let left = run_scale_qualification(&spec).unwrap();
        let right = run_scale_qualification(&spec).unwrap();
        assert_eq!(left, right);
        assert_eq!(
            left.decision,
            ScaleQualificationDecision::Passed,
            "failed checks: {:?}",
            left.checks
                .iter()
                .filter(|check| !check.passed)
                .collect::<Vec<_>>()
        );
        validate_scale_qualification(&left).unwrap();

        let output = directory.path().join("qualification.json");
        publish_scale_qualification(&output, &left).unwrap();
        assert!(publish_scale_qualification(&output, &left)
            .unwrap_err()
            .contains("refusing to replace"));
    }

    #[test]
    fn density_drift_fails_without_invalidating_evidence() {
        let directory = tempdir().unwrap();
        write_profile(&directory.path().join("small.toml"), 8, 1);
        write_profile(&directory.path().join("large.toml"), 16, 3);
        let spec = write_spec(directory.path(), 0.01);
        let report = run_scale_qualification(&spec).unwrap();
        assert_eq!(report.decision, ScaleQualificationDecision::Failed);
        assert!(report.checks.iter().any(|check| {
            check.profile.as_deref() == Some("large")
                && check.metric == "population_density_relative_deviation"
                && !check.passed
        }));
        validate_scale_qualification(&report).unwrap();
    }

    #[test]
    fn tampering_with_nested_metrics_is_rejected() {
        let directory = tempdir().unwrap();
        write_profile(&directory.path().join("small.toml"), 8, 1);
        write_profile(&directory.path().join("large.toml"), 16, 4);
        let spec = write_spec(directory.path(), 0.0);
        let mut report = run_scale_qualification(&spec).unwrap();
        report.profiles[0].metrics.plant_density += 1.0;
        assert!(validate_scale_qualification(&report)
            .unwrap_err()
            .contains("metrics mismatch"));
    }
}
