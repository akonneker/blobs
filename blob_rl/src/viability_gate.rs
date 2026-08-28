//! Deterministic pass/fail gates over immutable viability-matrix reports.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::config::OpponentProfile;
use crate::sweep::sha256;
use crate::telemetry::{ActionFamilyTelemetry, SideTelemetry};
use crate::viability::ViabilityReport;
use crate::viability_matrix::{
    validate_viability_matrix_report, PairedViabilityComparison, ViabilityMatrixReport,
    ViabilityMatrixVariant,
};

pub const VIABILITY_GATE_SCHEMA_VERSION: u32 = 2;
const MAX_MATRIX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_GATE_SPEC_BYTES: u64 = 1024 * 1024;
const MAX_GATE_REPORT_BYTES: u64 = 16 * 1024 * 1024;
static GATE_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct AbsoluteViabilityGates {
    pub max_timeout_rate: Option<f64>,
    pub max_extinction_without_combat_rate: Option<f64>,
    pub max_mass_energy_drift_episode_rate: Option<f64>,
    pub min_candidate_survival_rate: Option<f64>,
    pub min_attack_commits_per_episode_for_combat_profiles: Option<f64>,
    pub min_applied_damage_per_episode_for_combat_profiles: Option<f64>,
    pub max_commit_rejection_rate: Option<f64>,
    pub max_frustrated_resolution_rate: Option<f64>,
    pub min_forager_plant_occupancy_rate: Option<f64>,
    pub min_forager_consumed_energy_per_episode: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PairedViabilityGates {
    pub max_regressed_outcome_rate: Option<f64>,
    pub max_candidate_win_rate_drop: Option<f64>,
    pub max_final_candidate_cell_mean_drop: Option<f64>,
    pub max_timeout_rate_increase: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct GeometryViabilityGates {
    /// Variants listed here still receive reachability/endurance checks but
    /// are exempt from cross-team symmetry thresholds. Names are exact and
    /// must exist in the bound matrix.
    pub asymmetry_allowed_variants: Vec<String>,
    pub require_all_starting_cells_can_reach_plants: bool,
    pub require_team_p90_within_best_travel_endurance: bool,
    pub max_team_mean_plant_distance_gap: Option<f64>,
    pub max_team_p90_plant_distance_gap: Option<f64>,
    pub max_exclusive_plant_share_gap: Option<f64>,
    pub max_exclusive_major_food_share_gap: Option<f64>,
    pub max_starting_plant_occupancy_rate_gap: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityGateSpec {
    pub schema_version: u32,
    #[serde(default)]
    pub absolute: AbsoluteViabilityGates,
    #[serde(default)]
    pub geometry: GeometryViabilityGates,
    #[serde(default)]
    pub paired: PairedViabilityGates,
}

impl ViabilityGateSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != VIABILITY_GATE_SCHEMA_VERSION {
            return Err(format!(
                "unsupported viability gate schema {}; expected {}",
                self.schema_version, VIABILITY_GATE_SCHEMA_VERSION
            ));
        }
        let rates = [
            self.absolute.max_timeout_rate,
            self.absolute.max_extinction_without_combat_rate,
            self.absolute.max_mass_energy_drift_episode_rate,
            self.absolute.min_candidate_survival_rate,
            self.absolute.max_commit_rejection_rate,
            self.absolute.max_frustrated_resolution_rate,
            self.absolute.min_forager_plant_occupancy_rate,
            self.paired.max_regressed_outcome_rate,
            self.paired.max_candidate_win_rate_drop,
            self.paired.max_timeout_rate_increase,
            self.geometry.max_exclusive_plant_share_gap,
            self.geometry.max_exclusive_major_food_share_gap,
            self.geometry.max_starting_plant_occupancy_rate_gap,
        ];
        if rates
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err("viability gate rates must be finite and in [0, 1]".into());
        }
        let nonnegative = [
            self.absolute
                .min_attack_commits_per_episode_for_combat_profiles,
            self.absolute
                .min_applied_damage_per_episode_for_combat_profiles,
            self.paired.max_final_candidate_cell_mean_drop,
            self.geometry.max_team_mean_plant_distance_gap,
            self.geometry.max_team_p90_plant_distance_gap,
            self.absolute.min_forager_consumed_energy_per_episode,
        ];
        if nonnegative
            .into_iter()
            .flatten()
            .any(|value| !value.is_finite() || value < 0.0)
        {
            return Err(
                "viability gate counts and magnitudes must be finite and nonnegative".into(),
            );
        }
        if rates.into_iter().all(|value| value.is_none())
            && nonnegative.into_iter().all(|value| value.is_none())
            && !self.geometry.require_all_starting_cells_can_reach_plants
            && !self.geometry.require_team_p90_within_best_travel_endurance
        {
            return Err("viability gate specification enables no checks".into());
        }
        let mut asymmetry_variants = std::collections::HashSet::new();
        if self.geometry.asymmetry_allowed_variants.iter().any(|name| {
            name.is_empty()
                || !name.bytes().all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'-' | b'_')
                })
                || !asymmetry_variants.insert(name)
        }) {
            return Err("geometry asymmetry-allowed variants are invalid or duplicated".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ViabilityGateDecision {
    Passed,
    Failed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GateComparator {
    AtMost,
    AtLeast,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityGateSubject {
    pub variant: String,
    pub baseline_variant: Option<String>,
    pub candidate_profile: Option<OpponentProfile>,
    pub opponent_profile: Option<OpponentProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityGateCheck {
    pub metric: String,
    pub subject: ViabilityGateSubject,
    pub comparator: GateComparator,
    pub observed: f64,
    pub threshold: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityGateReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub matrix_file_sha256: String,
    pub matrix_package_version: String,
    pub matrix_code_revision: Option<String>,
    pub matrix_config_hash: String,
    pub manifest_sha256: String,
    pub gate_spec_sha256: String,
    pub gate_config_hash: String,
    pub decision: ViabilityGateDecision,
    pub total_checks: usize,
    pub failed_checks: usize,
    pub checks: Vec<ViabilityGateCheck>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ViabilityGateRequirement {
    pub decision_file: PathBuf,
    pub matrix_file: PathBuf,
    pub gate_spec_file: PathBuf,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Check the internal consistency of a decoded decision. Authenticity still
/// requires re-evaluating the bound matrix and policy with
/// `verify_viability_gate_requirement`.
pub fn validate_viability_gate_report(report: &ViabilityGateReport) -> Result<(), String> {
    if report.schema_version != VIABILITY_GATE_SCHEMA_VERSION {
        return Err(format!(
            "unsupported viability gate report schema {}; expected {}",
            report.schema_version, VIABILITY_GATE_SCHEMA_VERSION
        ));
    }
    if !valid_sha256(&report.matrix_file_sha256)
        || !valid_sha256(&report.matrix_config_hash)
        || !valid_sha256(&report.manifest_sha256)
        || !valid_sha256(&report.gate_spec_sha256)
        || !valid_sha256(&report.gate_config_hash)
    {
        return Err("viability gate report contains an invalid SHA-256 identity".into());
    }
    if report.package_version.is_empty() || report.matrix_package_version.is_empty() {
        return Err("viability gate report contains an empty package identity".into());
    }
    if report.total_checks == 0 || report.total_checks != report.checks.len() {
        return Err("viability gate report check count is inconsistent".into());
    }
    for check in &report.checks {
        if check.metric.is_empty() || !check.observed.is_finite() || !check.threshold.is_finite() {
            return Err("viability gate report contains an invalid check".into());
        }
        let expected = match check.comparator {
            GateComparator::AtMost => check.observed <= check.threshold,
            GateComparator::AtLeast => check.observed >= check.threshold,
        };
        if check.passed != expected {
            return Err("viability gate report check result is inconsistent".into());
        }
    }
    let failed_checks = report.checks.iter().filter(|check| !check.passed).count();
    let expected_decision = if failed_checks == 0 {
        ViabilityGateDecision::Passed
    } else {
        ViabilityGateDecision::Failed
    };
    if report.failed_checks != failed_checks || report.decision != expected_decision {
        return Err("viability gate report decision is inconsistent".into());
    }
    Ok(())
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

fn all_actions(families: &ActionFamilyTelemetry) -> [&crate::telemetry::ActionTelemetry; 9] {
    [
        &families.wait,
        &families.movement,
        &families.attack,
        &families.guard,
        &families.consume,
        &families.split,
        &families.regurgitate,
        &families.excavate,
        &families.deposit_terrain,
    ]
}

fn action_total(
    sides: [&SideTelemetry; 2],
    field: fn(&crate::telemetry::ActionTelemetry) -> u64,
) -> u64 {
    sides.into_iter().fold(0, |total, side| {
        all_actions(&side.actions)
            .into_iter()
            .fold(total, |total, action| total.saturating_add(field(action)))
    })
}

fn subject(report: &ViabilityReport, variant: &str) -> ViabilityGateSubject {
    ViabilityGateSubject {
        variant: variant.to_string(),
        baseline_variant: None,
        candidate_profile: Some(report.candidate),
        opponent_profile: Some(report.opponent),
    }
}

fn comparison_subject(comparison: &PairedViabilityComparison) -> ViabilityGateSubject {
    ViabilityGateSubject {
        variant: comparison.candidate_variant.clone(),
        baseline_variant: Some(comparison.baseline_variant.clone()),
        candidate_profile: Some(comparison.candidate_profile),
        opponent_profile: Some(comparison.opponent_profile),
    }
}

fn geometry_subject(variant: &ViabilityMatrixVariant) -> ViabilityGateSubject {
    ViabilityGateSubject {
        variant: variant.name.clone(),
        baseline_variant: None,
        candidate_profile: None,
        opponent_profile: None,
    }
}

fn push_check(
    checks: &mut Vec<ViabilityGateCheck>,
    metric: &str,
    subject: ViabilityGateSubject,
    comparator: GateComparator,
    observed: f64,
    threshold: f64,
) {
    let passed = match comparator {
        GateComparator::AtMost => observed <= threshold,
        GateComparator::AtLeast => observed >= threshold,
    };
    checks.push(ViabilityGateCheck {
        metric: metric.to_string(),
        subject,
        comparator,
        observed,
        threshold,
        passed,
    });
}

fn evaluate_absolute(
    checks: &mut Vec<ViabilityGateCheck>,
    variant: &str,
    report: &ViabilityReport,
    gates: &AbsoluteViabilityGates,
) {
    let episodes = report.episodes.len() as f64;
    let check_subject = subject(report, variant);
    if let Some(threshold) = gates.max_timeout_rate {
        push_check(
            checks,
            "timeout_rate",
            check_subject.clone(),
            GateComparator::AtMost,
            report.aggregate.timeouts as f64 / episodes,
            threshold,
        );
    }
    if report.aggregate.no_combat_profiles {
        if let Some(threshold) = gates.max_extinction_without_combat_rate {
            push_check(
                checks,
                "extinction_without_combat_rate",
                check_subject.clone(),
                GateComparator::AtMost,
                report.aggregate.extinctions_without_combat as f64 / episodes,
                threshold,
            );
        }
    }
    if let Some(threshold) = gates.max_mass_energy_drift_episode_rate {
        let drifted = report
            .episodes
            .iter()
            .filter(|episode| {
                episode.initial_tracked_mass_energy != episode.final_tracked_mass_energy
            })
            .count();
        push_check(
            checks,
            "mass_energy_drift_episode_rate",
            check_subject.clone(),
            GateComparator::AtMost,
            drifted as f64 / episodes,
            threshold,
        );
    }
    if let Some(threshold) = gates.min_candidate_survival_rate {
        let survived = report
            .episodes
            .iter()
            .filter(|episode| episode.candidate_cells > 0)
            .count();
        push_check(
            checks,
            "candidate_survival_rate",
            check_subject.clone(),
            GateComparator::AtLeast,
            survived as f64 / episodes,
            threshold,
        );
    }
    if report.candidate == OpponentProfile::Forager {
        if let Some(threshold) = gates.min_forager_plant_occupancy_rate {
            push_check(
                checks,
                "forager_plant_occupancy_rate",
                check_subject.clone(),
                GateComparator::AtLeast,
                if report.telemetry.sample_means.training_cells == 0.0 {
                    0.0
                } else {
                    report.telemetry.sample_means.training_cells_on_plants
                        / report.telemetry.sample_means.training_cells
                },
                threshold,
            );
        }
        if let Some(threshold) = gates.min_forager_consumed_energy_per_episode {
            push_check(
                checks,
                "forager_consumed_energy_per_episode",
                check_subject.clone(),
                GateComparator::AtLeast,
                report.telemetry.training.actions.consume.consumed_energy as f64 / episodes,
                threshold,
            );
        }
    }
    let sides = [&report.telemetry.training, &report.telemetry.opponents];
    if !report.aggregate.no_combat_profiles {
        if let Some(threshold) = gates.min_attack_commits_per_episode_for_combat_profiles {
            let attacks = sides.iter().fold(0u64, |total, side| {
                total.saturating_add(side.actions.attack.committed)
            });
            push_check(
                checks,
                "attack_commits_per_episode_for_combat_profiles",
                check_subject.clone(),
                GateComparator::AtLeast,
                attacks as f64 / episodes,
                threshold,
            );
        }
        if let Some(threshold) = gates.min_applied_damage_per_episode_for_combat_profiles {
            let damage = report
                .telemetry
                .training
                .damage
                .applied_dealt
                .saturating_add(report.telemetry.opponents.damage.applied_dealt);
            push_check(
                checks,
                "applied_damage_per_episode_for_combat_profiles",
                check_subject.clone(),
                GateComparator::AtLeast,
                damage as f64 / episodes,
                threshold,
            );
        }
    }
    let committed = action_total(sides, |action| action.committed);
    if let Some(threshold) = gates.max_commit_rejection_rate {
        let rejected = action_total(sides, |action| action.rejected_at_commit);
        push_check(
            checks,
            "commit_rejection_rate",
            check_subject.clone(),
            GateComparator::AtMost,
            if committed == 0 {
                0.0
            } else {
                rejected as f64 / committed as f64
            },
            threshold,
        );
    }
    if let Some(threshold) = gates.max_frustrated_resolution_rate {
        let completed = action_total(sides, |action| action.completed);
        let frustrated = action_total(sides, |action| action.frustrated);
        push_check(
            checks,
            "frustrated_resolution_rate",
            check_subject,
            GateComparator::AtMost,
            if completed == 0 {
                0.0
            } else {
                frustrated as f64 / completed as f64
            },
            threshold,
        );
    }
}

fn evaluate_geometry(
    checks: &mut Vec<ViabilityGateCheck>,
    variant: &ViabilityMatrixVariant,
    gates: &GeometryViabilityGates,
) {
    let ecology = &variant.ecological_characterization;
    let access = &ecology.resource_access;
    let check_subject = geometry_subject(variant);
    if gates.require_all_starting_cells_can_reach_plants {
        let samples = access
            .teams
            .iter()
            .map(|team| team.distance_to_plants.weighted_distance_units.samples)
            .sum::<usize>();
        let unreachable = access
            .teams
            .iter()
            .map(|team| {
                team.distance_to_plants
                    .weighted_distance_units
                    .unreachable_samples
            })
            .sum::<usize>();
        push_check(
            checks,
            "starting_cell_unreachable_plant_rate",
            check_subject.clone(),
            GateComparator::AtMost,
            if samples == 0 {
                1.0
            } else {
                unreachable as f64 / samples as f64
            },
            0.0,
        );
    }
    if gates.require_team_p90_within_best_travel_endurance {
        let best_travel = ecology
            .travel
            .iter()
            .filter(|travel| !travel.censored)
            .map(|travel| travel.completed_distance_q10 as f64 / 1024.0)
            .fold(0.0_f64, f64::max);
        let worst_p90 = access
            .teams
            .iter()
            .filter_map(|team| team.distance_to_plants.weighted_distance_units.p90)
            .fold(None, |maximum: Option<f64>, distance| {
                Some(maximum.map_or(distance, |maximum| maximum.max(distance)))
            });
        push_check(
            checks,
            "team_p90_plant_travel_endurance_margin",
            check_subject.clone(),
            GateComparator::AtLeast,
            worst_p90.map_or(-1.0, |distance| best_travel - distance),
            0.0,
        );
    }
    if gates.asymmetry_allowed_variants.contains(&variant.name) {
        return;
    }
    for (metric, observed, threshold) in [
        (
            "team_mean_plant_distance_gap",
            access.maximum_team_mean_plant_distance_gap,
            gates.max_team_mean_plant_distance_gap,
        ),
        (
            "team_p90_plant_distance_gap",
            access.maximum_team_p90_plant_distance_gap,
            gates.max_team_p90_plant_distance_gap,
        ),
        (
            "exclusive_plant_share_gap",
            access.maximum_exclusive_plant_share_gap,
            gates.max_exclusive_plant_share_gap,
        ),
        (
            "exclusive_major_food_share_gap",
            access.maximum_exclusive_major_food_share_gap,
            gates.max_exclusive_major_food_share_gap,
        ),
        (
            "starting_plant_occupancy_rate_gap",
            access.maximum_starting_plant_occupancy_rate_gap,
            gates.max_starting_plant_occupancy_rate_gap,
        ),
    ] {
        if let Some(threshold) = threshold {
            push_check(
                checks,
                metric,
                check_subject.clone(),
                GateComparator::AtMost,
                observed.unwrap_or(0.0),
                threshold,
            );
        }
    }
}

fn evaluate_paired(
    checks: &mut Vec<ViabilityGateCheck>,
    comparison: &PairedViabilityComparison,
    baseline: &ViabilityReport,
    candidate: &ViabilityReport,
    gates: &PairedViabilityGates,
) {
    let check_subject = comparison_subject(comparison);
    if let Some(threshold) = gates.max_regressed_outcome_rate {
        push_check(
            checks,
            "regressed_outcome_rate",
            check_subject.clone(),
            GateComparator::AtMost,
            comparison.regressed_outcomes as f64 / comparison.seeds.len() as f64,
            threshold,
        );
    }
    if let Some(threshold) = gates.max_candidate_win_rate_drop {
        push_check(
            checks,
            "candidate_win_rate_difference",
            check_subject.clone(),
            GateComparator::AtLeast,
            comparison.candidate_win_difference.mean,
            -threshold,
        );
    }
    if let Some(threshold) = gates.max_final_candidate_cell_mean_drop {
        push_check(
            checks,
            "final_candidate_cell_mean_difference",
            check_subject.clone(),
            GateComparator::AtLeast,
            comparison.final_candidate_cell_difference.mean,
            -threshold,
        );
    }
    if let Some(threshold) = gates.max_timeout_rate_increase {
        let baseline_timeout_rate =
            baseline.aggregate.timeouts as f64 / baseline.episodes.len() as f64;
        let candidate_timeout_rate =
            candidate.aggregate.timeouts as f64 / candidate.episodes.len() as f64;
        push_check(
            checks,
            "timeout_rate_difference",
            check_subject,
            GateComparator::AtMost,
            candidate_timeout_rate - baseline_timeout_rate,
            threshold,
        );
    }
}

fn evaluate(
    matrix: &ViabilityMatrixReport,
    spec: &ViabilityGateSpec,
    matrix_file_sha256: String,
    gate_spec_sha256: String,
) -> Result<ViabilityGateReport, String> {
    validate_viability_matrix_report(matrix)?;
    spec.validate()?;
    if spec
        .geometry
        .asymmetry_allowed_variants
        .iter()
        .any(|name| !matrix.variants.iter().any(|variant| &variant.name == name))
    {
        return Err("geometry gate names an asymmetry exception absent from the matrix".into());
    }
    let mut checks = Vec::new();
    let by_matchup = matrix
        .matchups
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
    for matchup in &matrix.matchups {
        evaluate_absolute(
            &mut checks,
            &matchup.variant,
            &matchup.report,
            &spec.absolute,
        );
    }
    for variant in &matrix.variants {
        evaluate_geometry(&mut checks, variant, &spec.geometry);
    }
    for comparison in &matrix.paired_comparisons {
        let baseline = by_matchup
            .get(&(
                comparison.baseline_variant.as_str(),
                comparison.candidate_profile,
                comparison.opponent_profile,
            ))
            .expect("validated matrix contains paired baseline");
        let candidate = by_matchup
            .get(&(
                comparison.candidate_variant.as_str(),
                comparison.candidate_profile,
                comparison.opponent_profile,
            ))
            .expect("validated matrix contains paired candidate");
        evaluate_paired(&mut checks, comparison, baseline, candidate, &spec.paired);
    }
    let failed_checks = checks.iter().filter(|check| !check.passed).count();
    let gate_config_hash = sha256(
        &serde_json::to_vec(spec)
            .map_err(|error| format!("failed to encode gate configuration: {error}"))?,
    );
    Ok(ViabilityGateReport {
        schema_version: VIABILITY_GATE_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        matrix_file_sha256,
        matrix_package_version: matrix.package_version.clone(),
        matrix_code_revision: matrix.code_revision.clone(),
        matrix_config_hash: matrix.matrix_config_hash.clone(),
        manifest_sha256: matrix.manifest_sha256.clone(),
        gate_spec_sha256,
        gate_config_hash,
        decision: if failed_checks == 0 {
            ViabilityGateDecision::Passed
        } else {
            ViabilityGateDecision::Failed
        },
        total_checks: checks.len(),
        failed_checks,
        checks,
    })
}

/// Load, bound, validate, and evaluate a matrix and strict TOML gate policy.
pub fn evaluate_viability_gates(
    matrix_file: &Path,
    gate_spec_file: &Path,
) -> Result<ViabilityGateReport, String> {
    let matrix_bytes = read_bounded(matrix_file, MAX_MATRIX_BYTES, "viability matrix")?;
    let matrix: ViabilityMatrixReport = serde_json::from_slice(&matrix_bytes)
        .map_err(|error| format!("failed to decode {}: {error}", matrix_file.display()))?;
    let spec_bytes = read_bounded(gate_spec_file, MAX_GATE_SPEC_BYTES, "viability gate spec")?;
    let spec_text = std::str::from_utf8(&spec_bytes)
        .map_err(|error| format!("viability gate spec is not UTF-8: {error}"))?;
    let spec: ViabilityGateSpec = toml::from_str(spec_text)
        .map_err(|error| format!("failed to parse viability gate spec: {error}"))?;
    evaluate(&matrix, &spec, sha256(&matrix_bytes), sha256(&spec_bytes))
}

/// Re-evaluate the exact matrix and policy named by a requirement, verify that
/// the published decision is identical, passing, and bound to the sweep about
/// to launch.
pub fn verify_viability_gate_requirement(
    requirement: &ViabilityGateRequirement,
    expected_manifest_sha256: &str,
) -> Result<ViabilityGateReport, String> {
    let published = reproduce_viability_gate_requirement(requirement, expected_manifest_sha256)?;
    if published.decision != ViabilityGateDecision::Passed {
        return Err(format!(
            "viability gate failed with {}/{} failed checks",
            published.failed_checks, published.total_checks
        ));
    }
    Ok(published)
}

/// Reproduce a published decision and all of its sweep/build bindings without
/// requiring the scientific result itself to have passed.
pub fn reproduce_viability_gate_requirement(
    requirement: &ViabilityGateRequirement,
    expected_manifest_sha256: &str,
) -> Result<ViabilityGateReport, String> {
    let expected = evaluate_viability_gates(&requirement.matrix_file, &requirement.gate_spec_file)?;
    let decision_bytes = read_bounded(
        &requirement.decision_file,
        MAX_GATE_REPORT_BYTES,
        "viability gate report",
    )?;
    let published: ViabilityGateReport =
        serde_json::from_slice(&decision_bytes).map_err(|error| {
            format!(
                "failed to decode viability gate report {}: {error}",
                requirement.decision_file.display()
            )
        })?;
    validate_viability_gate_report(&published)?;
    if published != expected {
        return Err("published viability gate report does not match fresh evaluation".into());
    }
    if published.manifest_sha256 != expected_manifest_sha256 {
        return Err("viability gate report belongs to a different sweep manifest".into());
    }
    if published.matrix_package_version != env!("CARGO_PKG_VERSION")
        || published.matrix_code_revision.as_deref() != option_env!("BLOB_CODE_REVISION")
    {
        return Err("viability matrix was produced by a different engine build".into());
    }
    Ok(published)
}

/// Publish a decision atomically without permitting policy-result replacement.
pub fn publish_viability_gate_report(
    output: &Path,
    report: &ViabilityGateReport,
) -> Result<PathBuf, String> {
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable viability gate report {}",
            output.display()
        ));
    }
    let nonce = GATE_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "viability gate output needs a UTF-8 file name".to_string())?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode viability gate report: {error}"))?;
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
    use crate::viability_matrix::{run_viability_matrix, ViabilityMatrixOptions};

    fn matrix() -> ViabilityMatrixReport {
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
            sim_time_limit_quanta = 6144

            [self_play]
            max_opponent_pool = 0
            "#,
        )
        .unwrap();
        let spec_file = temporary.path().join("sweep.toml");
        fs::write(
            &spec_file,
            r#"
            base_config = "base.toml"
            output_dir = "planned"
            seeds = [41, 42, 43]

            [[variants]]
            name = "baseline"

            [[variants]]
            name = "faster-digestion"
            [variants.rules]
            digestion_rate_numerator = 2
            "#,
        )
        .unwrap();
        let manifest = publish_rules_sweep(&spec_file).unwrap();
        run_viability_matrix(
            &PathBuf::from(manifest.output_directory).join("manifest.json"),
            &ViabilityMatrixOptions {
                candidates: vec![OpponentProfile::Random],
                opponents: vec![OpponentProfile::Wait],
                baseline_variant: None,
                max_parallel: 2,
                max_micro_actions: 1_000,
            },
        )
        .unwrap()
    }

    fn passing_spec(matrix: &ViabilityMatrixReport) -> ViabilityGateSpec {
        let maximum_timeout_rate = matrix
            .matchups
            .iter()
            .map(|matchup| {
                matchup.report.aggregate.timeouts as f64 / matchup.report.episodes.len() as f64
            })
            .fold(0.0, f64::max);
        ViabilityGateSpec {
            schema_version: VIABILITY_GATE_SCHEMA_VERSION,
            absolute: AbsoluteViabilityGates {
                max_timeout_rate: Some(maximum_timeout_rate),
                max_mass_energy_drift_episode_rate: Some(1.0),
                min_candidate_survival_rate: Some(0.0),
                max_commit_rejection_rate: Some(1.0),
                max_frustrated_resolution_rate: Some(1.0),
                ..AbsoluteViabilityGates::default()
            },
            geometry: GeometryViabilityGates::default(),
            paired: PairedViabilityGates {
                max_regressed_outcome_rate: Some(1.0),
                max_candidate_win_rate_drop: Some(1.0),
                max_final_candidate_cell_mean_drop: Some(f64::MAX / 4.0),
                max_timeout_rate_increase: Some(1.0),
            },
        }
    }

    #[test]
    fn inclusive_gates_are_reproducible_and_fail_closed() {
        let matrix = matrix();
        let passing = passing_spec(&matrix);
        let first = evaluate(&matrix, &passing, "matrix".into(), "policy".into()).unwrap();
        let second = evaluate(&matrix, &passing, "matrix".into(), "policy".into()).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.decision, ViabilityGateDecision::Passed);
        assert_eq!(first.failed_checks, 0);
        assert!(first.checks.iter().any(|check| {
            check.metric == "timeout_rate" && check.observed == check.threshold && check.passed
        }));

        let failing = ViabilityGateSpec {
            schema_version: VIABILITY_GATE_SCHEMA_VERSION,
            absolute: AbsoluteViabilityGates {
                min_applied_damage_per_episode_for_combat_profiles: Some(f64::MAX / 4.0),
                ..AbsoluteViabilityGates::default()
            },
            geometry: GeometryViabilityGates::default(),
            paired: PairedViabilityGates::default(),
        };
        let failed = evaluate(&matrix, &failing, "matrix".into(), "policy".into()).unwrap();
        assert_eq!(failed.decision, ViabilityGateDecision::Failed);
        assert_eq!(failed.failed_checks, matrix.matchups.len());

        let observed_gap = matrix
            .variants
            .iter()
            .filter_map(|variant| {
                variant
                    .ecological_characterization
                    .resource_access
                    .maximum_exclusive_plant_share_gap
            })
            .fold(0.0_f64, f64::max);
        assert!(observed_gap > 0.0);
        let mut geometry = passing_spec(&matrix);
        geometry.geometry.max_exclusive_plant_share_gap = Some(observed_gap);
        let at_boundary = evaluate(&matrix, &geometry, "matrix".into(), "policy".into()).unwrap();
        assert_eq!(at_boundary.decision, ViabilityGateDecision::Passed);
        assert!(at_boundary.checks.iter().any(|check| {
            check.metric == "exclusive_plant_share_gap"
                && check.observed == check.threshold
                && check.passed
                && check.subject.candidate_profile.is_none()
        }));

        geometry.geometry.max_exclusive_plant_share_gap = Some(observed_gap / 2.0);
        let asymmetric = evaluate(&matrix, &geometry, "matrix".into(), "policy".into()).unwrap();
        assert_eq!(asymmetric.decision, ViabilityGateDecision::Failed);
        geometry.geometry.asymmetry_allowed_variants = matrix
            .variants
            .iter()
            .map(|variant| variant.name.clone())
            .collect();
        let exempted = evaluate(&matrix, &geometry, "matrix".into(), "policy".into()).unwrap();
        assert_eq!(exempted.decision, ViabilityGateDecision::Passed);
    }

    #[test]
    fn strict_specs_tamper_detection_and_immutable_publication_hold() {
        assert!(toml::from_str::<ViabilityGateSpec>(
            r#"
            schema_version = 2
            [absolute]
            max_timeuot_rate = 0.5
            "#,
        )
        .unwrap_err()
        .to_string()
        .contains("unknown field"));
        let documented: ViabilityGateSpec =
            toml::from_str(include_str!("../config/viability_gates.toml")).unwrap();
        documented.validate().unwrap();
        assert!(ViabilityGateSpec {
            schema_version: VIABILITY_GATE_SCHEMA_VERSION,
            absolute: AbsoluteViabilityGates::default(),
            geometry: GeometryViabilityGates::default(),
            paired: PairedViabilityGates::default(),
        }
        .validate()
        .unwrap_err()
        .contains("enables no checks"));
        let matrix = matrix();
        let spec = passing_spec(&matrix);
        let mut tampered = matrix.clone();
        tampered.matchups[0].report.aggregate.timeouts += 1;
        assert!(evaluate(&tampered, &spec, "matrix".into(), "policy".into())
            .unwrap_err()
            .contains("aggregate"));

        let report = evaluate(&matrix, &spec, "matrix".into(), "policy".into()).unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let matrix_file = temporary.path().join("matrix.json");
        let spec_file = temporary.path().join("gates.toml");
        fs::write(&matrix_file, serde_json::to_vec_pretty(&matrix).unwrap()).unwrap();
        fs::write(&spec_file, toml::to_string_pretty(&spec).unwrap()).unwrap();
        let loaded = evaluate_viability_gates(&matrix_file, &spec_file).unwrap();
        assert_eq!(loaded.decision, report.decision);
        assert_eq!(loaded.checks, report.checks);
        let output = temporary.path().join("gate.json");
        publish_viability_gate_report(&output, &loaded).unwrap();
        let bytes = fs::read(&output).unwrap();
        assert!(publish_viability_gate_report(&output, &loaded).is_err());
        assert_eq!(fs::read(output).unwrap(), bytes);

        let requirement = ViabilityGateRequirement {
            decision_file: temporary.path().join("gate.json"),
            matrix_file,
            gate_spec_file: spec_file,
        };
        verify_viability_gate_requirement(&requirement, &matrix.manifest_sha256).unwrap();
        assert!(
            verify_viability_gate_requirement(&requirement, &"0".repeat(64))
                .unwrap_err()
                .contains("different sweep manifest")
        );

        let mut foreign_matrix = matrix.clone();
        foreign_matrix.code_revision = Some("different-engine-build".into());
        let foreign_matrix_file = temporary.path().join("foreign-matrix.json");
        fs::write(
            &foreign_matrix_file,
            serde_json::to_vec_pretty(&foreign_matrix).unwrap(),
        )
        .unwrap();
        let foreign_decision =
            evaluate_viability_gates(&foreign_matrix_file, &requirement.gate_spec_file).unwrap();
        let foreign_decision_file = temporary.path().join("foreign-decision.json");
        publish_viability_gate_report(&foreign_decision_file, &foreign_decision).unwrap();
        assert!(verify_viability_gate_requirement(
            &ViabilityGateRequirement {
                decision_file: foreign_decision_file,
                matrix_file: foreign_matrix_file,
                gate_spec_file: requirement.gate_spec_file.clone(),
            },
            &matrix.manifest_sha256,
        )
        .unwrap_err()
        .contains("different engine build"));

        let mut tampered_report = loaded;
        tampered_report.decision = ViabilityGateDecision::Failed;
        let tampered_file = temporary.path().join("tampered-gate.json");
        fs::write(
            &tampered_file,
            serde_json::to_vec_pretty(&tampered_report).unwrap(),
        )
        .unwrap();
        let tampered_requirement = ViabilityGateRequirement {
            decision_file: tampered_file,
            ..requirement
        };
        assert!(
            verify_viability_gate_requirement(&tampered_requirement, &matrix.manifest_sha256)
                .unwrap_err()
                .contains("decision is inconsistent")
        );
    }
}
