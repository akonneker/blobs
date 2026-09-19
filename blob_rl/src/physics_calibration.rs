//! Coarse, behavior-based feasibility gates over ecological characterizations.
//!
//! The characterization remains descriptive evidence. This module applies a
//! deliberately broad and separately versioned set of expectations so tuning
//! policy can evolve without changing canonical physics or rewriting evidence.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::ecological_characterization::{
    validate_ecological_characterization, CharacterizedEffort, EcologicalCharacterizationReport,
};

pub const PHYSICS_CALIBRATION_SCHEMA_VERSION: u32 = 1;
const MAX_CHARACTERIZATION_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SPEC_BYTES: u64 = 1024 * 1024;
const MAX_REPORT_BYTES: u64 = 4 * 1024 * 1024;
static CALIBRATION_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct PhysicsCalibrationSpec {
    pub schema_version: u32,
    pub min_reachable_major_food_fraction: f64,
    pub min_reachable_plant_fraction: f64,
    pub min_stationary_lifetime_time_units: f64,
    pub min_standard_travel_distance_units: f64,
    pub min_standard_travel_to_p90_major_food_ratio: f64,
    pub min_standard_travel_to_p90_plant_ratio: f64,
    pub min_feeding_net_energy: f64,
    pub min_feeding_net_fraction: f64,
    pub max_reproduction_break_even_initial_energy_fraction: f64,
    pub min_guarded_volley_additional_attackers: f64,
    pub min_sustained_siege_attackers: f64,
    pub max_sustained_siege_attackers: f64,
    pub require_high_attack_to_land: bool,
    pub require_terrain_cycle: bool,
}

impl Default for PhysicsCalibrationSpec {
    fn default() -> Self {
        Self {
            schema_version: PHYSICS_CALIBRATION_SCHEMA_VERSION,
            min_reachable_major_food_fraction: 1.0,
            min_reachable_plant_fraction: 1.0,
            min_stationary_lifetime_time_units: 20.0,
            min_standard_travel_distance_units: 12.0,
            // Enough for a round trip from a p90 start plus a 50% reserve.
            min_standard_travel_to_p90_major_food_ratio: 3.0,
            min_standard_travel_to_p90_plant_ratio: 2.0,
            min_feeding_net_energy: 1.0,
            min_feeding_net_fraction: 0.25,
            max_reproduction_break_even_initial_energy_fraction: 1.0,
            min_guarded_volley_additional_attackers: 1.0,
            min_sustained_siege_attackers: 2.0,
            max_sustained_siege_attackers: 4.0,
            require_high_attack_to_land: true,
            require_terrain_cycle: true,
        }
    }
}

impl PhysicsCalibrationSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != PHYSICS_CALIBRATION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported physics calibration schema {}; expected {}",
                self.schema_version, PHYSICS_CALIBRATION_SCHEMA_VERSION
            ));
        }
        for fraction in [
            self.min_reachable_major_food_fraction,
            self.min_reachable_plant_fraction,
            self.min_feeding_net_fraction,
            self.max_reproduction_break_even_initial_energy_fraction,
        ] {
            if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
                return Err("physics calibration fractions must be finite and in [0, 1]".into());
            }
        }
        for value in [
            self.min_stationary_lifetime_time_units,
            self.min_standard_travel_distance_units,
            self.min_standard_travel_to_p90_major_food_ratio,
            self.min_standard_travel_to_p90_plant_ratio,
            self.min_feeding_net_energy,
            self.min_guarded_volley_additional_attackers,
            self.min_sustained_siege_attackers,
            self.max_sustained_siege_attackers,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err("physics calibration thresholds must be finite and nonnegative".into());
            }
        }
        if self.min_sustained_siege_attackers > self.max_sustained_siege_attackers {
            return Err("physics calibration siege bounds are reversed".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationComparator {
    AtLeast,
    AtMost,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationDecision {
    Passed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PhysicsCalibrationCheck {
    pub metric: String,
    pub rationale: String,
    pub observed: Option<f64>,
    pub comparator: CalibrationComparator,
    pub threshold: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PhysicsCalibrationReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub characterization_file_sha256: String,
    pub characterization_hash: String,
    pub characterization_results_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub spec_file_sha256: String,
    pub spec: PhysicsCalibrationSpec,
    pub decision: CalibrationDecision,
    pub failed_checks: usize,
    pub checks: Vec<PhysicsCalibrationCheck>,
}

pub fn evaluate_physics_calibration(
    characterization: &EcologicalCharacterizationReport,
    characterization_file_sha256: String,
    spec: &PhysicsCalibrationSpec,
    spec_file_sha256: String,
) -> Result<PhysicsCalibrationReport, String> {
    validate_ecological_characterization(characterization)?;
    spec.validate()?;
    if !valid_sha256(&characterization_file_sha256) || !valid_sha256(&spec_file_sha256) {
        return Err("physics calibration inputs require lowercase SHA-256 identities".into());
    }

    let standard_travel = characterization
        .travel
        .iter()
        .find(|travel| travel.effort == CharacterizedEffort::Standard && !travel.censored);
    let travel_distance =
        standard_travel.map(|travel| travel.completed_distance_q10 as f64 / 1024.0);
    let stationary_lifetime = characterization.stationary_lifetime_quanta.map(|quanta| {
        quanta as f64
            / characterization
                .rules
                .time
                .arithmetic_quanta_per_unit
                .max(1) as f64
    });
    let major_reachable = reachable_fraction(
        characterization
            .distance_to_major_food
            .weighted_distance_units
            .samples,
        characterization
            .distance_to_major_food
            .weighted_distance_units
            .reachable_samples,
    );
    let plant_reachable = reachable_fraction(
        characterization
            .distance_to_plants
            .weighted_distance_units
            .samples,
        characterization
            .distance_to_plants
            .weighted_distance_units
            .reachable_samples,
    );
    let feeding_net = characterization
        .feeding_cycle
        .energy_after_digestion
        .map(|after| after as f64 - characterization.feeding_cycle.energy_before_consume as f64);
    let feeding_fraction = feeding_net.and_then(|net| {
        (characterization.feeding_cycle.consumed_energy > 0)
            .then_some(net / characterization.feeding_cycle.consumed_energy as f64)
    });
    let reproduction_fraction = characterization
        .reproduction_break_even
        .minimum_successful_parent_energy
        .and_then(|energy| {
            (characterization.scenario.initial_energy > 0)
                .then_some(energy as f64 / characterization.scenario.initial_energy as f64)
        });
    let high = characterization
        .attack_volleys
        .iter()
        .find(|volley| volley.effort == CharacterizedEffort::High)
        .ok_or("characterization has no high-effort attack volley")?;
    let guard_advantage = high
        .guarded_attackers_required
        .zip(high.unguarded_attackers_required)
        .map(|(guarded, unguarded)| guarded as f64 - unguarded as f64);
    let siege_attackers = characterization
        .sustained_siege
        .minimum_attackers_to_kill
        .map(|attackers| attackers as f64);

    let mut checks = Vec::with_capacity(14);
    push_check(
        &mut checks,
        "reachable_major_food_fraction",
        "Every starting cell should have a topologically reachable substantial food source.",
        major_reachable,
        CalibrationComparator::AtLeast,
        spec.min_reachable_major_food_fraction,
    );
    push_check(
        &mut checks,
        "reachable_plant_fraction",
        "Every starting cell should be able to reach a renewable plant.",
        plant_reachable,
        CalibrationComparator::AtLeast,
        spec.min_reachable_plant_fraction,
    );
    push_check(
        &mut checks,
        "stationary_lifetime_time_units",
        "A new cell needs a nontrivial planning and reaction window even before moving.",
        stationary_lifetime,
        CalibrationComparator::AtLeast,
        spec.min_stationary_lifetime_time_units,
    );
    push_check(&mut checks, "standard_travel_distance_units", "Standard movement should support meaningful exploration rather than only an adjacent step.", travel_distance, CalibrationComparator::AtLeast, spec.min_standard_travel_distance_units);
    push_check(
        &mut checks,
        "standard_travel_to_p90_major_food_ratio",
        "Standard endurance should cover a p90 food round trip with reserve.",
        ratio(
            travel_distance,
            characterization
                .distance_to_major_food
                .weighted_distance_units
                .p90,
        ),
        CalibrationComparator::AtLeast,
        spec.min_standard_travel_to_p90_major_food_ratio,
    );
    push_check(
        &mut checks,
        "standard_travel_to_p90_plant_ratio",
        "Standard endurance should normally permit reaching and leaving a renewable plant.",
        ratio(
            travel_distance,
            characterization
                .distance_to_plants
                .weighted_distance_units
                .p90,
        ),
        CalibrationComparator::AtLeast,
        spec.min_standard_travel_to_p90_plant_ratio,
    );
    push_check(
        &mut checks,
        "feeding_net_energy",
        "A maximum bite that is fully digested should repay its action and metabolic costs.",
        feeding_net.filter(|_| characterization.feeding_cycle.survived_full_digestion),
        CalibrationComparator::AtLeast,
        spec.min_feeding_net_energy,
    );
    push_check(&mut checks, "feeding_net_fraction", "Feeding should retain a useful fraction of extracted energy after digestion and metabolism.", feeding_fraction.filter(|_| characterization.feeding_cycle.survived_full_digestion), CalibrationComparator::AtLeast, spec.min_feeding_net_fraction);
    push_check(
        &mut checks,
        "reproduction_break_even_initial_energy_fraction",
        "Minimum viable reproduction should not require more energy than a new cell starts with.",
        reproduction_fraction,
        CalibrationComparator::AtMost,
        spec.max_reproduction_break_even_initial_energy_fraction,
    );
    push_check(
        &mut checks,
        "high_attack_lands",
        "The strongest attack family should be operationally usable.",
        Some(if high.single_attack_lands { 1.0 } else { 0.0 }),
        CalibrationComparator::AtLeast,
        if spec.require_high_attack_to_land {
            1.0
        } else {
            0.0
        },
    );
    push_check(
        &mut checks,
        "guarded_volley_additional_attackers",
        "Guard should materially increase the force needed for a simultaneous kill.",
        guard_advantage,
        CalibrationComparator::AtLeast,
        spec.min_guarded_volley_additional_attackers,
    );
    push_check(
        &mut checks,
        "sustained_siege_attackers_lower_bound",
        "A plant defender should not fall trivially to one attacker.",
        siege_attackers,
        CalibrationComparator::AtLeast,
        spec.min_sustained_siege_attackers,
    );
    push_check(
        &mut checks,
        "sustained_siege_attackers_upper_bound",
        "Plant turtling should remain breakable by a modest coordinated force.",
        siege_attackers,
        CalibrationComparator::AtMost,
        spec.max_sustained_siege_attackers,
    );
    push_check(
        &mut checks,
        "terrain_cycle_completable",
        "An initial cell should be able to lift and replace terrain without dying.",
        Some(
            if characterization.terrain_cycle.deposition_succeeded
                && characterization.terrain_cycle.elevation_restored
            {
                1.0
            } else {
                0.0
            },
        ),
        CalibrationComparator::AtLeast,
        if spec.require_terrain_cycle { 1.0 } else { 0.0 },
    );

    let failed_checks = checks.iter().filter(|check| !check.passed).count();
    let decision = if failed_checks == 0 {
        CalibrationDecision::Passed
    } else {
        CalibrationDecision::Failed
    };
    Ok(PhysicsCalibrationReport {
        schema_version: PHYSICS_CALIBRATION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").into(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_owned),
        characterization_file_sha256,
        characterization_hash: characterization.characterization_hash.clone(),
        characterization_results_hash: characterization.results_hash.clone(),
        compiled_ruleset_hash: characterization.compiled_ruleset_hash.clone(),
        scenario_hash: characterization.scenario_hash.clone(),
        spec_file_sha256,
        spec: spec.clone(),
        decision,
        failed_checks,
        checks,
    })
}

pub fn evaluate_physics_calibration_files(
    characterization_path: &Path,
    spec_path: &Path,
) -> Result<PhysicsCalibrationReport, String> {
    let characterization_bytes = read_bounded(
        characterization_path,
        MAX_CHARACTERIZATION_BYTES,
        "ecological characterization",
    )?;
    let characterization: EcologicalCharacterizationReport =
        serde_json::from_slice(&characterization_bytes).map_err(|error| {
            format!(
                "failed to decode ecological characterization {}: {error}",
                characterization_path.display()
            )
        })?;
    let spec_bytes = read_bounded(spec_path, MAX_SPEC_BYTES, "physics calibration spec")?;
    let spec_text = std::str::from_utf8(&spec_bytes)
        .map_err(|error| format!("physics calibration spec is not UTF-8: {error}"))?;
    let spec: PhysicsCalibrationSpec = toml::from_str(spec_text)
        .map_err(|error| format!("failed to parse physics calibration spec: {error}"))?;
    evaluate_physics_calibration(
        &characterization,
        sha256(&characterization_bytes),
        &spec,
        sha256(&spec_bytes),
    )
}

pub fn validate_physics_calibration_report(
    report: &PhysicsCalibrationReport,
) -> Result<(), String> {
    if report.schema_version != PHYSICS_CALIBRATION_SCHEMA_VERSION
        || report.package_version.is_empty()
        || !valid_sha256(&report.characterization_file_sha256)
        || !valid_sha256(&report.characterization_hash)
        || !valid_sha256(&report.characterization_results_hash)
        || !valid_sha256(&report.compiled_ruleset_hash)
        || !valid_sha256(&report.scenario_hash)
        || !valid_sha256(&report.spec_file_sha256)
    {
        return Err("physics calibration report identity is invalid".into());
    }
    report.spec.validate()?;
    if report.checks.is_empty() {
        return Err("physics calibration report has no checks".into());
    }
    for check in &report.checks {
        if check.metric.is_empty()
            || check.rationale.is_empty()
            || !check.threshold.is_finite()
            || check.observed.is_some_and(|value| !value.is_finite())
            || check.passed != check_passes(check.observed, check.comparator, check.threshold)
        {
            return Err("physics calibration report contains an invalid check".into());
        }
    }
    let failed = report.checks.iter().filter(|check| !check.passed).count();
    let decision = if failed == 0 {
        CalibrationDecision::Passed
    } else {
        CalibrationDecision::Failed
    };
    if report.failed_checks != failed || report.decision != decision {
        return Err("physics calibration report decision is inconsistent".into());
    }
    Ok(())
}

pub fn verify_physics_calibration_files(
    report_path: &Path,
    characterization_path: &Path,
    spec_path: &Path,
) -> Result<PhysicsCalibrationReport, String> {
    let report_bytes = read_bounded(report_path, MAX_REPORT_BYTES, "physics calibration report")?;
    let report: PhysicsCalibrationReport = serde_json::from_slice(&report_bytes)
        .map_err(|error| format!("failed to decode physics calibration report: {error}"))?;
    validate_physics_calibration_report(&report)?;
    let expected = evaluate_physics_calibration_files(characterization_path, spec_path)?;
    if report != expected {
        return Err("physics calibration report does not match its bound inputs".into());
    }
    Ok(report)
}

pub fn publish_physics_calibration(
    output: &Path,
    report: &PhysicsCalibrationReport,
) -> Result<PathBuf, String> {
    validate_physics_calibration_report(report)?;
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable physics calibration {}",
            output.display()
        ));
    }
    let nonce = CALIBRATION_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("physics calibration output needs a UTF-8 file name")?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode physics calibration report: {error}"))?;
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

fn push_check(
    checks: &mut Vec<PhysicsCalibrationCheck>,
    metric: &str,
    rationale: &str,
    observed: Option<f64>,
    comparator: CalibrationComparator,
    threshold: f64,
) {
    checks.push(PhysicsCalibrationCheck {
        metric: metric.into(),
        rationale: rationale.into(),
        observed,
        comparator,
        threshold,
        passed: check_passes(observed, comparator, threshold),
    });
}

fn check_passes(observed: Option<f64>, comparator: CalibrationComparator, threshold: f64) -> bool {
    observed.is_some_and(|value| {
        value.is_finite()
            && match comparator {
                CalibrationComparator::AtLeast => value >= threshold,
                CalibrationComparator::AtMost => value <= threshold,
            }
    })
}

fn reachable_fraction(samples: usize, reachable: usize) -> Option<f64> {
    (samples > 0 && reachable <= samples).then_some(reachable as f64 / samples as f64)
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    numerator
        .zip(denominator)
        .and_then(|(numerator, denominator)| (denominator > 0.0).then_some(numerator / denominator))
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn read_bounded(path: &Path, limit: u64, label: &str) -> Result<Vec<u8>, String> {
    crate::artifact_io::read_bounded(path, limit).map_err(|error| format!("{label}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EnvConfig, VictoryConfig};
    use crate::ecological_characterization::{
        characterize_ecology, EcologicalCharacterizationOptions,
    };

    fn characterized() -> EcologicalCharacterizationReport {
        let env = EnvConfig {
            world_size: 8,
            cells_per_team: 2,
            num_scattered_energy: 8,
            num_plants: 4,
            victory: VictoryConfig {
                sim_time_limit_quanta: 16_384,
                ..VictoryConfig::default()
            },
            ..EnvConfig::default()
        };
        characterize_ecology(
            &env,
            &EcologicalCharacterizationOptions {
                seeds: vec![10, 11, 12],
                max_micro_actions: 1_000,
            },
        )
        .unwrap()
    }

    #[test]
    fn broad_default_envelope_accepts_the_reference_profile() {
        let report = evaluate_physics_calibration(
            &characterized(),
            "a".repeat(64),
            &PhysicsCalibrationSpec::default(),
            "b".repeat(64),
        )
        .unwrap();
        assert_eq!(report.decision, CalibrationDecision::Passed);
        assert_eq!(report.failed_checks, 0);
        assert_eq!(report.checks.len(), 14);
        validate_physics_calibration_report(&report).unwrap();
    }

    #[test]
    fn impossible_endurance_and_turtle_windows_fail_independently() {
        let spec = PhysicsCalibrationSpec {
            min_standard_travel_distance_units: 1_000.0,
            min_sustained_siege_attackers: 1.0,
            max_sustained_siege_attackers: 1.0,
            ..PhysicsCalibrationSpec::default()
        };
        let report =
            evaluate_physics_calibration(&characterized(), "a".repeat(64), &spec, "b".repeat(64))
                .unwrap();
        assert_eq!(report.decision, CalibrationDecision::Failed);
        assert!(report
            .checks
            .iter()
            .find(|check| check.metric == "standard_travel_distance_units")
            .is_some_and(|check| !check.passed));
        assert!(report
            .checks
            .iter()
            .find(|check| check.metric == "sustained_siege_attackers_upper_bound")
            .is_some_and(|check| !check.passed));
    }

    #[test]
    fn report_validation_and_immutable_publication_detect_tampering() {
        let characterization = characterized();
        let temporary = tempfile::tempdir().unwrap();
        let characterization_path = temporary.path().join("characterization.json");
        let spec_path = temporary.path().join("spec.toml");
        fs::write(
            &characterization_path,
            serde_json::to_vec_pretty(&characterization).unwrap(),
        )
        .unwrap();
        fs::write(
            &spec_path,
            toml::to_string_pretty(&PhysicsCalibrationSpec::default()).unwrap(),
        )
        .unwrap();
        let report =
            evaluate_physics_calibration_files(&characterization_path, &spec_path).unwrap();
        let mut tampered = report.clone();
        tampered.checks[0].passed = !tampered.checks[0].passed;
        assert!(validate_physics_calibration_report(&tampered).is_err());

        let path = temporary.path().join("calibration.json");
        publish_physics_calibration(&path, &report).unwrap();
        assert_eq!(
            verify_physics_calibration_files(&path, &characterization_path, &spec_path).unwrap(),
            report
        );
        assert!(publish_physics_calibration(&path, &report).is_err());

        let changed_spec = PhysicsCalibrationSpec {
            min_standard_travel_distance_units: 13.0,
            ..PhysicsCalibrationSpec::default()
        };
        fs::write(&spec_path, toml::to_string_pretty(&changed_spec).unwrap()).unwrap();
        assert!(
            verify_physics_calibration_files(&path, &characterization_path, &spec_path).is_err()
        );
    }
}
