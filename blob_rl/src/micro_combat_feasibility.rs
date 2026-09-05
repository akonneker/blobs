//! Observation-legal witness-strategy audit for micro-combat scenarios.
//!
//! This is deliberately not called an optimal-policy oracle. A simulator
//! search that sees canonical state or a fixed opponent random stream only
//! establishes a privileged upper bound. These maintained baselines instead
//! pass through the public Mind observation boundary, so each successful row
//! is a realizable lower-bound witness for scenario feasibility.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{EnvConfig, OpponentProfile, RewardConfig};
use crate::micro_combat::{
    evaluate_micro_combat_baseline, MicroCombatObjective, MicroCombatScenarioMetrics,
    MicroCombatSuiteConfig,
};

pub const MICRO_COMBAT_FEASIBILITY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WitnessStrategyMetrics {
    pub strategy: OpponentProfile,
    pub objective_successes: usize,
    pub episodes: usize,
    pub objective_success_rate: f64,
    pub scientific_survival_rate: f64,
    pub average_final_training_cells: f64,
    pub average_final_stored_energy: f64,
    pub average_damage_dealt: f64,
    pub average_damage_received: f64,
    pub average_attacks_committed: f64,
    pub average_guards_committed: f64,
    pub average_births: f64,
    pub average_deaths: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScenarioFeasibilityAudit {
    pub scenario: String,
    pub objective: MicroCombatObjective,
    pub best_witness: OpponentProfile,
    pub best_witness_success_rate: f64,
    /// At least one legal observation-only maintained strategy succeeded.
    pub witnessed_feasible: bool,
    /// At least one maintained strategy met the requested reliability target.
    pub reliably_witnessed: bool,
    /// Waiting alone met the target, so the objective does not require skill.
    pub passive_timeout_degenerate: bool,
    pub witnesses: Vec<WitnessStrategyMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MicroCombatFeasibilityReport {
    pub schema_version: u32,
    pub ruleset_hash: String,
    pub suite_sha256: String,
    pub seeds: Vec<u64>,
    pub reliability_target: f64,
    pub scenarios: Vec<ScenarioFeasibilityAudit>,
}

impl MicroCombatFeasibilityReport {
    pub fn validate(&self, suite: &MicroCombatSuiteConfig) -> Result<(), String> {
        if self.schema_version != MICRO_COMBAT_FEASIBILITY_SCHEMA_VERSION
            || self.suite_sha256 != suite.semantic_hash()?
            || self.seeds.is_empty()
            || !self.reliability_target.is_finite()
            || !(0.0..=1.0).contains(&self.reliability_target)
            || self.scenarios.len() != suite.scenarios.len()
        {
            return Err("micro-combat feasibility report identity is invalid".into());
        }
        for (audit, scenario) in self.scenarios.iter().zip(&suite.scenarios) {
            if audit.scenario != scenario.name
                || audit.objective != scenario.objective
                || audit.witnesses.is_empty()
                || !audit.witnesses.iter().all(|row| {
                    row.episodes == self.seeds.len()
                        && row.objective_successes <= row.episodes
                        && row.objective_success_rate.is_finite()
                        && row.scientific_survival_rate.is_finite()
                })
            {
                return Err(format!(
                    "micro-combat feasibility row {} is invalid",
                    scenario.name
                ));
            }
            let best = audit
                .witnesses
                .iter()
                .max_by(|left, right| witness_order(left, right))
                .expect("non-empty witness rows");
            let wait_rate = audit
                .witnesses
                .iter()
                .find(|row| row.strategy == OpponentProfile::Wait)
                .map(|row| row.objective_success_rate);
            if best.strategy != audit.best_witness
                || best.objective_success_rate != audit.best_witness_success_rate
                || audit.witnessed_feasible != (best.objective_successes > 0)
                || audit.reliably_witnessed
                    != (best.objective_success_rate >= self.reliability_target)
                || audit.passive_timeout_degenerate
                    != (scenario.objective == MicroCombatObjective::Survival
                        && wait_rate.is_some_and(|rate| rate >= self.reliability_target))
            {
                return Err(format!(
                    "micro-combat feasibility classification {} is inconsistent",
                    scenario.name
                ));
            }
        }
        Ok(())
    }
}

fn average(value: impl Into<u128>, episodes: usize) -> f64 {
    if episodes == 0 {
        0.0
    } else {
        value.into() as f64 / episodes as f64
    }
}

fn witness_metrics(
    strategy: OpponentProfile,
    metrics: &MicroCombatScenarioMetrics,
) -> WitnessStrategyMetrics {
    WitnessStrategyMetrics {
        strategy,
        objective_successes: metrics.objective_successes,
        episodes: metrics.episodes,
        objective_success_rate: metrics.objective_success_rate,
        scientific_survival_rate: metrics.scientific_survival_rate,
        average_final_training_cells: average(metrics.final_training_cells_total, metrics.episodes),
        average_final_stored_energy: average(
            metrics.final_training_stored_energy_total,
            metrics.episodes,
        ),
        average_damage_dealt: average(metrics.training_damage_dealt, metrics.episodes),
        average_damage_received: average(metrics.training_damage_received, metrics.episodes),
        average_attacks_committed: average(metrics.training_attacks_committed, metrics.episodes),
        average_guards_committed: average(metrics.training_guards_committed, metrics.episodes),
        average_births: average(metrics.training_births, metrics.episodes),
        average_deaths: average(metrics.training_deaths, metrics.episodes),
    }
}

fn witness_order(
    left: &WitnessStrategyMetrics,
    right: &WitnessStrategyMetrics,
) -> std::cmp::Ordering {
    left.objective_success_rate
        .total_cmp(&right.objective_success_rate)
        .then_with(|| {
            left.average_final_stored_energy
                .total_cmp(&right.average_final_stored_energy)
        })
        // Stable, reproducible tie-break independent of input profile order.
        .then_with(|| right.strategy.as_str().cmp(left.strategy.as_str()))
}

pub fn audit_micro_combat_feasibility(
    base: &EnvConfig,
    reward: &RewardConfig,
    suite: &MicroCombatSuiteConfig,
    seeds: &[u64],
    strategies: &[OpponentProfile],
    reliability_target: f64,
) -> Result<MicroCombatFeasibilityReport, String> {
    suite.validate_against(base)?;
    if seeds.is_empty()
        || strategies.is_empty()
        || !strategies.contains(&OpponentProfile::Wait)
        || !reliability_target.is_finite()
        || !(0.0..=1.0).contains(&reliability_target)
    {
        return Err("invalid micro-combat feasibility audit settings".into());
    }

    let reports = strategies
        .iter()
        .copied()
        .map(|strategy| {
            (
                strategy,
                evaluate_micro_combat_baseline(strategy, base, reward, suite, seeds),
            )
        })
        .collect::<Vec<_>>();
    let ruleset_hash = reports[0].1.ruleset_hash.clone();
    let suite_sha256 = reports[0].1.suite_sha256.clone();
    if reports.iter().any(|(_, report)| {
        report.ruleset_hash != ruleset_hash || report.suite_sha256 != suite_sha256
    }) {
        return Err("witness evaluations do not share one scenario identity".into());
    }

    let scenarios = suite
        .scenarios
        .iter()
        .enumerate()
        .map(|(index, scenario)| {
            let witnesses = reports
                .iter()
                .map(|(strategy, report)| witness_metrics(*strategy, &report.scenarios[index]))
                .collect::<Vec<_>>();
            let best = witnesses
                .iter()
                .max_by(|left, right| witness_order(left, right))
                .expect("strategies are non-empty");
            let wait_rate = witnesses
                .iter()
                .find(|row| row.strategy == OpponentProfile::Wait)
                .expect("Wait is required")
                .objective_success_rate;
            ScenarioFeasibilityAudit {
                scenario: scenario.name.clone(),
                objective: scenario.objective,
                best_witness: best.strategy,
                best_witness_success_rate: best.objective_success_rate,
                witnessed_feasible: best.objective_successes > 0,
                reliably_witnessed: best.objective_success_rate >= reliability_target,
                passive_timeout_degenerate: scenario.objective == MicroCombatObjective::Survival
                    && wait_rate >= reliability_target,
                witnesses,
            }
        })
        .collect();
    let report = MicroCombatFeasibilityReport {
        schema_version: MICRO_COMBAT_FEASIBILITY_SCHEMA_VERSION,
        ruleset_hash,
        suite_sha256,
        seeds: seeds.to_vec(),
        reliability_target,
        scenarios,
    };
    report.validate(suite)?;
    Ok(report)
}

pub fn publish_micro_combat_feasibility(
    output: &Path,
    report: &MicroCombatFeasibilityReport,
) -> Result<PathBuf, String> {
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode feasibility report: {error}"))?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create feasibility report directory: {error}"))?;
    }
    let temporary = output.with_extension("json.tmp");
    fs::write(&temporary, bytes)
        .map_err(|error| format!("failed to write feasibility report: {error}"))?;
    fs::rename(&temporary, output)
        .map_err(|error| format!("failed to publish feasibility report: {error}"))?;
    Ok(output.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::micro_combat::{MicroCombatScenario, MICRO_COMBAT_SUITE_SCHEMA_VERSION};
    use blob_engine::engine::StartingCellLayout;

    #[test]
    fn audit_marks_wait_survival_as_passive_degeneracy() {
        let base = EnvConfig {
            max_episode_len: 64,
            ..EnvConfig::default()
        };
        let suite = MicroCombatSuiteConfig {
            schema_version: MICRO_COMBAT_SUITE_SCHEMA_VERSION,
            scenarios: vec![MicroCombatScenario {
                name: "wait-survival".into(),
                objective: MicroCombatObjective::Survival,
                world_size: 7,
                training_cells: 1,
                opponent_cells: 1,
                training_initial_energy: 100,
                opponent_initial_energy: 100,
                starting_layout: StartingCellLayout::PairedContact,
                opponent: OpponentProfile::Defensive,
                sim_time_limit_quanta: 64,
            }],
        };
        let report = audit_micro_combat_feasibility(
            &base,
            &RewardConfig::default(),
            &suite,
            &[17, 18],
            &[OpponentProfile::Wait, OpponentProfile::Defensive],
            1.0,
        )
        .unwrap();
        assert!(report.scenarios[0].witnessed_feasible);
        assert!(report.scenarios[0].reliably_witnessed);
        assert!(report.scenarios[0].passive_timeout_degenerate);
        report.validate(&suite).unwrap();
    }
}
