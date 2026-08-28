//! Strict host-side 1v1/1vN combat scenarios and held-out evidence.

use std::collections::HashSet;

use blob_engine::engine::StartingCellLayout;
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{EnvConfig, OpponentProfile, RewardConfig};
use crate::env::{BlobEnv, EpisodeOutcome, OpponentStartingState, StepOutput};
use crate::evaluation::greedy_policy_choices;
use crate::model::PolicyValueNet;
use crate::telemetry::TelemetryConfig;

pub const MICRO_COMBAT_SUITE_SCHEMA_VERSION: u32 = 1;
pub const MICRO_COMBAT_EVALUATION_SCHEMA_VERSION: u32 = 1;
pub const MICRO_COMBAT_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MicroCombatObjective {
    /// Keep at least one training cell alive through the scientific horizon or
    /// eliminate the opponent earlier.
    #[default]
    Survival,
    /// Eliminate every opponent cell before the scientific horizon.
    Elimination,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MicroCombatScenario {
    pub name: String,
    pub objective: MicroCombatObjective,
    pub world_size: usize,
    pub training_cells: usize,
    pub opponent_cells: usize,
    pub training_initial_energy: u32,
    pub opponent_initial_energy: u32,
    pub starting_layout: StartingCellLayout,
    pub opponent: OpponentProfile,
    pub sim_time_limit_quanta: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MicroCombatSuiteConfig {
    pub schema_version: u32,
    pub scenarios: Vec<MicroCombatScenario>,
}

impl MicroCombatSuiteConfig {
    pub fn from_toml_str(content: &str) -> Result<Self, String> {
        let suite: Self = toml::from_str(content)
            .map_err(|error| format!("failed to parse micro-combat suite: {error}"))?;
        suite.validate()?;
        Ok(suite)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != MICRO_COMBAT_SUITE_SCHEMA_VERSION {
            return Err("unsupported micro-combat suite schema".into());
        }
        if self.scenarios.is_empty() {
            return Err("micro-combat suite must contain at least one scenario".into());
        }
        let mut names = HashSet::with_capacity(self.scenarios.len());
        for scenario in &self.scenarios {
            if !valid_name(&scenario.name) || !names.insert(scenario.name.clone()) {
                return Err("micro-combat scenario names must be unique lowercase slugs".into());
            }
            if scenario.world_size < 2
                || scenario.training_cells == 0
                || scenario.opponent_cells == 0
                || scenario.sim_time_limit_quanta == 0
                || !matches!(
                    scenario.opponent,
                    OpponentProfile::Aggressive
                        | OpponentProfile::StochasticAggressive
                        | OpponentProfile::Defensive
                        | OpponentProfile::Random
                )
            {
                return Err(format!(
                    "micro-combat scenario {} is invalid",
                    scenario.name
                ));
            }
            match scenario.starting_layout {
                StartingCellLayout::PairedContact
                    if scenario.training_cells == 1 && scenario.opponent_cells == 1 => {}
                StartingCellLayout::OpposedLines
                    if scenario.training_cells <= scenario.world_size
                        && scenario.opponent_cells <= scenario.world_size => {}
                _ => {
                    return Err(format!(
                        "micro-combat scenario {} needs 1v1 paired_contact or bounded opposed_lines",
                        scenario.name
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn validate_against(&self, base: &EnvConfig) -> Result<(), String> {
        self.validate()?;
        if base.num_teams != 2 {
            return Err("micro-combat evaluation requires exactly two teams".into());
        }
        for scenario in &self.scenarios {
            for (side, energy) in [
                ("training", scenario.training_initial_energy),
                ("opponent", scenario.opponent_initial_energy),
            ] {
                if u64::from(energy) <= base.rules.minimum_survival_energy
                    || energy > base.max_energy
                {
                    return Err(format!(
                        "{} {side} energy is outside the configured viable range",
                        scenario.name
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn semantic_hash(&self) -> Result<String, String> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| format!("failed to encode micro-combat suite: {error}"))?;
        Ok(sha256(&encoded))
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

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MicroCombatScenarioMetrics {
    pub scenario: String,
    pub objective: MicroCombatObjective,
    pub episodes: usize,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub safety_aborts: usize,
    pub alive_at_end_episodes: usize,
    pub scientific_survival_episodes: usize,
    pub objective_successes: usize,
    pub objective_success_rate: f64,
    pub scientific_survival_rate: f64,
    pub survival_time_quanta_total: u128,
    pub final_training_cells_total: u64,
    pub final_training_stored_energy_total: u128,
    pub training_attacks_committed: u64,
    pub opponent_attacks_committed: u64,
    pub training_guards_committed: u64,
    pub training_damage_dealt: u128,
    pub training_damage_received: u128,
    pub training_guard_mitigation: u128,
    pub training_kills: u64,
}

impl MicroCombatScenarioMetrics {
    fn finalize(&mut self) {
        self.objective_success_rate = ratio(self.objective_successes, self.episodes);
        self.scientific_survival_rate = ratio(self.scientific_survival_episodes, self.episodes);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MicroCombatEvaluationReport {
    pub schema_version: u32,
    pub ruleset_hash: String,
    pub suite_sha256: String,
    pub seeds: Vec<u64>,
    pub scenarios: Vec<MicroCombatScenarioMetrics>,
}

impl MicroCombatEvaluationReport {
    pub fn validate_against(
        &self,
        expected_ruleset_hash: &str,
        suite: &MicroCombatSuiteConfig,
        expected_seeds: &[u64],
    ) -> Result<(), String> {
        suite.validate()?;
        if self.schema_version != MICRO_COMBAT_EVALUATION_SCHEMA_VERSION
            || self.ruleset_hash != expected_ruleset_hash
            || self.suite_sha256 != suite.semantic_hash()?
            || self.seeds != expected_seeds
            || self.seeds.is_empty()
            || self.scenarios.len() != suite.scenarios.len()
        {
            return Err("micro-combat report identity is inconsistent".into());
        }
        for (metrics, scenario) in self.scenarios.iter().zip(&suite.scenarios) {
            if metrics.scenario != scenario.name
                || metrics.objective != scenario.objective
                || metrics.episodes != self.seeds.len()
                || metrics.wins + metrics.losses + metrics.timeouts + metrics.safety_aborts
                    != metrics.episodes
                || metrics.alive_at_end_episodes > metrics.episodes
                || metrics.scientific_survival_episodes > metrics.alive_at_end_episodes
                || metrics.objective_successes > metrics.episodes
                || !close(
                    metrics.objective_success_rate,
                    ratio(metrics.objective_successes, metrics.episodes),
                )
                || !close(
                    metrics.scientific_survival_rate,
                    ratio(metrics.scientific_survival_episodes, metrics.episodes),
                )
            {
                return Err(format!(
                    "micro-combat metrics for {} are inconsistent",
                    scenario.name
                ));
            }
        }
        Ok(())
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn close(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= 1.0e-12
}

fn scenario_environment(base: &EnvConfig, scenario: &MicroCombatScenario) -> EnvConfig {
    let mut env = base.clone();
    env.world_size = scenario.world_size;
    env.cells_per_team = scenario.training_cells;
    env.starting_cell_layout = scenario.starting_layout;
    env.initial_energy = scenario.training_initial_energy;
    env.num_teams = 2;
    env.opponent = scenario.opponent;
    env.num_scattered_energy = 0;
    env.num_plants = 0;
    env.victory.sim_time_limit_quanta = scenario.sim_time_limit_quanta;
    env.max_episode_len = env.max_episode_len.max(256);
    env
}

fn evaluate_with(
    base: &EnvConfig,
    reward: &RewardConfig,
    suite: &MicroCombatSuiteConfig,
    seeds: &[u64],
    mut step_training: impl FnMut(&mut BlobEnv) -> StepOutput,
) -> MicroCombatEvaluationReport {
    suite
        .validate_against(base)
        .expect("invalid micro-combat suite");
    assert!(!seeds.is_empty(), "micro-combat evaluation requires seeds");
    let suite_sha256 = suite.semantic_hash().expect("validated suite must hash");
    let mut ruleset_hash = None;
    let mut all_metrics = Vec::with_capacity(suite.scenarios.len());
    for scenario in &suite.scenarios {
        let environment = scenario_environment(base, scenario);
        let mut metrics = MicroCombatScenarioMetrics {
            scenario: scenario.name.clone(),
            objective: scenario.objective,
            episodes: seeds.len(),
            ..MicroCombatScenarioMetrics::default()
        };
        for &seed in seeds {
            let mut env = BlobEnv::new_with_opponent_starting_state(
                environment.clone(),
                reward.clone(),
                seed,
                OpponentStartingState {
                    cells_per_team: scenario.opponent_cells,
                    initial_energy: scenario.opponent_initial_energy,
                },
            );
            let hash = env.compiled_ruleset_hash();
            if let Some(expected) = &ruleset_hash {
                assert_eq!(expected, &hash, "micro scenarios changed resolver rules");
            } else {
                ruleset_hash = Some(hash);
            }
            env.enable_telemetry(TelemetryConfig {
                enabled: true,
                state_sample_interval_steps: u64::MAX,
                max_state_samples_per_episode: 2,
                episode_log_stride: 0,
            });
            loop {
                let result = step_training(&mut env);
                let telemetry = result
                    .telemetry
                    .as_ref()
                    .expect("micro-combat telemetry is enabled");
                metrics.training_attacks_committed = metrics
                    .training_attacks_committed
                    .saturating_add(telemetry.training.actions.attack.committed);
                metrics.opponent_attacks_committed = metrics
                    .opponent_attacks_committed
                    .saturating_add(telemetry.opponents.actions.attack.committed);
                metrics.training_guards_committed = metrics
                    .training_guards_committed
                    .saturating_add(telemetry.training.actions.guard.committed);
                metrics.training_damage_dealt = metrics
                    .training_damage_dealt
                    .saturating_add(telemetry.training.damage.applied_dealt);
                metrics.training_damage_received = metrics
                    .training_damage_received
                    .saturating_add(telemetry.training.damage.applied_received);
                metrics.training_guard_mitigation = metrics
                    .training_guard_mitigation
                    .saturating_add(telemetry.training.damage.mitigated_by_own_guard);
                metrics.training_kills = metrics
                    .training_kills
                    .saturating_add(telemetry.training.kills);
                if !result.done {
                    continue;
                }
                let outcome = result
                    .outcome
                    .expect("completed micro-combat episode has an outcome");
                match outcome {
                    EpisodeOutcome::Win => metrics.wins += 1,
                    EpisodeOutcome::Loss => metrics.losses += 1,
                    EpisodeOutcome::Timeout => metrics.timeouts += 1,
                    EpisodeOutcome::SafetyAbort => metrics.safety_aborts += 1,
                }
                let alive = result.training_cells > 0;
                metrics.alive_at_end_episodes += usize::from(alive);
                let scientific_survival = alive && outcome != EpisodeOutcome::SafetyAbort;
                metrics.scientific_survival_episodes += usize::from(scientific_survival);
                let success = match scenario.objective {
                    MicroCombatObjective::Survival => scientific_survival,
                    MicroCombatObjective::Elimination => outcome == EpisodeOutcome::Win,
                };
                metrics.objective_successes += usize::from(success);
                metrics.survival_time_quanta_total = metrics
                    .survival_time_quanta_total
                    .saturating_add(u128::from(env.sim_time_quanta()));
                metrics.final_training_cells_total = metrics
                    .final_training_cells_total
                    .saturating_add(result.training_cells as u64);
                let final_sample = env
                    .telemetry_sample()
                    .expect("micro-combat telemetry omitted final state");
                metrics.final_training_stored_energy_total = metrics
                    .final_training_stored_energy_total
                    .saturating_add(final_sample.training_energy.assimilated_energy)
                    .saturating_add(final_sample.training_energy.gut_energy);
                break;
            }
        }
        metrics.finalize();
        all_metrics.push(metrics);
    }
    MicroCombatEvaluationReport {
        schema_version: MICRO_COMBAT_EVALUATION_SCHEMA_VERSION,
        ruleset_hash: ruleset_hash.expect("a nonempty suite and seed list create an environment"),
        suite_sha256,
        seeds: seeds.to_vec(),
        scenarios: all_metrics,
    }
}

pub fn evaluate_micro_combat<B: Backend>(
    model: &PolicyValueNet<B>,
    base: &EnvConfig,
    reward: &RewardConfig,
    suite: &MicroCombatSuiteConfig,
    seeds: &[u64],
    device: &B::Device,
) -> MicroCombatEvaluationReport
where
    f32: From<B::FloatElem>,
{
    evaluate_with(base, reward, suite, seeds, |env| {
        let observations = env.get_policy_observations();
        let actions = greedy_policy_choices(model, &observations, device);
        env.step_with_policy_memory(&actions)
    })
}

/// Deterministic baseline path used to characterize scenario difficulty and
/// verify the evaluator without granting either side privileged state.
pub fn evaluate_micro_combat_baseline(
    training_profile: OpponentProfile,
    base: &EnvConfig,
    reward: &RewardConfig,
    suite: &MicroCombatSuiteConfig,
    seeds: &[u64],
) -> MicroCombatEvaluationReport {
    evaluate_with(base, reward, suite, seeds, |env| {
        env.step_with_baseline(training_profile)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suite() -> MicroCombatSuiteConfig {
        MicroCombatSuiteConfig {
            schema_version: MICRO_COMBAT_SUITE_SCHEMA_VERSION,
            scenarios: vec![
                MicroCombatScenario {
                    name: "one-v-one-survival".into(),
                    objective: MicroCombatObjective::Survival,
                    world_size: 7,
                    training_cells: 1,
                    opponent_cells: 1,
                    training_initial_energy: 100,
                    opponent_initial_energy: 100,
                    starting_layout: StartingCellLayout::PairedContact,
                    opponent: OpponentProfile::Aggressive,
                    sim_time_limit_quanta: 1_024,
                },
                MicroCombatScenario {
                    name: "one-v-three-survival".into(),
                    objective: MicroCombatObjective::Survival,
                    world_size: 7,
                    training_cells: 1,
                    opponent_cells: 3,
                    training_initial_energy: 100,
                    opponent_initial_energy: 60,
                    starting_layout: StartingCellLayout::OpposedLines,
                    opponent: OpponentProfile::StochasticAggressive,
                    sim_time_limit_quanta: 1_024,
                },
            ],
        }
    }

    #[test]
    fn strict_suite_rejects_privileged_or_ambiguous_scenarios() {
        let mut suite = suite();
        suite.validate_against(&EnvConfig::default()).unwrap();
        assert_eq!(suite.semantic_hash().unwrap().len(), 64);

        suite.scenarios[1].name = suite.scenarios[0].name.clone();
        assert!(suite.validate().is_err());
        suite = self::suite();
        suite.scenarios[1].starting_layout = StartingCellLayout::PairedContact;
        assert!(suite.validate().is_err());
        suite = self::suite();
        suite.scenarios[0].training_initial_energy = 1;
        assert!(suite.validate_against(&EnvConfig::default()).is_err());
    }

    #[test]
    fn maintained_suite_covers_asymmetric_survival_and_elimination() {
        let suite = MicroCombatSuiteConfig::from_toml_str(include_str!(
            "../config/micro_combat_scenarios.toml"
        ))
        .unwrap();
        suite.validate_against(&EnvConfig::default()).unwrap();

        assert_eq!(suite.scenarios.len(), 6);
        assert!(suite.scenarios.iter().any(|scenario| {
            scenario.objective == MicroCombatObjective::Survival
                && scenario.training_cells == 1
                && scenario.opponent_cells == 3
                && scenario.opponent == OpponentProfile::StochasticAggressive
        }));
        assert!(suite.scenarios.iter().any(|scenario| {
            scenario.objective == MicroCombatObjective::Elimination
                && scenario.training_cells == 2
                && scenario.opponent_cells == 1
        }));
    }

    #[test]
    fn baseline_evidence_distinguishes_survival_from_offense_and_detects_tampering() {
        let suite = suite();
        let base = EnvConfig {
            max_episode_len: 512,
            ..EnvConfig::default()
        };
        let seeds = [71, 72, 73];
        let report = evaluate_micro_combat_baseline(
            OpponentProfile::Defensive,
            &base,
            &RewardConfig::default(),
            &suite,
            &seeds,
        );
        report
            .validate_against(&report.ruleset_hash, &suite, &seeds)
            .unwrap();
        assert_eq!(report.scenarios.len(), 2);
        assert!(report
            .scenarios
            .iter()
            .all(|metrics| metrics.alive_at_end_episodes <= metrics.episodes));
        assert!(report.scenarios[1].opponent_attacks_committed > 0);

        let mut tampered = report.clone();
        tampered.scenarios[0].scientific_survival_rate = 0.123;
        assert!(tampered
            .validate_against(&report.ruleset_hash, &suite, &seeds)
            .is_err());
    }
}
