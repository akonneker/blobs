//! Deterministic feeding-competency evaluation and promotion gating.
//!
//! These probes use the ordinary Mind observation/action boundary and the
//! canonical resolver.  Their only intervention is deterministic initial
//! resource placement, so passing cannot depend on privileged state.

use burn::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::{
    EnvConfig, FeedingCurriculumConfig, FeedingCurriculumStage, FeedingPromotionGateConfig,
    OpponentProfile, RewardConfig,
};
use crate::env::{BlobEnv, EpisodeOutcome};
use crate::evaluation::greedy_policy_choices;
use crate::model::PolicyValueNet;
use crate::telemetry::TelemetryConfig;

pub const FEEDING_PROMOTION_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingStageMetrics {
    pub stage: FeedingCurriculumStage,
    pub episodes: usize,
    pub successful_episodes: usize,
    pub initial_cells: usize,
    pub surviving_cells: usize,
    pub movement_successes: u64,
    pub consume_successes: u64,
    pub consumed_energy: u128,
    pub safety_aborts: usize,
    pub episode_success_rate: f64,
    pub survival_rate: f64,
    pub consumed_energy_per_initial_cell: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingGateCheck {
    pub name: String,
    pub observed: f64,
    pub minimum: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FeedingPromotionReport {
    pub schema_version: u32,
    pub ruleset_hash: String,
    pub seeds: Vec<u64>,
    pub stages: Vec<FeedingStageMetrics>,
    pub checks: Vec<FeedingGateCheck>,
    pub passed: bool,
}

impl FeedingPromotionReport {
    /// Recompute every derived rate and gate decision from the raw stage
    /// counters. This prevents artifact readers from trusting a self-reported
    /// `passed` bit or altered threshold/check rows.
    pub fn validate_against(
        &self,
        expected_ruleset_hash: &str,
        expected_seeds: &[u64],
        gate: &FeedingPromotionGateConfig,
    ) -> Result<(), String> {
        if self.schema_version != FEEDING_PROMOTION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported feeding-promotion schema {}; expected {}",
                self.schema_version, FEEDING_PROMOTION_SCHEMA_VERSION
            ));
        }
        if self.ruleset_hash != expected_ruleset_hash || self.seeds != expected_seeds {
            return Err("feeding-promotion identity does not match checkpoint evaluation".into());
        }
        if self.seeds.is_empty()
            || self.stages.len() != 2
            || self.stages[0].stage != FeedingCurriculumStage::OnFood
            || self.stages[1].stage != FeedingCurriculumStage::AdjacentFood
        {
            return Err("feeding-promotion stage suite is incomplete or misordered".into());
        }
        for stage in &self.stages {
            if stage.episodes != self.seeds.len()
                || stage.successful_episodes > stage.episodes
                || stage.surviving_cells > stage.initial_cells
                || stage.safety_aborts > stage.episodes
                || !stage.episode_success_rate.is_finite()
                || !stage.survival_rate.is_finite()
                || !stage.consumed_energy_per_initial_cell.is_finite()
            {
                return Err("feeding-promotion raw counters or rates are invalid".into());
            }
        }
        let expected = report_from_metrics(
            expected_ruleset_hash.to_owned(),
            expected_seeds.to_vec(),
            [self.stages[0].clone(), self.stages[1].clone()],
            gate,
        );
        if *self != expected {
            return Err(
                "feeding-promotion rates, thresholds, checks, or verdict were modified".into(),
            );
        }
        Ok(())
    }
}

/// Evaluate the two prerequisite skills independently of competitive win rate.
/// An adjacent-food episode succeeds only after both movement and extraction;
/// merely beginning beside food or surviving is insufficient.
pub fn evaluate_feeding_promotion<B: Backend>(
    model: &PolicyValueNet<B>,
    base_env: &EnvConfig,
    reward: &RewardConfig,
    curriculum: &FeedingCurriculumConfig,
    seeds: &[u64],
    device: &B::Device,
) -> FeedingPromotionReport
where
    f32: From<B::FloatElem>,
{
    assert!(
        !seeds.is_empty(),
        "feeding evaluation requires at least one seed"
    );
    let stages = [
        evaluate_stage(
            model,
            base_env,
            reward,
            curriculum,
            FeedingCurriculumStage::OnFood,
            seeds,
            device,
        ),
        evaluate_stage(
            model,
            base_env,
            reward,
            curriculum,
            FeedingCurriculumStage::AdjacentFood,
            seeds,
            device,
        ),
    ];
    let ruleset_hash = {
        let env = curriculum.environment_for_stage(base_env, FeedingCurriculumStage::OnFood);
        BlobEnv::new(env, reward.clone(), seeds[0]).compiled_ruleset_hash()
    };
    report_from_metrics(ruleset_hash, seeds.to_vec(), stages, &curriculum.promotion)
}

fn evaluate_stage<B: Backend>(
    model: &PolicyValueNet<B>,
    base_env: &EnvConfig,
    reward: &RewardConfig,
    curriculum: &FeedingCurriculumConfig,
    stage: FeedingCurriculumStage,
    seeds: &[u64],
    device: &B::Device,
) -> FeedingStageMetrics
where
    f32: From<B::FloatElem>,
{
    let mut successful_episodes = 0usize;
    let mut initial_cells = 0usize;
    let mut surviving_cells = 0usize;
    let mut movement_successes = 0u64;
    let mut consume_successes = 0u64;
    let mut consumed_energy = 0u128;
    let mut safety_aborts = 0usize;

    for &seed in seeds {
        let mut env_config = curriculum.environment_for_stage(base_env, stage);
        env_config.opponent = OpponentProfile::Wait;
        env_config.max_episode_len = curriculum.promotion.evaluation_max_episode_len;
        env_config.victory.sim_time_limit_quanta =
            curriculum.promotion.evaluation_sim_time_limit_quanta;
        let mut env = BlobEnv::new(env_config, reward.clone(), seed);
        env.enable_telemetry(TelemetryConfig {
            enabled: true,
            state_sample_interval_steps: u64::MAX,
            max_state_samples_per_episode: 2,
            episode_log_stride: 0,
        });
        let episode_initial_cells = env.training_cells_alive();
        initial_cells = initial_cells.saturating_add(episode_initial_cells);
        let mut episode_movement = 0u64;
        let mut episode_consumes = 0u64;
        let mut episode_energy = 0u128;
        let mut episode_safety_abort = false;
        let mut observations = env.get_policy_observations();

        loop {
            let actions = greedy_policy_choices(model, &observations, device);
            let result = env.step_with_policy_memory(&actions);
            let telemetry = result
                .telemetry
                .as_ref()
                .expect("feeding evaluation telemetry is enabled");
            episode_movement =
                episode_movement.saturating_add(telemetry.training.actions.movement.succeeded);
            episode_consumes =
                episode_consumes.saturating_add(telemetry.training.actions.consume.succeeded);
            episode_energy =
                episode_energy.saturating_add(telemetry.training.actions.consume.consumed_energy);

            if result.done {
                if result.outcome == Some(EpisodeOutcome::SafetyAbort) {
                    episode_safety_abort = true;
                    safety_aborts = safety_aborts.saturating_add(1);
                } else {
                    surviving_cells = surviving_cells.saturating_add(result.training_cells);
                }
                break;
            }
            observations = result.policy_observations;
        }

        let succeeded = !episode_safety_abort
            && match stage {
                FeedingCurriculumStage::OnFood => episode_consumes > 0 && episode_energy > 0,
                FeedingCurriculumStage::AdjacentFood => {
                    episode_movement > 0 && episode_consumes > 0 && episode_energy > 0
                }
                FeedingCurriculumStage::Contact | FeedingCurriculumStage::Skirmish => {
                    unreachable!("not a feeding prerequisite")
                }
                FeedingCurriculumStage::Competitive => unreachable!("not a feeding prerequisite"),
            };
        successful_episodes = successful_episodes.saturating_add(usize::from(succeeded));
        movement_successes = movement_successes.saturating_add(episode_movement);
        consume_successes = consume_successes.saturating_add(episode_consumes);
        consumed_energy = consumed_energy.saturating_add(episode_energy);
    }

    FeedingStageMetrics {
        stage,
        episodes: seeds.len(),
        successful_episodes,
        initial_cells,
        surviving_cells,
        movement_successes,
        consume_successes,
        consumed_energy,
        safety_aborts,
        episode_success_rate: ratio(successful_episodes, seeds.len()),
        survival_rate: ratio(surviving_cells, initial_cells),
        consumed_energy_per_initial_cell: ratio_u128(consumed_energy, initial_cells),
    }
}

pub fn report_from_metrics(
    ruleset_hash: String,
    seeds: Vec<u64>,
    stages: [FeedingStageMetrics; 2],
    gate: &FeedingPromotionGateConfig,
) -> FeedingPromotionReport {
    assert_eq!(stages[0].stage, FeedingCurriculumStage::OnFood);
    assert_eq!(stages[1].stage, FeedingCurriculumStage::AdjacentFood);
    let mut checks = Vec::with_capacity(6);
    add_check(
        &mut checks,
        "on_food_episode_success_rate",
        stages[0].episode_success_rate,
        gate.min_on_food_episode_success_rate,
    );
    add_check(
        &mut checks,
        "adjacent_food_episode_success_rate",
        stages[1].episode_success_rate,
        gate.min_adjacent_episode_success_rate,
    );
    for stage in &stages {
        add_check(
            &mut checks,
            &format!("{}_no_safety_aborts", stage.stage),
            f64::from(stage.safety_aborts == 0),
            1.0,
        );
        add_check(
            &mut checks,
            &format!("{}_survival_rate", stage.stage),
            stage.survival_rate,
            gate.min_survival_rate,
        );
        add_check(
            &mut checks,
            &format!("{}_consumed_energy_per_initial_cell", stage.stage),
            stage.consumed_energy_per_initial_cell,
            gate.min_consumed_energy_per_initial_cell,
        );
    }
    let passed = !checks.is_empty() && checks.iter().all(|check| check.passed);
    FeedingPromotionReport {
        schema_version: FEEDING_PROMOTION_SCHEMA_VERSION,
        ruleset_hash,
        seeds,
        stages: stages.into(),
        checks,
        passed,
    }
}

fn add_check(checks: &mut Vec<FeedingGateCheck>, name: &str, observed: f64, minimum: f64) {
    checks.push(FeedingGateCheck {
        name: name.to_owned(),
        observed,
        minimum,
        passed: observed.is_finite() && observed >= minimum,
    });
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn ratio_u128(numerator: u128, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(
        stage: FeedingCurriculumStage,
        success: f64,
        survival: f64,
        energy: f64,
    ) -> FeedingStageMetrics {
        FeedingStageMetrics {
            stage,
            episodes: 10,
            successful_episodes: (success * 10.0) as usize,
            initial_cells: 10,
            surviving_cells: (survival * 10.0) as usize,
            movement_successes: 10,
            consume_successes: 10,
            consumed_energy: (energy * 10.0) as u128,
            safety_aborts: 0,
            episode_success_rate: success,
            survival_rate: survival,
            consumed_energy_per_initial_cell: energy,
        }
    }

    #[test]
    fn promotion_gate_is_inclusive_at_every_threshold() {
        let gate = FeedingPromotionGateConfig::default();
        let report = report_from_metrics(
            "rules".into(),
            vec![1],
            [
                metrics(
                    FeedingCurriculumStage::OnFood,
                    gate.min_on_food_episode_success_rate,
                    gate.min_survival_rate,
                    gate.min_consumed_energy_per_initial_cell,
                ),
                metrics(
                    FeedingCurriculumStage::AdjacentFood,
                    gate.min_adjacent_episode_success_rate,
                    gate.min_survival_rate,
                    gate.min_consumed_energy_per_initial_cell,
                ),
            ],
            &gate,
        );
        assert!(report.passed);
        assert_eq!(report.checks.len(), 8);
    }

    #[test]
    fn promotion_gate_fails_closed_for_non_finite_or_weak_metrics() {
        let gate = FeedingPromotionGateConfig::default();
        let report = report_from_metrics(
            "rules".into(),
            vec![1],
            [
                metrics(FeedingCurriculumStage::OnFood, f64::NAN, 1.0, 2.0),
                metrics(FeedingCurriculumStage::AdjacentFood, 1.0, 0.7, 2.0),
            ],
            &gate,
        );
        assert!(!report.passed);
        assert_eq!(
            report.checks.iter().filter(|check| !check.passed).count(),
            2
        );
    }

    #[test]
    fn report_validation_recomputes_verdicts_and_thresholds() {
        let gate = FeedingPromotionGateConfig::default();
        let stages = [
            metrics(FeedingCurriculumStage::OnFood, 1.0, 1.0, 2.0),
            metrics(FeedingCurriculumStage::AdjacentFood, 1.0, 1.0, 2.0),
        ];
        let report = report_from_metrics("rules".into(), vec![7; 10], stages, &gate);
        assert!(report.validate_against("rules", &[7; 10], &gate).is_ok());

        let mut verdict = report.clone();
        verdict.passed = false;
        assert!(verdict.validate_against("rules", &[7; 10], &gate).is_err());
        let mut threshold = report.clone();
        threshold.checks[0].minimum = 0.0;
        assert!(threshold
            .validate_against("rules", &[7; 10], &gate)
            .is_err());
        let mut rate = report;
        rate.stages[0].episode_success_rate = 0.5;
        assert!(rate.validate_against("rules", &[7; 10], &gate).is_err());
    }

    #[test]
    fn safety_abort_is_a_hard_feeding_failure() {
        let gate = FeedingPromotionGateConfig::default();
        let mut on_food = metrics(FeedingCurriculumStage::OnFood, 1.0, 1.0, 2.0);
        on_food.safety_aborts = 1;
        let report = report_from_metrics(
            "rules".into(),
            vec![1],
            [
                on_food,
                metrics(FeedingCurriculumStage::AdjacentFood, 1.0, 1.0, 2.0),
            ],
            &gate,
        );
        assert!(!report.passed);
        assert!(report
            .checks
            .iter()
            .any(|check| check.name == "on_food_no_safety_aborts" && !check.passed));
    }
}
