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
use crate::control_matrix::MaintainedMindProfile;
use crate::env::{BlobEnv, EpisodeOutcome};
use crate::evaluation::greedy_policy_choices;
use crate::model::PolicyValueNet;
use crate::telemetry::TelemetryConfig;

pub const FEEDING_PROMOTION_SCHEMA_VERSION: u32 = 3;

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
    evaluate_feeding_promotion_with_progress(
        model,
        base_env,
        reward,
        curriculum,
        seeds,
        device,
        |_| {},
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedingEvaluationProgress {
    pub stage: FeedingCurriculumStage,
    pub seed: u64,
    pub completed_in_stage: usize,
    pub total_in_stage: usize,
    pub movement_successes: u64,
    pub consume_successes: u64,
    pub consumed_energy: u128,
    pub surviving_cells: usize,
    pub safety_abort: bool,
    pub succeeded: bool,
}

pub fn evaluate_feeding_promotion_with_progress<B: Backend>(
    model: &PolicyValueNet<B>,
    base_env: &EnvConfig,
    reward: &RewardConfig,
    curriculum: &FeedingCurriculumConfig,
    seeds: &[u64],
    device: &B::Device,
    mut progress: impl FnMut(FeedingEvaluationProgress),
) -> FeedingPromotionReport
where
    f32: From<B::FloatElem>,
{
    assert!(
        !seeds.is_empty(),
        "feeding evaluation requires at least one seed"
    );
    let context = FeedingEvaluationContext {
        base_env,
        reward,
        curriculum,
        seeds,
    };
    let stages = [
        evaluate_stage(
            model,
            &context,
            FeedingCurriculumStage::OnFood,
            device,
            Some(&mut progress),
        ),
        evaluate_stage(
            model,
            &context,
            FeedingCurriculumStage::AdjacentFood,
            device,
            Some(&mut progress),
        ),
    ];
    let ruleset_hash = {
        let env = curriculum.environment_for_stage(base_env, FeedingCurriculumStage::OnFood);
        BlobEnv::new(env, reward.clone(), seeds[0]).compiled_ruleset_hash()
    };
    report_from_metrics(ruleset_hash, seeds.to_vec(), stages, &curriculum.promotion)
}

/// Evaluate a maintained native Mind against the same full-population feeding
/// prerequisites used to qualify learned policies. This is a teacher-quality
/// check, not a shortcut around the canonical Mind boundary: each cell still
/// receives its own anonymous input and private randomness before its decision
/// is routed back to the resolver.
pub fn evaluate_feeding_teacher(
    teacher: MaintainedMindProfile,
    base_env: &EnvConfig,
    reward: &RewardConfig,
    curriculum: &FeedingCurriculumConfig,
    seeds: &[u64],
) -> FeedingPromotionReport {
    assert!(
        !seeds.is_empty(),
        "feeding teacher evaluation requires at least one seed"
    );
    let context = FeedingEvaluationContext {
        base_env,
        reward,
        curriculum,
        seeds,
    };
    let stages = [
        evaluate_teacher_stage(teacher, &context, FeedingCurriculumStage::OnFood),
        evaluate_teacher_stage(teacher, &context, FeedingCurriculumStage::AdjacentFood),
    ];
    let ruleset_hash = {
        let env = curriculum.environment_for_stage(base_env, FeedingCurriculumStage::OnFood);
        BlobEnv::new(env, reward.clone(), seeds[0]).compiled_ruleset_hash()
    };
    report_from_metrics(ruleset_hash, seeds.to_vec(), stages, &curriculum.promotion)
}

struct FeedingEvaluationContext<'a> {
    base_env: &'a EnvConfig,
    reward: &'a RewardConfig,
    curriculum: &'a FeedingCurriculumConfig,
    seeds: &'a [u64],
}

fn evaluate_stage<B: Backend>(
    model: &PolicyValueNet<B>,
    context: &FeedingEvaluationContext<'_>,
    stage: FeedingCurriculumStage,
    device: &B::Device,
    progress: Option<&mut dyn FnMut(FeedingEvaluationProgress)>,
) -> FeedingStageMetrics
where
    f32: From<B::FloatElem>,
{
    evaluate_stage_with(
        context,
        stage,
        |env| {
            let observations = env.get_policy_observations();
            let actions = greedy_policy_choices(model, &observations, device);
            env.step_with_policy_memory(&actions)
        },
        progress,
    )
}

fn evaluate_teacher_stage(
    teacher: MaintainedMindProfile,
    context: &FeedingEvaluationContext<'_>,
    stage: FeedingCurriculumStage,
) -> FeedingStageMetrics {
    evaluate_stage_with(
        context,
        stage,
        |env| {
            let decisions = env
                .prepare_training_reference_inputs()
                .expect("failed to prepare feeding-teacher Mind inputs")
                .into_iter()
                .map(|(cell_id, input)| (cell_id, teacher.decide(&input)))
                .collect();
            env.step_with_training_decisions(decisions)
        },
        None,
    )
}

fn evaluate_stage_with(
    context: &FeedingEvaluationContext<'_>,
    stage: FeedingCurriculumStage,
    mut step: impl FnMut(&mut BlobEnv) -> crate::env::StepOutput,
    mut progress: Option<&mut dyn FnMut(FeedingEvaluationProgress)>,
) -> FeedingStageMetrics {
    let mut successful_episodes = 0usize;
    let mut initial_cells = 0usize;
    let mut surviving_cells = 0usize;
    let mut movement_successes = 0u64;
    let mut consume_successes = 0u64;
    let mut consumed_energy = 0u128;
    let mut safety_aborts = 0usize;

    for (seed_index, &seed) in context.seeds.iter().enumerate() {
        let mut env_config = context
            .curriculum
            .environment_for_stage(context.base_env, stage);
        env_config.opponent = OpponentProfile::Wait;
        env_config.max_episode_len = context.curriculum.promotion.evaluation_max_episode_len;
        env_config.victory.sim_time_limit_quanta = context
            .curriculum
            .promotion
            .evaluation_sim_time_limit_quanta;
        let mut env = BlobEnv::new(env_config, context.reward.clone(), seed);
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

        loop {
            let result = step(&mut env);
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
        if let Some(progress) = progress.as_mut() {
            progress(FeedingEvaluationProgress {
                stage,
                seed,
                completed_in_stage: seed_index + 1,
                total_in_stage: context.seeds.len(),
                movement_successes: episode_movement,
                consume_successes: episode_consumes,
                consumed_energy: episode_energy,
                surviving_cells: if episode_safety_abort {
                    0
                } else {
                    env.training_cells_alive()
                },
                safety_abort: episode_safety_abort,
                succeeded,
            });
        }
    }

    FeedingStageMetrics {
        stage,
        episodes: context.seeds.len(),
        successful_episodes,
        initial_cells,
        surviving_cells,
        movement_successes,
        consume_successes,
        consumed_energy,
        safety_aborts,
        episode_success_rate: ratio(successful_episodes, context.seeds.len()),
        survival_rate: ratio(surviving_cells, initial_cells),
        consumed_energy_per_initial_cell: ratio_u128(consumed_energy, initial_cells),
    }
}

pub fn report_from_metrics(
    ruleset_hash: String,
    seeds: Vec<u64>,
    mut stages: [FeedingStageMetrics; 2],
    gate: &FeedingPromotionGateConfig,
) -> FeedingPromotionReport {
    assert_eq!(stages[0].stage, FeedingCurriculumStage::OnFood);
    assert_eq!(stages[1].stage, FeedingCurriculumStage::AdjacentFood);
    for stage in &mut stages {
        stage.episode_success_rate = ratio(stage.successful_episodes, stage.episodes);
        stage.survival_rate = ratio(stage.surviving_cells, stage.initial_cells);
        stage.consumed_energy_per_initial_cell =
            ratio_u128(stage.consumed_energy, stage.initial_cells);
    }
    let mut checks = Vec::with_capacity(8);
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
    use burn::backend::NdArray;

    use crate::config::TrainingConfig;
    use crate::model::PolicyValueNetConfig;

    fn metrics(
        stage: FeedingCurriculumStage,
        success: f64,
        survival: f64,
        energy: f64,
    ) -> FeedingStageMetrics {
        const SAMPLE_COUNT: usize = 100;
        FeedingStageMetrics {
            stage,
            episodes: SAMPLE_COUNT,
            successful_episodes: (success * SAMPLE_COUNT as f64).round() as usize,
            initial_cells: SAMPLE_COUNT,
            surviving_cells: (survival * SAMPLE_COUNT as f64).round() as usize,
            movement_successes: 10,
            consume_successes: 10,
            consumed_energy: (energy * SAMPLE_COUNT as f64).round() as u128,
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
        let report = report_from_metrics("rules".into(), vec![7; 100], stages, &gate);
        assert!(report.validate_against("rules", &[7; 100], &gate).is_ok());

        let mut verdict = report.clone();
        verdict.passed = false;
        assert!(verdict.validate_against("rules", &[7; 100], &gate).is_err());
        let mut threshold = report.clone();
        threshold.checks[0].minimum = 0.0;
        assert!(threshold
            .validate_against("rules", &[7; 100], &gate)
            .is_err());
        let mut rate = report;
        rate.stages[0].episode_success_rate = 0.5;
        assert!(rate.validate_against("rules", &[7; 100], &gate).is_err());
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

    #[test]
    fn progress_reports_each_completed_stage_seed_in_order() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let mut config = TrainingConfig::default();
        config.env.world_size = 4;
        config.env.cells_per_team = 1;
        config.env.num_plants = 1;
        config.env.num_scattered_energy = 0;
        config
            .feeding_curriculum
            .promotion
            .evaluation_max_episode_len = 32;
        config
            .feeding_curriculum
            .promotion
            .evaluation_sim_time_limit_quanta = 512;
        let device = Default::default();
        let model = PolicyValueNetConfig {
            hidden1: 8,
            hidden2: 8,
            recurrent_size: 8,
        }
        .init::<NdArray>(&device);
        let mut progress = Vec::new();

        evaluate_feeding_promotion_with_progress(
            &model,
            &config.env,
            &config.reward,
            &config.feeding_curriculum,
            &[11, 12],
            &device,
            |event| progress.push(event),
        );

        assert_eq!(progress.len(), 4);
        assert_eq!(progress[0].stage, FeedingCurriculumStage::OnFood);
        assert_eq!(progress[0].seed, 11);
        assert_eq!(progress[0].completed_in_stage, 1);
        assert_eq!(progress[1].seed, 12);
        assert_eq!(progress[2].stage, FeedingCurriculumStage::AdjacentFood);
        assert_eq!(progress[2].seed, 11);
        assert_eq!(progress[3].completed_in_stage, 2);
    }
}
