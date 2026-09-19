//! Feeding ecology using the exact portable deployment runtime and ordinary
//! cell-private Mind inputs. Host diagnostics never enter policy features.
use crate::{
    config::{FeedingCurriculumStage, TrainingConfig},
    env::{BlobEnv, EpisodeOutcome},
    feeding_curriculum::{report_from_metrics, FeedingPromotionReport, FeedingStageMetrics},
    feeding_layout_evaluation::FeedingQualificationLayout,
    telemetry::{ActionTelemetry, TelemetryConfig},
};
use blob_interface::reference_mind::*;
use blob_policy::{action::action_is_commit_legal, composite::DeployedPolicy};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FEEDING_DEPLOYED_SCHEMA_VERSION: u32 = 2;

/// Explicit host assessment semantics; canonical game victory is unchanged.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum FeedingAssessmentMode {
    #[default]
    CanonicalMatch,
    SustainedFeeding,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpponentExtinctionBoundary {
    pub elapsed_quanta: u64,
    pub steps: u64,
    pub training_cells: usize,
    pub state_hash: String,
    pub movement: ActionTelemetry,
    pub consume: ActionTelemetry,
    pub decisions: DecisionCounts,
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecisionCounts {
    pub decisions: u64,
    pub off_plant_visible_food: u64,
    pub moves_to_visible_food: u64,
    pub moves_elsewhere: u64,
    pub non_moves_with_visible_food: u64,
    pub on_plant: u64,
    pub moves_off_plant: u64,
    pub consumes_on_plant: u64,
    pub illegal_decisions: u64,
}
struct ObservedMind<'a> {
    policy: &'a mut dyn ReferenceMind,
    counts: DecisionCounts,
}
impl ReferenceMind for ObservedMind<'_> {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        let result = self.policy.decide(input);
        let c = &mut self.counts;
        c.decisions += 1;
        c.illegal_decisions += u64::from(!action_is_commit_legal(
            input,
            &result.action,
            result.signal.is_some(),
        ));
        let on_plant = input.current_tile.plant_capacity > 0;
        let food = |slot: &LocalObservation| {
            slot.reachable
                && slot.neighbor.is_none()
                && (slot.plant_energy.unwrap_or(0) > 0 || slot.loose_energy.unwrap_or(0) > 0)
        };
        if on_plant {
            c.on_plant += 1;
            c.moves_off_plant +=
                u64::from(matches!(result.action, ReferenceMindAction::Move { .. }));
            c.consumes_on_plant +=
                u64::from(matches!(result.action, ReferenceMindAction::Consume { .. }));
        } else if input.slots.iter().any(food) {
            c.off_plant_visible_food += 1;
            if let ReferenceMindAction::Move { target_slot, .. } = result.action {
                let to_food = input
                    .slots
                    .iter()
                    .any(|slot| slot.slot == target_slot && food(slot));
                c.moves_to_visible_food += u64::from(to_food);
                c.moves_elsewhere += u64::from(!to_food);
            } else {
                c.non_moves_with_visible_food += 1;
            }
        }
        result
    }
    fn reset(&mut self) -> Result<(), String> {
        self.policy.reset()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeployedFeedingEpisode {
    pub stage: FeedingCurriculumStage,
    pub seed: u64,
    pub effective_env_sha256: String,
    pub initial_state_hash: String,
    pub first_frontier_sha256: String,
    pub final_state_hash: String,
    pub opponent_extinction: Option<OpponentExtinctionBoundary>,
    pub ruleset_hash: String,
    pub elapsed_quanta: u64,
    pub steps: u64,
    pub outcome: String,
    pub end_reason: String,
    pub movement: ActionTelemetry,
    pub consume: ActionTelemetry,
    pub decisions: DecisionCounts,
    pub metrics: FeedingStageMetrics,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeployedFeedingTrial {
    pub schema_version: u32,
    pub execution_contract: String,
    pub assessment_mode: FeedingAssessmentMode,
    pub layout: FeedingQualificationLayout,
    pub seed: u64,
    pub episodes: Vec<DeployedFeedingEpisode>,
    pub promotion: FeedingPromotionReport,
}

pub fn evaluate_deployed_feeding(
    policy: &DeployedPolicy,
    config: &TrainingConfig,
    layout: FeedingQualificationLayout,
    seed: u64,
    progress: impl FnMut(FeedingCurriculumStage, u64, u64, usize),
) -> Result<DeployedFeedingTrial, String> {
    evaluate_deployed_feeding_with_mode(
        policy,
        config,
        layout,
        seed,
        FeedingAssessmentMode::CanonicalMatch,
        progress,
    )
}

pub fn evaluate_deployed_feeding_with_mode(
    policy: &DeployedPolicy,
    config: &TrainingConfig,
    layout: FeedingQualificationLayout,
    seed: u64,
    assessment_mode: FeedingAssessmentMode,
    progress: impl FnMut(FeedingCurriculumStage, u64, u64, usize),
) -> Result<DeployedFeedingTrial, String> {
    evaluate_feeding_mind(
        &mut policy.clone(),
        policy.execution_contract(),
        config,
        layout,
        seed,
        assessment_mode,
        progress,
    )
}

/// Evaluate an instrumented Mind through the same feeding boundary and telemetry.
/// `execution_contract` is caller-supplied provenance, not a qualification claim.
pub fn evaluate_feeding_mind(
    policy: &mut dyn ReferenceMind,
    execution_contract: &str,
    config: &TrainingConfig,
    layout: FeedingQualificationLayout,
    seed: u64,
    assessment_mode: FeedingAssessmentMode,
    mut progress: impl FnMut(FeedingCurriculumStage, u64, u64, usize),
) -> Result<DeployedFeedingTrial, String> {
    config.validate()?;
    let mut episodes = Vec::new();
    for stage in [
        FeedingCurriculumStage::OnFood,
        FeedingCurriculumStage::AdjacentFood,
    ] {
        let mut env_config = config
            .feeding_curriculum
            .environment_for_stage(&config.env, stage);
        env_config.starting_cell_layout = layout.starting_layout();
        env_config.max_episode_len = config
            .feeding_curriculum
            .promotion
            .evaluation_max_episode_len;
        env_config.victory.sim_time_limit_quanta = config
            .feeding_curriculum
            .promotion
            .evaluation_sim_time_limit_quanta;
        let effective_env_sha256 =
            crate::sweep::sha256(&serde_json::to_vec(&env_config).map_err(|e| e.to_string())?);
        let mut env = BlobEnv::new(env_config, config.reward.clone(), seed);
        let initial_state_hash = env
            .reference_simulation_for_diagnostics()?
            .state_hash()
            .to_hex();
        let ruleset_hash = env.compiled_ruleset_hash();
        let initial = env.training_cells_alive();
        // Bind paired initial observations, private memory and private draws
        // without publishing host cell identifiers or match-secret material.
        let mut frontier_hash = Sha256::new();
        for (_, input) in env.prepare_training_reference_inputs()? {
            let bytes = blob_interface::reference_mind_converter::reference_mind_input_to_capnp(
                &input,
                blob_interface::reference_mind_converter::ReferenceMindLimits::default(),
            )
            .map_err(|e| e.to_string())?;
            frontier_hash.update((bytes.len() as u64).to_le_bytes());
            frontier_hash.update(bytes);
        }
        let first_frontier_sha256 = format!("{:x}", frontier_hash.finalize());
        env.enable_telemetry(TelemetryConfig {
            enabled: true,
            state_sample_interval_steps: u64::MAX,
            max_state_samples_per_episode: 2,
            episode_log_stride: 0,
        });
        policy.reset()?;
        let mut mind = ObservedMind {
            policy,
            counts: DecisionCounts::default(),
        };
        let mut movement = ActionTelemetry::default();
        let mut consume = ActionTelemetry::default();
        let mut opponent_extinction = None;
        let result = loop {
            let result = match assessment_mode {
                FeedingAssessmentMode::CanonicalMatch => env.step_with_reference_mind(&mut mind),
                FeedingAssessmentMode::SustainedFeeding => {
                    env.step_with_survival_assessment(&mut mind)
                }
            };
            let telemetry = result
                .telemetry
                .as_ref()
                .expect("feeding telemetry enabled");
            movement.merge(&telemetry.training.actions.movement);
            consume.merge(&telemetry.training.actions.consume);
            if result.opponent_cells == 0 && opponent_extinction.is_none() {
                opponent_extinction = Some(OpponentExtinctionBoundary {
                    elapsed_quanta: env.sim_time_quanta(),
                    steps: result.episode_step,
                    training_cells: result.training_cells,
                    state_hash: env
                        .reference_simulation_for_diagnostics()?
                        .state_hash()
                        .to_hex(),
                    movement: movement.clone(),
                    consume: consume.clone(),
                    decisions: mind.counts.clone(),
                });
            }
            if result.episode_step.is_multiple_of(256) || result.done {
                progress(
                    stage,
                    result.episode_step,
                    env.sim_time_quanta(),
                    result.training_cells,
                );
            }
            if result.done {
                break result;
            }
        };
        let safety_abort = result.outcome == Some(EpisodeOutcome::SafetyAbort);
        let surviving = if safety_abort {
            0
        } else {
            result.training_cells
        };
        let succeeded = !safety_abort
            && consume.succeeded > 0
            && consume.consumed_energy > 0
            && (stage == FeedingCurriculumStage::OnFood || movement.succeeded > 0);
        let metrics = FeedingStageMetrics {
            stage,
            episodes: 1,
            successful_episodes: usize::from(succeeded),
            initial_cells: initial,
            surviving_cells: surviving,
            movement_successes: movement.succeeded,
            consume_successes: consume.succeeded,
            consumed_energy: consume.consumed_energy,
            safety_aborts: usize::from(safety_abort),
            episode_success_rate: f64::from(succeeded),
            survival_rate: if initial == 0 {
                0.
            } else {
                surviving as f64 / initial as f64
            },
            consumed_energy_per_initial_cell: if initial == 0 {
                0.
            } else {
                consume.consumed_energy as f64 / initial as f64
            },
        };
        episodes.push(DeployedFeedingEpisode {
            stage,
            seed,
            opponent_extinction,
            effective_env_sha256,
            initial_state_hash,
            first_frontier_sha256,
            final_state_hash: env
                .reference_simulation_for_diagnostics()?
                .state_hash()
                .to_hex(),
            ruleset_hash,
            elapsed_quanta: env.sim_time_quanta(),
            steps: result.episode_step,
            outcome: format!("{:?}", result.outcome),
            end_reason: format!("{:?}", result.end_reason),
            movement,
            consume,
            decisions: mind.counts,
            metrics,
        });
    }
    let promotion = report_from_metrics(
        episodes[0].ruleset_hash.clone(),
        vec![seed],
        [episodes[0].metrics.clone(), episodes[1].metrics.clone()],
        &config.feeding_curriculum.promotion,
    );
    promotion.validate_against(
        &episodes[0].ruleset_hash,
        &[seed],
        &config.feeding_curriculum.promotion,
    )?;
    Ok(DeployedFeedingTrial {
        schema_version: FEEDING_DEPLOYED_SCHEMA_VERSION,
        assessment_mode,
        execution_contract: execution_contract.into(),
        layout,
        seed,
        episodes,
        promotion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn instrumented_mind_reset_failure_prevents_any_decision() {
        struct Broken;
        impl ReferenceMind for Broken {
            fn reset(&mut self) -> Result<(), String> {
                Err("reset failed".into())
            }
            fn decide(&mut self, _: &ReferenceMindInput) -> ReferenceMindDecision {
                panic!("must not evaluate after reset failure")
            }
        }
        let config = TrainingConfig::default();
        let result = evaluate_feeding_mind(
            &mut Broken,
            "test",
            &config,
            FeedingQualificationLayout::Line,
            333,
            FeedingAssessmentMode::CanonicalMatch,
            |_, _, _, _| {},
        );
        assert_eq!(result.unwrap_err(), "reset failed");
    }

    #[test]
    fn deployed_wait_matches_baseline_at_the_same_frontier() {
        use blob_policy::runtime::*;
        let parent = FrozenPolicy::new(
            4,
            3,
            2,
            layer_shapes(4, 3, 2)
                .unwrap()
                .into_iter()
                .map(|(_, i, o)| Linear::new(i, o, vec![0.; i * o], vec![0.; o]).unwrap())
                .collect(),
        )
        .unwrap();
        let mut config = TrainingConfig::default();
        config.env.world_size = 16;
        config.env.cells_per_team = 4;
        config.env.starting_cell_layout = blob_engine::engine::StartingCellLayout::Line;
        config
            .feeding_curriculum
            .promotion
            .evaluation_max_episode_len = 32;
        config
            .feeding_curriculum
            .promotion
            .evaluation_sim_time_limit_quanta = 1024;
        let actual = evaluate_deployed_feeding(
            &DeployedPolicy::Greedy(parent),
            &config,
            FeedingQualificationLayout::Line,
            333,
            |_, _, _, _| {},
        )
        .unwrap();
        for episode in &actual.episodes {
            let mut ec = config
                .feeding_curriculum
                .environment_for_stage(&config.env, episode.stage);
            ec.max_episode_len = 32;
            ec.victory.sim_time_limit_quanta = 1024;
            let mut env = BlobEnv::new(ec, config.reward.clone(), 333);
            assert_eq!(
                episode.initial_state_hash,
                env.reference_simulation_for_diagnostics()
                    .unwrap()
                    .state_hash()
                    .to_hex()
            );
            let terminal = loop {
                let result = env.step_with_baseline(crate::config::OpponentProfile::Wait);
                if result.done {
                    break result;
                }
            };
            assert_eq!(episode.steps, terminal.episode_step);
            assert_eq!(episode.elapsed_quanta, env.sim_time_quanta());
            assert_eq!(episode.outcome, format!("{:?}", terminal.outcome));
            assert_eq!(episode.metrics.surviving_cells, terminal.training_cells);
            assert_eq!(episode.movement.completed, 0);
            assert_eq!(episode.consume.completed, 0);
            assert_eq!(episode.decisions.illegal_decisions, 0);
            assert!(episode.decisions.decisions > 0);
        }
    }
}
