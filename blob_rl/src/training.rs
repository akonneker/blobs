use burn::module::AutodiffModule;
use burn::optim::grad_clipping::GradientClipping;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::action::{
    compose_policy_action, policy_action_kind_mask, policy_effort_mask, policy_target_mask,
    policy_wait_action, HierarchicalActionChoice, PolicyActionKind, PolicyChoice,
    NUM_AMOUNT_CHOICES, NUM_POLICY_ACTION_KINDS, NUM_POLICY_AMOUNT_LOGITS, NUM_POLICY_EFFORTS,
    NUM_POLICY_EFFORT_LOGITS, NUM_POLICY_TARGETS, NUM_POLICY_TARGET_LOGITS, NUM_SIGNAL_CHOICES,
    NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::artifact::{
    load_checkpoint, load_policy_snapshot, load_rollout_snapshot, publish_best_pointer,
    publish_checkpoint, publish_rollout_pool_manifest, LeaguePromotion, PolicySnapshot,
    RolloutLeagueMember, RolloutOpponentAssignment, TrainingResumeState,
    TRAINING_ARTIFACT_SCHEMA_VERSION,
};
use crate::competency_frontier::{
    publish_competency_frontier, CompetencyCheckpointIdentity, CompetencyFrontier,
    CompetencyFrontierEntry, CompetencyMetrics, SpecialistTeacherSelection,
};
use crate::config::{FeedingCurriculumStage, OpponentProfile, SelfPlayConfig, TrainingConfig};
use crate::contact_evaluation::{evaluate_contact, ContactEvaluationReport};
use crate::env::{BlobEnv, EpisodeOutcome, OpponentStartingState, PolicyObservation};
use crate::evaluation::{evaluate_policy_suite, EvaluationMetrics, EvaluationOpponent};
use crate::feeding_curriculum::{evaluate_feeding_promotion, FeedingPromotionReport};
use crate::feeding_evaluation_artifact::verify_feeding_initial_policy;
use crate::micro_combat::{
    evaluate_micro_combat, MicroCombatEvaluationReport, MicroCombatRotationState,
    MicroCombatTrainingConfig,
};
use crate::model::{
    decode_policy_memory, encode_policy_memory, PolicyValueNet, PolicyValueNetConfig,
};
use crate::observation::OBS_DIM;
use crate::policy_artifact::load_behavior_clone;
use crate::ppo::{ppo_update_anchored, PolicyAnchorTarget, RolloutBuffer, Transition};
use crate::telemetry::{
    publish_episode_record, publish_training_summary, TelemetryEpisodeOutcome,
    TrainingTelemetryState,
};
use blob_interface::types::CellId;

#[derive(Debug, Clone, Copy)]
struct PendingTransition {
    rollout_index: usize,
    started_at: u64,
}

#[cfg(unix)]
fn peak_resident_set_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` initializes the supplied `rusage` on success, and
    // the pointer refers to writable storage of the exact C type.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the successful call above initialized the structure.
    let maximum = unsafe { usage.assume_init() }.ru_maxrss;
    let maximum = u64::try_from(maximum).ok()?;
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        Some(maximum)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        maximum.checked_mul(1024)
    }
}

#[cfg(not(unix))]
fn peak_resident_set_bytes() -> Option<u64> {
    None
}

#[derive(Debug, Clone, Copy)]
struct RolloutCurriculumAssignment {
    stage: FeedingCurriculumStage,
    simulation_time_quanta: u64,
    micro_combat_scenario: Option<usize>,
}

fn curriculum_label(
    config: &TrainingConfig,
    stage: FeedingCurriculumStage,
    micro_combat_scenario: Option<usize>,
) -> String {
    micro_combat_scenario.map_or_else(
        || stage.to_string(),
        |index| {
            let scenario = config
                .combat_curriculum
                .micro_combat
                .scenario(index)
                .expect("assigned micro-combat scenario exists");
            format!("{stage}:{}", scenario.name)
        },
    )
}

struct SpecialistTeachers<B: Backend> {
    ecology: Option<PolicySnapshot<B>>,
    combat: Option<PolicySnapshot<B>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnchorTeacher {
    Initial,
    Ecology,
    Combat,
}

fn anchor_teacher_for_stage(
    config: &TrainingConfig,
    stage: FeedingCurriculumStage,
    has_initial: bool,
    has_ecology: bool,
    has_combat: bool,
) -> Option<(AnchorTeacher, f32)> {
    if config.specialist_distillation.enabled {
        match stage {
            FeedingCurriculumStage::OnFood | FeedingCurriculumStage::AdjacentFood
                if has_ecology && config.specialist_distillation.ecology_coeff > 0.0 =>
            {
                return Some((
                    AnchorTeacher::Ecology,
                    config.specialist_distillation.ecology_coeff,
                ));
            }
            FeedingCurriculumStage::Contact | FeedingCurriculumStage::Skirmish
                if has_combat && config.specialist_distillation.combat_coeff > 0.0 =>
            {
                return Some((
                    AnchorTeacher::Combat,
                    config.specialist_distillation.combat_coeff,
                ));
            }
            _ => {}
        }
    }
    has_initial.then(|| {
        (
            AnchorTeacher::Initial,
            config.rollout_initial_policy_anchor_coeff(stage),
        )
    })
}

fn infer_anchor_targets<B: Backend>(
    model: &PolicyValueNet<B>,
    observations: &[f32],
    memory: &[f32],
    rows: &[usize],
    recurrent_size: usize,
    device: &B::Device,
) -> Vec<PolicyAnchorTarget>
where
    f32: From<B::FloatElem>,
{
    if rows.is_empty() {
        return Vec::new();
    }
    let mut selected_observations = Vec::with_capacity(rows.len() * OBS_DIM);
    let mut selected_memory = Vec::with_capacity(rows.len() * recurrent_size);
    for &row in rows {
        let observation_start = row * OBS_DIM;
        let memory_start = row * recurrent_size;
        selected_observations
            .extend_from_slice(&observations[observation_start..observation_start + OBS_DIM]);
        selected_memory.extend_from_slice(&memory[memory_start..memory_start + recurrent_size]);
    }
    let output = model.forward_with_memory(
        Tensor::<B, 2>::from_data(
            TensorData::new(selected_observations, [rows.len(), OBS_DIM]),
            device,
        ),
        Tensor::<B, 2>::from_data(
            TensorData::new(selected_memory, [rows.len(), recurrent_size]),
            device,
        ),
    );
    let width = NUM_POLICY_ACTION_KINDS
        + NUM_POLICY_TARGET_LOGITS
        + NUM_POLICY_EFFORT_LOGITS
        + NUM_POLICY_AMOUNT_LOGITS
        + NUM_SIGNAL_CHOICES
        + NUM_SIGNAL_STRENGTH_CHOICES
        + recurrent_size;
    let data: Vec<f32> = Tensor::cat(
        vec![
            output.action_kind_logits,
            output.target_logits,
            output.effort_logits,
            output.amount_logits,
            output.signal_logits,
            output.signal_strength_logits,
            output.next_memory,
        ],
        1,
    )
    .into_data()
    .to_vec()
    .unwrap();
    data.chunks_exact(width)
        .map(|row| {
            let target = NUM_POLICY_ACTION_KINDS;
            let effort = target + NUM_POLICY_TARGET_LOGITS;
            let amount = effort + NUM_POLICY_EFFORT_LOGITS;
            let signal = amount + NUM_POLICY_AMOUNT_LOGITS;
            let strength = signal + NUM_SIGNAL_CHOICES;
            let memory = strength + NUM_SIGNAL_STRENGTH_CHOICES;
            PolicyAnchorTarget {
                action_kind_logits: row[..target].to_vec(),
                target_logits: row[target..effort].to_vec(),
                effort_logits: row[effort..amount].to_vec(),
                amount_logits: row[amount..signal].to_vec(),
                signal_logits: row[signal..strength].to_vec(),
                signal_strength_logits: row[strength..memory].to_vec(),
                next_memory: row[memory..].to_vec(),
            }
        })
        .collect()
}

fn load_specialist_teacher<B: Backend>(
    identity: &CompetencyCheckpointIdentity,
    frontier: &CompetencyFrontier,
    device: &B::Device,
) -> Result<PolicySnapshot<B>, String> {
    if !frontier.contains_identity(identity) {
        return Err("specialist teacher is not a member of the competency frontier".into());
    }
    let checkpoint = Path::new(&identity.checkpoint_directory).join(&identity.checkpoint);
    let snapshot = load_policy_snapshot(&checkpoint, device)?;
    if snapshot.checkpoint != identity.checkpoint
        || snapshot.update != identity.update
        || snapshot.actions != identity.actions
    {
        return Err("specialist teacher identity does not match its checkpoint".into());
    }
    Ok(snapshot)
}

fn load_specialist_teachers<B: Backend>(
    selection: &SpecialistTeacherSelection,
    frontier: &CompetencyFrontier,
    device: &B::Device,
) -> Result<SpecialistTeachers<B>, String> {
    Ok(SpecialistTeachers {
        ecology: selection
            .ecology
            .as_ref()
            .map(|identity| load_specialist_teacher(identity, frontier, device))
            .transpose()?,
        combat: selection
            .combat
            .as_ref()
            .map(|identity| load_specialist_teacher(identity, frontier, device))
            .transpose()?,
    })
}

fn trim_rollout_pool<B: Backend>(
    pool: &mut Vec<PolicySnapshot<B>>,
    league: &mut Vec<RolloutLeagueMember>,
    retired: &mut Vec<PolicySnapshot<B>>,
    assignments: &[RolloutOpponentAssignment],
    maximum: usize,
) {
    assert_eq!(pool.len(), league.len(), "rollout league lost alignment");
    while pool.len() > maximum {
        // Retain the original anchor and the newest members. With a one-member
        // pool, the anchor is intentionally permanent.
        let index = if maximum > 1 { 1 } else { pool.len() - 1 };
        let removed = pool.remove(index);
        league.remove(index);
        if assignments.iter().any(|assignment| {
            matches!(assignment, RolloutOpponentAssignment::Snapshot { model_sha256 } if *model_sha256 == removed.model_sha256)
        }) {
            retired.push(removed);
        }
    }
}

fn prune_retired_snapshots<B: Backend>(
    retired: &mut Vec<PolicySnapshot<B>>,
    assignments: &[RolloutOpponentAssignment],
) {
    retired.retain(|snapshot| {
        assignments.iter().any(|assignment| {
            matches!(assignment, RolloutOpponentAssignment::Snapshot { model_sha256 } if *model_sha256 == snapshot.model_sha256)
        })
    });
}

fn choose_rollout_opponent(
    league: &mut [RolloutLeagueMember],
    baseline: OpponentProfile,
    config: &SelfPlayConfig,
    rng: &mut impl Rng,
) -> RolloutOpponentAssignment {
    if league.is_empty() || rng.random::<f64>() < config.baseline_probability {
        RolloutOpponentAssignment::Baseline { profile: baseline }
    } else {
        let max_rating = league
            .iter()
            .map(|member| member.rating)
            .reduce(f64::max)
            .unwrap_or(config.initial_rating);
        let weights = league
            .iter()
            .map(|member| {
                let rating_weight =
                    ((member.rating - max_rating) / config.rating_temperature).exp();
                let exposure_weight =
                    (1.0 + member.rollout_selections as f64).powf(-config.exposure_exponent);
                (rating_weight * exposure_weight).max(f64::MIN_POSITIVE)
            })
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        let mut draw = rng.random::<f64>() * total;
        let mut selected = weights.len() - 1;
        for (index, weight) in weights.into_iter().enumerate() {
            if draw < weight {
                selected = index;
                break;
            }
            draw -= weight;
        }
        league[selected].rollout_selections = league[selected].rollout_selections.saturating_add(1);
        RolloutOpponentAssignment::Snapshot {
            model_sha256: league[selected].snapshot.model_sha256.clone(),
        }
    }
}

fn choose_curriculum_rollout_opponent(
    stage: FeedingCurriculumStage,
    stage_baseline: OpponentProfile,
    league: &mut [RolloutLeagueMember],
    baseline: OpponentProfile,
    config: &SelfPlayConfig,
    rng: &mut impl Rng,
) -> RolloutOpponentAssignment {
    if stage == FeedingCurriculumStage::Competitive {
        choose_rollout_opponent(league, baseline, config, rng)
    } else {
        RolloutOpponentAssignment::Baseline {
            profile: stage_baseline,
        }
    }
}

fn update_league_ratings(
    league: &mut [RolloutLeagueMember],
    evaluation: Option<&EvaluationMetrics>,
    config: &SelfPlayConfig,
) -> LeaguePromotion {
    let mut candidate_rating = league
        .last()
        .map_or(config.initial_rating, |member| member.rating);
    let mut candidate_games = 0_u64;
    let Some(evaluation) = evaluation else {
        return LeaguePromotion {
            rating: candidate_rating,
            evaluation_games: 0,
        };
    };
    for metrics in &evaluation.opponents {
        let EvaluationOpponent::Snapshot { model_sha256, .. } = &metrics.opponent else {
            continue;
        };
        let Some(member) = league
            .iter_mut()
            .find(|member| member.snapshot.model_sha256 == *model_sha256)
        else {
            continue;
        };
        let games = u64::try_from(metrics.episodes).unwrap_or(u64::MAX);
        if games == 0 {
            continue;
        }
        let actual = (metrics.wins as f64 + 0.5 * metrics.timeouts as f64) / games as f64;
        let expected = 1.0 / (1.0 + 10_f64.powf((member.rating - candidate_rating) / 400.0));
        let delta = config.elo_k_factor * (actual - expected);
        candidate_rating += delta;
        member.rating -= delta;
        member.evaluation_games = member.evaluation_games.saturating_add(games);
        candidate_games = candidate_games.saturating_add(games);
    }
    LeaguePromotion {
        rating: candidate_rating,
        evaluation_games: candidate_games,
    }
}

fn new_rollout_env<B: Backend>(
    assignment: &RolloutOpponentAssignment,
    pool: &[PolicySnapshot<B>],
    retired: &[PolicySnapshot<B>],
    config: &TrainingConfig,
    curriculum: RolloutCurriculumAssignment,
    seed: u64,
    device: &B::Device,
) -> Result<BlobEnv, String>
where
    f32: From<B::FloatElem>,
{
    let scenario = curriculum
        .micro_combat_scenario
        .map(|index| {
            config
                .combat_curriculum
                .micro_combat
                .scenario(index)
                .ok_or_else(|| "rollout micro-combat scenario index is invalid".to_string())
        })
        .transpose()?;
    let stage_env = scenario.map_or_else(
        || config.rollout_environment(curriculum.stage, curriculum.simulation_time_quanta),
        |scenario| {
            config.combat_curriculum.micro_combat_rollout_environment(
                &config.env,
                scenario,
                curriculum.simulation_time_quanta,
            )
        },
    );
    let opponent_starting_state = scenario.map(|scenario| OpponentStartingState {
        cells_per_team: scenario.opponent_cells,
        initial_energy: scenario.opponent_initial_energy,
    });
    let mut env = match assignment {
        RolloutOpponentAssignment::Baseline { profile } => {
            let mut env_config = stage_env;
            env_config.opponent = *profile;
            if let Some(starting_state) = opponent_starting_state {
                BlobEnv::new_with_opponent_starting_state(
                    env_config,
                    config.reward.clone(),
                    seed,
                    starting_state,
                )
            } else {
                BlobEnv::new(env_config, config.reward.clone(), seed)
            }
        }
        RolloutOpponentAssignment::Snapshot { model_sha256 } => {
            if scenario.is_some() {
                return Err("micro-combat rollout cannot use a snapshot opponent".into());
            }
            let snapshot = pool
                .iter()
                .chain(retired)
                .find(|snapshot| snapshot.model_sha256 == *model_sha256)
                .ok_or_else(|| "rollout opponent is absent from the snapshot pool".to_string())?;
            BlobEnv::new_with_snapshot(
                stage_env,
                config.reward.clone(),
                seed,
                snapshot.model.clone(),
                device.clone(),
            )
        }
    };
    env.enable_telemetry(config.telemetry.clone());
    Ok(env)
}

fn restore_rollout_env<B: Backend>(
    assignment: &RolloutOpponentAssignment,
    pool: &[PolicySnapshot<B>],
    retired: &[PolicySnapshot<B>],
    config: &TrainingConfig,
    curriculum: RolloutCurriculumAssignment,
    checkpoint: crate::env::BlobEnvCheckpoint,
    device: &B::Device,
) -> Result<BlobEnv, String>
where
    f32: From<B::FloatElem>,
{
    let scenario = curriculum
        .micro_combat_scenario
        .map(|index| {
            config
                .combat_curriculum
                .micro_combat
                .scenario(index)
                .ok_or_else(|| "restored micro-combat scenario index is invalid".to_string())
        })
        .transpose()?;
    let stage_env = scenario.map_or_else(
        || config.rollout_environment(curriculum.stage, curriculum.simulation_time_quanta),
        |scenario| {
            config.combat_curriculum.micro_combat_rollout_environment(
                &config.env,
                scenario,
                curriculum.simulation_time_quanta,
            )
        },
    );
    let mut env = match assignment {
        RolloutOpponentAssignment::Baseline { profile } => {
            let mut env_config = stage_env;
            env_config.opponent = *profile;
            if let Some(scenario) = scenario {
                BlobEnv::from_checkpoint_with_opponent_starting_state(
                    env_config,
                    config.reward.clone(),
                    OpponentStartingState {
                        cells_per_team: scenario.opponent_cells,
                        initial_energy: scenario.opponent_initial_energy,
                    },
                    checkpoint,
                )
            } else {
                BlobEnv::from_checkpoint(env_config, config.reward.clone(), checkpoint)
            }
        }
        RolloutOpponentAssignment::Snapshot { model_sha256 } => {
            if scenario.is_some() {
                return Err("restored micro-combat rollout has a snapshot opponent".into());
            }
            let snapshot = pool
                .iter()
                .chain(retired)
                .find(|snapshot| snapshot.model_sha256 == *model_sha256)
                .ok_or_else(|| "restored rollout opponent is absent from the pool".to_string())?;
            BlobEnv::from_checkpoint_with_snapshot(
                stage_env,
                config.reward.clone(),
                snapshot.model.clone(),
                device.clone(),
                checkpoint,
            )
        }
    }?;
    env.enable_telemetry(config.telemetry.clone());
    Ok(env)
}

/// Aggregated episode stats for logging.
struct EpisodeStats {
    wins: u64,
    losses: u64,
    timeouts: u64,
    total_episode_len: u64,
    total_reward: f64,
    episodes_completed: u64,
}

impl EpisodeStats {
    fn new() -> Self {
        EpisodeStats {
            wins: 0,
            losses: 0,
            timeouts: 0,
            total_episode_len: 0,
            total_reward: 0.0,
            episodes_completed: 0,
        }
    }

    fn record(&mut self, outcome: EpisodeOutcome, episode_len: u64, total_reward: f32) {
        match outcome {
            EpisodeOutcome::Win => self.wins += 1,
            EpisodeOutcome::Loss => self.losses += 1,
            EpisodeOutcome::Timeout => self.timeouts += 1,
            EpisodeOutcome::SafetyAbort => {
                panic!("host decision-frontier safety limit is not an episode outcome")
            }
        }
        self.total_episode_len += episode_len;
        self.total_reward += total_reward as f64;
        self.episodes_completed += 1;
    }

    fn avg_episode_len(&self) -> f64 {
        if self.episodes_completed == 0 {
            0.0
        } else {
            self.total_episode_len as f64 / self.episodes_completed as f64
        }
    }

    fn avg_reward(&self) -> f64 {
        if self.episodes_completed == 0 {
            0.0
        } else {
            self.total_reward / self.episodes_completed as f64
        }
    }

    fn win_rate(&self) -> f64 {
        if self.episodes_completed == 0 {
            0.0
        } else {
            self.wins as f64 / self.episodes_completed as f64
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }
}

fn contact_wins(report: &ContactEvaluationReport) -> usize {
    report.variants.iter().map(|variant| variant.wins).sum()
}

fn contact_is_better(
    candidate: Option<&ContactEvaluationReport>,
    incumbent: Option<&ContactEvaluationReport>,
) -> bool {
    match (candidate, incumbent) {
        (Some(candidate), Some(incumbent)) => {
            // Local opponents can exhaust themselves while attacking, so a
            // scenario win is not necessarily attributable to the learned
            // policy. Resolver-attributed kills are the stronger tie-break.
            candidate.kills > incumbent.kills
                || (candidate.kills == incumbent.kills
                    && contact_wins(candidate) > contact_wins(incumbent))
                || (candidate.kills == incumbent.kills
                    && contact_wins(candidate) == contact_wins(incumbent)
                    && candidate.damage_dealt > incumbent.damage_dealt)
                || (candidate.kills == incumbent.kills
                    && contact_wins(candidate) == contact_wins(incumbent)
                    && candidate.damage_dealt == incumbent.damage_dealt
                    && candidate.damaging_episode_rate > incumbent.damaging_episode_rate)
        }
        (Some(_), None) => true,
        (None, Some(_) | None) => false,
    }
}

fn contact_is_equal(
    candidate: Option<&ContactEvaluationReport>,
    incumbent: Option<&ContactEvaluationReport>,
) -> bool {
    match (candidate, incumbent) {
        (Some(candidate), Some(incumbent)) => {
            contact_wins(candidate) == contact_wins(incumbent)
                && candidate.kills == incumbent.kills
                && candidate.damage_dealt == incumbent.damage_dealt
                && candidate.damaging_episode_rate == incumbent.damaging_episode_rate
        }
        (None, None) => true,
        _ => false,
    }
}

fn micro_combat_gate_passed(
    config: &MicroCombatTrainingConfig,
    report: Option<&MicroCombatEvaluationReport>,
) -> bool {
    if config.enabled {
        report.is_some_and(|report| report.meets_promotion_thresholds(config))
    } else {
        report.is_none()
    }
}

fn micro_combat_is_better(
    candidate: Option<&MicroCombatEvaluationReport>,
    incumbent: Option<&MicroCombatEvaluationReport>,
) -> bool {
    match (candidate, incumbent) {
        (Some(candidate), Some(incumbent)) => {
            let candidate = candidate.gate_summary();
            let incumbent = incumbent.gate_summary();
            candidate.elimination_success_rate > incumbent.elimination_success_rate
                || (candidate.elimination_success_rate == incumbent.elimination_success_rate
                    && candidate.survival_success_rate > incumbent.survival_success_rate)
        }
        (Some(_), None) => true,
        (None, Some(_) | None) => false,
    }
}

fn micro_combat_is_equal(
    candidate: Option<&MicroCombatEvaluationReport>,
    incumbent: Option<&MicroCombatEvaluationReport>,
) -> bool {
    match (candidate, incumbent) {
        (Some(candidate), Some(incumbent)) => {
            let candidate = candidate.gate_summary();
            let incumbent = incumbent.gate_summary();
            candidate.elimination_success_rate == incumbent.elimination_success_rate
                && candidate.survival_success_rate == incumbent.survival_success_rate
        }
        (None, None) => true,
        _ => false,
    }
}

fn evaluation_is_better(
    candidate: &EvaluationMetrics,
    candidate_contact: Option<&ContactEvaluationReport>,
    candidate_micro: Option<&MicroCombatEvaluationReport>,
    incumbent: &EvaluationMetrics,
    incumbent_contact: Option<&ContactEvaluationReport>,
    incumbent_micro: Option<&MicroCombatEvaluationReport>,
) -> bool {
    candidate.worst_case_win_rate > incumbent.worst_case_win_rate
        || (candidate.worst_case_win_rate == incumbent.worst_case_win_rate
            && candidate.win_rate > incumbent.win_rate)
        || (candidate.worst_case_win_rate == incumbent.worst_case_win_rate
            && candidate.win_rate == incumbent.win_rate
            && contact_is_better(candidate_contact, incumbent_contact))
        || (candidate.worst_case_win_rate == incumbent.worst_case_win_rate
            && candidate.win_rate == incumbent.win_rate
            && contact_is_equal(candidate_contact, incumbent_contact)
            && micro_combat_is_better(candidate_micro, incumbent_micro))
        || (candidate.worst_case_win_rate == incumbent.worst_case_win_rate
            && candidate.win_rate == incumbent.win_rate
            && contact_is_equal(candidate_contact, incumbent_contact)
            && micro_combat_is_equal(candidate_micro, incumbent_micro)
            && candidate.average_reward > incumbent.average_reward)
        || (candidate.worst_case_win_rate == incumbent.worst_case_win_rate
            && candidate.win_rate == incumbent.win_rate
            && contact_is_equal(candidate_contact, incumbent_contact)
            && micro_combat_is_equal(candidate_micro, incumbent_micro)
            && candidate.average_reward == incumbent.average_reward
            && candidate.average_episode_len < incumbent.average_episode_len)
}

fn resume_config_is_compatible(saved: &TrainingConfig, requested: &TrainingConfig) -> bool {
    let mut saved = saved.clone();
    saved.total_timesteps = requested.total_timesteps;
    saved.total_simulation_quanta_per_env = requested.total_simulation_quanta_per_env;
    saved.eval_interval = requested.eval_interval;
    saved.checkpoint_interval = requested.checkpoint_interval;
    saved.checkpoint_dir.clone_from(&requested.checkpoint_dir);
    saved == *requested
}

fn simulation_budget_complete(exposure: &[u64], target_per_env: u64) -> bool {
    !exposure.is_empty() && exposure.iter().all(|quanta| *quanta >= target_per_env)
}

fn training_budget_complete(
    config: &TrainingConfig,
    total_actions: u64,
    cumulative_sim_time_quanta: &[u64],
) -> bool {
    config.total_simulation_quanta_per_env.map_or_else(
        || total_actions >= config.total_timesteps,
        |target| simulation_budget_complete(cumulative_sim_time_quanta, target),
    )
}

fn checkpoint_publication_due(
    periodic_checkpoint: bool,
    is_new_best: bool,
    promote_to_rollout_pool: bool,
    terminal_boundary: bool,
    competency_frontier_changed: bool,
    world_time_competency_evaluation: bool,
) -> bool {
    periodic_checkpoint
        || is_new_best
        || promote_to_rollout_pool
        || terminal_boundary
        || competency_frontier_changed
        || world_time_competency_evaluation
}

fn self_play_promotion_due(
    config: &TrainingConfig,
    minimum_sim_time_quanta: u64,
    update_count: usize,
) -> bool {
    config.self_play.max_opponent_pool > 0
        && minimum_sim_time_quanta >= config.self_play.start_after_sim_time_quanta_per_env
        && update_count.is_multiple_of(config.self_play.opponent_update_interval)
}

fn open_metrics_file(path: &Path, resume: bool, header: &str) -> Result<File, String> {
    let already_has_content = resume && path.metadata().is_ok_and(|metadata| metadata.len() > 0);
    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if resume {
        options.append(true);
    } else {
        options.truncate(true);
    }
    let mut file = options
        .open(path)
        .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    if !already_has_content {
        writeln!(file, "{header}")
            .map_err(|error| format!("failed to initialize {}: {error}", path.display()))?;
    }
    Ok(file)
}

fn write_fixed_evaluation_rows(
    file: &mut File,
    update: usize,
    actions: u64,
    evaluation: &EvaluationMetrics,
    seed_start: u64,
    seed_count: usize,
) -> Result<(), String> {
    for opponent in &evaluation.opponents {
        writeln!(
            file,
            "{update},{actions},{},{},{},{},{},{:.6},,{:.2},{:.6},{},{seed_start},{seed_count}",
            opponent.opponent,
            opponent.episodes,
            opponent.wins,
            opponent.losses,
            opponent.timeouts,
            opponent.win_rate,
            opponent.average_episode_len,
            opponent.average_reward,
            opponent.actions,
        )
        .map_err(|error| format!("failed to write opponent evaluation: {error}"))?;
    }
    writeln!(
        file,
        "{update},{actions},suite,{},{},{},{},{:.6},{:.6},{:.2},{:.6},{},{seed_start},{seed_count}",
        evaluation.episodes,
        evaluation.wins,
        evaluation.losses,
        evaluation.timeouts,
        evaluation.win_rate,
        evaluation.worst_case_win_rate,
        evaluation.average_episode_len,
        evaluation.average_reward,
        evaluation.actions,
    )
    .map_err(|error| format!("failed to write suite evaluation: {error}"))?;
    file.flush()
        .map_err(|error| format!("failed to flush evaluation metrics: {error}"))
}

fn write_feeding_evaluation_rows(
    file: &mut File,
    update: usize,
    actions: u64,
    suite: &str,
    report: &FeedingPromotionReport,
) -> Result<(), String> {
    for stage in &report.stages {
        writeln!(
            file,
            "{update},{actions},{suite},{},{},{},{},{:.6},{:.6},{:.6},{},{},{},{},{}",
            report.passed,
            stage.stage,
            stage.episodes,
            stage.successful_episodes,
            stage.episode_success_rate,
            stage.survival_rate,
            stage.consumed_energy_per_initial_cell,
            stage.movement_successes,
            stage.consume_successes,
            stage.consumed_energy,
            stage.safety_aborts,
            report.seeds.len(),
        )
        .map_err(|error| format!("failed to write feeding evaluation: {error}"))?;
    }
    file.flush()
        .map_err(|error| format!("failed to flush feeding evaluation: {error}"))
}

fn write_contact_evaluation_rows(
    file: &mut File,
    update: usize,
    actions: u64,
    report: &ContactEvaluationReport,
) -> Result<(), String> {
    for variant in &report.variants {
        writeln!(
            file,
            "{update},{actions},{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17:.6},{18:.6},{19:.6},{20}",
            variant.stage,
            variant.cells_per_team,
            variant.initial_energy,
            variant.opponent,
            variant.episodes,
            variant.wins,
            variant.losses,
            variant.timeouts,
            variant.safety_aborts,
            variant.attacking_episodes,
            variant.damaging_episodes,
            variant.attacks_committed,
            variant.attacks_succeeded,
            variant.damage_dealt,
            variant.kills,
            variant.attacks_frustrated,
            variant.attacks_interrupted,
            variant.attacking_episode_rate,
            variant.damaging_episode_rate,
            variant.attack_success_rate,
            report.seeds.len(),
        )
        .map_err(|error| format!("failed to write contact evaluation: {error}"))?;
    }
    file.flush()
        .map_err(|error| format!("failed to flush contact evaluation: {error}"))
}

fn write_micro_combat_evaluation_rows(
    file: &mut File,
    update: usize,
    actions: u64,
    report: &MicroCombatEvaluationReport,
) -> Result<(), String> {
    for metrics in &report.scenarios {
        writeln!(
            file,
            "{update},{actions},{},{:?},{},{},{},{},{},{},{},{:.6},{:.6},{},{},{},{},{},{}",
            metrics.scenario,
            metrics.objective,
            metrics.episodes,
            metrics.wins,
            metrics.losses,
            metrics.timeouts,
            metrics.safety_aborts,
            metrics.alive_at_end_episodes,
            metrics.objective_successes,
            metrics.objective_success_rate,
            metrics.scientific_survival_rate,
            metrics.survival_time_quanta_total,
            metrics.training_attacks_committed,
            metrics.training_damage_dealt,
            metrics.training_damage_received,
            metrics.training_kills,
            report.seeds.len(),
        )
        .map_err(|error| format!("failed to write micro-combat evaluation: {error}"))?;
    }
    file.flush()
        .map_err(|error| format!("failed to flush micro-combat evaluation: {error}"))
}

#[allow(clippy::too_many_arguments)]
fn write_competency_timeline_row(
    file: &mut File,
    update: usize,
    actions: u64,
    minimum_sim_time_quanta_per_env: u64,
    maximum_sim_time_quanta_per_env: u64,
    total_sim_time_quanta: u128,
    scheduled_world_time_frontier: Option<u64>,
    trigger: &str,
    metrics: &CompetencyMetrics,
) -> Result<(), String> {
    writeln!(
        file,
        "{update},{actions},{minimum_sim_time_quanta_per_env},{maximum_sim_time_quanta_per_env},{total_sim_time_quanta},{},{trigger},{},{},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{},{},{},{},{:.6},{:.6},{:.6}",
        scheduled_world_time_frontier.map_or_else(String::new, |value| value.to_string()),
        metrics.configured_feeding_passed,
        metrics.retention_feeding_passed,
        metrics.on_food_survival_rate,
        metrics.adjacent_food_survival_rate,
        metrics.on_food_intake_per_initial_cell,
        metrics.adjacent_food_intake_per_initial_cell,
        metrics.combat_passed,
        metrics.micro_survival_success_rate,
        metrics.micro_elimination_success_rate,
        metrics.micro_attack_commitments_per_episode,
        metrics.contact_damage,
        metrics.contact_kills,
        metrics.skirmish_damage,
        metrics.skirmish_kills,
        metrics.fixed_worst_case_win_rate,
        metrics.fixed_win_rate,
        metrics.fixed_average_reward,
    )
    .map_err(|error| format!("failed to write competency timeline: {error}"))?;
    file.flush()
        .map_err(|error| format!("failed to flush competency timeline: {error}"))
}

/// Run the full training loop.
///
/// - `load_model_path`: If Some, load weights from this path to continue training.
/// - `resume_checkpoint`: If Some, restore an exact update-boundary checkpoint.
/// - `save_model_path`: If Some, save the trained model to this path at the end.
pub fn train<B: AutodiffBackend>(
    config: TrainingConfig,
    device: B::Device,
    load_model_path: Option<&str>,
    resume_checkpoint: Option<&Path>,
    save_model_path: Option<&str>,
) where
    B::FloatElem: From<f32>,
    f32: From<B::FloatElem>,
{
    config
        .validate()
        .expect("invalid RL training configuration");
    assert!(
        !config.uses_initial_policy_anchor() || config.initial_policy.is_some(),
        "functional policy anchoring requires a verified initial_policy at training startup"
    );
    assert!(
        !config.specialist_distillation.enabled || config.initial_policy.is_some(),
        "specialist distillation requires a verified initial_policy as its pre-frontier and competitive-stage fallback"
    );
    assert!(
        load_model_path.is_none() || resume_checkpoint.is_none(),
        "load_model_path and resume_checkpoint are mutually exclusive"
    );
    assert!(
        load_model_path.is_none() || config.initial_policy.is_none(),
        "load_model_path and config.initial_policy are mutually exclusive"
    );
    let initial_qualification = config.initial_policy.as_ref().and_then(|initial| {
        initial.qualification.as_ref().map(|qualification| {
            verify_feeding_initial_policy(
                Path::new(&qualification.path),
                &qualification.artifact_hash,
                &initial.artifact_sha256,
                &config,
            )
            .unwrap_or_else(|error| panic!("invalid initial-policy qualification: {error}"))
        })
    });
    let load_model_path = load_model_path.map(PathBuf::from);
    let evaluation_snapshots = if config.fixed_evaluation_enabled()
        || config.self_play.max_opponent_pool > 0
    {
        config
            .evaluation_snapshots
            .iter()
            .map(|path| {
                load_policy_snapshot::<B::InnerBackend>(Path::new(path), &device).unwrap_or_else(
                    |error| panic!("failed to load evaluation snapshot {path}: {error}"),
                )
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║           blob_rl Training                          ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!(
        "║  Envs: {:>4}  Timesteps: {:>10}                    ║",
        config.num_envs, config.total_timesteps
    );
    println!(
        "║  Model: {}→r{}→{}  LR: {:.0e}                  ║",
        config.model.hidden1,
        config.model.recurrent_size,
        config.model.hidden2,
        config.ppo.learning_rate
    );
    println!("╚══════════════════════════════════════════════════════╝");

    // Initialize model
    B::seed(&device, config.seed);
    let model_config = PolicyValueNetConfig {
        hidden1: config.model.hidden1,
        hidden2: config.model.hidden2,
        recurrent_size: config.model.recurrent_size,
    };
    let reference_model: Option<PolicyValueNet<B::InnerBackend>> = (config
        .uses_initial_policy_anchor()
        || config.specialist_distillation.enabled)
        .then(|| {
            let initial = config
                .initial_policy
                .as_ref()
                .expect("validated initial-policy anchoring has an initial policy");
            println!(
                "  Anchoring policy to: {} (ordinary {}, contact {})",
                initial.directory,
                config.ppo.initial_policy_anchor_coeff,
                config
                    .combat_curriculum
                    .contact_initial_policy_anchor_coeff
                    .unwrap_or(config.ppo.initial_policy_anchor_coeff),
            );
            load_behavior_clone::<B::InnerBackend>(
                Path::new(&initial.directory),
                &initial.artifact_sha256,
                &config.model,
                &device,
            )
            .expect("failed to load initial-policy anchor")
        });
    let initial_model: Option<PolicyValueNet<B>> = if resume_checkpoint.is_some() {
        None
    } else if let Some(path) = load_model_path.as_ref() {
        println!("  Loading model from: {}", path.display());
        Some(
            model_config
                .init(&device)
                .load_file(path, &CompactRecorder::new(), &device)
                .expect("Failed to load model"),
        )
    } else if let Some(initial) = config.initial_policy.as_ref() {
        Some(
            load_behavior_clone::<B>(
                Path::new(&initial.directory),
                &initial.artifact_sha256,
                &config.model,
                &device,
            )
            .expect("failed to load configured initial policy"),
        )
    } else {
        Some(model_config.init(&device))
    };
    let initial_optimizer = AdamWConfig::new()
        .init()
        .with_grad_clipping(GradientClipping::Norm(config.ppo.max_grad_norm));

    let (
        mut model,
        mut optimizer,
        mut envs,
        mut action_rng,
        mut optimizer_rng,
        mut opponent_rng,
        mut env_episode_ids,
        mut env_episode_rewards,
        mut total_timesteps,
        mut cumulative_sim_time_quanta,
        mut update_count,
        mut best_evaluation,
        mut best_contact_evaluation,
        mut best_micro_combat_evaluation,
        mut competency_frontier,
        mut specialist_teacher_selection,
        mut rollout_pool,
        mut rollout_league,
        mut retired_rollout_snapshots,
        mut environment_opponents,
        mut environment_curriculum_stages,
        mut micro_combat_rotation,
        mut environment_micro_combat_scenarios,
        mut training_telemetry,
    ) = if let Some(checkpoint) = resume_checkpoint {
        let (model, optimizer, state, metadata) =
            load_checkpoint::<B, _>(checkpoint, &model_config, initial_optimizer, &device)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to load checkpoint {}: {error}",
                        checkpoint.display()
                    )
                });
        assert!(
            resume_config_is_compatible(&metadata.config, &config),
            "resume configuration changes training dynamics; only training-budget targets, eval_interval, checkpoint_interval, and checkpoint_dir may change"
        );
        assert!(
            config.total_timesteps > state.total_timesteps,
            "resume target must exceed the checkpoint action count ({})",
            state.total_timesteps
        );
        assert_eq!(
            state.environments.len(),
            config.num_envs,
            "checkpoint environment count mismatch"
        );
        assert_eq!(
            state.env_episode_ids.len(),
            config.num_envs,
            "checkpoint episode-id count mismatch"
        );
        assert_eq!(
            state.env_episode_rewards.len(),
            config.num_envs,
            "checkpoint reward-accumulator count mismatch"
        );
        assert_eq!(
            state.cumulative_sim_time_quanta.len(),
            config.num_envs,
            "checkpoint simulation-exposure count mismatch"
        );
        assert!(
            !training_budget_complete(
                &config,
                state.total_timesteps,
                &state.cumulative_sim_time_quanta
            ),
            "resume target is already complete"
        );
        assert_eq!(
            state.environment_opponents.len(),
            config.num_envs,
            "checkpoint rollout-opponent assignment count mismatch"
        );
        assert_eq!(
            state.environment_curriculum_stages.len(),
            config.num_envs,
            "checkpoint curriculum-stage assignment count mismatch"
        );
        assert_eq!(
            state.environment_micro_combat_scenarios.len(),
            config.num_envs,
            "checkpoint micro-combat scenario assignment count mismatch"
        );
        let mut rollout_league = state.rollout_pool.clone();
        let mut rollout_pool = rollout_league
            .iter()
            .map(|member| load_rollout_snapshot::<B::InnerBackend>(&member.snapshot, &device))
            .collect::<Result<Vec<_>, _>>()
            .expect("failed to restore rollout-opponent pool");
        let environment_opponents = state.environment_opponents.clone();
        let environment_curriculum_stages = state.environment_curriculum_stages.clone();
        let environment_micro_combat_scenarios = state.environment_micro_combat_scenarios.clone();
        let mut retired_rollout_snapshots = state
            .active_retired_snapshots
            .iter()
            .map(|descriptor| load_rollout_snapshot::<B::InnerBackend>(descriptor, &device))
            .collect::<Result<Vec<_>, _>>()
            .expect("failed to restore active retired rollout opponents");
        let envs = state
            .environments
            .into_iter()
            .enumerate()
            .map(|(index, environment)| {
                restore_rollout_env(
                    &environment_opponents[index],
                    &rollout_pool,
                    &retired_rollout_snapshots,
                    &config,
                    RolloutCurriculumAssignment {
                        stage: environment_curriculum_stages[index],
                        simulation_time_quanta: state.cumulative_sim_time_quanta[index],
                        micro_combat_scenario: environment_micro_combat_scenarios[index],
                    },
                    environment,
                    &device,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("failed to restore RL environments");
        if let Some(promotion) = metadata.rollout_pool_promotion {
            let snapshot = load_policy_snapshot::<B::InnerBackend>(checkpoint, &device)
                .expect("failed to restore the checkpoint's rollout-pool promotion");
            rollout_league.push(RolloutLeagueMember {
                snapshot: snapshot.descriptor(),
                rating: promotion.rating,
                evaluation_games: promotion.evaluation_games,
                rollout_selections: 0,
            });
            rollout_pool.push(snapshot);
            trim_rollout_pool(
                &mut rollout_pool,
                &mut rollout_league,
                &mut retired_rollout_snapshots,
                &environment_opponents,
                config.self_play.max_opponent_pool,
            );
        }
        println!(
            "  Resumed exact boundary: update {} / {} actions from {}",
            state.update_count,
            state.total_timesteps,
            checkpoint.display()
        );
        (
            model,
            optimizer,
            envs,
            state.action_rng,
            state.optimizer_rng,
            state.opponent_rng,
            state.env_episode_ids,
            state.env_episode_rewards,
            state.total_timesteps,
            state.cumulative_sim_time_quanta,
            state.update_count,
            state.best_evaluation,
            state.best_contact_evaluation,
            state.best_micro_combat_evaluation,
            state.competency_frontier,
            state.specialist_teachers,
            rollout_pool,
            rollout_league,
            retired_rollout_snapshots,
            environment_opponents,
            environment_curriculum_stages,
            state.micro_combat_rotation,
            environment_micro_combat_scenarios,
            state.telemetry,
        )
    } else {
        let initial_stage = config.rollout_stage(0);
        let initial_assignment = RolloutOpponentAssignment::Baseline {
            profile: config.rollout_baseline_opponent(initial_stage, 0),
        };
        let initial_env_config = config.rollout_environment(initial_stage, 0);
        let envs = (0..config.num_envs)
            .map(|i| {
                BlobEnv::new(
                    initial_env_config.clone(),
                    config.reward.clone(),
                    config.seed.wrapping_add(i as u64),
                )
            })
            .collect();
        (
            initial_model.expect("fresh training always initializes a model"),
            initial_optimizer,
            envs,
            ChaCha12Rng::seed_from_u64(config.seed ^ 0x524c_504f_4c49_4359),
            ChaCha12Rng::seed_from_u64(config.seed ^ 0x5050_4f5f_4241_5443),
            ChaCha12Rng::seed_from_u64(config.seed ^ 0x5345_4c46_504c_4159),
            vec![0_u64; config.num_envs],
            vec![0.0; config.num_envs],
            0,
            vec![0_u64; config.num_envs],
            0,
            None,
            None,
            None,
            CompetencyFrontier::default(),
            SpecialistTeacherSelection::default(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![initial_assignment; config.num_envs],
            vec![initial_stage; config.num_envs],
            MicroCombatRotationState::default(),
            vec![None; config.num_envs],
            None,
        )
    };

    let mut specialist_teachers = if config.specialist_distillation.enabled {
        load_specialist_teachers::<B::InnerBackend>(
            &specialist_teacher_selection,
            &competency_frontier,
            &device,
        )
        .expect("failed to restore specialist distillation teachers")
    } else {
        SpecialistTeachers {
            ecology: None,
            combat: None,
        }
    };

    for env in &mut envs {
        env.enable_telemetry(config.telemetry.clone());
    }
    if config.telemetry.enabled {
        if let Some(telemetry) = training_telemetry.as_ref() {
            telemetry
                .validate(config.num_envs)
                .expect("checkpoint telemetry state mismatch");
        } else {
            let initial = envs
                .iter()
                .enumerate()
                .map(|(index, env)| {
                    (
                        config.seed.wrapping_add(index as u64),
                        curriculum_label(
                            &config,
                            environment_curriculum_stages[index],
                            environment_micro_combat_scenarios[index],
                        ),
                        env.telemetry_sample()
                            .expect("enabled telemetry has an initial sample"),
                    )
                })
                .collect();
            training_telemetry = Some(TrainingTelemetryState::new(initial));
        }
    } else {
        training_telemetry = None;
    }

    // Get initial observations
    let mut env_observations: Vec<Vec<PolicyObservation>> = envs
        .iter_mut()
        .map(BlobEnv::get_policy_observations)
        .collect();

    let start_time = Instant::now();
    let mut episode_stats = EpisodeStats::new();

    // Create metrics log file
    std::fs::create_dir_all(&config.checkpoint_dir).expect("failed to create checkpoint directory");
    let artifact_root = Path::new(&config.checkpoint_dir);
    let frontier_checkpoint_directory = std::fs::canonicalize(artifact_root)
        .expect("failed to resolve checkpoint directory")
        .to_string_lossy()
        .into_owned();
    if !competency_frontier.entries.is_empty() {
        publish_competency_frontier(artifact_root, &competency_frontier)
            .expect("failed to restore competency-frontier pointer");
    }
    let metrics_path = artifact_root.join("metrics.csv");
    let mut metrics_file = open_metrics_file(
        &metrics_path,
        resume_checkpoint.is_some(),
        "update,actions,policy_loss,value_loss,entropy,approx_kl,anchor_loss,clip_fraction,explained_variance,ppo_optimizer_steps,ppo_epochs_completed,kl_early_stop,ppo_recurrent_unroll_steps,ppo_recurrent_chunks,episodes,wins,losses,timeouts,win_rate,avg_ep_len,avg_reward,training_cells_alive,completed_transitions,discarded_tails,mean_elapsed_time,actions_per_second,min_sim_time_quanta,max_sim_time_quanta,total_sim_time_quanta,simulation_quanta_per_second,peak_resident_set_bytes,curriculum_stage",
    )
    .expect("failed to initialize training metrics");
    println!("  Logging metrics to: {}", metrics_path.display());
    let evaluation_path = artifact_root.join("evaluation.csv");
    let mut evaluation_file = open_metrics_file(
        &evaluation_path,
        resume_checkpoint.is_some(),
        "update,actions,opponent,episodes,wins,losses,timeouts,win_rate,worst_case_win_rate,avg_ep_len,avg_reward,evaluation_actions,seed_start,seed_count",
    )
    .expect("failed to initialize evaluation metrics");
    let feeding_evaluation_path = artifact_root.join("feeding-evaluation.csv");
    let mut feeding_evaluation_file = open_metrics_file(
        &feeding_evaluation_path,
        resume_checkpoint.is_some(),
        "update,actions,suite,gate_passed,stage,episodes,successful_episodes,episode_success_rate,survival_rate,consumed_energy_per_initial_cell,movement_successes,consume_successes,consumed_energy,safety_aborts,seed_count",
    )
    .expect("failed to initialize feeding evaluation metrics");
    let contact_evaluation_path = artifact_root.join("contact-evaluation.csv");
    let mut contact_evaluation_file = open_metrics_file(
        &contact_evaluation_path,
        resume_checkpoint.is_some(),
        "update,actions,stage,cells_per_team,initial_energy,opponent,episodes,wins,losses,timeouts,safety_aborts,attacking_episodes,damaging_episodes,attacks_committed,attacks_succeeded,damage_dealt,kills,attacks_frustrated,attacks_interrupted,attacking_episode_rate,damaging_episode_rate,attack_success_rate,seed_count",
    )
    .expect("failed to initialize contact evaluation metrics");
    let micro_combat_evaluation_path = artifact_root.join("micro-combat-evaluation.csv");
    let mut micro_combat_evaluation_file = open_metrics_file(
        &micro_combat_evaluation_path,
        resume_checkpoint.is_some(),
        "update,actions,scenario,objective,episodes,wins,losses,timeouts,safety_aborts,alive_at_end,objective_successes,objective_success_rate,scientific_survival_rate,survival_time_quanta,attacks_committed,damage_dealt,damage_received,kills,seed_count",
    )
    .expect("failed to initialize micro-combat evaluation metrics");
    let competency_timeline_path = artifact_root.join("competency-timeline.csv");
    let mut competency_timeline_file = open_metrics_file(
        &competency_timeline_path,
        resume_checkpoint.is_some(),
        "update,actions,min_sim_time_quanta_per_env,max_sim_time_quanta_per_env,total_sim_time_quanta,scheduled_world_time_frontier,trigger,configured_feeding_passed,retention_feeding_passed,on_food_survival_rate,adjacent_food_survival_rate,on_food_intake_per_initial_cell,adjacent_food_intake_per_initial_cell,combat_passed,micro_survival_success_rate,micro_elimination_success_rate,micro_attack_commitments_per_episode,contact_damage,contact_kills,skirmish_damage,skirmish_kills,fixed_worst_case_win_rate,fixed_win_rate,fixed_average_reward",
    )
    .expect("failed to initialize competency timeline");
    let evaluation_seeds = (0..config.eval_episodes)
        .map(|index| config.evaluation_seed.wrapping_add(index as u64))
        .collect::<Vec<_>>();
    let retention_evaluation_seeds = initial_qualification
        .as_ref()
        .map(|qualification| qualification.report.seeds.clone())
        .filter(|seeds| config.feeding_curriculum.enabled && *seeds != evaluation_seeds);
    let compiled_ruleset_hash = envs
        .first()
        .map(BlobEnv::compiled_ruleset_hash)
        .unwrap_or_else(|| "uninitialized".to_string());
    let mut last_fixed_evaluation_actions = None;

    println!(
        "\n  Update │ Timesteps │  Policy Loss │  Value Loss │  Entropy │       KL │  Win Rate │ Avg Len │ Cells"
    );
    println!(
        "  ──────┼───────────┼──────────────┼─────────────┼──────────┼──────────┼───────────┼─────────┼──────"
    );

    while !training_budget_complete(&config, total_timesteps, &cumulative_sim_time_quanta) {
        let minimum_sim_time_before_update = cumulative_sim_time_quanta
            .iter()
            .copied()
            .min()
            .unwrap_or(0);
        let mut rollout = RolloutBuffer::new();
        // One open action interval per cell. Rewards from every intervening
        // event frontier accrue here until that same cell decides again or
        // dies. Tails still pending when collection ends are discarded rather
        // than trained as false zero-value terminals.
        let mut pending_transitions: Vec<HashMap<CellId, PendingTransition>> =
            vec![HashMap::new(); config.num_envs];

        // Collect rollout across all environments. Policy inference is batched
        // across the complete environment set for each event frontier. This
        // keeps GPU dispatches large enough to amortize launch overhead and
        // preserves the stable env/cell action-sampling order used by exact
        // continuation. Detached teachers use disjoint stage-routed
        // sub-batches so no row pays for an inapplicable specialist.
        let mut collection_steps = 0u64;
        loop {
            let draining_terminal_tails = config.total_simulation_quanta_per_env.is_some()
                && training_budget_complete(&config, total_timesteps, &cumulative_sim_time_quanta)
                && pending_transitions
                    .iter()
                    .any(|pending| !pending.is_empty());
            if collection_steps >= config.rollout_length && !draining_terminal_tails {
                break;
            }
            collection_steps = collection_steps.saturating_add(1);
            let mut inference_observations = Vec::new();
            let mut inference_memory = Vec::new();
            let mut environment_rows = vec![None; config.num_envs];
            let mut inference_rows = 0usize;

            for (env_idx, obs_list) in env_observations.iter().enumerate() {
                if config
                    .total_simulation_quanta_per_env
                    .is_some_and(|target| cumulative_sim_time_quanta[env_idx] >= target)
                    && pending_transitions[env_idx].is_empty()
                {
                    continue;
                }
                if obs_list.is_empty() {
                    continue;
                }
                let row_start = inference_rows;
                inference_rows += obs_list.len();
                environment_rows[env_idx] = Some(row_start..inference_rows);
                inference_observations.extend(
                    obs_list
                        .iter()
                        .flat_map(|input| input.observation.data.iter().copied()),
                );
                inference_memory.extend(obs_list.iter().flat_map(|input| {
                    decode_policy_memory(&input.private_memory, config.model.recurrent_size)
                }));
            }

            if inference_rows == 0 {
                continue;
            }

            let mut initial_anchor_rows = Vec::new();
            let mut ecology_anchor_rows = Vec::new();
            let mut combat_anchor_rows = Vec::new();
            for (env_idx, rows) in environment_rows.iter().enumerate() {
                let Some(rows) = rows else { continue };
                let route = anchor_teacher_for_stage(
                    &config,
                    environment_curriculum_stages[env_idx],
                    reference_model.is_some(),
                    specialist_teachers.ecology.is_some(),
                    specialist_teachers.combat.is_some(),
                );
                match route.map(|(teacher, _)| teacher) {
                    Some(AnchorTeacher::Initial) => initial_anchor_rows.extend(rows.clone()),
                    Some(AnchorTeacher::Ecology) => ecology_anchor_rows.extend(rows.clone()),
                    Some(AnchorTeacher::Combat) => combat_anchor_rows.extend(rows.clone()),
                    None => {}
                }
            }
            debug_assert_eq!(
                initial_anchor_rows.len() + ecology_anchor_rows.len() + combat_anchor_rows.len(),
                if reference_model.is_some() {
                    inference_rows
                } else {
                    0
                },
                "every anchored inference row must route to exactly one teacher"
            );
            let mut anchor_targets = vec![None; inference_rows];
            let mut scatter_targets = |rows: &[usize], targets: Vec<PolicyAnchorTarget>| {
                assert_eq!(rows.len(), targets.len());
                for (&row, target) in rows.iter().zip(targets) {
                    assert!(anchor_targets[row].replace(target).is_none());
                }
            };
            if let Some(reference) = reference_model.as_ref() {
                scatter_targets(
                    &initial_anchor_rows,
                    infer_anchor_targets(
                        reference,
                        &inference_observations,
                        &inference_memory,
                        &initial_anchor_rows,
                        config.model.recurrent_size,
                        &device,
                    ),
                );
            }
            if let Some(teacher) = specialist_teachers.ecology.as_ref() {
                scatter_targets(
                    &ecology_anchor_rows,
                    infer_anchor_targets(
                        &teacher.model,
                        &inference_observations,
                        &inference_memory,
                        &ecology_anchor_rows,
                        config.model.recurrent_size,
                        &device,
                    ),
                );
            }
            if let Some(teacher) = specialist_teachers.combat.as_ref() {
                scatter_targets(
                    &combat_anchor_rows,
                    infer_anchor_targets(
                        &teacher.model,
                        &inference_observations,
                        &inference_memory,
                        &combat_anchor_rows,
                        config.model.recurrent_size,
                        &device,
                    ),
                );
            }

            let obs_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(inference_observations, [inference_rows, OBS_DIM]),
                &device,
            );
            let memory_tensor = Tensor::<B::InnerBackend, 2>::from_data(
                TensorData::new(
                    inference_memory,
                    [inference_rows, config.model.recurrent_size],
                ),
                &device,
            );
            let inner_observations = obs_tensor.inner();
            let output = model
                .valid()
                .forward_with_memory(inner_observations, memory_tensor);
            // Target, effort, amount, and signal masks depend on earlier
            // sampled heads, so logits are transferred once and normalized
            // row-wise on the host.
            let inference_width = NUM_POLICY_ACTION_KINDS
                + NUM_POLICY_TARGET_LOGITS
                + NUM_POLICY_EFFORT_LOGITS
                + NUM_POLICY_AMOUNT_LOGITS
                + NUM_SIGNAL_CHOICES
                + NUM_SIGNAL_STRENGTH_CHOICES
                + 1
                + config.model.recurrent_size;
            let inference_data: Vec<f32> = Tensor::cat(
                vec![
                    output.action_kind_logits,
                    output.target_logits,
                    output.effort_logits,
                    output.amount_logits,
                    output.signal_logits,
                    output.signal_strength_logits,
                    output.values,
                    output.next_memory,
                ],
                1,
            )
            .into_data()
            .to_vec()
            .unwrap();
            for (env_idx, env) in envs.iter_mut().enumerate() {
                let Some(rows) = environment_rows[env_idx].clone() else {
                    continue;
                };
                let obs_list = &env_observations[env_idx];
                let n_cells = obs_list.len();
                debug_assert_eq!(rows.len(), n_cells);
                let collect_new_transitions = config
                    .total_simulation_quanta_per_env
                    .is_none_or(|target| cumulative_sim_time_quanta[env_idx] < target);

                let mut actions: Vec<(CellId, PolicyChoice, Option<Vec<u8>>)> =
                    Vec::with_capacity(n_cells);
                let decision_time = env.sim_time_quanta();
                let action_kind_exploration_floor = config
                    .rollout_action_kind_exploration_floor(environment_curriculum_stages[env_idx]);
                let configured_attack_action_kind_exploration_floor = config
                    .rollout_attack_action_kind_exploration_floor(
                        environment_micro_combat_scenarios[env_idx],
                    );
                let stage = environment_curriculum_stages[env_idx];
                let initial_policy_anchor_coeff = anchor_teacher_for_stage(
                    &config,
                    stage,
                    reference_model.is_some(),
                    specialist_teachers.ecology.is_some(),
                    specialist_teachers.combat.is_some(),
                )
                .map_or(0.0, |(_, coefficient)| coefficient);

                for (cell_idx, input) in obs_list.iter().enumerate() {
                    let cell_id = input.cell_id;
                    let obs = &input.observation;
                    let inference_row = rows.start + cell_idx;
                    let row_start = inference_row * inference_width;
                    let row = &inference_data[row_start..row_start + inference_width];
                    let target_logits_start = NUM_POLICY_ACTION_KINDS;
                    let effort_logits_start = target_logits_start + NUM_POLICY_TARGET_LOGITS;
                    let amount_logits_start = effort_logits_start + NUM_POLICY_EFFORT_LOGITS;
                    let signal_start = amount_logits_start + NUM_POLICY_AMOUNT_LOGITS;
                    let kind_mask = policy_action_kind_mask(&obs.action_mask);
                    let attack_action_kind_exploration_floor =
                        if kind_mask[PolicyActionKind::Attack.index()] {
                            configured_attack_action_kind_exploration_floor
                        } else {
                            0.0
                        };
                    let (kind_probs, kind_log_probs) = masked_action_kind_distribution(
                        &row[..target_logits_start],
                        &kind_mask,
                        action_kind_exploration_floor,
                        attack_action_kind_exploration_floor,
                    );
                    let kind = sample_action(&kind_probs, &mut action_rng);
                    let target_start = target_logits_start + kind * NUM_POLICY_TARGETS;
                    let (target_probs, target_log_probs) = masked_distribution(
                        &row[target_start..target_start + NUM_POLICY_TARGETS],
                        &policy_target_mask(&obs.action_mask, kind),
                        0.0,
                    );
                    let target = sample_action(&target_probs, &mut action_rng);
                    let effort_start = effort_logits_start + kind * NUM_POLICY_EFFORTS;
                    let (effort_probs, effort_log_probs) = masked_distribution(
                        &row[effort_start..effort_start + NUM_POLICY_EFFORTS],
                        &policy_effort_mask(&obs.action_mask, kind, target),
                        0.0,
                    );
                    let effort = sample_action(&effort_probs, &mut action_rng);
                    let action = compose_policy_action(HierarchicalActionChoice {
                        kind,
                        target,
                        effort,
                    })
                    .expect("projected hierarchical masks must compose to a flat action");
                    let amount_mask = obs.amount_mask(action);
                    let amount_start = amount_logits_start + kind * NUM_AMOUNT_CHOICES;
                    let (amount_probs, amount_log_probs) = masked_distribution(
                        &row[amount_start..amount_start + NUM_AMOUNT_CHOICES],
                        &amount_mask,
                        0.0,
                    );
                    let amount = sample_action(&amount_probs, &mut action_rng);
                    let signal_mask = obs.signal_mask(action, amount);
                    let (signal_probs, signal_log_probs) = masked_distribution(
                        &row[signal_start..signal_start + NUM_SIGNAL_CHOICES],
                        &signal_mask,
                        0.0,
                    );
                    let signal = sample_action(&signal_probs, &mut action_rng);
                    let signal_strength_mask = obs.signal_strength_mask(action, amount, signal);
                    let signal_strength_start = signal_start + NUM_SIGNAL_CHOICES;
                    let (signal_strength_probs, signal_strength_log_probs) = masked_distribution(
                        &row[signal_strength_start
                            ..signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES],
                        &signal_strength_mask,
                        0.0,
                    );
                    let signal_strength = sample_action(&signal_strength_probs, &mut action_rng);
                    let log_prob = kind_log_probs[kind]
                        + target_log_probs[target]
                        + effort_log_probs[effort]
                        + amount_log_probs[amount]
                        + signal_log_probs[signal]
                        + signal_strength_log_probs[signal_strength];
                    let value = row[signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES];

                    if let Some(pending) = pending_transitions[env_idx].remove(&cell_id) {
                        let transition = &mut rollout.transitions[pending.rollout_index];
                        transition.next_value = value;
                        transition.elapsed_time =
                            env.elapsed_time_units(pending.started_at).max(f32::EPSILON);
                        transition.complete = true;
                    }

                    let memory_start = NUM_POLICY_ACTION_KINDS
                        + NUM_POLICY_TARGET_LOGITS
                        + NUM_POLICY_EFFORT_LOGITS
                        + NUM_POLICY_AMOUNT_LOGITS
                        + NUM_SIGNAL_CHOICES
                        + NUM_SIGNAL_STRENGTH_CHOICES
                        + 1;
                    // Once this environment has crossed the scientific
                    // frontier, advance only far enough to close actions that
                    // were already part of the rollout. Deterministic waits
                    // cannot inject new combat/resource effects into those
                    // pending rewards and are not added as training samples.
                    let submitted_choice = if collect_new_transitions {
                        PolicyChoice {
                            action,
                            amount,
                            signal,
                            signal_strength,
                        }
                    } else {
                        PolicyChoice {
                            action: policy_wait_action(),
                            amount: 0,
                            signal: 0,
                            signal_strength: 0,
                        }
                    };
                    actions.push((
                        cell_id,
                        submitted_choice,
                        Some(encode_policy_memory(
                            &row[memory_start..memory_start + config.model.recurrent_size],
                        )),
                    ));

                    if collect_new_transitions {
                        let rollout_index = rollout.len();
                        rollout.push(Transition {
                            observation: obs.to_vec(),
                            policy_memory: decode_policy_memory(
                                &input.private_memory,
                                config.model.recurrent_size,
                            ),
                            action_mask: obs.action_mask.to_vec(),
                            action,
                            amount_mask: amount_mask.to_vec(),
                            amount,
                            signal_mask: signal_mask.to_vec(),
                            signal,
                            signal_strength_mask: signal_strength_mask.to_vec(),
                            signal_strength,
                            action_kind_exploration_floor,
                            attack_action_kind_exploration_floor,
                            reward: 0.0,
                            value,
                            next_value: 0.0,
                            log_prob,
                            anchor: anchor_targets[inference_row].take(),
                            initial_policy_anchor_coeff,
                            done: false,
                            elapsed_time: 0.0,
                            complete: false,
                            trajectory_id: (env_idx, cell_id.0),
                        });
                        pending_transitions[env_idx].insert(
                            cell_id,
                            PendingTransition {
                                rollout_index,
                                started_at: decision_time,
                            },
                        );
                    }
                }

                // Step environment
                if config.total_simulation_quanta_per_env.is_some()
                    && total_timesteps.saturating_add(actions.len() as u64) > config.total_timesteps
                {
                    panic!(
                        "training exhausted its {}-action safety cap before every environment reached the simulation-time target",
                        config.total_timesteps
                    );
                }
                let step_result = env.step_with_policy_memory(&actions);
                total_timesteps += actions.len() as u64;
                let elapsed_quanta = env
                    .sim_time_quanta()
                    .checked_sub(decision_time)
                    .expect("environment simulation clock moved backwards");
                cumulative_sim_time_quanta[env_idx] = cumulative_sim_time_quanta[env_idx]
                    .checked_add(elapsed_quanta)
                    .expect("cumulative simulation-time exposure overflowed");
                if let (Some(state), Some(step)) =
                    (training_telemetry.as_mut(), step_result.telemetry.as_ref())
                {
                    state.apply_step(
                        env_idx,
                        step,
                        config.telemetry.max_state_samples_per_episode,
                    );
                }

                // Every open cell action owns rewards until its next decision,
                // including frontiers where only a different cell acted.
                for (cell_id, reward) in &step_result.rewards {
                    if let Some(pending) = pending_transitions[env_idx].get(cell_id) {
                        rollout.transitions[pending.rollout_index].reward += *reward;
                    }
                    env_episode_rewards[env_idx] += *reward;
                }

                let dead_cells = pending_transitions[env_idx]
                    .keys()
                    .copied()
                    .filter(|cell_id| !env.cell_is_alive(*cell_id))
                    .collect::<Vec<_>>();
                for cell_id in dead_cells {
                    let pending = pending_transitions[env_idx]
                        .remove(&cell_id)
                        .expect("dead pending cell disappeared from the collector");
                    let transition = &mut rollout.transitions[pending.rollout_index];
                    transition.next_value = 0.0;
                    transition.done = true;
                    transition.elapsed_time =
                        env.elapsed_time_units(pending.started_at).max(f32::EPSILON);
                    transition.complete = true;
                }

                // Update observations for next step
                if step_result.done {
                    for (_, pending) in pending_transitions[env_idx].drain() {
                        let transition = &mut rollout.transitions[pending.rollout_index];
                        transition.next_value = 0.0;
                        transition.done = true;
                        transition.elapsed_time =
                            env.elapsed_time_units(pending.started_at).max(f32::EPSILON);
                        transition.complete = true;
                    }
                    // Record episode outcome
                    if let Some(outcome) = step_result.outcome {
                        episode_stats.record(
                            outcome,
                            step_result.episode_step,
                            env_episode_rewards[env_idx],
                        );
                        if let (Some(state), Some(step)) =
                            (training_telemetry.as_mut(), step_result.telemetry.as_ref())
                        {
                            let telemetry_outcome = match outcome {
                                EpisodeOutcome::Win => TelemetryEpisodeOutcome::Win,
                                EpisodeOutcome::Loss => TelemetryEpisodeOutcome::Loss,
                                EpisodeOutcome::Timeout => TelemetryEpisodeOutcome::Timeout,
                                EpisodeOutcome::SafetyAbort => panic!(
                                    "training reached the non-scientific decision-frontier safety limit"
                                ),
                            };
                            let final_sample = step
                                .state_sample
                                .clone()
                                .expect("a terminal telemetry step has a state sample");
                            let record = state.finish_episode(
                                env_idx,
                                telemetry_outcome,
                                step_result.episode_step,
                                final_sample,
                                config.telemetry.max_state_samples_per_episode,
                            );
                            if config
                                .telemetry
                                .should_log_episode(env_episode_ids[env_idx])
                            {
                                publish_episode_record(artifact_root, &record)
                                    .expect("failed to publish immutable episode telemetry");
                            }
                        }
                    }
                    env_episode_rewards[env_idx] = 0.0;

                    env_episode_ids[env_idx] = env_episode_ids[env_idx].wrapping_add(1);
                    let reset_seed = config.seed
                        ^ (env_idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ env_episode_ids[env_idx].wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    let curriculum_time = cumulative_sim_time_quanta[env_idx];
                    let curriculum_stage = config.rollout_stage(curriculum_time);
                    let micro_combat_scenario = micro_combat_rotation
                        .assign(&config.combat_curriculum.micro_combat, curriculum_stage);
                    let stage_baseline = micro_combat_scenario.map_or_else(
                        || config.rollout_baseline_opponent(curriculum_stage, curriculum_time),
                        |index| {
                            config
                                .combat_curriculum
                                .micro_combat
                                .scenario(index)
                                .expect("selected micro-combat scenario exists")
                                .opponent
                        },
                    );
                    let assignment = choose_curriculum_rollout_opponent(
                        curriculum_stage,
                        stage_baseline,
                        &mut rollout_league,
                        config.env.opponent,
                        &config.self_play,
                        &mut opponent_rng,
                    );
                    *env = new_rollout_env(
                        &assignment,
                        &rollout_pool,
                        &retired_rollout_snapshots,
                        &config,
                        RolloutCurriculumAssignment {
                            stage: curriculum_stage,
                            simulation_time_quanta: curriculum_time,
                            micro_combat_scenario,
                        },
                        reset_seed,
                        &device,
                    )
                    .expect("failed to rotate rollout opponent");
                    environment_opponents[env_idx] = assignment;
                    environment_curriculum_stages[env_idx] = curriculum_stage;
                    environment_micro_combat_scenarios[env_idx] = micro_combat_scenario;
                    prune_retired_snapshots(&mut retired_rollout_snapshots, &environment_opponents);
                    if let Some(state) = training_telemetry.as_mut() {
                        state.start_episode(
                            env_idx,
                            env_episode_ids[env_idx],
                            reset_seed,
                            curriculum_label(&config, curriculum_stage, micro_combat_scenario),
                            env.telemetry_sample()
                                .expect("enabled telemetry has a reset sample"),
                        );
                    }
                    env_observations[env_idx] = env.get_policy_observations();
                } else {
                    env_observations[env_idx] = step_result.policy_observations;
                }
            }
            if config.total_simulation_quanta_per_env.is_some()
                && training_budget_complete(&config, total_timesteps, &cumulative_sim_time_quanta)
                && pending_transitions.iter().all(HashMap::is_empty)
            {
                break;
            }
        }

        let collected_transitions = rollout.len();
        rollout.retain_complete();
        let discarded_tails = collected_transitions.saturating_sub(rollout.len());
        let mean_elapsed_time = if rollout.is_empty() {
            0.0
        } else {
            rollout
                .transitions
                .iter()
                .map(|transition| transition.elapsed_time as f64)
                .sum::<f64>()
                / rollout.len() as f64
        };

        // PPO update
        if !rollout.is_empty() {
            // Diagnostic: print reward and advantage stats for first few updates
            if update_count < 5 {
                let rewards: Vec<f32> = rollout.transitions.iter().map(|t| t.reward).collect();
                let nonzero_rewards = rewards.iter().filter(|&&r| r.abs() > 1e-6).count();
                let max_reward = rewards.iter().cloned().fold(0.0f32, f32::max);
                let min_reward = rewards.iter().cloned().fold(0.0f32, f32::min);
                let mean_reward: f32 = rewards.iter().sum::<f32>() / rewards.len() as f32;
                let unique_trajs: std::collections::HashSet<(usize, usize)> = rollout
                    .transitions
                    .iter()
                    .map(|t| t.trajectory_id)
                    .collect();
                eprintln!(
                    "  [DEBUG] Rollout: {} transitions, {} trajectories",
                    rollout.len(),
                    unique_trajs.len()
                );
                eprintln!(
                    "  [DEBUG] Rewards: mean={:.6} min={:.4} max={:.4} nonzero={}/{}",
                    mean_reward,
                    min_reward,
                    max_reward,
                    nonzero_rewards,
                    rewards.len()
                );
            }

            let (updated_model, ppo_metrics) = ppo_update_anchored(
                model,
                &mut optimizer,
                &rollout,
                &config.ppo,
                &mut optimizer_rng,
                &device,
            );
            model = updated_model;
            update_count += 1;

            let total_cells: usize = envs.iter().map(BlobEnv::training_cells_alive).sum();
            let actions_per_second = total_timesteps as f64 / start_time.elapsed().as_secs_f64();
            let minimum_sim_time_quanta = cumulative_sim_time_quanta
                .iter()
                .copied()
                .min()
                .unwrap_or(0);
            let maximum_sim_time_quanta = cumulative_sim_time_quanta
                .iter()
                .copied()
                .max()
                .unwrap_or(0);
            let total_sim_time_quanta = cumulative_sim_time_quanta
                .iter()
                .map(|quanta| u128::from(*quanta))
                .sum::<u128>();
            let simulation_quanta_per_second =
                total_sim_time_quanta as f64 / start_time.elapsed().as_secs_f64();
            let peak_resident_set_bytes =
                peak_resident_set_bytes().map_or_else(String::new, |bytes| bytes.to_string());

            if update_count.is_multiple_of(10) || update_count <= 5 {
                println!(
                    "  {:>5} │ {:>9} │ {:>12.6} │ {:>11.6} │ {:>8.4} │ {:>8.5} │ {:>8.1}% │ {:>7.1} │ {:>4}",
                    update_count,
                    total_timesteps,
                    ppo_metrics.policy_loss,
                    ppo_metrics.value_loss,
                    ppo_metrics.entropy,
                    ppo_metrics.approx_kl,
                    episode_stats.win_rate() * 100.0,
                    episode_stats.avg_episode_len(),
                    total_cells,
                );
            }

            // Write metrics CSV row
            writeln!(
                metrics_file,
                "{},{},{:.6},{:.6},{:.4},{:.6},{:.6},{:.4},{:.4},{},{},{},{},{},{},{},{},{},{:.4},{:.1},{:.4},{},{},{},{:.4},{:.1},{},{},{},{:.1},{},{}",
                update_count,
                total_timesteps,
                ppo_metrics.policy_loss,
                ppo_metrics.value_loss,
                ppo_metrics.entropy,
                ppo_metrics.approx_kl,
                ppo_metrics.anchor_loss,
                ppo_metrics.clip_fraction,
                ppo_metrics.explained_variance,
                ppo_metrics.optimizer_steps,
                ppo_metrics.epochs_completed,
                ppo_metrics.early_stopped,
                ppo_metrics.recurrent_unroll_steps,
                ppo_metrics.recurrent_chunks,
                episode_stats.episodes_completed,
                episode_stats.wins,
                episode_stats.losses,
                episode_stats.timeouts,
                episode_stats.win_rate(),
                episode_stats.avg_episode_len(),
                episode_stats.avg_reward(),
                total_cells,
                rollout.len(),
                discarded_tails,
                mean_elapsed_time,
                actions_per_second,
                minimum_sim_time_quanta,
                maximum_sim_time_quanta,
                total_sim_time_quanta,
                simulation_quanta_per_second,
                peak_resident_set_bytes,
                config.rollout_stage(minimum_sim_time_quanta),
            )
            .unwrap();
            metrics_file
                .flush()
                .expect("failed to flush training metrics");

            let promotion_due =
                self_play_promotion_due(&config, minimum_sim_time_quanta, update_count);
            let terminal_boundary =
                training_budget_complete(&config, total_timesteps, &cumulative_sim_time_quanta);
            let update_interval_evaluation_due =
                config.eval_interval > 0 && update_count.is_multiple_of(config.eval_interval);
            let scheduled_world_time_frontier = config
                .combat_curriculum
                .crossed_competency_evaluation_frontier(
                    minimum_sim_time_before_update,
                    minimum_sim_time_quanta,
                );
            let world_time_competency_evaluation = scheduled_world_time_frontier.is_some();
            let regular_evaluation_due = update_interval_evaluation_due
                || world_time_competency_evaluation
                || (terminal_boundary && config.fixed_evaluation_enabled());
            let (
                evaluation,
                feeding_evaluation,
                retention_feeding_evaluation,
                contact_evaluation,
                micro_combat_evaluation,
            ) = if regular_evaluation_due || promotion_due {
                let valid_model = model.valid();
                let evaluation = evaluate_policy_suite(
                    &valid_model,
                    &config.env,
                    &config.reward,
                    &config.evaluation_opponents,
                    &evaluation_snapshots,
                    &evaluation_seeds,
                    &device,
                );
                write_fixed_evaluation_rows(
                    &mut evaluation_file,
                    update_count,
                    total_timesteps,
                    &evaluation,
                    config.evaluation_seed,
                    evaluation_seeds.len(),
                )
                .expect("failed to publish fixed evaluation metrics");
                last_fixed_evaluation_actions = Some(total_timesteps);
                println!(
                    "  eval {:>5} │ suite win {:>6.1}% │ worst {:>6.1}% │ reward {:>9.3} │ W/L/T {}/{}/{}",
                    update_count,
                    evaluation.win_rate * 100.0,
                    evaluation.worst_case_win_rate * 100.0,
                    evaluation.average_reward,
                    evaluation.wins,
                    evaluation.losses,
                    evaluation.timeouts,
                );
                for opponent in &evaluation.opponents {
                    println!(
                        "             │ {:>10} {:>6.1}% │ reward {:>9.3} │ len {:>7.1}",
                        opponent.opponent,
                        opponent.win_rate * 100.0,
                        opponent.average_reward,
                        opponent.average_episode_len,
                    );
                }
                let feeding_evaluation = config.feeding_curriculum.enabled.then(|| {
                    let report = evaluate_feeding_promotion(
                        &valid_model,
                        &config.env,
                        &config.reward,
                        &config.feeding_curriculum,
                        &evaluation_seeds,
                        &device,
                    );
                    write_feeding_evaluation_rows(
                        &mut feeding_evaluation_file,
                        update_count,
                        total_timesteps,
                        "configured",
                        &report,
                    )
                    .expect("failed to publish feeding evaluation metrics");
                    println!(
                        "  feed gate   │ {} │ on-food {:>6.1}% │ adjacent {:>6.1}%",
                        if report.passed { "passed" } else { "failed" },
                        report.stages[0].episode_success_rate * 100.0,
                        report.stages[1].episode_success_rate * 100.0,
                    );
                    report
                });
                let retention_feeding_evaluation =
                    retention_evaluation_seeds.as_ref().map(|seeds| {
                        let report = evaluate_feeding_promotion(
                            &valid_model,
                            &config.env,
                            &config.reward,
                            &config.feeding_curriculum,
                            seeds,
                            &device,
                        );
                        write_feeding_evaluation_rows(
                            &mut feeding_evaluation_file,
                            update_count,
                            total_timesteps,
                            "initial_qualification",
                            &report,
                        )
                        .expect("failed to publish retention feeding evaluation metrics");
                        println!(
                            "  retain gate │ {} │ on-food survival {:>6.1}% │ adjacent {:>6.1}%",
                            if report.passed { "passed" } else { "failed" },
                            report.stages[0].survival_rate * 100.0,
                            report.stages[1].survival_rate * 100.0,
                        );
                        report
                    });
                let contact_evaluation = config.combat_curriculum.enabled.then(|| {
                        let report = evaluate_contact(
                            &valid_model,
                            &config.env,
                            &config.reward,
                            &config.feeding_curriculum,
                            &config.combat_curriculum,
                            &evaluation_seeds,
                            &device,
                        );
                        write_contact_evaluation_rows(
                            &mut contact_evaluation_file,
                            update_count,
                            total_timesteps,
                            &report,
                        )
                        .expect("failed to publish contact evaluation metrics");
                        println!(
                            "  combat      │ {} wins │ {} damage │ {} kills (contact {}/{}, skirmish {}/{}) │ damaging {:>6.1}%",
                            contact_wins(&report),
                            report.damage_dealt,
                            report.kills,
                            report.kills_for_stage(FeedingCurriculumStage::Contact),
                            config.combat_curriculum.min_contact_kills_for_promotion,
                            report.kills_for_stage(FeedingCurriculumStage::Skirmish),
                            config.combat_curriculum.min_skirmish_kills_for_promotion,
                            report.damaging_episode_rate * 100.0,
                        );
                        report
                    });
                let micro_combat_evaluation = config
                        .combat_curriculum
                        .micro_combat
                        .enabled
                        .then(|| {
                            let report = evaluate_micro_combat(
                                &valid_model,
                                &config.env,
                                &config.reward,
                                config
                                    .combat_curriculum
                                    .micro_combat
                                    .suite
                                    .as_ref()
                                    .expect("validated micro-combat suite exists"),
                                &evaluation_seeds,
                                &device,
                            );
                            write_micro_combat_evaluation_rows(
                                &mut micro_combat_evaluation_file,
                                update_count,
                                total_timesteps,
                                &report,
                            )
                            .expect("failed to publish micro-combat evaluation metrics");
                            let gate = report.gate_summary();
                            println!(
                                "  micro gate  │ survival {:>6.1}% / {:>6.1}% │ elimination {:>6.1}% / {:>6.1}% │ attacks {:>6.2} / {:>6.2} per episode",
                                gate.survival_success_rate * 100.0,
                                config
                                    .combat_curriculum
                                    .micro_combat
                                    .min_survival_objective_success_rate
                                    * 100.0,
                                gate.elimination_success_rate * 100.0,
                                config
                                    .combat_curriculum
                                    .micro_combat
                                    .min_elimination_objective_success_rate
                                    * 100.0,
                                gate.attack_commitments_per_episode,
                                config
                                    .combat_curriculum
                                    .micro_combat
                                    .min_attack_commitments_per_episode,
                            );
                            report
                        });
                (
                    Some(evaluation),
                    feeding_evaluation,
                    retention_feeding_evaluation,
                    contact_evaluation,
                    micro_combat_evaluation,
                )
            } else {
                (None, None, None, None, None)
            };

            let competency_metrics = evaluation.as_ref().and_then(|evaluation| {
                CompetencyMetrics::from_reports(
                    evaluation,
                    feeding_evaluation.as_ref()?,
                    retention_feeding_evaluation.as_ref(),
                    contact_evaluation.as_ref()?,
                    micro_combat_evaluation.as_ref(),
                    &config.combat_curriculum,
                )
            });
            if let Some(metrics) = competency_metrics.as_ref() {
                let mut triggers = Vec::with_capacity(4);
                if update_interval_evaluation_due {
                    triggers.push("update_interval");
                }
                if world_time_competency_evaluation {
                    triggers.push("world_time_frontier");
                }
                if promotion_due {
                    triggers.push("promotion");
                }
                if terminal_boundary {
                    triggers.push("terminal");
                }
                write_competency_timeline_row(
                    &mut competency_timeline_file,
                    update_count,
                    total_timesteps,
                    minimum_sim_time_quanta,
                    maximum_sim_time_quanta,
                    total_sim_time_quanta,
                    scheduled_world_time_frontier,
                    &triggers.join("+"),
                    metrics,
                )
                .expect("failed to publish competency timeline");
            }
            let competency_candidate = competency_metrics.map(|metrics| CompetencyFrontierEntry {
                checkpoint_directory: frontier_checkpoint_directory.clone(),
                checkpoint: format!("checkpoint-{update_count:08}"),
                update: update_count,
                actions: total_timesteps,
                minimum_sim_time_quanta_per_env: minimum_sim_time_quanta,
                maximum_sim_time_quanta_per_env: maximum_sim_time_quanta,
                total_sim_time_quanta,
                metrics,
            });
            let mut next_competency_frontier = competency_frontier.clone();
            let competency_frontier_changed = competency_candidate
                .is_some_and(|candidate| next_competency_frontier.insert(candidate));
            let next_specialist_teacher_selection = if config.specialist_distillation.enabled {
                next_competency_frontier.specialist_teachers(
                    config
                        .specialist_distillation
                        .combat_precursor_min_skirmish_damage,
                )
            } else {
                SpecialistTeacherSelection::default()
            };

            let pool_evaluation = if promotion_due && !rollout_pool.is_empty() {
                let valid_model = model.valid();
                Some(evaluate_policy_suite(
                    &valid_model,
                    &config.env,
                    &config.reward,
                    &[],
                    &rollout_pool,
                    &evaluation_seeds,
                    &device,
                ))
            } else {
                None
            };
            if let Some(pool_metrics) = &pool_evaluation {
                for opponent in &pool_metrics.opponents {
                    writeln!(
                        evaluation_file,
                        "{},{},{},{},{},{},{},{:.6},,{:.2},{:.6},{},{},{}",
                        update_count,
                        total_timesteps,
                        opponent.opponent,
                        opponent.episodes,
                        opponent.wins,
                        opponent.losses,
                        opponent.timeouts,
                        opponent.win_rate,
                        opponent.average_episode_len,
                        opponent.average_reward,
                        opponent.actions,
                        config.evaluation_seed,
                        evaluation_seeds.len(),
                    )
                    .unwrap();
                }
                writeln!(
                    evaluation_file,
                    "{},{},pool_gate,{},{},{},{},{:.6},{:.6},{:.2},{:.6},{},{},{}",
                    update_count,
                    total_timesteps,
                    pool_metrics.episodes,
                    pool_metrics.wins,
                    pool_metrics.losses,
                    pool_metrics.timeouts,
                    pool_metrics.win_rate,
                    pool_metrics.worst_case_win_rate,
                    pool_metrics.average_episode_len,
                    pool_metrics.average_reward,
                    pool_metrics.actions,
                    config.evaluation_seed,
                    evaluation_seeds.len(),
                )
                .unwrap();
                evaluation_file.flush().unwrap();
            }
            let promote_to_rollout_pool = promotion_due
                && feeding_evaluation
                    .as_ref()
                    .is_none_or(|report| report.passed)
                && retention_feeding_evaluation
                    .as_ref()
                    .is_none_or(|report| report.passed)
                && contact_evaluation.as_ref().is_none_or(|report| {
                    report.meets_promotion_thresholds(&config.combat_curriculum)
                })
                && micro_combat_gate_passed(
                    &config.combat_curriculum.micro_combat,
                    micro_combat_evaluation.as_ref(),
                )
                && evaluation.as_ref().is_some_and(|candidate| {
                    rollout_pool.is_empty()
                        || (best_evaluation.as_ref().is_none_or(|incumbent| {
                            candidate.worst_case_win_rate
                                + config.self_play.max_fixed_worst_case_regression
                                >= incumbent.worst_case_win_rate
                        }) && pool_evaluation.as_ref().is_some_and(|pool_metrics| {
                            pool_metrics.opponents.iter().all(|metrics| {
                                metrics.win_rate >= config.self_play.min_pool_win_rate
                            })
                        }))
                });
            let league_promotion = promotion_due.then(|| {
                update_league_ratings(
                    &mut rollout_league,
                    pool_evaluation.as_ref(),
                    &config.self_play,
                )
            });
            let accepted_promotion = promote_to_rollout_pool.then(|| {
                league_promotion
                    .clone()
                    .expect("a pool promotion must have a rating")
            });
            if promotion_due {
                if let Some(pool_metrics) = &pool_evaluation {
                    println!(
                        "  pool gate   │ worst {:>6.1}% │ required {:>6.1}% │ members {}",
                        pool_metrics.worst_case_win_rate * 100.0,
                        config.self_play.min_pool_win_rate * 100.0,
                        rollout_pool.len(),
                    );
                }
                println!(
                    "  promotion   │ {} │ candidate rating {:.1}",
                    if promote_to_rollout_pool {
                        "accepted"
                    } else {
                        "rejected"
                    },
                    league_promotion
                        .as_ref()
                        .expect("a due promotion must have a rating")
                        .rating,
                );
            }

            let is_new_best = evaluation.as_ref().is_some_and(|candidate| {
                feeding_evaluation
                    .as_ref()
                    .is_none_or(|report| report.passed)
                    && retention_feeding_evaluation
                        .as_ref()
                        .is_none_or(|report| report.passed)
                    && contact_evaluation.as_ref().is_none_or(|report| {
                        report.meets_promotion_thresholds(&config.combat_curriculum)
                    })
                    && micro_combat_gate_passed(
                        &config.combat_curriculum.micro_combat,
                        micro_combat_evaluation.as_ref(),
                    )
                    && best_evaluation.as_ref().is_none_or(|incumbent| {
                        evaluation_is_better(
                            candidate,
                            contact_evaluation.as_ref(),
                            micro_combat_evaluation.as_ref(),
                            incumbent,
                            best_contact_evaluation.as_ref(),
                            best_micro_combat_evaluation.as_ref(),
                        )
                    })
            });
            let periodic_checkpoint = config.checkpoint_interval > 0
                && update_count.is_multiple_of(config.checkpoint_interval);
            if is_new_best {
                best_evaluation = evaluation.clone();
                best_contact_evaluation = contact_evaluation.clone();
                best_micro_combat_evaluation = micro_combat_evaluation.clone();
            }

            // A published state always describes the clean boundary between
            // updates. Per-update metrics have already been emitted and are
            // intentionally not carried into the next update.
            episode_stats.reset();
            if checkpoint_publication_due(
                periodic_checkpoint,
                is_new_best,
                promote_to_rollout_pool,
                terminal_boundary,
                competency_frontier_changed,
                world_time_competency_evaluation,
            ) {
                let resume_state = TrainingResumeState {
                    schema_version: TRAINING_ARTIFACT_SCHEMA_VERSION,
                    total_timesteps,
                    cumulative_sim_time_quanta: cumulative_sim_time_quanta.clone(),
                    update_count,
                    action_rng: action_rng.clone(),
                    optimizer_rng: optimizer_rng.clone(),
                    opponent_rng: opponent_rng.clone(),
                    env_episode_ids: env_episode_ids.clone(),
                    env_episode_rewards: env_episode_rewards.clone(),
                    environments: envs
                        .iter()
                        .map(BlobEnv::checkpoint)
                        .collect::<Result<Vec<_>, _>>()
                        .expect("failed to checkpoint RL environments"),
                    best_evaluation: best_evaluation.clone(),
                    best_contact_evaluation: best_contact_evaluation.clone(),
                    best_micro_combat_evaluation: best_micro_combat_evaluation.clone(),
                    competency_frontier: next_competency_frontier.clone(),
                    specialist_teachers: next_specialist_teacher_selection.clone(),
                    rollout_pool: rollout_league.clone(),
                    active_retired_snapshots: retired_rollout_snapshots
                        .iter()
                        .map(PolicySnapshot::descriptor)
                        .collect(),
                    environment_opponents: environment_opponents.clone(),
                    environment_curriculum_stages: environment_curriculum_stages.clone(),
                    micro_combat_rotation: micro_combat_rotation.clone(),
                    environment_micro_combat_scenarios: environment_micro_combat_scenarios.clone(),
                    telemetry: training_telemetry.clone(),
                };
                let checkpoint = publish_checkpoint(
                    artifact_root,
                    &model,
                    &optimizer,
                    &config,
                    update_count,
                    total_timesteps,
                    &compiled_ruleset_hash,
                    evaluation.as_ref(),
                    feeding_evaluation.as_ref(),
                    retention_feeding_evaluation.as_ref(),
                    contact_evaluation.as_ref(),
                    micro_combat_evaluation.as_ref(),
                    accepted_promotion.as_ref(),
                    &resume_state,
                )
                .expect("failed to publish immutable training artifact");
                println!("  Checkpoint published: {}", checkpoint.display());
                if competency_frontier_changed {
                    competency_frontier = next_competency_frontier;
                    let frontier_path =
                        publish_competency_frontier(artifact_root, &competency_frontier)
                            .expect("failed to publish competency frontier");
                    println!(
                        "  Frontier    │ {} nondominated checkpoints ({})",
                        competency_frontier.entries.len(),
                        frontier_path.display(),
                    );
                }
                if specialist_teacher_selection != next_specialist_teacher_selection {
                    specialist_teachers = load_specialist_teachers::<B::InnerBackend>(
                        &next_specialist_teacher_selection,
                        &competency_frontier,
                        &device,
                    )
                    .expect("failed to activate specialist distillation teachers");
                    specialist_teacher_selection = next_specialist_teacher_selection;
                    let combat_teacher = specialist_teacher_selection.combat.as_ref().map_or(
                        "none".to_string(),
                        |teacher| {
                            let precursor = competency_frontier.entries.iter().any(|entry| {
                                entry.identity() == *teacher && !entry.metrics.combat_passed
                            });
                            if precursor {
                                format!("{} (precursor)", teacher.checkpoint)
                            } else {
                                teacher.checkpoint.clone()
                            }
                        },
                    );
                    println!(
                        "  Teachers    │ ecology {} │ combat {}",
                        specialist_teacher_selection
                            .ecology
                            .as_ref()
                            .map_or("none".to_string(), |teacher| teacher.checkpoint.clone()),
                        combat_teacher,
                    );
                }
                if is_new_best {
                    publish_best_pointer(artifact_root, &checkpoint)
                        .expect("failed to publish best checkpoint pointer");
                }
                if promote_to_rollout_pool {
                    let snapshot = load_policy_snapshot::<B::InnerBackend>(&checkpoint, &device)
                        .expect("failed to load newly promoted rollout opponent");
                    let promotion = accepted_promotion
                        .expect("an accepted rollout promotion must have a rating");
                    rollout_league.push(RolloutLeagueMember {
                        snapshot: snapshot.descriptor(),
                        rating: promotion.rating,
                        evaluation_games: promotion.evaluation_games,
                        rollout_selections: 0,
                    });
                    rollout_pool.push(snapshot);
                    trim_rollout_pool(
                        &mut rollout_pool,
                        &mut rollout_league,
                        &mut retired_rollout_snapshots,
                        &environment_opponents,
                        config.self_play.max_opponent_pool,
                    );
                    let manifest = publish_rollout_pool_manifest(
                        artifact_root,
                        update_count,
                        total_timesteps,
                        &compiled_ruleset_hash,
                        &config.self_play,
                        feeding_evaluation.as_ref(),
                        retention_feeding_evaluation.as_ref(),
                        contact_evaluation.as_ref(),
                        micro_combat_evaluation.as_ref(),
                        &rollout_league,
                    )
                    .expect("failed to publish immutable rollout-pool manifest");
                    println!(
                        "  Rollout pool: {} members ({})",
                        rollout_pool.len(),
                        manifest.display()
                    );
                }
            }
        }
    }

    if config.fixed_evaluation_enabled() && last_fixed_evaluation_actions != Some(total_timesteps) {
        let valid_model = model.valid();
        let evaluation = evaluate_policy_suite(
            &valid_model,
            &config.env,
            &config.reward,
            &config.evaluation_opponents,
            &evaluation_snapshots,
            &evaluation_seeds,
            &device,
        );
        write_fixed_evaluation_rows(
            &mut evaluation_file,
            update_count,
            total_timesteps,
            &evaluation,
            config.evaluation_seed,
            evaluation_seeds.len(),
        )
        .expect("failed to publish terminal evaluation metrics");
        println!(
            "  final eval  │ suite win {:>6.1}% │ worst {:>6.1}% │ reward {:>9.3}",
            evaluation.win_rate * 100.0,
            evaluation.worst_case_win_rate * 100.0,
            evaluation.average_reward,
        );
        if config.feeding_curriculum.enabled {
            let report = evaluate_feeding_promotion(
                &valid_model,
                &config.env,
                &config.reward,
                &config.feeding_curriculum,
                &evaluation_seeds,
                &device,
            );
            write_feeding_evaluation_rows(
                &mut feeding_evaluation_file,
                update_count,
                total_timesteps,
                "configured",
                &report,
            )
            .expect("failed to publish terminal feeding evaluation metrics");
            println!(
                "  final feed  │ {} │ on-food {:>6.1}% │ adjacent {:>6.1}%",
                if report.passed { "passed" } else { "failed" },
                report.stages[0].episode_success_rate * 100.0,
                report.stages[1].episode_success_rate * 100.0,
            );
            if let Some(seeds) = &retention_evaluation_seeds {
                let report = evaluate_feeding_promotion(
                    &valid_model,
                    &config.env,
                    &config.reward,
                    &config.feeding_curriculum,
                    seeds,
                    &device,
                );
                write_feeding_evaluation_rows(
                    &mut feeding_evaluation_file,
                    update_count,
                    total_timesteps,
                    "initial_qualification",
                    &report,
                )
                .expect("failed to publish terminal retention feeding evaluation metrics");
                println!(
                    "  final retain │ {} │ on-food survival {:>6.1}% │ adjacent {:>6.1}%",
                    if report.passed { "passed" } else { "failed" },
                    report.stages[0].survival_rate * 100.0,
                    report.stages[1].survival_rate * 100.0,
                );
            }
        }
        if config.combat_curriculum.enabled {
            let report = evaluate_contact(
                &valid_model,
                &config.env,
                &config.reward,
                &config.feeding_curriculum,
                &config.combat_curriculum,
                &evaluation_seeds,
                &device,
            );
            write_contact_evaluation_rows(
                &mut contact_evaluation_file,
                update_count,
                total_timesteps,
                &report,
            )
            .expect("failed to publish terminal contact evaluation metrics");
            println!(
                "  final combat │ {} wins │ {} damage │ {} kills (contact {}/{}, skirmish {}/{}) │ damaging {:>6.1}%",
                contact_wins(&report),
                report.damage_dealt,
                report.kills,
                report.kills_for_stage(FeedingCurriculumStage::Contact),
                config.combat_curriculum.min_contact_kills_for_promotion,
                report.kills_for_stage(FeedingCurriculumStage::Skirmish),
                config.combat_curriculum.min_skirmish_kills_for_promotion,
                report.damaging_episode_rate * 100.0,
            );
        }
        if config.combat_curriculum.micro_combat.enabled {
            let report = evaluate_micro_combat(
                &valid_model,
                &config.env,
                &config.reward,
                config
                    .combat_curriculum
                    .micro_combat
                    .suite
                    .as_ref()
                    .expect("validated micro-combat suite exists"),
                &evaluation_seeds,
                &device,
            );
            write_micro_combat_evaluation_rows(
                &mut micro_combat_evaluation_file,
                update_count,
                total_timesteps,
                &report,
            )
            .expect("failed to publish terminal micro-combat evaluation metrics");
            let gate = report.gate_summary();
            println!(
                "  final micro │ survival {:>6.1}% │ elimination {:>6.1}%",
                gate.survival_success_rate * 100.0,
                gate.elimination_success_rate * 100.0,
            );
        }
    }

    if let Some(telemetry) = training_telemetry.as_mut() {
        for (environment, env) in envs.iter().enumerate() {
            telemetry.sample_active_episode(
                environment,
                env.telemetry_sample()
                    .expect("enabled telemetry has a terminal active sample"),
                config.telemetry.max_state_samples_per_episode,
            );
        }
        let summary = telemetry.summary();
        let path = publish_training_summary(artifact_root, &summary)
            .expect("failed to publish training telemetry summary");
        println!(
            "  Telemetry: {} completed episodes, {} state samples ({})",
            summary.completed_episodes,
            summary.state_samples,
            path.display()
        );
    }

    let total_time = start_time.elapsed().as_secs_f64();
    println!(
        "\n  Training complete: {} timesteps in {:.1}s ({:.0} steps/sec)",
        total_timesteps,
        total_time,
        total_timesteps as f64 / total_time
    );

    // Save the trained model
    if let Some(path) = save_model_path {
        model
            .clone()
            .save_file(path, &CompactRecorder::new())
            .expect("Failed to save model");
        println!("  Model saved to: {}", path);
    }
}

fn masked_distribution<const N: usize>(
    logits: &[f32],
    mask: &[bool; N],
    exploration_floor: f32,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(logits.len(), N);
    let maximum = logits
        .iter()
        .copied()
        .zip(mask)
        .filter_map(|(logit, allowed)| (*allowed && logit.is_finite()).then_some(logit))
        .max_by(f32::total_cmp)
        .unwrap_or(0.0);
    let mut weights = logits
        .iter()
        .copied()
        .zip(mask)
        .map(|(logit, allowed)| {
            if *allowed && logit.is_finite() {
                (logit - maximum).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let denominator = weights.iter().sum::<f32>();
    if denominator <= 0.0 || !denominator.is_finite() {
        weights.fill(0.0);
        weights[0] = 1.0;
    } else {
        for weight in &mut weights {
            *weight /= denominator;
        }
    }
    let legal = mask.iter().filter(|allowed| **allowed).count();
    if exploration_floor > 0.0 && legal > 0 {
        let uniform = exploration_floor / legal as f32;
        for (weight, allowed) in weights.iter_mut().zip(mask) {
            *weight = if *allowed {
                (1.0 - exploration_floor) * *weight + uniform
            } else {
                0.0
            };
        }
    }
    let log_probs = weights
        .iter()
        .map(|probability| {
            if *probability > 0.0 {
                probability.ln()
            } else {
                -1.0e9
            }
        })
        .collect();
    (weights, log_probs)
}

fn masked_action_kind_distribution(
    logits: &[f32],
    mask: &[bool; NUM_POLICY_ACTION_KINDS],
    uniform_floor: f32,
    attack_floor: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (mut probabilities, _) = masked_distribution(logits, mask, uniform_floor);
    if attack_floor > 0.0 {
        debug_assert!(mask[PolicyActionKind::Attack.index()]);
        for probability in &mut probabilities {
            *probability *= 1.0 - attack_floor;
        }
        probabilities[PolicyActionKind::Attack.index()] += attack_floor;
    }
    let log_probabilities = probabilities
        .iter()
        .map(|probability| {
            if *probability > 0.0 {
                probability.ln()
            } else {
                -1.0e9
            }
        })
        .collect();
    (probabilities, log_probabilities)
}

/// Sample an action from a probability distribution.
fn sample_action(probs: &[f32], rng: &mut impl Rng) -> usize {
    let rng_val: f32 = rng.random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rng_val <= cumsum {
            return i;
        }
    }
    probs.len() - 1 // fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn host_peak_resident_set_is_available_for_training_cost_comparisons() {
        assert!(peak_resident_set_bytes().is_some_and(|bytes| bytes > 0));
    }
    use crate::artifact::RolloutSnapshotDescriptor;
    use crate::evaluation::OpponentEvaluationMetrics;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray<f32>>;

    fn league_member(hash: &str, rating: f64, selections: u64) -> RolloutLeagueMember {
        RolloutLeagueMember {
            snapshot: RolloutSnapshotDescriptor {
                directory: format!("/snapshot/{hash}"),
                checkpoint: format!("checkpoint-{hash}"),
                update: 1,
                actions: 10,
                model_sha256: hash.to_string(),
            },
            rating,
            evaluation_games: 0,
            rollout_selections: selections,
        }
    }

    #[test]
    fn specialist_anchor_routes_are_stage_local_and_fall_back_to_initial() {
        let mut config = TrainingConfig::default();
        config.feeding_curriculum.enabled = true;
        config.combat_curriculum.enabled = true;
        config.specialist_distillation.enabled = true;
        config.ppo.initial_policy_anchor_coeff = 0.25;

        assert_eq!(
            anchor_teacher_for_stage(&config, FeedingCurriculumStage::OnFood, true, false, true),
            Some((AnchorTeacher::Initial, 0.25))
        );
        assert_eq!(
            anchor_teacher_for_stage(
                &config,
                FeedingCurriculumStage::AdjacentFood,
                true,
                true,
                true
            ),
            Some((AnchorTeacher::Ecology, 0.1))
        );
        assert_eq!(
            anchor_teacher_for_stage(&config, FeedingCurriculumStage::Skirmish, true, true, true),
            Some((AnchorTeacher::Combat, 0.1))
        );
        assert_eq!(
            anchor_teacher_for_stage(
                &config,
                FeedingCurriculumStage::Competitive,
                true,
                true,
                true
            ),
            Some((AnchorTeacher::Initial, 0.25))
        );
    }

    #[test]
    fn compact_anchor_subbatches_match_full_batch_outputs() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        NdArray::<f32>::seed(&device, 71);
        let model = crate::model::PolicyValueNetConfig::new().init::<NdArray<f32>>(&device);
        let recurrent_size = model.recurrent_size();
        let observations = (0..4 * OBS_DIM)
            .map(|index| (index as f32 * 0.013).sin())
            .collect::<Vec<_>>();
        let memory = (0..4 * recurrent_size)
            .map(|index| (index as f32 * 0.017).cos())
            .collect::<Vec<_>>();
        let full = infer_anchor_targets(
            &model,
            &observations,
            &memory,
            &[0, 1, 2, 3],
            recurrent_size,
            &device,
        );
        let even = infer_anchor_targets(
            &model,
            &observations,
            &memory,
            &[0, 2],
            recurrent_size,
            &device,
        );
        let odd = infer_anchor_targets(
            &model,
            &observations,
            &memory,
            &[1, 3],
            recurrent_size,
            &device,
        );
        let compare = |left: &PolicyAnchorTarget, right: &PolicyAnchorTarget| {
            for (left, right) in [
                (&left.action_kind_logits, &right.action_kind_logits),
                (&left.target_logits, &right.target_logits),
                (&left.effort_logits, &right.effort_logits),
                (&left.amount_logits, &right.amount_logits),
                (&left.signal_logits, &right.signal_logits),
                (&left.signal_strength_logits, &right.signal_strength_logits),
                (&left.next_memory, &right.next_memory),
            ] {
                assert_eq!(left.len(), right.len());
                assert!(left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| (left - right).abs() < 1.0e-5));
            }
        };
        compare(&full[0], &even[0]);
        compare(&full[2], &even[1]);
        compare(&full[1], &odd[0]);
        compare(&full[3], &odd[1]);
    }

    #[test]
    fn action_kind_exploration_floor_only_spreads_probability_over_legal_kinds() {
        let (probabilities, log_probabilities) =
            masked_distribution(&[20.0, -20.0, 4.0, 0.0], &[true, true, false, true], 0.3);

        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        assert_eq!(probabilities[2], 0.0);
        assert_eq!(log_probabilities[2], -1.0e9);
        for index in [0, 1, 3] {
            assert!(probabilities[index] >= 0.1 - 1.0e-6);
            assert!((log_probabilities[index].exp() - probabilities[index]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn micro_combat_attack_mixture_is_applied_after_uniform_legal_exploration() {
        let mut mask = [false; NUM_POLICY_ACTION_KINDS];
        mask[PolicyActionKind::Wait.index()] = true;
        mask[PolicyActionKind::Guard.index()] = true;
        mask[PolicyActionKind::Attack.index()] = true;
        let (base, _) = masked_distribution(&[0.0; NUM_POLICY_ACTION_KINDS], &mask, 0.3);
        let (mixed, logs) =
            masked_action_kind_distribution(&[0.0; NUM_POLICY_ACTION_KINDS], &mask, 0.3, 0.5);

        assert!((mixed.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        for kind in 0..NUM_POLICY_ACTION_KINDS {
            let expected = 0.5 * base[kind]
                + if kind == PolicyActionKind::Attack.index() {
                    0.5
                } else {
                    0.0
                };
            assert!((mixed[kind] - expected).abs() < 1.0e-6);
            if mixed[kind] > 0.0 {
                assert!((logs[kind].exp() - mixed[kind]).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn action_sampling_is_seed_reproducible() {
        let probabilities = [0.1, 0.2, 0.3, 0.4];
        let mut left = ChaCha12Rng::seed_from_u64(7);
        let mut right = ChaCha12Rng::seed_from_u64(7);
        let left = (0..64)
            .map(|_| sample_action(&probabilities, &mut left))
            .collect::<Vec<_>>();
        let right = (0..64)
            .map(|_| sample_action(&probabilities, &mut right))
            .collect::<Vec<_>>();
        assert_eq!(left, right);
    }

    #[test]
    fn league_rating_update_uses_held_out_snapshot_results() {
        let mut league = vec![league_member("incumbent", 1_000.0, 0)];
        let opponent = OpponentEvaluationMetrics {
            opponent: EvaluationOpponent::Snapshot {
                checkpoint: "checkpoint-incumbent".into(),
                update: 1,
                model_sha256: "incumbent".into(),
            },
            episodes: 10,
            wins: 10,
            losses: 0,
            timeouts: 0,
            win_rate: 1.0,
            average_episode_len: 1.0,
            average_reward: 1.0,
            actions: 10,
        };
        let evaluation = EvaluationMetrics {
            seeds: (0..10).collect(),
            opponents: vec![opponent],
            episodes: 10,
            wins: 10,
            losses: 0,
            timeouts: 0,
            worst_case_win_rate: 1.0,
            win_rate: 1.0,
            average_episode_len: 1.0,
            average_reward: 1.0,
            actions: 10,
        };

        let promotion =
            update_league_ratings(&mut league, Some(&evaluation), &SelfPlayConfig::default());

        assert_eq!(promotion.rating, 1_016.0);
        assert_eq!(promotion.evaluation_games, 10);
        assert_eq!(league[0].rating, 984.0);
        assert_eq!(league[0].evaluation_games, 10);
    }

    #[test]
    fn league_selection_is_reproducible_and_reduces_overexposure() {
        let config = SelfPlayConfig {
            baseline_probability: 0.0,
            exposure_exponent: 1.0,
            ..SelfPlayConfig::default()
        };
        let initial = vec![
            league_member("overexposed", 1_000.0, 10_000),
            league_member("underexposed", 1_000.0, 0),
        ];
        let mut left_league = initial.clone();
        let mut right_league = initial;
        let mut left_rng = ChaCha12Rng::seed_from_u64(19);
        let mut right_rng = ChaCha12Rng::seed_from_u64(19);
        let left = (0..256)
            .map(|_| {
                choose_rollout_opponent(
                    &mut left_league,
                    OpponentProfile::Wait,
                    &config,
                    &mut left_rng,
                )
            })
            .collect::<Vec<_>>();
        let right = (0..256)
            .map(|_| {
                choose_rollout_opponent(
                    &mut right_league,
                    OpponentProfile::Wait,
                    &config,
                    &mut right_rng,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(left, right);
        assert_eq!(left_league, right_league);
        let overexposed_draws = left_league[0].rollout_selections - 10_000;
        assert!(left_league[1].rollout_selections > overexposed_draws);
        assert_eq!(left_league[1].rollout_selections + overexposed_draws, 256);
    }

    #[test]
    fn noncompetitive_curriculum_uses_its_baseline_without_exposing_pool_members() {
        let mut league = vec![league_member("snapshot", 1_000.0, 7)];
        let mut rng = ChaCha12Rng::seed_from_u64(91);
        for stage in [
            FeedingCurriculumStage::OnFood,
            FeedingCurriculumStage::AdjacentFood,
        ] {
            assert_eq!(
                choose_curriculum_rollout_opponent(
                    stage,
                    OpponentProfile::Wait,
                    &mut league,
                    OpponentProfile::Aggressive,
                    &SelfPlayConfig {
                        baseline_probability: 0.0,
                        ..SelfPlayConfig::default()
                    },
                    &mut rng,
                ),
                RolloutOpponentAssignment::Baseline {
                    profile: OpponentProfile::Wait,
                }
            );
        }
        assert_eq!(
            choose_curriculum_rollout_opponent(
                FeedingCurriculumStage::Contact,
                OpponentProfile::Defensive,
                &mut league,
                OpponentProfile::Aggressive,
                &SelfPlayConfig {
                    baseline_probability: 0.0,
                    ..SelfPlayConfig::default()
                },
                &mut rng,
            ),
            RolloutOpponentAssignment::Baseline {
                profile: OpponentProfile::Defensive,
            }
        );
        assert_eq!(
            choose_curriculum_rollout_opponent(
                FeedingCurriculumStage::Skirmish,
                OpponentProfile::Aggressive,
                &mut league,
                OpponentProfile::Defensive,
                &SelfPlayConfig {
                    baseline_probability: 0.0,
                    ..SelfPlayConfig::default()
                },
                &mut rng,
            ),
            RolloutOpponentAssignment::Baseline {
                profile: OpponentProfile::Aggressive,
            }
        );
        assert_eq!(league[0].rollout_selections, 7);
    }

    #[test]
    #[ignore = "manual league-selection overhead diagnostic"]
    fn benchmark_league_opponent_selection() {
        let config = SelfPlayConfig::default();
        let mut league = (0..10)
            .map(|index| league_member(&format!("member-{index}"), 980.0 + index as f64 * 5.0, 0))
            .collect::<Vec<_>>();
        let mut rng = ChaCha12Rng::seed_from_u64(23);
        let draws = 1_000_000_u64;
        let started = Instant::now();
        for _ in 0..draws {
            std::hint::black_box(choose_rollout_opponent(
                &mut league,
                OpponentProfile::Wait,
                &config,
                &mut rng,
            ));
        }
        let elapsed = started.elapsed();
        println!(
            "league selection: {draws} draws in {:.3}s ({:.1} ns/draw)",
            elapsed.as_secs_f64(),
            elapsed.as_nanos() as f64 / draws as f64,
        );
    }

    #[test]
    fn best_evaluation_orders_by_wins_reward_then_shorter_games() {
        let baseline = EvaluationMetrics {
            seeds: vec![1],
            opponents: Vec::new(),
            episodes: 1,
            wins: 0,
            losses: 1,
            timeouts: 0,
            worst_case_win_rate: 0.0,
            win_rate: 0.0,
            average_episode_len: 10.0,
            average_reward: -2.0,
            actions: 5,
        };
        let mut candidate = baseline.clone();
        candidate.average_reward = -1.0;
        assert!(evaluation_is_better(
            &candidate, None, None, &baseline, None, None
        ));
        candidate.average_reward = -2.0;
        candidate.average_episode_len = 9.0;
        assert!(evaluation_is_better(
            &candidate, None, None, &baseline, None, None
        ));
        candidate.win_rate = 1.0;
        candidate.average_reward = -100.0;
        assert!(evaluation_is_better(
            &candidate, None, None, &baseline, None, None
        ));

        let mut robust = baseline.clone();
        robust.worst_case_win_rate = 0.5;
        robust.win_rate = 0.1;
        let mut brittle = baseline.clone();
        brittle.worst_case_win_rate = 0.25;
        brittle.win_rate = 1.0;
        brittle.average_reward = 1_000.0;
        assert!(evaluation_is_better(
            &robust, None, None, &brittle, None, None
        ));

        let contact = |damage, kills| ContactEvaluationReport {
            schema_version: crate::contact_evaluation::CONTACT_EVALUATION_SCHEMA_VERSION,
            ruleset_hash: "rules".into(),
            seeds: vec![1],
            contact_sim_time_limit_quanta: 32_768,
            skirmish_sim_time_limit_quanta: 65_536,
            variants: Vec::new(),
            episodes: 1,
            attacking_episodes: 1,
            damaging_episodes: 1,
            attacks_committed: 1,
            attacks_succeeded: 1,
            damage_dealt: damage,
            kills,
            attacking_episode_rate: 1.0,
            damaging_episode_rate: 1.0,
            attack_success_rate: 1.0,
        };
        let combat_candidate = contact(20, 1);
        let combat_incumbent = contact(100, 0);
        let mut self_exhaustion_winner = contact(100, 0);
        self_exhaustion_winner
            .variants
            .push(crate::contact_evaluation::ContactVariantMetrics {
                stage: FeedingCurriculumStage::Skirmish,
                cells_per_team: 4,
                initial_energy: 60,
                opponent: OpponentProfile::Aggressive,
                episodes: 8,
                wins: 8,
                losses: 0,
                timeouts: 0,
                safety_aborts: 0,
                attacking_episodes: 0,
                damaging_episodes: 0,
                attacks_committed: 0,
                attacks_succeeded: 0,
                attacks_frustrated: 0,
                attacks_interrupted: 0,
                damage_dealt: 0,
                kills: 0,
                attacking_episode_rate: 0.0,
                damaging_episode_rate: 0.0,
                attack_success_rate: 0.0,
            });
        assert!(contact_is_better(
            Some(&combat_candidate),
            Some(&self_exhaustion_winner)
        ));
        let mut lower_reward = candidate.clone();
        lower_reward.win_rate = baseline.win_rate;
        lower_reward.average_reward = baseline.average_reward - 100.0;
        assert!(evaluation_is_better(
            &lower_reward,
            Some(&combat_candidate),
            None,
            &baseline,
            Some(&combat_incumbent),
            None,
        ));

        let micro = |elimination_successes| MicroCombatEvaluationReport {
            schema_version: crate::micro_combat::MICRO_COMBAT_EVALUATION_SCHEMA_VERSION,
            ruleset_hash: "rules".into(),
            suite_sha256: "suite".into(),
            seeds: vec![1],
            scenarios: vec![crate::micro_combat::MicroCombatScenarioMetrics {
                scenario: "elimination".into(),
                objective: crate::micro_combat::MicroCombatObjective::Elimination,
                episodes: 1,
                objective_successes: elimination_successes,
                objective_success_rate: elimination_successes as f64,
                ..crate::micro_combat::MicroCombatScenarioMetrics::default()
            }],
        };
        let worse_micro = micro(0);
        let better_micro = micro(1);
        let mut shorter = baseline.clone();
        shorter.average_episode_len -= 1.0;
        assert!(!evaluation_is_better(
            &shorter,
            None,
            Some(&worse_micro),
            &baseline,
            None,
            Some(&better_micro),
        ));
    }

    #[test]
    fn simulation_budget_requires_every_environment_while_legacy_uses_actions() {
        let mut config = TrainingConfig {
            total_timesteps: 100,
            total_simulation_quanta_per_env: Some(64),
            ..TrainingConfig::default()
        };
        assert!(!training_budget_complete(&config, 100, &[64, 63]));
        assert!(training_budget_complete(&config, 12, &[64, 80]));
        config.total_simulation_quanta_per_env = None;
        assert!(!training_budget_complete(&config, 99, &[u64::MAX]));
        assert!(training_budget_complete(&config, 100, &[0]));
    }

    #[test]
    fn enabled_micro_combat_gate_fails_closed_without_evidence() {
        let disabled = MicroCombatTrainingConfig::default();
        assert!(micro_combat_gate_passed(&disabled, None));
        let enabled = MicroCombatTrainingConfig {
            enabled: true,
            rollout_enabled: true,
            suite: Some(
                crate::micro_combat::MicroCombatSuiteConfig::from_toml_str(include_str!(
                    "../config/micro_combat_scenarios.toml"
                ))
                .unwrap(),
            ),
            ..MicroCombatTrainingConfig::default()
        };
        assert!(!micro_combat_gate_passed(&enabled, None));
    }

    #[test]
    fn asymmetric_micro_rollout_restores_its_inflight_scenario_and_future_resets() {
        let suite = crate::micro_combat::MicroCombatSuiteConfig::from_toml_str(include_str!(
            "../config/micro_combat_scenarios.toml"
        ))
        .unwrap();
        let mut config = TrainingConfig::default();
        config.combat_curriculum.micro_combat = crate::micro_combat::MicroCombatTrainingConfig {
            enabled: true,
            rollout_enabled: true,
            attack_action_kind_exploration_floor: 0.0,
            suite: Some(suite),
            min_attack_commitments_per_episode: 0.0,
            min_survival_objective_success_rate: 0.75,
            min_elimination_objective_success_rate: 0.25,
        };
        let scenario_index = config
            .combat_curriculum
            .micro_combat
            .suite
            .as_ref()
            .unwrap()
            .scenarios
            .iter()
            .position(|scenario| scenario.opponent_cells == 3)
            .unwrap();
        let scenario = config
            .combat_curriculum
            .micro_combat
            .scenario(scenario_index)
            .unwrap();
        let assignment = RolloutOpponentAssignment::Baseline {
            profile: scenario.opponent,
        };
        let curriculum = RolloutCurriculumAssignment {
            stage: FeedingCurriculumStage::Skirmish,
            simulation_time_quanta: 100_000,
            micro_combat_scenario: Some(scenario_index),
        };
        let mut environment = new_rollout_env::<NdArray<f32>>(
            &assignment,
            &[],
            &[],
            &config,
            curriculum,
            818,
            &Default::default(),
        )
        .unwrap();
        let initial = environment.initial_ecology_snapshot().unwrap();
        assert_eq!(initial.starts_by_team[0].len(), 1);
        assert_eq!(initial.starts_by_team[1].len(), 3);

        let checkpoint = environment.checkpoint().unwrap();
        let mut restored = restore_rollout_env::<NdArray<f32>>(
            &assignment,
            &[],
            &[],
            &config,
            curriculum,
            checkpoint,
            &Default::default(),
        )
        .unwrap();
        assert_eq!(
            restored.initial_ecology_snapshot().unwrap().starts_by_team,
            initial.starts_by_team
        );
        restored.reset(819);
        environment.reset(819);
        assert_eq!(
            restored.initial_ecology_snapshot().unwrap().starts_by_team,
            environment
                .initial_ecology_snapshot()
                .unwrap()
                .starts_by_team
        );
    }

    #[test]
    fn micro_combat_curriculum_publishes_gate_evidence_and_rotation_state() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut suite = crate::micro_combat::MicroCombatSuiteConfig::from_toml_str(include_str!(
            "../config/micro_combat_scenarios.toml"
        ))
        .unwrap();
        for scenario in &mut suite.scenarios {
            scenario.sim_time_limit_quanta = 256;
        }
        let mut config = TrainingConfig {
            seed: 821,
            num_envs: 1,
            rollout_length: 32,
            total_timesteps: 2_000,
            total_simulation_quanta_per_env: Some(4_096),
            eval_interval: 0,
            eval_episodes: 1,
            evaluation_opponents: vec![OpponentProfile::Wait],
            checkpoint_interval: 0,
            checkpoint_dir: temporary.path().to_string_lossy().into_owned(),
            ..TrainingConfig::default()
        };
        config.model.hidden1 = 8;
        config.model.hidden2 = 8;
        config.model.recurrent_size = 8;
        config.ppo.epochs_per_update = 1;
        config.ppo.minibatch_size = 32;
        config.feeding_curriculum.enabled = true;
        config.combat_curriculum.enabled = true;
        config.combat_curriculum.cycle_sim_time_quanta_per_env = 4_096;
        config.combat_curriculum.on_food_sim_time_quanta_per_cycle = 512;
        config
            .combat_curriculum
            .adjacent_food_sim_time_quanta_per_cycle = 512;
        config.combat_curriculum.contact_sim_time_quanta_per_cycle = 1_024;
        config.combat_curriculum.skirmish_sim_time_quanta_per_cycle = 1_024;
        config
            .combat_curriculum
            .retention_episode_sim_time_limit_quanta = 256;
        config
            .combat_curriculum
            .contact_episode_sim_time_limit_quanta = 256;
        config
            .combat_curriculum
            .skirmish_episode_sim_time_limit_quanta = 256;
        config
            .combat_curriculum
            .contact_evaluation_sim_time_limit_quanta = 256;
        config
            .combat_curriculum
            .skirmish_evaluation_sim_time_limit_quanta = 256;
        config.combat_curriculum.contact_initial_energies = vec![100];
        config.combat_curriculum.contact_opponents = vec![OpponentProfile::Aggressive];
        config.combat_curriculum.micro_combat = MicroCombatTrainingConfig {
            enabled: true,
            rollout_enabled: true,
            attack_action_kind_exploration_floor: 0.0,
            suite: Some(suite),
            min_attack_commitments_per_episode: 0.0,
            min_survival_objective_success_rate: 1.0,
            min_elimination_objective_success_rate: 1.0,
        };
        config.env.world_size = 7;
        config.env.cells_per_team = 1;
        config.env.max_episode_len = 1_024;
        config.env.victory.sim_time_limit_quanta = 256;
        config.env.num_scattered_energy = 2;
        config.env.num_plants = 1;
        config.self_play.max_opponent_pool = 0;
        config.validate().unwrap();

        train::<TestBackend>(config, Default::default(), None, None, None);

        let metrics = std::fs::read_to_string(temporary.path().join("metrics.csv")).unwrap();
        let update = metrics
            .lines()
            .last()
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let checkpoint = temporary.path().join(format!("checkpoint-{update:08}"));
        let metadata = crate::artifact::verify_checkpoint_metadata(&checkpoint).unwrap();
        let micro = metadata
            .micro_combat_evaluation
            .expect("terminal checkpoint must bind micro-combat evidence");
        assert_eq!(micro.scenarios.len(), 6);
        assert!(micro.gate_summary().survival_episodes > 0);
        assert!(micro.gate_summary().elimination_episodes > 0);
        let resume: TrainingResumeState =
            rmp_serde::from_slice(&std::fs::read(checkpoint.join("resume.mpk")).unwrap()).unwrap();
        assert!(resume.micro_combat_rotation.contact_assignments > 0);
        assert!(resume.micro_combat_rotation.skirmish_assignments > 0);
        assert_eq!(resume.environment_micro_combat_scenarios.len(), 1);
    }

    #[test]
    fn terminal_boundary_is_checkpointed_even_when_not_periodic_best_or_promoted() {
        assert!(checkpoint_publication_due(
            false, false, false, true, false, false
        ));
        assert!(checkpoint_publication_due(
            false, false, false, false, true, false
        ));
        assert!(checkpoint_publication_due(
            false, false, false, false, false, true
        ));
        assert!(!checkpoint_publication_due(
            false, false, false, false, false, false
        ));
    }

    #[test]
    fn self_play_activation_uses_world_time_not_population_actions() {
        let mut config = TrainingConfig::default();
        config.self_play.start_after_sim_time_quanta_per_env = 10_000;
        config.self_play.opponent_update_interval = 5;
        assert!(!self_play_promotion_due(&config, 9_999, 10));
        assert!(!self_play_promotion_due(&config, 10_000, 9));
        assert!(self_play_promotion_due(&config, 10_000, 10));

        config.self_play.max_opponent_pool = 0;
        assert!(!self_play_promotion_due(&config, u64::MAX, 10));
    }

    #[test]
    fn training_stops_on_cumulative_world_time_across_fields_and_parallel_environments() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        for world_size in [4, 8] {
            let artifact_root = temporary.path().join(format!("field-{world_size}"));
            let mut config = TrainingConfig {
                seed: 74,
                num_envs: 2,
                rollout_length: 8,
                total_timesteps: 128,
                total_simulation_quanta_per_env: Some(256),
                eval_interval: 0,
                checkpoint_interval: 0,
                checkpoint_dir: artifact_root.to_string_lossy().into_owned(),
                ..TrainingConfig::default()
            };
            config.model.hidden1 = 8;
            config.model.hidden2 = 8;
            config.ppo.epochs_per_update = 1;
            config.ppo.minibatch_size = 32;
            config.env.world_size = world_size;
            config.env.cells_per_team = 1;
            config.env.max_episode_len = 1_024;
            config.env.victory.sim_time_limit_quanta = 64;
            config.env.num_scattered_energy = 2;
            config.env.num_plants = 1;
            config.self_play.max_opponent_pool = 0;

            train::<TestBackend>(config, Default::default(), None, None, None);

            let metrics = std::fs::read_to_string(artifact_root.join("metrics.csv")).unwrap();
            let headers = metrics
                .lines()
                .next()
                .unwrap()
                .split(',')
                .collect::<Vec<_>>();
            let values = metrics
                .lines()
                .last()
                .unwrap()
                .split(',')
                .collect::<Vec<_>>();
            let row = headers.into_iter().zip(values).collect::<HashMap<_, _>>();
            assert!(row["min_sim_time_quanta"].parse::<u64>().unwrap() >= 256);
            assert!(row["actions"].parse::<u64>().unwrap() <= 128);
            assert!(row["total_sim_time_quanta"].parse::<u128>().unwrap() >= 512);
            assert_eq!(row["discarded_tails"], "0");
            let update = row["update"].parse::<usize>().unwrap();
            let checkpoint = artifact_root.join(format!("checkpoint-{update:08}"));
            let metadata = crate::artifact::verify_checkpoint_metadata(&checkpoint).unwrap();
            assert!(metadata
                .config
                .total_simulation_quanta_per_env
                .is_some_and(|target| {
                    metadata.actions == row["actions"].parse::<u64>().unwrap() && target == 256
                }));
        }
    }

    #[test]
    fn completion_between_intervals_publishes_terminal_evaluation() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut config = TrainingConfig {
            seed: 72,
            num_envs: 1,
            rollout_length: 4,
            total_timesteps: 4,
            eval_interval: 100,
            eval_episodes: 1,
            evaluation_opponents: vec![crate::config::OpponentProfile::Wait],
            checkpoint_interval: 0,
            checkpoint_dir: temporary.path().to_string_lossy().into_owned(),
            ..TrainingConfig::default()
        };
        config.model.hidden1 = 8;
        config.model.hidden2 = 8;
        config.ppo.epochs_per_update = 1;
        config.ppo.minibatch_size = 32;
        config.env.world_size = 4;
        config.env.cells_per_team = 1;
        config.env.max_episode_len = 64;
        config.env.victory.sim_time_limit_quanta = 1_024;
        config.env.num_scattered_energy = 2;
        config.env.num_plants = 1;
        config.self_play.max_opponent_pool = 0;

        train::<TestBackend>(config.clone(), Default::default(), None, None, None);

        let evaluations = std::fs::read_to_string(temporary.path().join("evaluation.csv")).unwrap();
        let suites = evaluations
            .lines()
            .filter(|line| line.contains(",suite,"))
            .collect::<Vec<_>>();
        assert_eq!(suites.len(), 1);
        let training = std::fs::read_to_string(temporary.path().join("metrics.csv")).unwrap();
        let final_actions = training.lines().last().unwrap().split(',').nth(1).unwrap();
        assert_eq!(suites[0].split(',').nth(1).unwrap(), final_actions);
        let checkpoint = temporary.path().join("checkpoint-00000001");
        let metadata = crate::artifact::verify_checkpoint_metadata(&checkpoint).unwrap();
        assert!(metadata.evaluation.is_some());
        assert!(temporary.path().join("best.json").is_file());
        let telemetry: crate::telemetry::TrainingTelemetrySummary = serde_json::from_slice(
            &std::fs::read(temporary.path().join("telemetry/summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            telemetry.schema_version,
            crate::telemetry::TELEMETRY_SCHEMA_VERSION
        );
        assert!(telemetry.state_samples >= 2);
        let actions = &telemetry.training.actions;
        let committed = actions.wait.committed
            + actions.movement.committed
            + actions.attack.committed
            + actions.guard.committed
            + actions.consume.committed
            + actions.split.committed
            + actions.regurgitate.committed
            + actions.excavate.committed
            + actions.deposit_terrain.committed;
        assert!(committed > 0);
    }

    #[test]
    fn feeding_curriculum_publishes_gate_metrics_and_blocks_best_label() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut config = TrainingConfig {
            seed: 75,
            num_envs: 1,
            rollout_length: 4,
            total_timesteps: 4,
            eval_interval: 1,
            eval_episodes: 1,
            evaluation_opponents: vec![crate::config::OpponentProfile::Wait],
            checkpoint_interval: 1,
            checkpoint_dir: temporary.path().to_string_lossy().into_owned(),
            ..TrainingConfig::default()
        };
        config.model.hidden1 = 8;
        config.model.hidden2 = 8;
        config.ppo.epochs_per_update = 1;
        config.ppo.minibatch_size = 32;
        config.env.world_size = 4;
        config.env.cells_per_team = 1;
        config.env.max_episode_len = 16;
        config.env.victory.sim_time_limit_quanta = 1_024;
        config.env.num_scattered_energy = 0;
        config.env.num_plants = 0;
        config.self_play.max_opponent_pool = 0;
        config.feeding_curriculum.enabled = true;
        config
            .feeding_curriculum
            .on_food_until_sim_time_quanta_per_env = 10_000;
        config
            .feeding_curriculum
            .adjacent_food_until_sim_time_quanta_per_env = 20_000;
        config
            .feeding_curriculum
            .promotion
            .evaluation_max_episode_len = 8;
        config
            .feeding_curriculum
            .promotion
            .evaluation_sim_time_limit_quanta = 1_024;
        config
            .feeding_curriculum
            .promotion
            .min_consumed_energy_per_initial_cell = 1_000_000.0;

        train::<TestBackend>(config.clone(), Default::default(), None, None, None);

        let feeding =
            std::fs::read_to_string(temporary.path().join("feeding-evaluation.csv")).unwrap();
        let rows = feeding.lines().skip(1).collect::<Vec<_>>();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|row| row.contains(",false,")));
        assert!(rows.iter().any(|row| row.contains(",on_food,")));
        assert!(rows.iter().any(|row| row.contains(",adjacent_food,")));
        assert!(!temporary.path().join("best.json").exists());
        let checkpoint = temporary.path().join("checkpoint-00000001");
        let checkpoint_metadata = crate::artifact::verify_checkpoint_metadata(&checkpoint).unwrap();
        assert_eq!(
            checkpoint_metadata
                .feeding_evaluation
                .as_ref()
                .map(|report| report.passed),
            Some(false)
        );

        let metrics = std::fs::read_to_string(temporary.path().join("metrics.csv")).unwrap();
        assert_eq!(
            metrics.lines().next().unwrap().split(',').next_back(),
            Some("curriculum_stage")
        );
        assert_eq!(
            metrics.lines().last().unwrap().split(',').next_back(),
            Some("on_food")
        );

        let resumed_root = temporary.path().join("resumed");
        config.total_timesteps = 8;
        config.checkpoint_dir = resumed_root.to_string_lossy().into_owned();
        train::<TestBackend>(config, Default::default(), None, Some(&checkpoint), None);
        let resumed =
            crate::artifact::verify_checkpoint_metadata(&resumed_root.join("checkpoint-00000002"))
                .unwrap();
        assert_eq!(resumed.actions, 8);
        assert!(resumed.feeding_evaluation.is_some());
    }

    #[test]
    fn combat_curriculum_publishes_contact_metrics_and_checkpoint_evidence() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let mut config = TrainingConfig {
            seed: 76,
            num_envs: 1,
            rollout_length: 4,
            total_timesteps: 8,
            eval_interval: 0,
            eval_episodes: 1,
            evaluation_opponents: vec![crate::config::OpponentProfile::Wait],
            checkpoint_interval: 0,
            checkpoint_dir: temporary.path().to_string_lossy().into_owned(),
            ..TrainingConfig::default()
        };
        config.model.hidden1 = 8;
        config.model.hidden2 = 8;
        config.ppo.epochs_per_update = 1;
        config.ppo.minibatch_size = 32;
        config.env.world_size = 4;
        config.env.cells_per_team = 1;
        config.env.max_episode_len = 16;
        config.env.victory.sim_time_limit_quanta = 1_024;
        config.env.num_scattered_energy = 0;
        config.env.num_plants = 0;
        config.self_play.max_opponent_pool = 0;
        config.feeding_curriculum.enabled = true;
        config.combat_curriculum.enabled = true;
        config
            .combat_curriculum
            .retention_episode_sim_time_limit_quanta = 1_024;
        config
            .combat_curriculum
            .contact_episode_sim_time_limit_quanta = 1_024;
        config
            .combat_curriculum
            .skirmish_episode_sim_time_limit_quanta = 1_024;
        config
            .combat_curriculum
            .competency_evaluation_frontiers_sim_time_quanta_per_cycle = vec![1];

        let evaluation_seed = config.evaluation_seed;
        train::<TestBackend>(config, Default::default(), None, None, None);

        let csv = std::fs::read_to_string(temporary.path().join("contact-evaluation.csv"))
            .expect("contact metrics should be published");
        assert_eq!(
            csv.lines().skip(1).count(),
            2 * 2 * 3 * 2,
            "both world-time and terminal evaluations need every combat variant"
        );
        let checkpoint = temporary.path().join("checkpoint-00000001");
        let metadata = crate::artifact::verify_checkpoint_metadata(&checkpoint).unwrap();
        let report = metadata
            .contact_evaluation
            .as_ref()
            .expect("evaluated combat checkpoint must embed contact evidence");
        assert_eq!(report.seeds, vec![evaluation_seed]);
        assert_eq!(report.variants.len(), 12);
        assert_eq!(
            report.episodes,
            report
                .variants
                .iter()
                .map(|variant| variant.episodes)
                .sum::<usize>()
        );
        assert!(metadata.minimum_sim_time_quanta_per_env >= 1);
        let timeline = std::fs::read_to_string(temporary.path().join("competency-timeline.csv"))
            .expect("world-time competency timeline should be published");
        let first_measurement = timeline
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .collect::<Vec<_>>();
        assert_eq!(first_measurement[5], "1");
        assert!(first_measurement[6].contains("world_time_frontier"));
        assert!(!temporary.path().join("best.json").exists());
        let frontier = crate::competency_frontier::load_competency_frontier(
            &temporary.path().join("competency-frontier.json"),
        )
        .unwrap();
        assert!(!frontier.entries.is_empty());
        assert!(frontier
            .entries
            .iter()
            .all(|entry| !entry.metrics.joint_qualified()));
        let mut tampered = frontier;
        tampered.entries[0].metrics.skirmish_damage += 1;
        std::fs::write(
            temporary.path().join("competency-frontier.json"),
            serde_json::to_vec_pretty(&tampered).unwrap(),
        )
        .unwrap();
        assert!(crate::competency_frontier::load_competency_frontier(
            &temporary.path().join("competency-frontier.json"),
        )
        .unwrap_err()
        .contains("does not match its immutable checkpoint"));
    }

    #[test]
    fn update_boundary_resume_matches_uninterrupted_training() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let continuous_root = temporary.path().join("continuous");
        let resumed_root = temporary.path().join("resumed");
        let mut config = TrainingConfig {
            seed: 73,
            num_envs: 1,
            rollout_length: 8,
            // Keep the target one action beyond the first update's current
            // nine-action frontier so this regression reaches two boundaries
            // without depending on the old flat head's sampled population.
            total_timesteps: 10,
            eval_interval: 1,
            eval_episodes: 1,
            evaluation_opponents: vec![crate::config::OpponentProfile::Wait],
            checkpoint_interval: 1,
            checkpoint_dir: continuous_root.to_string_lossy().into_owned(),
            ..TrainingConfig::default()
        };
        config.model.hidden1 = 8;
        config.model.hidden2 = 8;
        config.ppo.epochs_per_update = 1;
        config.ppo.minibatch_size = 32;
        config.env.world_size = 4;
        config.env.cells_per_team = 1;
        // The safety guard is not the scientific horizon. Keep it comfortably
        // above policy-dependent exploratory trajectories so this regression
        // tests resume identity rather than a particular initialization's
        // early action preferences.
        config.env.max_episode_len = 64;
        config.env.num_scattered_energy = 4;
        config.env.num_plants = 2;
        config.self_play.start_after_sim_time_quanta_per_env = 0;
        config.self_play.opponent_update_interval = 1;
        config.self_play.max_opponent_pool = 2;
        config.self_play.baseline_probability = 0.0;
        config.self_play.min_pool_win_rate = 0.0;
        config.self_play.max_fixed_worst_case_regression = 1.0;

        let device = Default::default();
        train::<TestBackend>(config.clone(), device, None, None, None);

        let first = continuous_root.join("checkpoint-00000001");
        let second = continuous_root.join("checkpoint-00000002");
        assert!(first.is_dir(), "test setup did not reach update one");
        assert!(second.is_dir(), "test setup did not reach update two");
        assert!(continuous_root
            .join("opponent-pool/manifest-00000001.json")
            .is_file());
        assert!(continuous_root
            .join("opponent-pool/manifest-00000002.json")
            .is_file());
        let second_metadata: crate::artifact::CheckpointMetadata =
            serde_json::from_slice(&std::fs::read(second.join("metadata.json")).unwrap()).unwrap();
        let evaluation_csv = std::fs::read_to_string(continuous_root.join("evaluation.csv"))
            .expect("evaluation metrics should be published");
        assert!(evaluation_csv.lines().any(|line| line.contains(",wait,")));
        assert!(evaluation_csv.lines().any(|line| line.contains(",suite,")));
        assert!(second_metadata.rollout_pool_promotion.is_some());

        config.total_timesteps = second_metadata.actions;
        config.checkpoint_dir = resumed_root.to_string_lossy().into_owned();
        let device = Default::default();
        train::<TestBackend>(config, device, None, Some(&first), None);

        let resumed_second = resumed_root.join("checkpoint-00000002");
        assert_eq!(
            std::fs::read(second.join("model.mpk")).unwrap(),
            std::fs::read(resumed_second.join("model.mpk")).unwrap(),
            "resumed model diverged at the next update"
        );
        assert_eq!(
            std::fs::read(second.join("resume.mpk")).unwrap(),
            std::fs::read(resumed_second.join("resume.mpk")).unwrap(),
            "resumed environment or RNG state diverged at the next update"
        );
    }
}
