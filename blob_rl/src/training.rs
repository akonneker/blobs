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
    PolicyChoice, NUM_ACTIONS, NUM_AMOUNT_CHOICES, NUM_SIGNAL_CHOICES, NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::artifact::{
    load_checkpoint, load_policy_snapshot, load_rollout_snapshot, publish_best_pointer,
    publish_checkpoint, publish_rollout_pool_manifest, LeaguePromotion, PolicySnapshot,
    RolloutLeagueMember, RolloutOpponentAssignment, TrainingResumeState,
    TRAINING_ARTIFACT_SCHEMA_VERSION,
};
use crate::behavior_cloning::verify_behavior_clone_artifact;
use crate::config::{OpponentProfile, SelfPlayConfig, TrainingConfig};
use crate::env::{BlobEnv, EpisodeOutcome, PolicyObservation};
use crate::evaluation::{evaluate_policy_suite, EvaluationMetrics, EvaluationOpponent};
use crate::model::{
    decode_policy_memory, encode_policy_memory, PolicyValueNet, PolicyValueNetConfig,
};
use crate::observation::OBS_DIM;
use crate::ppo::{ppo_update, RolloutBuffer, Transition};
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
    seed: u64,
    device: &B::Device,
) -> Result<BlobEnv, String>
where
    f32: From<B::FloatElem>,
{
    let mut env = match assignment {
        RolloutOpponentAssignment::Baseline { profile } => {
            let mut env_config = config.env.clone();
            env_config.opponent = *profile;
            BlobEnv::new(env_config, config.reward.clone(), seed)
        }
        RolloutOpponentAssignment::Snapshot { model_sha256 } => {
            let snapshot = pool
                .iter()
                .chain(retired)
                .find(|snapshot| snapshot.model_sha256 == *model_sha256)
                .ok_or_else(|| "rollout opponent is absent from the snapshot pool".to_string())?;
            BlobEnv::new_with_snapshot(
                config.env.clone(),
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
    checkpoint: crate::env::BlobEnvCheckpoint,
    device: &B::Device,
) -> Result<BlobEnv, String>
where
    f32: From<B::FloatElem>,
{
    let mut env = match assignment {
        RolloutOpponentAssignment::Baseline { profile } => {
            let mut env_config = config.env.clone();
            env_config.opponent = *profile;
            BlobEnv::from_checkpoint(env_config, config.reward.clone(), checkpoint)
        }
        RolloutOpponentAssignment::Snapshot { model_sha256 } => {
            let snapshot = pool
                .iter()
                .chain(retired)
                .find(|snapshot| snapshot.model_sha256 == *model_sha256)
                .ok_or_else(|| "restored rollout opponent is absent from the pool".to_string())?;
            BlobEnv::from_checkpoint_with_snapshot(
                config.env.clone(),
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

fn evaluation_is_better(candidate: &EvaluationMetrics, incumbent: &EvaluationMetrics) -> bool {
    candidate.worst_case_win_rate > incumbent.worst_case_win_rate
        || (candidate.worst_case_win_rate == incumbent.worst_case_win_rate
            && candidate.win_rate > incumbent.win_rate)
        || (candidate.worst_case_win_rate == incumbent.worst_case_win_rate
            && candidate.win_rate == incumbent.win_rate
            && candidate.average_reward > incumbent.average_reward)
        || (candidate.worst_case_win_rate == incumbent.worst_case_win_rate
            && candidate.win_rate == incumbent.win_rate
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
        load_model_path.is_none() || resume_checkpoint.is_none(),
        "load_model_path and resume_checkpoint are mutually exclusive"
    );
    assert!(
        load_model_path.is_none() || config.initial_policy.is_none(),
        "load_model_path and config.initial_policy are mutually exclusive"
    );
    let configured_initial_model = if resume_checkpoint.is_none() {
        config.initial_policy.as_ref().map(|initial| {
            verify_behavior_clone_artifact(
                Path::new(&initial.directory),
                &initial.artifact_sha256,
                &config.model,
            )
            .unwrap_or_else(|error| panic!("invalid configured initial policy: {error}"))
        })
    } else {
        None
    };
    let load_model_path = load_model_path
        .map(PathBuf::from)
        .or(configured_initial_model);
    let evaluation_snapshots = if config.eval_interval > 0 || config.self_play.max_opponent_pool > 0
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
        mut rollout_pool,
        mut rollout_league,
        mut retired_rollout_snapshots,
        mut environment_opponents,
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
        let mut rollout_league = state.rollout_pool.clone();
        let mut rollout_pool = rollout_league
            .iter()
            .map(|member| load_rollout_snapshot::<B::InnerBackend>(&member.snapshot, &device))
            .collect::<Result<Vec<_>, _>>()
            .expect("failed to restore rollout-opponent pool");
        let environment_opponents = state.environment_opponents.clone();
        let mut retired_rollout_snapshots = state
            .active_retired_snapshots
            .iter()
            .map(|descriptor| load_rollout_snapshot::<B::InnerBackend>(descriptor, &device))
            .collect::<Result<Vec<_>, _>>()
            .expect("failed to restore active retired rollout opponents");
        let envs = state
            .environments
            .into_iter()
            .zip(&environment_opponents)
            .map(|(environment, opponent)| {
                restore_rollout_env(
                    opponent,
                    &rollout_pool,
                    &retired_rollout_snapshots,
                    &config,
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
            rollout_pool,
            rollout_league,
            retired_rollout_snapshots,
            environment_opponents,
            state.telemetry,
        )
    } else {
        let envs = (0..config.num_envs)
            .map(|i| {
                BlobEnv::new(
                    config.env.clone(),
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
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![
                RolloutOpponentAssignment::Baseline {
                    profile: config.env.opponent,
                };
                config.num_envs
            ],
            None,
        )
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
    let mut env_observations: Vec<Vec<PolicyObservation>> =
        envs.iter().map(BlobEnv::get_policy_observations).collect();

    let start_time = Instant::now();
    let mut episode_stats = EpisodeStats::new();

    // Create metrics log file
    std::fs::create_dir_all(&config.checkpoint_dir).expect("failed to create checkpoint directory");
    let artifact_root = Path::new(&config.checkpoint_dir);
    let metrics_path = artifact_root.join("metrics.csv");
    let mut metrics_file = open_metrics_file(
        &metrics_path,
        resume_checkpoint.is_some(),
        "update,actions,policy_loss,value_loss,entropy,approx_kl,clip_fraction,explained_variance,ppo_optimizer_steps,ppo_epochs_completed,kl_early_stop,ppo_recurrent_unroll_steps,ppo_recurrent_chunks,episodes,wins,losses,timeouts,win_rate,avg_ep_len,avg_reward,training_cells_alive,completed_transitions,discarded_tails,mean_elapsed_time,actions_per_second,min_sim_time_quanta,max_sim_time_quanta,total_sim_time_quanta,simulation_quanta_per_second",
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
    let evaluation_seeds = (0..config.eval_episodes)
        .map(|index| config.evaluation_seed.wrapping_add(index as u64))
        .collect::<Vec<_>>();
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
        // continuation.
        for _step in 0..config.rollout_length {
            let mut inference_observations = Vec::new();
            let mut inference_memory = Vec::new();
            let mut inference_mask_bias = Vec::new();
            let mut environment_rows = vec![None; config.num_envs];
            let mut inference_rows = 0usize;

            for (env_idx, obs_list) in env_observations.iter().enumerate() {
                if config
                    .total_simulation_quanta_per_env
                    .is_some_and(|target| cumulative_sim_time_quanta[env_idx] >= target)
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
                inference_mask_bias.extend(obs_list.iter().flat_map(|input| {
                    input
                        .observation
                        .action_mask
                        .iter()
                        .map(|allowed| if *allowed { 0.0 } else { -1.0e9 })
                }));
            }

            if inference_rows == 0 {
                continue;
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
            let output = model
                .valid()
                .forward_with_memory(obs_tensor.inner(), memory_tensor);
            let mask_bias = Tensor::<B::InnerBackend, 2>::from_data(
                TensorData::new(inference_mask_bias, [inference_rows, NUM_ACTIONS]),
                &device,
            );
            let log_probs =
                burn::tensor::activation::log_softmax(output.policy_logits + mask_bias, 1);
            let probabilities = log_probs.clone().exp();
            // Signal masking depends on the sampled physical action, so signal
            // logits are transferred once and masked row-wise on the host.
            let inference_width = NUM_ACTIONS * 2
                + NUM_AMOUNT_CHOICES
                + NUM_SIGNAL_CHOICES
                + NUM_SIGNAL_STRENGTH_CHOICES
                + 1
                + config.model.recurrent_size;
            let inference_data: Vec<f32> = Tensor::cat(
                vec![
                    probabilities,
                    log_probs,
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

                let mut actions: Vec<(CellId, PolicyChoice, Option<Vec<u8>>)> =
                    Vec::with_capacity(n_cells);
                let decision_time = env.sim_time_quanta();

                for (cell_idx, input) in obs_list.iter().enumerate() {
                    let cell_id = input.cell_id;
                    let obs = &input.observation;
                    let inference_row = rows.start + cell_idx;
                    let row_start = inference_row * inference_width;
                    let row = &inference_data[row_start..row_start + inference_width];
                    let cell_probs = &row[..NUM_ACTIONS];
                    let cell_log_probs = &row[NUM_ACTIONS..NUM_ACTIONS * 2];

                    // Sample action
                    let action = sample_action(cell_probs, &mut action_rng);
                    let amount_mask = obs.amount_mask(action);
                    let amount_start = NUM_ACTIONS * 2;
                    let (amount_probs, amount_log_probs) = masked_distribution(
                        &row[amount_start..amount_start + NUM_AMOUNT_CHOICES],
                        &amount_mask,
                    );
                    let amount = sample_action(&amount_probs, &mut action_rng);
                    let signal_mask = obs.signal_mask(action, amount);
                    let signal_start = amount_start + NUM_AMOUNT_CHOICES;
                    let (signal_probs, signal_log_probs) = masked_distribution(
                        &row[signal_start..signal_start + NUM_SIGNAL_CHOICES],
                        &signal_mask,
                    );
                    let signal = sample_action(&signal_probs, &mut action_rng);
                    let signal_strength_mask = obs.signal_strength_mask(action, amount, signal);
                    let signal_strength_start = signal_start + NUM_SIGNAL_CHOICES;
                    let (signal_strength_probs, signal_strength_log_probs) = masked_distribution(
                        &row[signal_strength_start
                            ..signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES],
                        &signal_strength_mask,
                    );
                    let signal_strength = sample_action(&signal_strength_probs, &mut action_rng);
                    let log_prob = cell_log_probs[action]
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

                    let memory_start = NUM_ACTIONS * 2
                        + NUM_AMOUNT_CHOICES
                        + NUM_SIGNAL_CHOICES
                        + NUM_SIGNAL_STRENGTH_CHOICES
                        + 1;
                    actions.push((
                        cell_id,
                        PolicyChoice {
                            action,
                            amount,
                            signal,
                            signal_strength,
                        },
                        Some(encode_policy_memory(
                            &row[memory_start..memory_start + config.model.recurrent_size],
                        )),
                    ));

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
                        reward: 0.0,
                        value,
                        next_value: 0.0,
                        log_prob,
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
                    let assignment = choose_rollout_opponent(
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
                        reset_seed,
                        &device,
                    )
                    .expect("failed to rotate rollout opponent");
                    environment_opponents[env_idx] = assignment;
                    prune_retired_snapshots(&mut retired_rollout_snapshots, &environment_opponents);
                    if let Some(state) = training_telemetry.as_mut() {
                        state.start_episode(
                            env_idx,
                            env_episode_ids[env_idx],
                            reset_seed,
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

            let (updated_model, ppo_metrics) = ppo_update(
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
                "{},{},{:.6},{:.6},{:.4},{:.6},{:.4},{:.4},{},{},{},{},{},{},{},{},{},{:.4},{:.1},{:.4},{},{},{},{:.4},{:.1},{},{},{},{:.1}",
                update_count,
                total_timesteps,
                ppo_metrics.policy_loss,
                ppo_metrics.value_loss,
                ppo_metrics.entropy,
                ppo_metrics.approx_kl,
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
            )
            .unwrap();
            metrics_file
                .flush()
                .expect("failed to flush training metrics");

            let promotion_due = config.self_play.max_opponent_pool > 0
                && total_timesteps >= config.self_play.start_after_timesteps
                && update_count.is_multiple_of(config.self_play.opponent_update_interval);
            let regular_evaluation_due =
                config.eval_interval > 0 && update_count.is_multiple_of(config.eval_interval);
            let evaluation = if regular_evaluation_due || promotion_due {
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
                Some(evaluation)
            } else {
                None
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
                best_evaluation
                    .as_ref()
                    .is_none_or(|incumbent| evaluation_is_better(candidate, incumbent))
            });
            let periodic_checkpoint = config.checkpoint_interval > 0
                && update_count.is_multiple_of(config.checkpoint_interval);
            if is_new_best {
                best_evaluation = evaluation.clone();
            }

            // A published state always describes the clean boundary between
            // updates. Per-update metrics have already been emitted and are
            // intentionally not carried into the next update.
            episode_stats.reset();
            if periodic_checkpoint || is_new_best || promote_to_rollout_pool {
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
                    rollout_pool: rollout_league.clone(),
                    active_retired_snapshots: retired_rollout_snapshots
                        .iter()
                        .map(PolicySnapshot::descriptor)
                        .collect(),
                    environment_opponents: environment_opponents.clone(),
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
                    accepted_promotion.as_ref(),
                    &resume_state,
                )
                .expect("failed to publish immutable training artifact");
                println!("  Checkpoint published: {}", checkpoint.display());
                if is_new_best {
                    let evaluation = evaluation
                        .as_ref()
                        .expect("a best checkpoint always has evaluation metrics");
                    publish_best_pointer(
                        artifact_root,
                        &checkpoint,
                        update_count,
                        total_timesteps,
                        evaluation,
                    )
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

    if config.eval_interval > 0 && last_fixed_evaluation_actions != Some(total_timesteps) {
        let evaluation = evaluate_policy_suite(
            &model.valid(),
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

fn masked_distribution<const N: usize>(logits: &[f32], mask: &[bool; N]) -> (Vec<f32>, Vec<f32>) {
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
        assert!(evaluation_is_better(&candidate, &baseline));
        candidate.average_reward = -2.0;
        candidate.average_episode_len = 9.0;
        assert!(evaluation_is_better(&candidate, &baseline));
        candidate.win_rate = 1.0;
        candidate.average_reward = -100.0;
        assert!(evaluation_is_better(&candidate, &baseline));

        let mut robust = baseline.clone();
        robust.worst_case_win_rate = 0.5;
        robust.win_rate = 0.1;
        let mut brittle = baseline;
        brittle.worst_case_win_rate = 0.25;
        brittle.win_rate = 1.0;
        brittle.average_reward = 1_000.0;
        assert!(evaluation_is_better(&robust, &brittle));
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

        train::<TestBackend>(config, Default::default(), None, None, None);

        let evaluations = std::fs::read_to_string(temporary.path().join("evaluation.csv")).unwrap();
        let suites = evaluations
            .lines()
            .filter(|line| line.contains(",suite,"))
            .collect::<Vec<_>>();
        assert_eq!(suites.len(), 1);
        let training = std::fs::read_to_string(temporary.path().join("metrics.csv")).unwrap();
        let final_actions = training.lines().last().unwrap().split(',').nth(1).unwrap();
        assert_eq!(suites[0].split(',').nth(1).unwrap(), final_actions);
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
    fn update_boundary_resume_matches_uninterrupted_training() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let temporary = tempfile::tempdir().unwrap();
        let continuous_root = temporary.path().join("continuous");
        let resumed_root = temporary.path().join("resumed");
        let mut config = TrainingConfig {
            seed: 73,
            num_envs: 1,
            rollout_length: 8,
            // This fixture produces eight actions in its first update, so 9
            // reaches exactly the two boundaries needed by this regression.
            total_timesteps: 9,
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
        config.self_play.start_after_timesteps = 0;
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
