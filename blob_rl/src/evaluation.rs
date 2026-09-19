//! Deterministic, gradient-free evaluation on fixed held-out environment seeds.

use burn::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::action::{
    PolicyChoice, NUM_AMOUNT_CHOICES, NUM_POLICY_ACTION_KINDS, NUM_POLICY_AMOUNT_LOGITS,
    NUM_POLICY_EFFORT_LOGITS, NUM_POLICY_TARGET_LOGITS, NUM_SIGNAL_CHOICES,
    NUM_SIGNAL_STRENGTH_CHOICES,
};
use crate::artifact::PolicySnapshot;
use crate::config::{EnvConfig, OpponentProfile, RewardConfig};
use crate::env::{BlobEnv, EpisodeEndReason, EpisodeOutcome};
use crate::match_explorer::{MatchExplorerConfig, RecordedExplorerMatch};
use crate::model::{decode_policy_memory, encode_policy_memory, PolicyValueNet};
#[cfg(test)]
use crate::observation::Observation;
use crate::observation::OBS_DIM;

/// Stable identity of one held-out opponent. Snapshot hashes make similarly
/// named checkpoints unambiguous in logs and published artifact metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvaluationOpponent {
    Baseline {
        profile: OpponentProfile,
    },
    Snapshot {
        checkpoint: String,
        update: usize,
        model_sha256: String,
    },
}

impl fmt::Display for EvaluationOpponent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Baseline { profile } => profile.fmt(formatter),
            Self::Snapshot {
                checkpoint,
                update,
                model_sha256,
            } => write!(
                formatter,
                "snapshot:{checkpoint}:u{update}:{}",
                &model_sha256[..model_sha256.len().min(12)]
            ),
        }
    }
}

/// Results against one fixed opponent on a shared held-out seed suite.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpponentEvaluationMetrics {
    pub opponent: EvaluationOpponent,
    pub episodes: usize,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub win_rate: f64,
    pub average_episode_len: f64,
    pub average_reward: f64,
    pub actions: u64,
}

/// Aggregate and per-opponent results for one policy evaluation boundary.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvaluationMetrics {
    pub seeds: Vec<u64>,
    pub opponents: Vec<OpponentEvaluationMetrics>,
    pub episodes: usize,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub worst_case_win_rate: f64,
    pub win_rate: f64,
    pub average_episode_len: f64,
    pub average_reward: f64,
    pub actions: u64,
}

pub(crate) use blob_policy::greedy::{
    greedy_amount, greedy_policy_action, greedy_signal, greedy_signal_strength,
};

/// Run one deterministic, batched policy frontier. Keeping this decoder in
/// one place ensures scientific counterfactual evaluators use exactly the
/// same greedy action and recurrent-memory semantics as checkpoint selection.
pub(crate) fn greedy_policy_choices<B: Backend>(
    model: &PolicyValueNet<B>,
    observations: &[crate::env::PolicyObservation],
    device: &B::Device,
) -> Vec<(blob_interface::types::CellId, PolicyChoice, Option<Vec<u8>>)>
where
    f32: From<B::FloatElem>,
{
    greedy_policy_choices_with_kind_logits(model, observations, device)
        .into_iter()
        .map(|(cell_id, choice, memory, _)| (cell_id, choice, memory))
        .collect()
}

/// The deployed greedy decision plus the exact authoritatively-routed action
/// kind logits that produced it. Correction diagnostics use this single
/// forward pass so telemetry cannot observe a different recurrent boundary.
pub(crate) type GreedyPolicyChoiceWithKindLogits = (
    blob_interface::types::CellId,
    PolicyChoice,
    Option<Vec<u8>>,
    [f32; NUM_POLICY_ACTION_KINDS],
);

pub(crate) fn greedy_policy_choices_with_kind_logits<B: Backend>(
    model: &PolicyValueNet<B>,
    observations: &[crate::env::PolicyObservation],
    device: &B::Device,
) -> Vec<GreedyPolicyChoiceWithKindLogits>
where
    f32: From<B::FloatElem>,
{
    let batch_size = observations.len();
    if batch_size == 0 {
        return Vec::new();
    }
    let obs_data = observations
        .iter()
        .flat_map(|input| input.observation.data.iter().copied())
        .collect::<Vec<_>>();
    let recurrent_size = model.recurrent_size();
    let memory = observations
        .iter()
        .flat_map(|input| decode_policy_memory(&input.private_memory, recurrent_size))
        .collect::<Vec<_>>();
    let obs_tensor =
        Tensor::<B, 2>::from_data(TensorData::new(obs_data, [batch_size, OBS_DIM]), device);
    let output = model.forward_with_memory(
        obs_tensor,
        Tensor::<B, 2>::from_data(
            TensorData::new(memory, [batch_size, recurrent_size]),
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
    let output = Tensor::cat(
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
    .expect("evaluation logits should use a supported float type")
    .into_iter()
    .map(f32::from)
    .collect::<Vec<_>>();

    observations
        .iter()
        .enumerate()
        .map(|(cell_index, input)| {
            let start = cell_index * width;
            let target_start = start + NUM_POLICY_ACTION_KINDS;
            let effort_start = target_start + NUM_POLICY_TARGET_LOGITS;
            let amount_start = effort_start + NUM_POLICY_EFFORT_LOGITS;
            let action = greedy_policy_action(
                &output[start..target_start],
                &output[target_start..effort_start],
                &output[effort_start..amount_start],
                &input.observation,
            );
            let kind = crate::action::decompose_policy_action(action)
                .expect("greedy action is in the policy catalog")
                .kind;
            let amount = greedy_amount(
                &output[amount_start + kind * NUM_AMOUNT_CHOICES
                    ..amount_start + (kind + 1) * NUM_AMOUNT_CHOICES],
                &input.observation,
                action,
            );
            let signal_start = amount_start + NUM_POLICY_AMOUNT_LOGITS;
            let signal = greedy_signal(
                &output[signal_start..signal_start + NUM_SIGNAL_CHOICES],
                &input.observation,
                action,
                amount,
            );
            let signal_strength_start = signal_start + NUM_SIGNAL_CHOICES;
            let signal_strength = greedy_signal_strength(
                &output[signal_strength_start..signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES],
                &input.observation,
                action,
                amount,
                signal,
            );
            (
                input.cell_id,
                PolicyChoice {
                    action,
                    amount,
                    signal,
                    signal_strength,
                },
                Some(encode_policy_memory(
                    &output[signal_strength_start + NUM_SIGNAL_STRENGTH_CHOICES..start + width],
                )),
                output[start..target_start]
                    .try_into()
                    .expect("action-kind logit row has the fixed catalog width"),
            )
        })
        .collect()
}

/// Evaluate a policy without autodiff or action sampling. Environment seed
/// `seeds[i]` determines the entire episode and greedy ties resolve to the
/// lowest canonical action index.
pub fn evaluate_policy<B: Backend>(
    model: &PolicyValueNet<B>,
    env_config: &EnvConfig,
    reward_config: &RewardConfig,
    opponent: OpponentProfile,
    seeds: &[u64],
    device: &B::Device,
) -> OpponentEvaluationMetrics
where
    f32: From<B::FloatElem>,
{
    evaluate_policy_with_env(
        model,
        seeds,
        device,
        EvaluationOpponent::Baseline { profile: opponent },
        |seed| {
            let mut profile_env = env_config.clone();
            profile_env.opponent = opponent;
            BlobEnv::new(profile_env, reward_config.clone(), seed)
        },
    )
}

/// Evaluate against an integrity-checked policy snapshot. Each episode owns a
/// fresh environment and every opponent cell owns an isolated model clone.
pub fn evaluate_policy_against_snapshot<B: Backend>(
    model: &PolicyValueNet<B>,
    env_config: &EnvConfig,
    reward_config: &RewardConfig,
    snapshot: &PolicySnapshot<B>,
    seeds: &[u64],
    device: &B::Device,
) -> OpponentEvaluationMetrics
where
    f32: From<B::FloatElem>,
{
    evaluate_policy_with_env(
        model,
        seeds,
        device,
        EvaluationOpponent::Snapshot {
            checkpoint: snapshot.checkpoint.clone(),
            update: snapshot.update,
            model_sha256: snapshot.model_sha256.clone(),
        },
        |seed| {
            BlobEnv::new_with_snapshot(
                env_config.clone(),
                reward_config.clone(),
                seed,
                snapshot.model.clone(),
                device.clone(),
            )
        },
    )
}

/// Record one deterministic greedy-policy evaluation at every canonical
/// resolution batch. This is intentionally separate from bulk evaluation so
/// verified hashing, replay copies, and presentation patches are paid only for
/// the explicitly selected match.
pub fn record_greedy_policy_match<B: Backend>(
    model: &PolicyValueNet<B>,
    env_config: &EnvConfig,
    reward_config: &RewardConfig,
    opponent: OpponentProfile,
    seed: u64,
    device: &B::Device,
    explorer: MatchExplorerConfig,
) -> Result<RecordedExplorerMatch, String>
where
    f32: From<B::FloatElem>,
{
    let mut profile_env = env_config.clone();
    profile_env.opponent = opponent;
    let mut env = BlobEnv::new(profile_env, reward_config.clone(), seed);
    env.enable_match_explorer_recording(explorer)?;
    let mut observations = env.get_policy_observations();
    loop {
        let actions = greedy_policy_choices(model, &observations, device);
        let result = env.step_with_policy_memory(&actions);
        if result.done {
            let outcome = match result
                .outcome
                .ok_or("completed recorded match has no outcome")?
            {
                EpisodeOutcome::Win => "training_team_win",
                EpisodeOutcome::Loss => "opponent_win",
                EpisodeOutcome::Timeout => "timeout",
                EpisodeOutcome::SafetyAbort => "safety_abort",
            };
            let end_reason = match result
                .end_reason
                .ok_or("completed recorded match has no end reason")?
            {
                EpisodeEndReason::Extermination => "extermination",
                EpisodeEndReason::SimTimeDeadline => "sim_time_deadline",
                EpisodeEndReason::DecisionFrontierSafetyLimit => "decision_frontier_safety_limit",
            };
            return env.finish_match_explorer_recording(outcome, end_reason);
        }
        observations = result.policy_observations;
    }
}

fn evaluate_policy_with_env<B: Backend>(
    model: &PolicyValueNet<B>,
    seeds: &[u64],
    device: &B::Device,
    opponent: EvaluationOpponent,
    mut create_env: impl FnMut(u64) -> BlobEnv,
) -> OpponentEvaluationMetrics
where
    f32: From<B::FloatElem>,
{
    let mut wins = 0usize;
    let mut losses = 0usize;
    let mut timeouts = 0usize;
    let mut total_episode_len = 0u64;
    let mut total_reward = 0.0f64;
    let mut total_actions = 0u64;

    for &seed in seeds {
        let mut env = create_env(seed);
        let mut observations = env.get_policy_observations();

        loop {
            let actions = greedy_policy_choices(model, &observations, device);

            total_actions += actions.len() as u64;
            let result = env.step_with_policy_memory(&actions);
            total_reward += result
                .rewards
                .values()
                .map(|reward| *reward as f64)
                .sum::<f64>();

            if result.done {
                total_episode_len += result.episode_step;
                match result
                    .outcome
                    .expect("a completed evaluation episode must have an outcome")
                {
                    EpisodeOutcome::Win => wins += 1,
                    EpisodeOutcome::Loss => losses += 1,
                    EpisodeOutcome::Timeout => timeouts += 1,
                    EpisodeOutcome::SafetyAbort => panic!(
                        "evaluation reached the non-scientific decision-frontier safety limit"
                    ),
                }
                break;
            }
            observations = result.policy_observations;
        }
    }

    let episodes = seeds.len();
    let denominator = episodes.max(1) as f64;
    OpponentEvaluationMetrics {
        opponent,
        episodes,
        wins,
        losses,
        timeouts,
        win_rate: wins as f64 / denominator,
        average_episode_len: total_episode_len as f64 / denominator,
        average_reward: total_reward / denominator,
        actions: total_actions,
    }
}

/// Evaluate every baseline and policy snapshot independently on identical
/// seeds, then aggregate
/// with equal weight per episode. The worst-case win rate is retained for
/// robust checkpoint selection instead of hiding a weak matchup in the mean.
pub fn evaluate_policy_suite<B: Backend>(
    model: &PolicyValueNet<B>,
    env_config: &EnvConfig,
    reward_config: &RewardConfig,
    opponents: &[OpponentProfile],
    snapshots: &[PolicySnapshot<B>],
    seeds: &[u64],
    device: &B::Device,
) -> EvaluationMetrics
where
    f32: From<B::FloatElem>,
{
    let mut results = opponents
        .iter()
        .copied()
        .map(|opponent| evaluate_policy(model, env_config, reward_config, opponent, seeds, device))
        .collect::<Vec<_>>();
    results.extend(snapshots.iter().map(|snapshot| {
        evaluate_policy_against_snapshot(model, env_config, reward_config, snapshot, seeds, device)
    }));
    let episodes = results
        .iter()
        .map(|metrics| metrics.episodes)
        .sum::<usize>();
    let wins = results.iter().map(|metrics| metrics.wins).sum::<usize>();
    let losses = results.iter().map(|metrics| metrics.losses).sum::<usize>();
    let timeouts = results
        .iter()
        .map(|metrics| metrics.timeouts)
        .sum::<usize>();
    let actions = results.iter().map(|metrics| metrics.actions).sum::<u64>();
    let denominator = episodes.max(1) as f64;
    let weighted_episode_len = results
        .iter()
        .map(|metrics| metrics.average_episode_len * metrics.episodes as f64)
        .sum::<f64>();
    let weighted_reward = results
        .iter()
        .map(|metrics| metrics.average_reward * metrics.episodes as f64)
        .sum::<f64>();
    let worst_case_win_rate = results
        .iter()
        .map(|metrics| metrics.win_rate)
        .reduce(f64::min)
        .unwrap_or(0.0);
    EvaluationMetrics {
        seeds: seeds.to_vec(),
        opponents: results,
        episodes,
        wins,
        losses,
        timeouts,
        worst_case_win_rate,
        win_rate: wins as f64 / denominator,
        average_episode_len: weighted_episode_len / denominator,
        average_reward: weighted_reward / denominator,
        actions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::NUM_ACTIONS;
    use burn::backend::NdArray;

    #[test]
    fn greedy_action_respects_masks_and_has_a_stable_tie_break() {
        let mut observation = Observation {
            data: [0.0; OBS_DIM],
            action_mask: [false; NUM_ACTIONS],
            amount_choice_bits: [1; NUM_ACTIONS],
            sidecar_strength_bits: [0; NUM_ACTIONS],
            explicit_signal_strength_bits: [0; 4],
        };
        observation.action_mask[3] = true;
        observation.action_mask[7] = true;
        let mut kinds = vec![100.0; NUM_POLICY_ACTION_KINDS];
        kinds[1] = 5.0;
        kinds[3] = 5.0;
        let targets = vec![100.0; NUM_POLICY_TARGET_LOGITS];
        let efforts = vec![100.0; NUM_POLICY_EFFORT_LOGITS];
        assert_eq!(
            greedy_policy_action(&kinds, &targets, &efforts, &observation),
            3
        );
    }

    #[test]
    fn fixed_seed_evaluation_is_reproducible() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        NdArray::<f32>::seed(&device, 91);
        let model: PolicyValueNet<NdArray<f32>> =
            crate::model::PolicyValueNetConfig::new().init(&device);
        let env = EnvConfig {
            world_size: 8,
            cells_per_team: 1,
            max_episode_len: 64,
            victory: crate::config::VictoryConfig {
                sim_time_limit_quanta: 8_192,
                ..crate::config::VictoryConfig::default()
            },
            num_scattered_energy: 4,
            num_plants: 2,
            ..EnvConfig::default()
        };
        let seeds = [8001, 8002];

        let left = evaluate_policy_suite(
            &model,
            &env,
            &RewardConfig::default(),
            &OpponentProfile::ALL,
            &[],
            &seeds,
            &device,
        );
        let right = evaluate_policy_suite(
            &model,
            &env,
            &RewardConfig::default(),
            &OpponentProfile::ALL,
            &[],
            &seeds,
            &device,
        );
        assert_eq!(left, right);
        assert_eq!(left.episodes, seeds.len() * OpponentProfile::ALL.len());
        assert_eq!(left.opponents.len(), OpponentProfile::ALL.len());
        assert_eq!(
            left.opponents
                .iter()
                .map(|metrics| metrics.opponent.clone())
                .collect::<Vec<_>>(),
            OpponentProfile::ALL.map(|profile| EvaluationOpponent::Baseline { profile })
        );
    }

    #[test]
    fn snapshot_opponent_is_reproducible_and_hash_identified() {
        let _backend_guard = crate::BACKEND_TEST_LOCK.lock().unwrap();
        let device = Default::default();
        NdArray::<f32>::seed(&device, 92);
        let candidate: PolicyValueNet<NdArray<f32>> =
            crate::model::PolicyValueNetConfig::new().init(&device);
        NdArray::<f32>::seed(&device, 93);
        let opponent: PolicyValueNet<NdArray<f32>> =
            crate::model::PolicyValueNetConfig::new().init(&device);
        let snapshot = PolicySnapshot {
            model: opponent,
            directory: "/immutable/checkpoint-00000012".into(),
            checkpoint: "checkpoint-00000012".into(),
            update: 12,
            actions: 4_096,
            model_sha256: "0123456789abcdef".repeat(4),
        };
        let env = EnvConfig {
            world_size: 8,
            cells_per_team: 1,
            max_episode_len: 64,
            victory: crate::config::VictoryConfig {
                sim_time_limit_quanta: 8_192,
                ..crate::config::VictoryConfig::default()
            },
            num_scattered_energy: 4,
            num_plants: 2,
            ..EnvConfig::default()
        };
        let seeds = [9001, 9002];

        let left = evaluate_policy_against_snapshot(
            &candidate,
            &env,
            &RewardConfig::default(),
            &snapshot,
            &seeds,
            &device,
        );
        let right = evaluate_policy_against_snapshot(
            &candidate,
            &env,
            &RewardConfig::default(),
            &snapshot,
            &seeds,
            &device,
        );

        assert_eq!(left, right);
        assert_eq!(left.episodes, seeds.len());
        assert_eq!(
            left.opponent.to_string(),
            "snapshot:checkpoint-00000012:u12:0123456789ab"
        );
    }
}
