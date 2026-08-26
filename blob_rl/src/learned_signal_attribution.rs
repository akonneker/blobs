//! Paired counterfactual signal attribution for one immutable learned policy.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use blob_engine::resolution::SlotMask;
use burn::prelude::*;
use serde::{Deserialize, Serialize};

use crate::artifact::{CheckpointMetadata, PolicySnapshot};
use crate::config::{OpponentProfile, ScenarioProfile};
use crate::env::{BlobEnv, EpisodeOutcome};
use crate::evaluation::greedy_policy_choices;
use crate::signal_policy_ablation::PairedOutcomeComparison;
use crate::sweep::sha256;
use crate::telemetry::{EcologySample, StepTelemetry, TelemetryConfig};

pub const LEARNED_SIGNAL_ATTRIBUTION_SCHEMA_VERSION: u32 = 1;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AttributionOutcome {
    Win,
    Loss,
    Timeout,
}

impl AttributionOutcome {
    fn score(self) -> u8 {
        match self {
            Self::Win => 2,
            Self::Timeout => 1,
            Self::Loss => 0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvaluationSignalTelemetry {
    pub emitted_decisions: u64,
    pub explicit_signal_actions: u64,
    pub sidecar_emissions: u64,
    pub channel_energy: [u128; 4],
    pub decayed_energy: [u128; 4],
    pub terrain_erased_energy: [u128; 4],
    pub terrain_erasure_events: u64,
    pub state_samples: u64,
    pub mean_environment_signal_energy: f64,
    pub mean_observable_signal_variation: f64,
    pub final_environment_signal_energy: u128,
}

#[derive(Default)]
struct SignalTelemetryAccumulator {
    metrics: EvaluationSignalTelemetry,
    environment_signal_energy_sum: f64,
    observable_signal_variation_sum: f64,
}

impl SignalTelemetryAccumulator {
    fn observe_sample(&mut self, sample: &EcologySample) {
        let environment_signal_energy = sample.environment_energy.signal_energy;
        let observable_variation = sample
            .signal_total_variation_by_channel
            .into_iter()
            .sum::<u128>();
        self.metrics.state_samples = self.metrics.state_samples.saturating_add(1);
        self.environment_signal_energy_sum += environment_signal_energy as f64;
        self.observable_signal_variation_sum += observable_variation as f64;
        self.metrics.final_environment_signal_energy = environment_signal_energy;
    }

    fn observe_step(&mut self, step: StepTelemetry) {
        let signals = step.training.signals;
        self.metrics.emitted_decisions = self
            .metrics
            .emitted_decisions
            .saturating_add(signals.emitted_decisions);
        self.metrics.explicit_signal_actions = self
            .metrics
            .explicit_signal_actions
            .saturating_add(signals.explicit_signal_actions);
        self.metrics.sidecar_emissions = self
            .metrics
            .sidecar_emissions
            .saturating_add(signals.sidecar_emissions);
        for channel in 0..4 {
            self.metrics.channel_energy[channel] = self.metrics.channel_energy[channel]
                .saturating_add(signals.channel_energy[channel]);
            self.metrics.decayed_energy[channel] = self.metrics.decayed_energy[channel]
                .saturating_add(step.signal_field.decayed_energy[channel]);
            self.metrics.terrain_erased_energy[channel] = self.metrics.terrain_erased_energy
                [channel]
                .saturating_add(step.signal_field.terrain_erased_energy[channel]);
        }
        self.metrics.terrain_erasure_events = self
            .metrics
            .terrain_erasure_events
            .saturating_add(step.signal_field.terrain_erasure_events);
        if let Some(sample) = step.state_sample.as_ref() {
            self.observe_sample(sample);
        }
    }

    fn finish(mut self) -> EvaluationSignalTelemetry {
        if self.metrics.state_samples > 0 {
            let denominator = self.metrics.state_samples as f64;
            self.metrics.mean_environment_signal_energy =
                self.environment_signal_energy_sum / denominator;
            self.metrics.mean_observable_signal_variation =
                self.observable_signal_variation_sum / denominator;
        }
        self.metrics
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AttributionEpisode {
    pub opponent: OpponentProfile,
    pub seed: u64,
    /// The current RL environment always owns team zero. This is explicit so
    /// fixed-seat evidence cannot be mistaken for a seat-swapped tournament.
    pub candidate_seat: String,
    pub outcome: AttributionOutcome,
    pub episode_steps: u64,
    pub policy_actions: u64,
    pub reward: f64,
    pub signals: EvaluationSignalTelemetry,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AttributionVariant {
    pub name: String,
    pub signal_observation_mask_bits: u32,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub episodes: Vec<AttributionEpisode>,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub win_rate: f64,
    pub average_episode_steps: f64,
    pub average_reward: f64,
    pub policy_actions: u64,
    pub signal_energy: u128,
    pub signal_energy_per_1000_policy_actions: f64,
    pub mean_environment_signal_energy: f64,
    pub mean_observable_signal_variation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearnedSignalAttributionReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub checkpoint: String,
    pub checkpoint_update: usize,
    pub checkpoint_actions: u64,
    pub model_sha256: String,
    pub checkpoint_compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub reward_config_sha256: String,
    /// RewardConfig has no signal-emission or signal-observation term. Indirect
    /// effects through survival, food, combat, and outcomes remain intentional.
    pub direct_signal_reward_term_present: bool,
    pub trained_signal_observation_mask_bits: u32,
    pub candidate_seat: String,
    pub opponents: Vec<OpponentProfile>,
    pub seeds: Vec<u64>,
    pub full_visibility: AttributionVariant,
    pub hidden_neighbor_visibility: AttributionVariant,
    /// Positive means hidden visibility improved the outcome relative to full.
    pub hidden_vs_full_outcomes: PairedOutcomeComparison,
    pub verification_scope: String,
    pub server_verified: bool,
    pub replay_committed: bool,
    pub results_sha256: String,
}

fn run_episode<B: Backend>(
    model: &crate::model::PolicyValueNet<B>,
    mut env: BlobEnv,
    opponent: OpponentProfile,
    seed: u64,
    telemetry_config: &TelemetryConfig,
    device: &B::Device,
) -> Result<AttributionEpisode, String>
where
    f32: From<B::FloatElem>,
{
    env.enable_telemetry(telemetry_config.clone());
    let mut telemetry = SignalTelemetryAccumulator::default();
    if let Some(initial) = env.telemetry_sample() {
        telemetry.observe_sample(&initial);
    }
    let mut observations = env.get_policy_observations();
    let mut policy_actions = 0_u64;
    let mut reward = 0.0_f64;

    loop {
        let actions = greedy_policy_choices(model, &observations, device);
        policy_actions = policy_actions.saturating_add(actions.len() as u64);
        let result = env.step_with_policy_memory(&actions);
        reward += result
            .rewards
            .values()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        if let Some(step) = result.telemetry {
            telemetry.observe_step(step);
        }
        if result.done {
            let outcome = match result
                .outcome
                .ok_or_else(|| "completed attribution episode has no outcome".to_string())?
            {
                EpisodeOutcome::Win => AttributionOutcome::Win,
                EpisodeOutcome::Loss => AttributionOutcome::Loss,
                EpisodeOutcome::Timeout => AttributionOutcome::Timeout,
                EpisodeOutcome::SafetyAbort => {
                    return Err(format!(
                        "attribution episode {seed} against {opponent} reached the non-scientific safety limit"
                    ));
                }
            };
            return Ok(AttributionEpisode {
                opponent,
                seed,
                candidate_seat: "team_zero".into(),
                outcome,
                episode_steps: result.episode_step,
                policy_actions,
                reward,
                signals: telemetry.finish(),
            });
        }
        observations = result.policy_observations;
    }
}

fn summarize_variant(
    name: &str,
    mask: SlotMask,
    semantic_ruleset_hash: String,
    compiled_ruleset_hash: String,
    episodes: Vec<AttributionEpisode>,
) -> AttributionVariant {
    let wins = episodes
        .iter()
        .filter(|episode| episode.outcome == AttributionOutcome::Win)
        .count();
    let losses = episodes
        .iter()
        .filter(|episode| episode.outcome == AttributionOutcome::Loss)
        .count();
    let timeouts = episodes.len().saturating_sub(wins + losses);
    let denominator = episodes.len().max(1) as f64;
    let policy_actions = episodes.iter().map(|episode| episode.policy_actions).sum();
    let signal_energy = episodes
        .iter()
        .flat_map(|episode| episode.signals.channel_energy)
        .sum();
    let state_samples = episodes
        .iter()
        .map(|episode| episode.signals.state_samples)
        .sum::<u64>();
    let weighted_sample_mean = |value: fn(&EvaluationSignalTelemetry) -> f64| {
        if state_samples == 0 {
            0.0
        } else {
            episodes
                .iter()
                .map(|episode| value(&episode.signals) * episode.signals.state_samples as f64)
                .sum::<f64>()
                / state_samples as f64
        }
    };
    AttributionVariant {
        name: name.into(),
        signal_observation_mask_bits: mask.bits(),
        semantic_ruleset_hash,
        compiled_ruleset_hash,
        wins,
        losses,
        timeouts,
        win_rate: wins as f64 / denominator,
        average_episode_steps: episodes
            .iter()
            .map(|episode| episode.episode_steps as f64)
            .sum::<f64>()
            / denominator,
        average_reward: episodes.iter().map(|episode| episode.reward).sum::<f64>() / denominator,
        policy_actions,
        signal_energy,
        signal_energy_per_1000_policy_actions: if policy_actions == 0 {
            0.0
        } else {
            signal_energy as f64 * 1_000.0 / policy_actions as f64
        },
        mean_environment_signal_energy: weighted_sample_mean(|signals| {
            signals.mean_environment_signal_energy
        }),
        mean_observable_signal_variation: weighted_sample_mean(|signals| {
            signals.mean_observable_signal_variation
        }),
        episodes,
    }
}

fn evaluate_variant<B: Backend>(
    variant: (&str, SlotMask),
    snapshot: &PolicySnapshot<B>,
    metadata: &CheckpointMetadata,
    opponents: &[OpponentProfile],
    seeds: &[u64],
    telemetry_config: &TelemetryConfig,
    device: &B::Device,
) -> Result<AttributionVariant, String>
where
    f32: From<B::FloatElem>,
{
    let (name, mask) = variant;
    let mut env_config = metadata.config.env.clone();
    env_config.rules.neighborhood.observations.signal = mask;
    env_config
        .rules
        .validate_for_world(env_config.world_size, env_config.world_size)
        .map_err(|error| format!("invalid {name} ruleset: {error}"))?;
    let semantic_ruleset_hash = env_config.rules.semantic_hash().to_string();
    let mut compiled_ruleset_hash = None;
    let mut episodes = Vec::with_capacity(opponents.len().saturating_mul(seeds.len()));
    for &opponent in opponents {
        for &seed in seeds {
            let mut profile_env = env_config.clone();
            profile_env.opponent = opponent;
            let env = BlobEnv::new(profile_env, metadata.config.reward.clone(), seed);
            let hash = env.compiled_ruleset_hash();
            if compiled_ruleset_hash
                .as_ref()
                .is_some_and(|expected| expected != &hash)
            {
                return Err(format!("{name} produced multiple compiled ruleset hashes"));
            }
            compiled_ruleset_hash = Some(hash);
            episodes.push(run_episode(
                &snapshot.model,
                env,
                opponent,
                seed,
                telemetry_config,
                device,
            )?);
        }
    }
    Ok(summarize_variant(
        name,
        mask,
        semantic_ruleset_hash,
        compiled_ruleset_hash.ok_or_else(|| format!("{name} has no episodes"))?,
        episodes,
    ))
}

pub fn evaluate_learned_signal_attribution<B: Backend>(
    snapshot: &PolicySnapshot<B>,
    metadata: &CheckpointMetadata,
    opponents: &[OpponentProfile],
    seeds: &[u64],
    device: &B::Device,
) -> Result<LearnedSignalAttributionReport, String>
where
    f32: From<B::FloatElem>,
{
    if opponents.is_empty() || seeds.is_empty() {
        return Err("learned signal attribution requires opponents and seeds".into());
    }
    if snapshot.model_sha256 != metadata.model_sha256
        || snapshot.update != metadata.update
        || snapshot.actions != metadata.actions
    {
        return Err("policy snapshot identity does not match checkpoint metadata".into());
    }
    let slot_count = metadata.config.env.rules.neighborhood.slots.len();
    let trained_signal_observation_mask_bits = metadata
        .config
        .env
        .rules
        .neighborhood
        .observations
        .signal
        .bits();
    let full_mask = SlotMask::first(slot_count);
    let hidden_mask = SlotMask::empty();
    let mut telemetry_config = if metadata.config.telemetry.enabled {
        metadata.config.telemetry.clone()
    } else {
        // A checkpoint may legitimately have disabled training telemetry and
        // therefore retain zero-valued cadence bounds. Attribution still needs
        // host observations, so use the bounded default rather than mutating
        // an otherwise-valid disabled profile into an invalid enabled one.
        TelemetryConfig::default()
    };
    telemetry_config.enabled = true;
    telemetry_config.validate()?;

    let full_visibility = evaluate_variant(
        ("full_visibility", full_mask),
        snapshot,
        metadata,
        opponents,
        seeds,
        &telemetry_config,
        device,
    )?;
    let hidden_neighbor_visibility = evaluate_variant(
        ("hidden_neighbor_visibility", hidden_mask),
        snapshot,
        metadata,
        opponents,
        seeds,
        &telemetry_config,
        device,
    )?;
    if full_visibility.episodes.len() != hidden_neighbor_visibility.episodes.len() {
        return Err("visibility variants do not contain the same episode count".into());
    }
    let mut hidden_vs_full_outcomes = PairedOutcomeComparison {
        improved: 0,
        worsened: 0,
        unchanged: 0,
    };
    for (full, hidden) in full_visibility
        .episodes
        .iter()
        .zip(&hidden_neighbor_visibility.episodes)
    {
        if full.opponent != hidden.opponent || full.seed != hidden.seed {
            return Err("visibility variants do not contain identical paired episodes".into());
        }
        match hidden.outcome.score().cmp(&full.outcome.score()) {
            std::cmp::Ordering::Greater => hidden_vs_full_outcomes.improved += 1,
            std::cmp::Ordering::Less => hidden_vs_full_outcomes.worsened += 1,
            std::cmp::Ordering::Equal => hidden_vs_full_outcomes.unchanged += 1,
        }
    }
    let reward_config_sha256 = sha256(
        &serde_json::to_vec(&metadata.config.reward)
            .map_err(|error| format!("failed to encode reward config: {error}"))?,
    );
    let scenario_hash = ScenarioProfile::from(&metadata.config.env).semantic_hash()?;
    let results_sha256 = sha256(
        &serde_json::to_vec(&(
            &full_visibility,
            &hidden_neighbor_visibility,
            &hidden_vs_full_outcomes,
        ))
        .map_err(|error| format!("failed to encode attribution results: {error}"))?,
    );
    Ok(LearnedSignalAttributionReport {
        schema_version: LEARNED_SIGNAL_ATTRIBUTION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").into(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        checkpoint: snapshot.checkpoint.clone(),
        checkpoint_update: snapshot.update,
        checkpoint_actions: snapshot.actions,
        model_sha256: snapshot.model_sha256.clone(),
        checkpoint_compiled_ruleset_hash: metadata.compiled_ruleset_hash.clone(),
        scenario_hash,
        reward_config_sha256,
        direct_signal_reward_term_present: false,
        trained_signal_observation_mask_bits,
        candidate_seat: "team_zero".into(),
        opponents: opponents.to_vec(),
        seeds: seeds.to_vec(),
        full_visibility,
        hidden_neighbor_visibility,
        hidden_vs_full_outcomes,
        verification_scope: "local_deterministic_frozen_policy".into(),
        server_verified: false,
        replay_committed: false,
        results_sha256,
    })
}

pub fn publish_learned_signal_attribution(
    output: &Path,
    report: &LearnedSignalAttributionReport,
) -> Result<(), String> {
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable attribution report {}",
            output.display()
        ));
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{}.tmp-{}-{nonce}",
        output
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("learned-signal-attribution"),
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode attribution report: {error}"))?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
    file.write_all(&bytes)
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
    fs::rename(&temporary, output)
        .map_err(|error| format!("failed to publish {}: {error}", output.display()))?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to sync {}: {error}", parent.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_is_weighted_by_state_samples_and_actions() {
        let episode = |seed, samples, mean, energy, actions| AttributionEpisode {
            opponent: OpponentProfile::Wait,
            seed,
            candidate_seat: "team_zero".into(),
            outcome: AttributionOutcome::Timeout,
            episode_steps: 10,
            policy_actions: actions,
            reward: 0.0,
            signals: EvaluationSignalTelemetry {
                state_samples: samples,
                mean_environment_signal_energy: mean,
                channel_energy: [energy, 0, 0, 0],
                ..EvaluationSignalTelemetry::default()
            },
        };
        let summary = summarize_variant(
            "test",
            SlotMask::empty(),
            "semantic".into(),
            "compiled".into(),
            vec![episode(1, 1, 10.0, 2, 10), episode(2, 3, 30.0, 4, 20)],
        );
        assert_eq!(summary.signal_energy, 6);
        assert_eq!(summary.signal_energy_per_1000_policy_actions, 200.0);
        assert_eq!(summary.mean_environment_signal_energy, 25.0);
    }

    #[test]
    fn immutable_publication_refuses_replacement() {
        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("attribution.json");
        let report = LearnedSignalAttributionReport {
            schema_version: LEARNED_SIGNAL_ATTRIBUTION_SCHEMA_VERSION,
            package_version: "test".into(),
            code_revision: None,
            checkpoint: "checkpoint-1".into(),
            checkpoint_update: 1,
            checkpoint_actions: 2,
            model_sha256: "model".into(),
            checkpoint_compiled_ruleset_hash: "trained".into(),
            scenario_hash: "scenario".into(),
            reward_config_sha256: "reward".into(),
            direct_signal_reward_term_present: false,
            trained_signal_observation_mask_bits: 0,
            candidate_seat: "team_zero".into(),
            opponents: vec![OpponentProfile::Wait],
            seeds: vec![1],
            full_visibility: summarize_variant(
                "full_visibility",
                SlotMask::empty(),
                "semantic".into(),
                "compiled".into(),
                Vec::new(),
            ),
            hidden_neighbor_visibility: summarize_variant(
                "hidden_neighbor_visibility",
                SlotMask::empty(),
                "semantic".into(),
                "compiled".into(),
                Vec::new(),
            ),
            hidden_vs_full_outcomes: PairedOutcomeComparison {
                improved: 0,
                worsened: 0,
                unchanged: 0,
            },
            verification_scope: "test".into(),
            server_verified: false,
            replay_committed: false,
            results_sha256: "results".into(),
        };
        publish_learned_signal_attribution(&output, &report).unwrap();
        assert!(publish_learned_signal_attribution(&output, &report).is_err());
    }
}
