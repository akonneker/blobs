//! Deterministic non-learning baseline-versus-baseline viability evaluation.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{EnvConfig, OpponentProfile, RewardConfig, ScenarioProfile};
use crate::env::{BlobEnv, EpisodeOutcome};
use crate::telemetry::{
    TelemetryConfig, TelemetryEpisodeOutcome, TrainingTelemetryState, TrainingTelemetrySummary,
};

pub const VIABILITY_REPORT_SCHEMA_VERSION: u32 = 2;
static VIABILITY_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ViabilityOutcome {
    CandidateWin,
    CandidateLoss,
    Timeout,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityEpisode {
    pub seed: u64,
    pub outcome: ViabilityOutcome,
    pub environment_steps: u64,
    pub final_sim_time_quanta: u64,
    pub candidate_cells: usize,
    pub opponent_cells: usize,
    pub initial_tracked_mass_energy: u128,
    pub final_tracked_mass_energy: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityAggregate {
    pub episodes: usize,
    pub candidate_wins: usize,
    pub candidate_losses: usize,
    pub timeouts: usize,
    pub candidate_win_rate: f64,
    pub average_environment_steps: f64,
    pub average_final_sim_time_quanta: f64,
    pub average_final_candidate_cells: f64,
    pub average_final_opponent_cells: f64,
    pub no_combat_profiles: bool,
    pub extinctions_without_combat: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ViabilityReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub mind_abi_version: u16,
    pub mind_abi_hash: String,
    pub viability_config_hash: String,
    pub semantic_ruleset_hash: String,
    pub compiled_ruleset_hash: String,
    pub scenario_hash: String,
    pub scenario: ScenarioProfile,
    pub candidate: OpponentProfile,
    pub opponent: OpponentProfile,
    pub seeds: Vec<u64>,
    pub telemetry_config: TelemetryConfig,
    pub episodes: Vec<ViabilityEpisode>,
    pub aggregate: ViabilityAggregate,
    pub telemetry: TrainingTelemetrySummary,
}

#[derive(Serialize)]
struct ViabilityIdentity<'a> {
    mind_abi_version: u16,
    mind_abi_hash: &'a str,
    semantic_ruleset_hash: &'a str,
    compiled_ruleset_hash: &'a str,
    scenario_hash: &'a str,
    candidate: OpponentProfile,
    opponent: OpponentProfile,
    seeds: &'a [u64],
    telemetry_config: &'a TelemetryConfig,
}

pub(crate) fn hash_json(value: &impl Serialize) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("failed to encode viability identity: {error}"))?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

pub(crate) fn mind_abi_hash() -> String {
    blob_interface::abi::reference_mind_abi_hash()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn zero_reward() -> RewardConfig {
    RewardConfig {
        survive_tick: 0.0,
        eat_energy: 0.0,
        kill_enemy: 0.0,
        cell_died: 0.0,
        split_success: 0.0,
        team_wins: 0.0,
        team_loses: 0.0,
        move_toward_food: 0.0,
        move_toward_enemy: 0.0,
        proximity_search_radius: 0,
        deadline: Default::default(),
    }
}

fn aggregate_episodes(
    episodes: &[ViabilityEpisode],
    candidate: OpponentProfile,
    opponent: OpponentProfile,
) -> ViabilityAggregate {
    let candidate_wins = episodes
        .iter()
        .filter(|episode| episode.outcome == ViabilityOutcome::CandidateWin)
        .count();
    let candidate_losses = episodes
        .iter()
        .filter(|episode| episode.outcome == ViabilityOutcome::CandidateLoss)
        .count();
    let timeouts = episodes.len() - candidate_wins - candidate_losses;
    let denominator = episodes.len() as f64;
    let no_combat_profiles = !candidate.can_attack() && !opponent.can_attack();
    ViabilityAggregate {
        episodes: episodes.len(),
        candidate_wins,
        candidate_losses,
        timeouts,
        candidate_win_rate: candidate_wins as f64 / denominator,
        average_environment_steps: episodes
            .iter()
            .map(|episode| episode.environment_steps as f64)
            .sum::<f64>()
            / denominator,
        average_final_sim_time_quanta: episodes
            .iter()
            .map(|episode| episode.final_sim_time_quanta as f64)
            .sum::<f64>()
            / denominator,
        average_final_candidate_cells: episodes
            .iter()
            .map(|episode| episode.candidate_cells as f64)
            .sum::<f64>()
            / denominator,
        average_final_opponent_cells: episodes
            .iter()
            .map(|episode| episode.opponent_cells as f64)
            .sum::<f64>()
            / denominator,
        no_combat_profiles,
        extinctions_without_combat: if no_combat_profiles {
            candidate_wins + candidate_losses
        } else {
            0
        },
    }
}

/// Validate the self-contained identities and all episode-derived fields of a
/// viability report before using it as a scientific control input.
pub fn validate_viability_report(report: &ViabilityReport) -> Result<(), String> {
    if report.schema_version != VIABILITY_REPORT_SCHEMA_VERSION {
        return Err(format!(
            "unsupported viability report schema {}; expected {}",
            report.schema_version, VIABILITY_REPORT_SCHEMA_VERSION
        ));
    }
    if report.mind_abi_version != blob_interface::abi::REFERENCE_MIND_ABI_VERSION
        || report.mind_abi_hash != mind_abi_hash()
    {
        return Err("viability report Mind ABI identity mismatch".into());
    }
    if report.seeds.is_empty() || report.episodes.len() != report.seeds.len() {
        return Err("viability report seed/episode matrix is incomplete".into());
    }
    let mut seeds = HashSet::with_capacity(report.seeds.len());
    if report.seeds.iter().any(|seed| !seeds.insert(*seed))
        || report
            .episodes
            .iter()
            .zip(&report.seeds)
            .any(|(episode, seed)| episode.seed != *seed)
    {
        return Err("viability report seeds are duplicated or out of order".into());
    }
    report.telemetry_config.validate()?;
    if !report.telemetry_config.enabled {
        return Err("viability report requires enabled telemetry".into());
    }
    if report.scenario.semantic_hash()? != report.scenario_hash {
        return Err("viability report scenario hash mismatch".into());
    }
    let expected_aggregate =
        aggregate_episodes(&report.episodes, report.candidate, report.opponent);
    if expected_aggregate != report.aggregate {
        return Err(format!(
            "viability report aggregate does not match its episodes: expected {expected_aggregate:?}, found {:?}",
            report.aggregate
        ));
    }
    if report.telemetry.completed_episodes != report.episodes.len() as u64
        || report.telemetry.censored_active_episodes != 0
        || report.telemetry.wins != report.aggregate.candidate_wins as u64
        || report.telemetry.losses != report.aggregate.candidate_losses as u64
        || report.telemetry.timeouts != report.aggregate.timeouts as u64
    {
        return Err("viability report telemetry outcome totals are inconsistent".into());
    }
    let expected_identity = hash_json(&ViabilityIdentity {
        mind_abi_version: report.mind_abi_version,
        mind_abi_hash: &report.mind_abi_hash,
        semantic_ruleset_hash: &report.semantic_ruleset_hash,
        compiled_ruleset_hash: &report.compiled_ruleset_hash,
        scenario_hash: &report.scenario_hash,
        candidate: report.candidate,
        opponent: report.opponent,
        seeds: &report.seeds,
        telemetry_config: &report.telemetry_config,
    })?;
    if expected_identity != report.viability_config_hash {
        return Err("viability report configuration hash mismatch".into());
    }
    Ok(())
}

/// Run two native baseline profiles against each other through the canonical
/// anonymous Mind boundary. Team zero is the candidate; every other team uses
/// the opponent profile.
pub fn run_baseline_viability(
    env_config: &EnvConfig,
    telemetry_config: &TelemetryConfig,
    candidate: OpponentProfile,
    opponent: OpponentProfile,
    seeds: &[u64],
) -> Result<ViabilityReport, String> {
    if seeds.is_empty() {
        return Err("viability evaluation requires at least one seed".into());
    }
    let mut unique = HashSet::with_capacity(seeds.len());
    if seeds.iter().any(|seed| !unique.insert(*seed)) {
        return Err("viability evaluation seeds must be unique".into());
    }
    let validation = crate::config::TrainingConfig {
        env: env_config.clone(),
        telemetry: telemetry_config.clone(),
        ..crate::config::TrainingConfig::default()
    };
    validation.validate()?;
    if !telemetry_config.enabled {
        return Err("viability evaluation requires telemetry".into());
    }

    let mut profile_env = env_config.clone();
    profile_env.opponent = opponent;
    let scenario = ScenarioProfile::from(&profile_env);
    let scenario_hash = scenario.semantic_hash()?;
    let semantic_ruleset_hash = profile_env.rules.semantic_hash().to_string();
    let expected_candidate_cells = profile_env.cells_per_team;
    let expected_opponent_cells = profile_env
        .cells_per_team
        .checked_mul(profile_env.num_teams - 1)
        .ok_or("expected opponent population overflows usize")?;
    let mut episodes = Vec::with_capacity(seeds.len());
    let mut telemetry_state: Option<TrainingTelemetryState> = None;
    let mut compiled_ruleset_hash = None;

    for (episode_id, &seed) in seeds.iter().enumerate() {
        let mut env = BlobEnv::new(profile_env.clone(), zero_reward(), seed);
        env.enable_telemetry(telemetry_config.clone());
        let initial = env
            .telemetry_sample()
            .ok_or("enabled viability telemetry did not produce an initial sample")?;
        if initial.training_cells != expected_candidate_cells
            || initial.opponent_cells != expected_opponent_cells
        {
            return Err(format!(
                "engine placed {}/{} candidate cells and {}/{} opponent cells for seed {seed}",
                initial.training_cells,
                expected_candidate_cells,
                initial.opponent_cells,
                expected_opponent_cells
            ));
        }
        if let Some(state) = telemetry_state.as_mut() {
            state.start_episode(0, episode_id as u64, seed, initial.clone());
        } else {
            telemetry_state = Some(TrainingTelemetryState::new(vec![(seed, initial.clone())]));
        }
        let compiled = env.compiled_ruleset_hash();
        if compiled_ruleset_hash
            .as_ref()
            .is_some_and(|expected| expected != &compiled)
        {
            return Err("identical viability configuration compiled to different rulesets".into());
        }
        compiled_ruleset_hash = Some(compiled);

        loop {
            let mut result = env.step_with_baseline(candidate);
            let step_telemetry = result
                .telemetry
                .take()
                .ok_or("enabled viability environment omitted step telemetry")?;
            telemetry_state
                .as_mut()
                .expect("initialized above")
                .apply_step(
                    0,
                    &step_telemetry,
                    telemetry_config.max_state_samples_per_episode,
                );
            if !result.done {
                continue;
            }
            let outcome = match result
                .outcome
                .ok_or("completed viability episode omitted its outcome")?
            {
                EpisodeOutcome::Win => ViabilityOutcome::CandidateWin,
                EpisodeOutcome::Loss => ViabilityOutcome::CandidateLoss,
                EpisodeOutcome::Timeout => ViabilityOutcome::Timeout,
                EpisodeOutcome::SafetyAbort => {
                    return Err(format!(
                        "seed {seed} reached the non-scientific decision-frontier safety limit"
                    ));
                }
            };
            let telemetry_outcome = match outcome {
                ViabilityOutcome::CandidateWin => TelemetryEpisodeOutcome::Win,
                ViabilityOutcome::CandidateLoss => TelemetryEpisodeOutcome::Loss,
                ViabilityOutcome::Timeout => TelemetryEpisodeOutcome::Timeout,
            };
            let final_sample = env
                .telemetry_sample()
                .ok_or("enabled viability telemetry omitted the terminal sample")?;
            telemetry_state
                .as_mut()
                .expect("initialized above")
                .finish_episode(
                    0,
                    telemetry_outcome,
                    result.episode_step,
                    final_sample.clone(),
                    telemetry_config.max_state_samples_per_episode,
                );
            episodes.push(ViabilityEpisode {
                seed,
                outcome,
                environment_steps: result.episode_step,
                final_sim_time_quanta: final_sample.sim_time_quanta,
                candidate_cells: result.training_cells,
                opponent_cells: result.opponent_cells,
                initial_tracked_mass_energy: initial.tracked_mass_energy,
                final_tracked_mass_energy: final_sample.tracked_mass_energy,
            });
            break;
        }
    }

    let compiled_ruleset_hash = compiled_ruleset_hash.expect("at least one seed was required");
    let mind_abi_version = blob_interface::abi::REFERENCE_MIND_ABI_VERSION;
    let mind_abi_hash = mind_abi_hash();
    let viability_config_hash = hash_json(&ViabilityIdentity {
        mind_abi_version,
        mind_abi_hash: &mind_abi_hash,
        semantic_ruleset_hash: &semantic_ruleset_hash,
        compiled_ruleset_hash: &compiled_ruleset_hash,
        scenario_hash: &scenario_hash,
        candidate,
        opponent,
        seeds,
        telemetry_config,
    })?;
    let mut telemetry = telemetry_state
        .expect("at least one seed was required")
        .summary();
    // Unlike a training snapshot, the viability report is published only
    // after its final requested episode has completed.
    telemetry.censored_active_episodes = 0;
    Ok(ViabilityReport {
        schema_version: VIABILITY_REPORT_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        mind_abi_version,
        mind_abi_hash,
        viability_config_hash,
        semantic_ruleset_hash,
        compiled_ruleset_hash,
        scenario_hash,
        scenario,
        candidate,
        opponent,
        seeds: seeds.to_vec(),
        telemetry_config: telemetry_config.clone(),
        aggregate: aggregate_episodes(&episodes, candidate, opponent),
        episodes,
        telemetry,
    })
}

/// Publish one immutable report without exposing a partially written JSON
/// file. A hard-link publication refuses to replace an existing result.
pub fn publish_viability_report(
    output: &Path,
    report: &ViabilityReport,
) -> Result<PathBuf, String> {
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable viability report {}",
            output.display()
        ));
    }
    let nonce = VIABILITY_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "viability output needs a UTF-8 file name".to_string())?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode viability report: {error}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn published_aggregate_floats_round_trip_exactly() {
        // Immutable reports are re-derived and compared exactly when reused.
        // This repeating value exercises serde_json's exact float parser.
        let original = 373.0_f64 / 3.0;
        let encoded = serde_json::to_vec(&original).unwrap();
        let decoded: f64 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.to_bits(), original.to_bits());
    }

    fn small_env() -> EnvConfig {
        EnvConfig {
            world_size: 8,
            cells_per_team: 1,
            max_episode_len: 64,
            victory: crate::config::VictoryConfig {
                sim_time_limit_quanta: 12_288,
                ..crate::config::VictoryConfig::default()
            },
            num_scattered_energy: 4,
            num_plants: 2,
            ..EnvConfig::default()
        }
    }

    #[test]
    fn baseline_viability_is_reproducible_and_accounts_for_every_seed() {
        let telemetry = TelemetryConfig {
            state_sample_interval_steps: 2,
            max_state_samples_per_episode: 8,
            episode_log_stride: 0,
            ..TelemetryConfig::default()
        };
        let seeds = [71, 72, 73];
        let left = run_baseline_viability(
            &small_env(),
            &telemetry,
            OpponentProfile::Forager,
            OpponentProfile::Forager,
            &seeds,
        )
        .unwrap();
        let right = run_baseline_viability(
            &small_env(),
            &telemetry,
            OpponentProfile::Forager,
            OpponentProfile::Forager,
            &seeds,
        )
        .unwrap();
        assert_eq!(left, right);
        assert_eq!(left.aggregate.episodes, seeds.len());
        assert_eq!(left.telemetry.completed_episodes, seeds.len() as u64);
        assert_eq!(left.telemetry.censored_active_episodes, 0);
        assert!(left.aggregate.no_combat_profiles);
        assert_eq!(left.telemetry.training.actions.attack.committed, 0);
        assert_eq!(left.telemetry.opponents.actions.attack.committed, 0);
    }

    #[test]
    fn viability_identity_changes_with_seed_suite_and_reports_are_immutable() {
        let telemetry = TelemetryConfig {
            state_sample_interval_steps: 2,
            max_state_samples_per_episode: 8,
            episode_log_stride: 0,
            ..TelemetryConfig::default()
        };
        let first = run_baseline_viability(
            &small_env(),
            &telemetry,
            OpponentProfile::Random,
            OpponentProfile::Wait,
            &[101],
        )
        .unwrap();
        let second = run_baseline_viability(
            &small_env(),
            &telemetry,
            OpponentProfile::Random,
            OpponentProfile::Wait,
            &[102],
        )
        .unwrap();
        assert_ne!(first.viability_config_hash, second.viability_config_hash);

        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("report.json");
        publish_viability_report(&output, &first).unwrap();
        let published = fs::read(&output).unwrap();
        assert!(publish_viability_report(&output, &second).is_err());
        assert_eq!(fs::read(&output).unwrap(), published);
        let decoded: ViabilityReport = serde_json::from_slice(&published).unwrap();
        assert_eq!(decoded.viability_config_hash, first.viability_config_hash);
        assert_eq!(decoded.seeds, first.seeds);
        assert_eq!(decoded.episodes, first.episodes);
        validate_viability_report(&decoded).unwrap();
        let mut tampered = decoded;
        tampered.aggregate.timeouts += 1;
        assert!(validate_viability_report(&tampered)
            .unwrap_err()
            .contains("aggregate"));
    }
}
