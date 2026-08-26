//! Paired analysis of signal-observation visibility variants.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::control_matrix::{
    decode_control_matrix_report, ControlMatrixMatchup, ControlSeat, MaintainedMindProfile,
};
use crate::signal_policy_ablation::PairedOutcomeComparison;
use crate::sweep::sha256;
use crate::viability::ViabilityOutcome;

pub const SIGNAL_VISIBILITY_ABLATION_SCHEMA_VERSION: u32 = 2;
static SIGNAL_VISIBILITY_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);
const EXPECTED_VARIANTS: [&str; 3] = [
    "full-visibility",
    "cardinal-visibility",
    "no-neighbor-visibility",
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SignalVisibilityRow {
    pub variant: String,
    pub semantic_ruleset_hash: String,
    pub episodes: usize,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub sidecar_emissions: u64,
    pub signal_energy: u128,
    pub completed_actions: u64,
    pub mean_environment_signal_energy: f64,
    pub mean_observable_signal_variation: f64,
    pub paired_vs_full_visibility: PairedOutcomeComparison,
    pub paired_vs_disabled: PairedOutcomeComparison,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SignalVisibilityAblationReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub source_report_sha256: String,
    pub disabled_source_report_sha256: String,
    pub manifest_sha256: String,
    pub mind_abi_version: u16,
    pub mind_abi_hash: String,
    pub candidate: MaintainedMindProfile,
    pub verification_scope: String,
    pub server_verified: bool,
    pub replay_committed: bool,
    pub opponents: Vec<MaintainedMindProfile>,
    pub seeds: Vec<u64>,
    pub rows: Vec<SignalVisibilityRow>,
    pub results_sha256: String,
}

type EpisodeKey = (MaintainedMindProfile, u64, ControlSeat);

fn outcome_score(outcome: ViabilityOutcome) -> u8 {
    match outcome {
        ViabilityOutcome::CandidateWin => 2,
        ViabilityOutcome::Timeout => 1,
        ViabilityOutcome::CandidateLoss => 0,
    }
}

fn outcomes(matchups: &[&ControlMatrixMatchup]) -> HashMap<EpisodeKey, ViabilityOutcome> {
    matchups
        .iter()
        .flat_map(|matchup| {
            matchup.report.episodes.iter().map(|episode| {
                (
                    (matchup.report.opponent, episode.seed, episode.seat),
                    episode.outcome,
                )
            })
        })
        .collect()
}

fn action_count(matchups: &[&ControlMatrixMatchup]) -> u64 {
    matchups
        .iter()
        .map(|matchup| {
            let actions = &matchup.report.telemetry.training.actions;
            [
                &actions.wait,
                &actions.movement,
                &actions.attack,
                &actions.guard,
                &actions.consume,
                &actions.split,
                &actions.regurgitate,
                &actions.signal,
                &actions.excavate,
                &actions.deposit_terrain,
            ]
            .into_iter()
            .map(|action| action.completed)
            .sum::<u64>()
        })
        .sum()
}

fn variant_row(
    variant: &str,
    matchups: &[&ControlMatrixMatchup],
    full: &HashMap<EpisodeKey, ViabilityOutcome>,
    disabled: &HashMap<EpisodeKey, ViabilityOutcome>,
) -> Result<SignalVisibilityRow, String> {
    let observed = outcomes(matchups);
    if observed.len() != full.len() || observed.keys().any(|key| !full.contains_key(key)) {
        return Err(format!(
            "variant {variant} does not contain every paired episode"
        ));
    }
    let mut paired = PairedOutcomeComparison {
        improved: 0,
        worsened: 0,
        unchanged: 0,
    };
    let mut paired_disabled = PairedOutcomeComparison {
        improved: 0,
        worsened: 0,
        unchanged: 0,
    };
    for (key, outcome) in &observed {
        match outcome_score(*outcome).cmp(&outcome_score(full[key])) {
            std::cmp::Ordering::Greater => paired.improved += 1,
            std::cmp::Ordering::Less => paired.worsened += 1,
            std::cmp::Ordering::Equal => paired.unchanged += 1,
        }
        match outcome_score(*outcome).cmp(&outcome_score(disabled[key])) {
            std::cmp::Ordering::Greater => paired_disabled.improved += 1,
            std::cmp::Ordering::Less => paired_disabled.worsened += 1,
            std::cmp::Ordering::Equal => paired_disabled.unchanged += 1,
        }
    }
    let state_samples = matchups
        .iter()
        .map(|matchup| matchup.report.telemetry.state_samples)
        .sum::<u64>();
    let weighted_mean = |value: fn(&crate::telemetry::TelemetrySampleMeans) -> f64| {
        if state_samples == 0 {
            0.0
        } else {
            matchups
                .iter()
                .map(|matchup| {
                    value(&matchup.report.telemetry.sample_means)
                        * matchup.report.telemetry.state_samples as f64
                })
                .sum::<f64>()
                / state_samples as f64
        }
    };
    let wins = observed
        .values()
        .filter(|outcome| **outcome == ViabilityOutcome::CandidateWin)
        .count();
    let losses = observed
        .values()
        .filter(|outcome| **outcome == ViabilityOutcome::CandidateLoss)
        .count();
    let semantic_ruleset_hash = matchups
        .first()
        .ok_or_else(|| format!("variant {variant} has no matchups"))?
        .report
        .semantic_ruleset_hash
        .clone();
    if matchups
        .iter()
        .any(|matchup| matchup.report.semantic_ruleset_hash != semantic_ruleset_hash)
    {
        return Err(format!("variant {variant} has multiple semantic rulesets"));
    }
    Ok(SignalVisibilityRow {
        variant: variant.into(),
        semantic_ruleset_hash,
        episodes: observed.len(),
        wins,
        losses,
        timeouts: observed.len().saturating_sub(wins + losses),
        sidecar_emissions: matchups
            .iter()
            .map(|matchup| matchup.report.telemetry.training.signals.sidecar_emissions)
            .sum(),
        signal_energy: matchups
            .iter()
            .flat_map(|matchup| matchup.report.telemetry.training.signals.channel_energy)
            .sum(),
        completed_actions: action_count(matchups),
        mean_environment_signal_energy: weighted_mean(|means| means.environment_signal_energy),
        mean_observable_signal_variation: weighted_mean(|means| {
            means.signal_observation_total_variation
        }),
        paired_vs_full_visibility: paired,
        paired_vs_disabled: paired_disabled,
    })
}

pub fn analyze_signal_visibility_report(
    source: &Path,
    disabled_source: &Path,
) -> Result<SignalVisibilityAblationReport, String> {
    let bytes = fs::read(source)
        .map_err(|error| format!("failed to read {}: {error}", source.display()))?;
    let report = decode_control_matrix_report(&bytes)
        .map_err(|error| format!("{}: {error}", source.display()))?;
    let disabled_bytes = fs::read(disabled_source)
        .map_err(|error| format!("failed to read {}: {error}", disabled_source.display()))?;
    let disabled_report = decode_control_matrix_report(&disabled_bytes)
        .map_err(|error| format!("{}: {error}", disabled_source.display()))?;
    if report.candidate != MaintainedMindProfile::ColonySignalOneQuantum {
        return Err("visibility ablation requires the one-quantum colony candidate".into());
    }
    if disabled_report.candidate != MaintainedMindProfile::ColonySignalDisabled
        || disabled_report.variants.len() != 1
        || disabled_report.opponents != report.opponents
        || disabled_report.seeds != report.seeds
        || disabled_report.mind_abi_version != report.mind_abi_version
        || disabled_report.mind_abi_hash != report.mind_abi_hash
    {
        return Err("disabled control does not share the visibility evaluation identity".into());
    }
    let names = report
        .variants
        .iter()
        .map(|variant| variant.name.as_str())
        .collect::<Vec<_>>();
    if names != EXPECTED_VARIANTS {
        return Err(format!(
            "visibility variants must be {:?}, found {names:?}",
            EXPECTED_VARIANTS
        ));
    }
    let grouped = EXPECTED_VARIANTS
        .iter()
        .map(|variant| {
            report
                .matchups
                .iter()
                .filter(|matchup| matchup.variant == *variant)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if grouped
        .iter()
        .any(|matchups| matchups.len() != report.opponents.len())
    {
        return Err("visibility ablation matchup grid is incomplete".into());
    }
    let full = outcomes(&grouped[0]);
    let disabled_matchups = disabled_report.matchups.iter().collect::<Vec<_>>();
    let disabled = outcomes(&disabled_matchups);
    if disabled.len() != full.len()
        || full.keys().any(|key| !disabled.contains_key(key))
        || grouped[0].iter().any(|full_matchup| {
            disabled_matchups.iter().all(|disabled_matchup| {
                disabled_matchup.report.opponent != full_matchup.report.opponent
                    || disabled_matchup.report.scenario_hash != full_matchup.report.scenario_hash
                    || disabled_matchup.report.semantic_ruleset_hash
                        != full_matchup.report.semantic_ruleset_hash
            })
        })
    {
        return Err("disabled control does not match the full-visibility world rules".into());
    }
    let rows = EXPECTED_VARIANTS
        .iter()
        .zip(&grouped)
        .map(|(variant, matchups)| variant_row(variant, matchups, &full, &disabled))
        .collect::<Result<Vec<_>, _>>()?;
    let results_sha256 = sha256(
        &serde_json::to_vec(&rows)
            .map_err(|error| format!("failed to hash visibility rows: {error}"))?,
    );
    Ok(SignalVisibilityAblationReport {
        schema_version: SIGNAL_VISIBILITY_ABLATION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").into(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        source_report_sha256: sha256(&bytes),
        disabled_source_report_sha256: sha256(&disabled_bytes),
        manifest_sha256: report.manifest_sha256,
        mind_abi_version: report.mind_abi_version,
        mind_abi_hash: report.mind_abi_hash,
        candidate: report.candidate,
        verification_scope: "local_deterministic".into(),
        server_verified: false,
        replay_committed: false,
        opponents: report.opponents,
        seeds: report.seeds,
        rows,
        results_sha256,
    })
}

pub fn publish_signal_visibility_ablation(
    output: &Path,
    report: &SignalVisibilityAblationReport,
) -> Result<(), String> {
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable {}",
            output.display()
        ));
    }
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode visibility report: {error}"))?;
    bytes.push(b'\n');
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} needs a UTF-8 file name", output.display()))?;
    let nonce = SIGNAL_VISIBILITY_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
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
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_outcome_order_is_win_timeout_loss() {
        assert!(
            outcome_score(ViabilityOutcome::CandidateWin)
                > outcome_score(ViabilityOutcome::Timeout)
        );
        assert!(
            outcome_score(ViabilityOutcome::Timeout)
                > outcome_score(ViabilityOutcome::CandidateLoss)
        );
    }
}
