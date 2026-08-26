//! Validated paired analysis of native colony signaling-policy controls.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::control_matrix::{
    decode_control_matrix_report, ControlMatrixReport, ControlSeat, MaintainedMindProfile,
};
use crate::sweep::sha256;
use crate::viability::ViabilityOutcome;

pub const SIGNAL_POLICY_ABLATION_SCHEMA_VERSION: u32 = 1;
static SIGNAL_POLICY_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

const EXPECTED_CANDIDATES: [MaintainedMindProfile; 5] = [
    MaintainedMindProfile::ColonySignalDisabled,
    MaintainedMindProfile::ColonySignalOneQuantum,
    MaintainedMindProfile::ColonySignalSidecar,
    MaintainedMindProfile::ColonySignalCadenced,
    MaintainedMindProfile::Colony,
];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SignalPolicySource {
    pub candidate: MaintainedMindProfile,
    pub report_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PairedOutcomeComparison {
    pub improved: usize,
    pub worsened: usize,
    pub unchanged: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SignalPolicyAblationRow {
    pub candidate: MaintainedMindProfile,
    pub episodes: usize,
    pub wins: usize,
    pub losses: usize,
    pub timeouts: usize,
    pub explicit_signal_actions: u64,
    pub sidecar_emissions: u64,
    pub signal_energy: u128,
    pub signal_energy_decayed: u128,
    pub signal_energy_erased_by_terrain: u128,
    pub completed_actions: u64,
    pub mean_environment_signal_energy: f64,
    pub mean_observable_signal_variation: f64,
    pub paired_vs_disabled: PairedOutcomeComparison,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SignalPolicyAblationReport {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub manifest_sha256: String,
    pub mind_abi_version: u16,
    pub mind_abi_hash: String,
    pub verification_scope: String,
    pub server_verified: bool,
    pub replay_committed: bool,
    pub opponents: Vec<MaintainedMindProfile>,
    pub seeds: Vec<u64>,
    pub sources: Vec<SignalPolicySource>,
    pub rows: Vec<SignalPolicyAblationRow>,
    pub results_sha256: String,
}

type EpisodeKey = (String, MaintainedMindProfile, u64, ControlSeat);

fn outcome_score(outcome: ViabilityOutcome) -> u8 {
    match outcome {
        ViabilityOutcome::CandidateWin => 2,
        ViabilityOutcome::Timeout => 1,
        ViabilityOutcome::CandidateLoss => 0,
    }
}

fn episode_outcomes(report: &ControlMatrixReport) -> HashMap<EpisodeKey, ViabilityOutcome> {
    report
        .matchups
        .iter()
        .flat_map(|matchup| {
            matchup.report.episodes.iter().map(|episode| {
                (
                    (
                        matchup.variant.clone(),
                        matchup.report.opponent,
                        episode.seed,
                        episode.seat,
                    ),
                    episode.outcome,
                )
            })
        })
        .collect()
}

fn completed_actions(report: &ControlMatrixReport) -> u64 {
    report
        .matchups
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

fn row(
    report: &ControlMatrixReport,
    disabled: &HashMap<EpisodeKey, ViabilityOutcome>,
) -> Result<SignalPolicyAblationRow, String> {
    let outcomes = episode_outcomes(report);
    if outcomes.len() != disabled.len() || outcomes.keys().any(|key| !disabled.contains_key(key)) {
        return Err(format!(
            "candidate {} does not contain the disabled profile's paired episodes",
            report.candidate
        ));
    }
    let mut paired = PairedOutcomeComparison {
        improved: 0,
        worsened: 0,
        unchanged: 0,
    };
    for (key, outcome) in &outcomes {
        match outcome_score(*outcome).cmp(&outcome_score(disabled[key])) {
            std::cmp::Ordering::Greater => paired.improved += 1,
            std::cmp::Ordering::Less => paired.worsened += 1,
            std::cmp::Ordering::Equal => paired.unchanged += 1,
        }
    }
    let state_samples = report
        .matchups
        .iter()
        .map(|matchup| matchup.report.telemetry.state_samples)
        .sum::<u64>();
    let weighted_mean = |value: fn(&crate::telemetry::TelemetrySampleMeans) -> f64| {
        if state_samples == 0 {
            0.0
        } else {
            report
                .matchups
                .iter()
                .map(|matchup| {
                    value(&matchup.report.telemetry.sample_means)
                        * matchup.report.telemetry.state_samples as f64
                })
                .sum::<f64>()
                / state_samples as f64
        }
    };
    Ok(SignalPolicyAblationRow {
        candidate: report.candidate,
        episodes: outcomes.len(),
        wins: report
            .summaries
            .iter()
            .map(|summary| summary.colony_wins)
            .sum(),
        losses: report
            .summaries
            .iter()
            .map(|summary| summary.colony_losses)
            .sum(),
        timeouts: report
            .summaries
            .iter()
            .map(|summary| summary.timeouts)
            .sum(),
        explicit_signal_actions: report
            .matchups
            .iter()
            .map(|matchup| {
                matchup
                    .report
                    .telemetry
                    .training
                    .signals
                    .explicit_signal_actions
            })
            .sum(),
        sidecar_emissions: report
            .matchups
            .iter()
            .map(|matchup| matchup.report.telemetry.training.signals.sidecar_emissions)
            .sum(),
        signal_energy: report
            .matchups
            .iter()
            .flat_map(|matchup| matchup.report.telemetry.training.signals.channel_energy)
            .sum(),
        signal_energy_decayed: report
            .matchups
            .iter()
            .flat_map(|matchup| matchup.report.telemetry.signal_field.decayed_energy)
            .sum(),
        signal_energy_erased_by_terrain: report
            .matchups
            .iter()
            .flat_map(|matchup| matchup.report.telemetry.signal_field.terrain_erased_energy)
            .sum(),
        completed_actions: completed_actions(report),
        mean_environment_signal_energy: weighted_mean(|means| means.environment_signal_energy),
        mean_observable_signal_variation: weighted_mean(|means| {
            means.signal_observation_total_variation
        }),
        paired_vs_disabled: paired,
    })
}

pub fn analyze_signal_policy_reports(
    paths: &[PathBuf],
) -> Result<SignalPolicyAblationReport, String> {
    if paths.len() != EXPECTED_CANDIDATES.len() {
        return Err(format!(
            "signal-policy ablation requires {} reports",
            EXPECTED_CANDIDATES.len()
        ));
    }
    let mut loaded = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let report = decode_control_matrix_report(&bytes)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        loaded.push((report, sha256(&bytes)));
    }
    loaded.sort_by_key(|(report, _)| {
        EXPECTED_CANDIDATES
            .iter()
            .position(|candidate| *candidate == report.candidate)
            .unwrap_or(usize::MAX)
    });
    let candidates = loaded
        .iter()
        .map(|(report, _)| report.candidate)
        .collect::<Vec<_>>();
    if candidates != EXPECTED_CANDIDATES {
        return Err(
            "signal-policy reports do not contain each expected candidate exactly once".into(),
        );
    }
    let baseline = &loaded[0].0;
    if loaded.iter().skip(1).any(|(report, _)| {
        report.manifest_sha256 != baseline.manifest_sha256
            || report.mind_abi_version != baseline.mind_abi_version
            || report.mind_abi_hash != baseline.mind_abi_hash
            || report.opponents != baseline.opponents
            || report.seeds != baseline.seeds
            || report.variants != baseline.variants
            || report.verification_scope != baseline.verification_scope
            || report.server_verified != baseline.server_verified
            || report.replay_committed != baseline.replay_committed
    }) {
        return Err("signal-policy reports do not share one evaluation identity".into());
    }
    let disabled = episode_outcomes(baseline);
    let rows = loaded
        .iter()
        .map(|(report, _)| row(report, &disabled))
        .collect::<Result<Vec<_>, _>>()?;
    let sources = loaded
        .iter()
        .map(|(report, hash)| SignalPolicySource {
            candidate: report.candidate,
            report_sha256: hash.clone(),
        })
        .collect::<Vec<_>>();
    let results_sha256 = sha256(
        &serde_json::to_vec(&rows)
            .map_err(|error| format!("failed to hash signal-policy rows: {error}"))?,
    );
    Ok(SignalPolicyAblationReport {
        schema_version: SIGNAL_POLICY_ABLATION_SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION").into(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        manifest_sha256: baseline.manifest_sha256.clone(),
        mind_abi_version: baseline.mind_abi_version,
        mind_abi_hash: baseline.mind_abi_hash.clone(),
        verification_scope: "local_deterministic".into(),
        server_verified: false,
        replay_committed: false,
        opponents: baseline.opponents.clone(),
        seeds: baseline.seeds.clone(),
        sources,
        rows,
        results_sha256,
    })
}

pub fn publish_signal_policy_ablation(
    output: &Path,
    report: &SignalPolicyAblationReport,
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
        .map_err(|error| format!("failed to encode signal-policy report: {error}"))?;
    bytes.push(b'\n');
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} needs a UTF-8 file name", output.display()))?;
    let nonce = SIGNAL_POLICY_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
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

    #[test]
    fn analysis_requires_the_complete_policy_family() {
        assert!(analyze_signal_policy_reports(&[])
            .unwrap_err()
            .contains("requires 5 reports"));
    }
}
