//! Offline sensitivity analysis for hypothetical deadline adjudications.
//!
//! This module never changes the canonical outcome stored in a control matrix.
//! It applies an explicit, hash-bound grid of integer compartment weights only
//! to episodes that canonically ended in a deadline draw.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::control_matrix::{
    decode_control_matrix_report, ControlEpisode, ControlMatrixReport, ControlSeat,
    MaintainedMindProfile,
};
use crate::env::EpisodeEndReason;
use crate::telemetry::CellEnergyCompartments;
use crate::viability::ViabilityOutcome;

pub const ADJUDICATION_SENSITIVITY_SCHEMA_VERSION: u32 = 2;
pub const COMPARTMENT_WEIGHT_SCALE: u32 = 10_000;
const MAX_POLICY_FILE_BYTES: u64 = 1024 * 1024;
const MAX_SOURCE_MATRIX_BYTES: u64 = 64 * 1024 * 1024;
static SENSITIVITY_TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct CompartmentWeights {
    pub core_basis_points: u32,
    pub assimilated_basis_points: u32,
    pub gut_basis_points: u32,
    pub carried_material_basis_points: u32,
    pub payload_escrow_basis_points: u32,
}

impl CompartmentWeights {
    fn validate(&self) -> Result<(), String> {
        let scale = COMPARTMENT_WEIGHT_SCALE;
        if [
            self.core_basis_points,
            self.assimilated_basis_points,
            self.gut_basis_points,
            self.carried_material_basis_points,
            self.payload_escrow_basis_points,
        ]
        .into_iter()
        .any(|weight| weight > scale)
        {
            return Err(format!(
                "compartment weights must be in 0..={COMPARTMENT_WEIGHT_SCALE} basis points"
            ));
        }
        if self.core_basis_points == 0
            && self.assimilated_basis_points == 0
            && self.gut_basis_points == 0
            && self.carried_material_basis_points == 0
            && self.payload_escrow_basis_points == 0
        {
            return Err("at least one compartment weight must be positive".into());
        }
        Ok(())
    }

    fn score(&self, energy: &CellEnergyCompartments) -> Result<u128, String> {
        let terms = [
            (energy.core_mass, self.core_basis_points),
            (energy.assimilated_energy, self.assimilated_basis_points),
            (energy.gut_energy, self.gut_basis_points),
            (
                energy.carried_material_mass,
                self.carried_material_basis_points,
            ),
            (energy.payload_escrow, self.payload_escrow_basis_points),
        ];
        terms
            .into_iter()
            .try_fold(0u128, |total, (amount, weight)| {
                amount
                    .checked_mul(u128::from(weight))
                    .and_then(|term| total.checked_add(term))
                    .ok_or_else(|| "weighted terminal score overflows u128".to_string())
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationPolicy {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub weights: CompartmentWeights,
    /// Required lead in unweighted mass-energy units. It is converted to the
    /// fixed basis-point score scale before comparison.
    #[serde(default)]
    pub minimum_victory_margin_mass_energy: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationSensitivitySpec {
    pub policies: Vec<AdjudicationPolicy>,
}

impl AdjudicationSensitivitySpec {
    fn validate(&self) -> Result<(), String> {
        if self.policies.is_empty() {
            return Err("adjudication sensitivity requires at least one policy".into());
        }
        let mut names = HashSet::new();
        let mut configurations = HashSet::new();
        for policy in &self.policies {
            if !valid_name(&policy.name) {
                return Err(format!(
                    "invalid adjudication policy name {}; use lowercase letters, digits, '-' or '_'",
                    policy.name
                ));
            }
            if !names.insert(policy.name.as_str()) {
                return Err(format!(
                    "duplicate adjudication policy name {}",
                    policy.name
                ));
            }
            if policy
                .description
                .as_ref()
                .is_some_and(|description| description.trim().is_empty())
            {
                return Err(format!("policy {} has an empty description", policy.name));
            }
            policy.weights.validate()?;
            if !configurations.insert((&policy.weights, policy.minimum_victory_margin_mass_energy))
            {
                return Err(format!(
                    "policy {} duplicates another policy's weights and margin",
                    policy.name
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WeightedDeadlineOutcome {
    ColonyWin,
    ControlWin,
    Draw,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationAggregate {
    pub episodes: usize,
    pub extermination_colony_wins: usize,
    pub extermination_colony_losses: usize,
    pub deadline_colony_wins: usize,
    pub deadline_colony_losses: usize,
    pub deadline_draws: usize,
    pub hypothetical_colony_wins: usize,
    pub hypothetical_colony_losses: usize,
    pub hypothetical_draws: usize,
}

impl AdjudicationAggregate {
    fn observe_extermination(&mut self, outcome: ViabilityOutcome) -> Result<(), String> {
        self.episodes += 1;
        match outcome {
            ViabilityOutcome::CandidateWin => {
                self.extermination_colony_wins += 1;
                self.hypothetical_colony_wins += 1;
            }
            ViabilityOutcome::CandidateLoss => {
                self.extermination_colony_losses += 1;
                self.hypothetical_colony_losses += 1;
            }
            ViabilityOutcome::Timeout => {
                return Err("extermination episode is marked as a timeout".into());
            }
        }
        Ok(())
    }

    fn observe_deadline(&mut self, outcome: WeightedDeadlineOutcome) {
        self.episodes += 1;
        match outcome {
            WeightedDeadlineOutcome::ColonyWin => {
                self.deadline_colony_wins += 1;
                self.hypothetical_colony_wins += 1;
            }
            WeightedDeadlineOutcome::ControlWin => {
                self.deadline_colony_losses += 1;
                self.hypothetical_colony_losses += 1;
            }
            WeightedDeadlineOutcome::Draw => {
                self.deadline_draws += 1;
                self.hypothetical_draws += 1;
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdjudicatedDeadlineEpisode {
    pub variant: String,
    pub opponent: MaintainedMindProfile,
    pub seed: u64,
    pub seat: ControlSeat,
    pub sim_time_quanta: u64,
    /// Exact weighted score with `COMPARTMENT_WEIGHT_SCALE` units per fully
    /// counted mass-energy unit.
    pub colony_score_basis_units: u128,
    pub control_score_basis_units: u128,
    pub outcome: WeightedDeadlineOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationGroup {
    pub variant: String,
    pub opponent: MaintainedMindProfile,
    pub sim_time_limit_quanta: u64,
    pub aggregate: AdjudicationAggregate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationPolicyReport {
    pub name: String,
    pub description: Option<String>,
    pub weights: CompartmentWeights,
    pub minimum_victory_margin_mass_energy: u64,
    pub overall: AdjudicationAggregate,
    pub groups: Vec<AdjudicationGroup>,
    pub deadline_episodes: Vec<AdjudicatedDeadlineEpisode>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdjudicationSensitivityReport {
    pub schema_version: u32,
    pub report_kind: String,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub source_report_sha256: String,
    pub source_matrix_config_hash: String,
    pub policy_spec_sha256: String,
    pub expanded_policy_sha256: String,
    pub analysis_config_sha256: String,
    pub weight_scale: u32,
    pub verification_scope: String,
    pub server_verified: bool,
    pub replay_committed: bool,
    pub source_episodes: usize,
    pub source_deadline_draws: usize,
    pub policies: Vec<AdjudicationPolicyReport>,
}

#[derive(Serialize)]
struct AnalysisIdentity<'a> {
    schema_version: u32,
    source_report_sha256: &'a str,
    policy_spec_sha256: &'a str,
    expanded_policy_sha256: &'a str,
    weight_scale: u32,
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().enumerate().all(|(index, byte)| match byte {
            b'a'..=b'z' | b'0'..=b'9' => true,
            b'-' | b'_' => index > 0,
            _ => false,
        })
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hash_json(value: &impl Serialize) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256(&bytes))
        .map_err(|error| format!("failed to encode adjudication identity: {error}"))
}

fn load_spec(path: &Path) -> Result<(AdjudicationSensitivitySpec, String), String> {
    let bytes = crate::artifact_io::read_bounded(path, MAX_POLICY_FILE_BYTES)?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("policy file is not UTF-8: {error}"))?;
    let spec: AdjudicationSensitivitySpec = toml::from_str(text)
        .map_err(|error| format!("failed to parse adjudication policies: {error}"))?;
    spec.validate()?;
    Ok((spec, sha256(&bytes)))
}

fn deadline_outcome(
    policy: &AdjudicationPolicy,
    episode: &ControlEpisode,
) -> Result<(u128, u128, WeightedDeadlineOutcome), String> {
    let colony = policy.weights.score(&episode.terminal_colony_energy)?;
    let control = policy.weights.score(&episode.terminal_control_energy)?;
    let margin = u128::from(policy.minimum_victory_margin_mass_energy)
        .checked_mul(u128::from(COMPARTMENT_WEIGHT_SCALE))
        .ok_or_else(|| "minimum victory margin overflows u128".to_string())?;
    Ok((colony, control, compare_scores(colony, control, margin)))
}

fn compare_scores(colony: u128, control: u128, margin: u128) -> WeightedDeadlineOutcome {
    if colony > control && colony - control >= margin {
        WeightedDeadlineOutcome::ColonyWin
    } else if control > colony && control - colony >= margin {
        WeightedDeadlineOutcome::ControlWin
    } else {
        WeightedDeadlineOutcome::Draw
    }
}

fn analyze_policy(
    source: &ControlMatrixReport,
    policy: &AdjudicationPolicy,
) -> Result<AdjudicationPolicyReport, String> {
    let mut overall = AdjudicationAggregate::default();
    let mut groups = Vec::with_capacity(source.matchups.len());
    let mut deadline_episodes = Vec::new();
    for matchup in &source.matchups {
        let mut aggregate = AdjudicationAggregate::default();
        for episode in &matchup.report.episodes {
            match episode.end_reason {
                EpisodeEndReason::Extermination => {
                    aggregate.observe_extermination(episode.outcome)?;
                    overall.observe_extermination(episode.outcome)?;
                }
                EpisodeEndReason::SimTimeDeadline => {
                    let (colony_score, control_score, outcome) = deadline_outcome(policy, episode)?;
                    aggregate.observe_deadline(outcome);
                    overall.observe_deadline(outcome);
                    deadline_episodes.push(AdjudicatedDeadlineEpisode {
                        variant: matchup.variant.clone(),
                        opponent: matchup.report.opponent,
                        seed: episode.seed,
                        seat: episode.seat,
                        sim_time_quanta: episode.final_sim_time_quanta,
                        colony_score_basis_units: colony_score,
                        control_score_basis_units: control_score,
                        outcome,
                    });
                }
                EpisodeEndReason::DecisionFrontierSafetyLimit => {
                    return Err("source report contains a host safety abort".into());
                }
            }
        }
        groups.push(AdjudicationGroup {
            variant: matchup.variant.clone(),
            opponent: matchup.report.opponent,
            sim_time_limit_quanta: matchup.report.scenario.victory.sim_time_limit_quanta,
            aggregate,
        });
    }
    Ok(AdjudicationPolicyReport {
        name: policy.name.clone(),
        description: policy.description.clone(),
        weights: policy.weights.clone(),
        minimum_victory_margin_mass_energy: policy.minimum_victory_margin_mass_energy,
        overall,
        groups,
        deadline_episodes,
    })
}

pub fn analyze_adjudication_sensitivity(
    matrix_file: &Path,
    policy_file: &Path,
) -> Result<AdjudicationSensitivityReport, String> {
    let matrix_bytes = crate::artifact_io::read_bounded(matrix_file, MAX_SOURCE_MATRIX_BYTES)?;
    let source_report_sha256 = sha256(&matrix_bytes);
    let source = decode_control_matrix_report(&matrix_bytes)
        .map_err(|error| format!("failed to decode {}: {error}", matrix_file.display()))?;
    let (spec, policy_spec_sha256) = load_spec(policy_file)?;
    let expanded_policy_sha256 = hash_json(&spec.policies)?;
    let analysis_config_sha256 = hash_json(&AnalysisIdentity {
        schema_version: ADJUDICATION_SENSITIVITY_SCHEMA_VERSION,
        source_report_sha256: &source_report_sha256,
        policy_spec_sha256: &policy_spec_sha256,
        expanded_policy_sha256: &expanded_policy_sha256,
        weight_scale: COMPARTMENT_WEIGHT_SCALE,
    })?;
    let source_episodes = source
        .matchups
        .iter()
        .map(|matchup| matchup.report.episodes.len())
        .sum();
    let source_deadline_draws = source
        .matchups
        .iter()
        .flat_map(|matchup| &matchup.report.episodes)
        .filter(|episode| episode.end_reason == EpisodeEndReason::SimTimeDeadline)
        .count();
    let policies = spec
        .policies
        .iter()
        .map(|policy| analyze_policy(&source, policy))
        .collect::<Result<Vec<_>, _>>()?;
    let report = AdjudicationSensitivityReport {
        schema_version: ADJUDICATION_SENSITIVITY_SCHEMA_VERSION,
        report_kind: "blob_adjudication_sensitivity".into(),
        package_version: env!("CARGO_PKG_VERSION").into(),
        code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
        source_report_sha256,
        source_matrix_config_hash: source.matrix_config_hash,
        policy_spec_sha256,
        expanded_policy_sha256,
        analysis_config_sha256,
        weight_scale: COMPARTMENT_WEIGHT_SCALE,
        verification_scope: "local_counterfactual".into(),
        server_verified: false,
        replay_committed: false,
        source_episodes,
        source_deadline_draws,
        policies,
    };
    validate_adjudication_sensitivity_report(&report)?;
    Ok(report)
}

pub fn validate_adjudication_sensitivity_report(
    report: &AdjudicationSensitivityReport,
) -> Result<(), String> {
    if report.schema_version != ADJUDICATION_SENSITIVITY_SCHEMA_VERSION
        || report.report_kind != "blob_adjudication_sensitivity"
        || report.weight_scale != COMPARTMENT_WEIGHT_SCALE
        || report.verification_scope != "local_counterfactual"
        || report.server_verified
        || report.replay_committed
        || report.policies.is_empty()
    {
        return Err("adjudication sensitivity identity or trust scope is invalid".into());
    }
    let expected_hash = hash_json(&AnalysisIdentity {
        schema_version: report.schema_version,
        source_report_sha256: &report.source_report_sha256,
        policy_spec_sha256: &report.policy_spec_sha256,
        expanded_policy_sha256: &report.expanded_policy_sha256,
        weight_scale: report.weight_scale,
    })?;
    if report.analysis_config_sha256 != expected_hash
        || !is_sha256(&report.source_report_sha256)
        || !is_sha256(&report.source_matrix_config_hash)
        || !is_sha256(&report.policy_spec_sha256)
        || !is_sha256(&report.expanded_policy_sha256)
        || !is_sha256(&report.analysis_config_sha256)
    {
        return Err("adjudication sensitivity configuration hash mismatch".into());
    }
    let expanded_policies = report
        .policies
        .iter()
        .map(|policy| AdjudicationPolicy {
            name: policy.name.clone(),
            description: policy.description.clone(),
            weights: policy.weights.clone(),
            minimum_victory_margin_mass_energy: policy.minimum_victory_margin_mass_energy,
        })
        .collect::<Vec<_>>();
    if hash_json(&expanded_policies)? != report.expanded_policy_sha256 {
        return Err("expanded adjudication policies do not match their hash".into());
    }
    let mut names = HashSet::new();
    for policy in &report.policies {
        if !names.insert(policy.name.as_str()) || !valid_name(&policy.name) {
            return Err("adjudication sensitivity policy names are invalid".into());
        }
        policy.weights.validate()?;
        if policy
            .groups
            .iter()
            .map(|group| group.aggregate.episodes)
            .sum::<usize>()
            != report.source_episodes
            || policy.deadline_episodes.len() != report.source_deadline_draws
            || policy.overall.episodes != report.source_episodes
            || policy.overall.deadline_colony_wins
                + policy.overall.deadline_colony_losses
                + policy.overall.deadline_draws
                != report.source_deadline_draws
            || policy.overall.hypothetical_colony_wins
                + policy.overall.hypothetical_colony_losses
                + policy.overall.hypothetical_draws
                != report.source_episodes
        {
            return Err(format!(
                "policy {} aggregates are inconsistent",
                policy.name
            ));
        }
        let mut reconstructed = AdjudicationAggregate::default();
        let mut groups = HashMap::new();
        for group in &policy.groups {
            validate_aggregate(&group.aggregate)?;
            if group.sim_time_limit_quanta == 0
                || groups
                    .insert((group.variant.as_str(), group.opponent), &group.aggregate)
                    .is_some()
            {
                return Err(format!("policy {} groups are invalid", policy.name));
            }
            merge_aggregate(&mut reconstructed, &group.aggregate);
        }
        if reconstructed != policy.overall {
            return Err(format!(
                "policy {} overall total is inconsistent",
                policy.name
            ));
        }
        validate_aggregate(&policy.overall)?;
        let mut deadline_keys = HashSet::new();
        let margin = u128::from(policy.minimum_victory_margin_mass_energy)
            .checked_mul(u128::from(COMPARTMENT_WEIGHT_SCALE))
            .ok_or_else(|| "minimum victory margin overflows u128".to_string())?;
        let mut deadline_counts: HashMap<(&str, MaintainedMindProfile), (usize, usize, usize)> =
            HashMap::new();
        for episode in &policy.deadline_episodes {
            if !deadline_keys.insert((
                episode.variant.as_str(),
                episode.opponent,
                episode.seed,
                episode.seat,
            )) || !groups.contains_key(&(episode.variant.as_str(), episode.opponent))
            {
                return Err(format!(
                    "policy {} deadline episode identity is invalid",
                    policy.name
                ));
            }
            if compare_scores(
                episode.colony_score_basis_units,
                episode.control_score_basis_units,
                margin,
            ) != episode.outcome
            {
                return Err(format!(
                    "policy {} deadline score outcome is inconsistent",
                    policy.name
                ));
            }
            let counts = deadline_counts
                .entry((episode.variant.as_str(), episode.opponent))
                .or_default();
            match episode.outcome {
                WeightedDeadlineOutcome::ColonyWin => counts.0 += 1,
                WeightedDeadlineOutcome::ControlWin => counts.1 += 1,
                WeightedDeadlineOutcome::Draw => counts.2 += 1,
            }
        }
        for (key, group) in groups {
            let counts = deadline_counts.get(&key).copied().unwrap_or_default();
            if counts
                != (
                    group.deadline_colony_wins,
                    group.deadline_colony_losses,
                    group.deadline_draws,
                )
            {
                return Err(format!(
                    "policy {} deadline episode counts are inconsistent",
                    policy.name
                ));
            }
        }
    }
    Ok(())
}

fn validate_aggregate(aggregate: &AdjudicationAggregate) -> Result<(), String> {
    if aggregate.extermination_colony_wins
        + aggregate.extermination_colony_losses
        + aggregate.deadline_colony_wins
        + aggregate.deadline_colony_losses
        + aggregate.deadline_draws
        != aggregate.episodes
        || aggregate.hypothetical_colony_wins
            != aggregate.extermination_colony_wins + aggregate.deadline_colony_wins
        || aggregate.hypothetical_colony_losses
            != aggregate.extermination_colony_losses + aggregate.deadline_colony_losses
        || aggregate.hypothetical_draws != aggregate.deadline_draws
    {
        return Err("adjudication aggregate accounting is invalid".into());
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn merge_aggregate(target: &mut AdjudicationAggregate, source: &AdjudicationAggregate) {
    target.episodes += source.episodes;
    target.extermination_colony_wins += source.extermination_colony_wins;
    target.extermination_colony_losses += source.extermination_colony_losses;
    target.deadline_colony_wins += source.deadline_colony_wins;
    target.deadline_colony_losses += source.deadline_colony_losses;
    target.deadline_draws += source.deadline_draws;
    target.hypothetical_colony_wins += source.hypothetical_colony_wins;
    target.hypothetical_colony_losses += source.hypothetical_colony_losses;
    target.hypothetical_draws += source.hypothetical_draws;
}

pub fn publish_adjudication_sensitivity_report(
    output: &Path,
    report: &AdjudicationSensitivityReport,
) -> Result<PathBuf, String> {
    validate_adjudication_sensitivity_report(report)?;
    let parent = output
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", output.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if output.exists() {
        return Err(format!(
            "refusing to replace immutable adjudication sensitivity report {}",
            output.display()
        ));
    }
    let nonce = SENSITIVITY_TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "adjudication sensitivity output needs a UTF-8 name".to_string())?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{nonce}", std::process::id()));
    let mut bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode adjudication sensitivity: {error}"))?;
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

    fn energy(core: u128, assimilated: u128, gut: u128) -> CellEnergyCompartments {
        CellEnergyCompartments {
            core_mass: core,
            assimilated_energy: assimilated,
            gut_energy: gut,
            carried_material_mass: 0,
            payload_escrow: 0,
        }
    }

    #[test]
    fn exact_weighting_and_margin_keep_gut_assumptions_explicit() {
        let mut episode = ControlEpisode {
            seed: 7,
            seat: ControlSeat::ColonyTeamZero,
            outcome: ViabilityOutcome::Timeout,
            end_reason: EpisodeEndReason::SimTimeDeadline,
            environment_steps: 1,
            final_sim_time_quanta: 1024,
            colony_cells: 1,
            control_cells: 1,
            initial_tracked_mass_energy: 100,
            final_tracked_mass_energy: 100,
            terminal_colony_energy: energy(10, 20, 40),
            terminal_control_energy: energy(10, 25, 0),
        };
        let body = AdjudicationPolicy {
            name: "body".into(),
            description: None,
            weights: CompartmentWeights {
                core_basis_points: 10_000,
                assimilated_basis_points: 10_000,
                gut_basis_points: 0,
                carried_material_basis_points: 0,
                payload_escrow_basis_points: 0,
            },
            minimum_victory_margin_mass_energy: 0,
        };
        assert_eq!(
            deadline_outcome(&body, &episode).unwrap().2,
            WeightedDeadlineOutcome::ControlWin
        );
        let half_gut = AdjudicationPolicy {
            name: "half-gut".into(),
            weights: CompartmentWeights {
                gut_basis_points: 5_000,
                ..body.weights.clone()
            },
            ..body.clone()
        };
        assert_eq!(
            deadline_outcome(&half_gut, &episode).unwrap(),
            (500_000, 350_000, WeightedDeadlineOutcome::ColonyWin)
        );
        episode.terminal_colony_energy = energy(10, 20, 12);
        let margin = AdjudicationPolicy {
            name: "margin".into(),
            minimum_victory_margin_mass_energy: 2,
            ..half_gut
        };
        assert_eq!(
            deadline_outcome(&margin, &episode).unwrap().2,
            WeightedDeadlineOutcome::Draw
        );
    }

    #[test]
    fn policy_spec_rejects_typos_duplicates_and_unbounded_weights() {
        let temporary = tempfile::tempdir().unwrap();
        let typo = temporary.path().join("typo.toml");
        fs::write(
            &typo,
            r#"
            [[policies]]
            name = "body"
            minimum_margin = 1
            [policies.weights]
            core_basis_points = 10000
            assimilated_basis_points = 10000
            gut_basis_points = 0
            carried_material_basis_points = 0
            payload_escrow_basis_points = 0
            "#,
        )
        .unwrap();
        assert!(load_spec(&typo).unwrap_err().contains("minimum_margin"));

        let invalid = temporary.path().join("invalid.toml");
        fs::write(
            &invalid,
            r#"
            [[policies]]
            name = "body"
            [policies.weights]
            core_basis_points = 10001
            assimilated_basis_points = 10000
            gut_basis_points = 0
            carried_material_basis_points = 0
            payload_escrow_basis_points = 0
            "#,
        )
        .unwrap();
        assert!(load_spec(&invalid).unwrap_err().contains("0..=10000"));
    }

    #[test]
    fn report_validation_detects_tampering_and_publication_is_immutable() {
        let aggregate = AdjudicationAggregate {
            episodes: 1,
            deadline_colony_wins: 1,
            hypothetical_colony_wins: 1,
            ..AdjudicationAggregate::default()
        };
        let source_report_sha256 = "11".repeat(32);
        let policy_spec_sha256 = "22".repeat(32);
        let expanded_policy = AdjudicationPolicy {
            name: "body".into(),
            description: None,
            weights: CompartmentWeights {
                core_basis_points: 10_000,
                assimilated_basis_points: 10_000,
                gut_basis_points: 0,
                carried_material_basis_points: 0,
                payload_escrow_basis_points: 0,
            },
            minimum_victory_margin_mass_energy: 0,
        };
        let expanded_policy_sha256 = hash_json(&vec![expanded_policy.clone()]).unwrap();
        let analysis_config_sha256 = hash_json(&AnalysisIdentity {
            schema_version: ADJUDICATION_SENSITIVITY_SCHEMA_VERSION,
            source_report_sha256: &source_report_sha256,
            policy_spec_sha256: &policy_spec_sha256,
            expanded_policy_sha256: &expanded_policy_sha256,
            weight_scale: COMPARTMENT_WEIGHT_SCALE,
        })
        .unwrap();
        let report = AdjudicationSensitivityReport {
            schema_version: ADJUDICATION_SENSITIVITY_SCHEMA_VERSION,
            report_kind: "blob_adjudication_sensitivity".into(),
            package_version: "test".into(),
            code_revision: None,
            source_report_sha256,
            source_matrix_config_hash: "33".repeat(32),
            policy_spec_sha256,
            expanded_policy_sha256,
            analysis_config_sha256,
            weight_scale: COMPARTMENT_WEIGHT_SCALE,
            verification_scope: "local_counterfactual".into(),
            server_verified: false,
            replay_committed: false,
            source_episodes: 1,
            source_deadline_draws: 1,
            policies: vec![AdjudicationPolicyReport {
                name: expanded_policy.name,
                description: expanded_policy.description,
                weights: expanded_policy.weights,
                minimum_victory_margin_mass_energy: expanded_policy
                    .minimum_victory_margin_mass_energy,
                overall: aggregate.clone(),
                groups: vec![AdjudicationGroup {
                    variant: "horizon".into(),
                    opponent: MaintainedMindProfile::Simple,
                    sim_time_limit_quanta: 1024,
                    aggregate,
                }],
                deadline_episodes: vec![AdjudicatedDeadlineEpisode {
                    variant: "horizon".into(),
                    opponent: MaintainedMindProfile::Simple,
                    seed: 7,
                    seat: ControlSeat::ColonyTeamZero,
                    sim_time_quanta: 1024,
                    colony_score_basis_units: 200_000,
                    control_score_basis_units: 100_000,
                    outcome: WeightedDeadlineOutcome::ColonyWin,
                }],
            }],
        };
        validate_adjudication_sensitivity_report(&report).unwrap();
        let mut tampered = report.clone();
        tampered.policies[0].deadline_episodes[0].outcome = WeightedDeadlineOutcome::ControlWin;
        assert!(validate_adjudication_sensitivity_report(&tampered)
            .unwrap_err()
            .contains("deadline"));

        let temporary = tempfile::tempdir().unwrap();
        let output = temporary.path().join("sensitivity.json");
        assert_eq!(
            publish_adjudication_sensitivity_report(&output, &report).unwrap(),
            output
        );
        assert!(publish_adjudication_sensitivity_report(&output, &report)
            .unwrap_err()
            .contains("refusing to replace"));
    }
}
