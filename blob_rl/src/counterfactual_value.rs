//! Explicit multi-objective interpretation of immutable counterfactual branches.
//!
//! This layer never changes physics and never enters a Mind input. It turns
//! future branch outcomes into a declared lexicographic Pareto comparison. A
//! conflict between metrics or continuation controllers remains incomparable;
//! it is never collapsed into a convenient scalar label.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::action::{decompose_policy_action, PolicyChoice};
use crate::counterfactual_branch::{
    BranchContinuation, BranchHorizonResult, CounterfactualBranchArtifact, CounterfactualState,
};
use crate::sweep::sha256;

pub const COUNTERFACTUAL_VALUE_SCHEMA_VERSION: u32 = 3;
const MAX_VALUE_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_WEIGHT_PPM: u32 = 1_000_000;
static TEMP_NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ValuePerspective {
    Cell,
    Colony,
    Conservative,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EnergyValuation {
    pub core_weight_ppm: u32,
    pub assimilated_weight_ppm: u32,
    pub gut_weight_ppm: u32,
}

impl Default for EnergyValuation {
    fn default() -> Self {
        Self {
            core_weight_ppm: MAX_WEIGHT_PPM,
            assimilated_weight_ppm: MAX_WEIGHT_PPM,
            gut_weight_ppm: 0,
        }
    }
}

impl EnergyValuation {
    fn validate(self) -> Result<(), String> {
        if self.core_weight_ppm > MAX_WEIGHT_PPM
            || self.assimilated_weight_ppm > MAX_WEIGHT_PPM
            || self.gut_weight_ppm > MAX_WEIGHT_PPM
            || self.core_weight_ppm == 0
                && self.assimilated_weight_ppm == 0
                && self.gut_weight_ppm == 0
        {
            return Err("counterfactual energy valuation is invalid".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualValueOptions {
    pub perspectives: Vec<ValuePerspective>,
    pub horizon_quanta: Vec<u64>,
    pub continuations: Vec<BranchContinuation>,
    pub energy: EnergyValuation,
}

impl CounterfactualValueOptions {
    pub fn validate_against(&self, source: &CounterfactualBranchArtifact) -> Result<(), String> {
        self.energy.validate()?;
        if self.perspectives.is_empty()
            || self
                .perspectives
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                != self.perspectives.len()
            || self.horizon_quanta.is_empty()
            || self
                .horizon_quanta
                .iter()
                .any(|horizon| !source.options.horizon_quanta.contains(horizon))
            || self
                .horizon_quanta
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                != self.horizon_quanta.len()
            || self.continuations.is_empty()
            || self
                .continuations
                .iter()
                .any(|continuation| !source.options.continuations.contains(continuation))
            || self
                .continuations
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                != self.continuations.len()
        {
            return Err(
                "counterfactual value options are invalid or absent from the source".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ValuePreference {
    Teacher,
    Policy,
    Tie,
    Incomparable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContinuationPreference {
    pub continuation: BranchContinuation,
    pub preference: ValuePreference,
}

/// Target-free physical action identity. It can supervise an action family and
/// effort without choosing a direction from hidden future state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(deny_unknown_fields)]
pub struct ActionArchetype {
    pub kind: usize,
    pub effort: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StateValueVerdict {
    pub seed: u64,
    pub state_ordinal: usize,
    pub continuation_preferences: Vec<ContinuationPreference>,
    pub robust_preference: ValuePreference,
    /// Source-order stable actions not robustly dominated by another evaluated
    /// candidate under every selected continuation.
    pub pareto_frontier: Vec<PolicyChoice>,
    pub pareto_archetypes: Vec<ActionArchetype>,
    pub dominated_candidates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SeedValueVerdict {
    pub seed: u64,
    pub selected_states: usize,
    pub continuation_preferences: Vec<ContinuationPreference>,
    pub robust_preference: ValuePreference,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreferenceCounts {
    pub teacher: usize,
    pub policy: usize,
    pub ties: usize,
    pub incomparable: usize,
}

impl PreferenceCounts {
    fn observe(&mut self, preference: ValuePreference) {
        match preference {
            ValuePreference::Teacher => self.teacher += 1,
            ValuePreference::Policy => self.policy += 1,
            ValuePreference::Tie => self.ties += 1,
            ValuePreference::Incomparable => self.incomparable += 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ValueEvaluation {
    pub perspective: ValuePerspective,
    pub requested_elapsed_quanta: u64,
    pub state_counts: PreferenceCounts,
    pub seed_counts: PreferenceCounts,
    pub states: Vec<StateValueVerdict>,
    pub seeds: Vec<SeedValueVerdict>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualValueReport {
    pub schema_version: u32,
    pub evaluations: Vec<ValueEvaluation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualValueArtifact {
    pub schema_version: u32,
    pub package_version: String,
    pub code_revision: Option<String>,
    pub artifact_hash: String,
    pub source_counterfactual_hash: String,
    pub options: CounterfactualValueOptions,
    pub report: CounterfactualValueReport,
}

#[derive(Serialize)]
struct ArtifactIdentity<'a> {
    schema_version: u32,
    package_version: &'a str,
    code_revision: &'a Option<String>,
    source_counterfactual_hash: &'a str,
    options: &'a CounterfactualValueOptions,
    report: &'a CounterfactualValueReport,
}

impl CounterfactualValueArtifact {
    pub fn new(
        source: &CounterfactualBranchArtifact,
        options: CounterfactualValueOptions,
    ) -> Result<Self, String> {
        source.validate()?;
        options.validate_against(source)?;
        let report = evaluate_counterfactual_values(source, &options)?;
        let mut artifact = Self {
            schema_version: COUNTERFACTUAL_VALUE_SCHEMA_VERSION,
            package_version: env!("CARGO_PKG_VERSION").into(),
            code_revision: option_env!("BLOB_CODE_REVISION").map(str::to_string),
            artifact_hash: String::new(),
            source_counterfactual_hash: source.artifact_hash.clone(),
            options,
            report,
        };
        artifact.artifact_hash = artifact.recompute_hash()?;
        artifact.validate_against(source)?;
        Ok(artifact)
    }

    fn recompute_hash(&self) -> Result<String, String> {
        serde_json::to_vec(&ArtifactIdentity {
            schema_version: self.schema_version,
            package_version: &self.package_version,
            code_revision: &self.code_revision,
            source_counterfactual_hash: &self.source_counterfactual_hash,
            options: &self.options,
            report: &self.report,
        })
        .map(|bytes| sha256(&bytes))
        .map_err(|error| format!("failed to hash counterfactual value artifact: {error}"))
    }

    pub fn validate_against(&self, source: &CounterfactualBranchArtifact) -> Result<(), String> {
        source.validate()?;
        self.options.validate_against(source)?;
        if self.schema_version != COUNTERFACTUAL_VALUE_SCHEMA_VERSION
            || self.package_version.is_empty()
            || self.source_counterfactual_hash != source.artifact_hash
            || !valid_sha256(&self.artifact_hash)
            || self.report != evaluate_counterfactual_values(source, &self.options)?
            || self.artifact_hash != self.recompute_hash()?
        {
            return Err("counterfactual value artifact identity or report is invalid".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct ValueVector {
    target_survival: u128,
    team_population: u128,
    target_energy: u128,
    team_energy: u128,
}

impl ValueVector {
    fn add(&mut self, other: Self) -> Result<(), String> {
        self.target_survival = checked_add(self.target_survival, other.target_survival)?;
        self.team_population = checked_add(self.team_population, other.team_population)?;
        self.target_energy = checked_add(self.target_energy, other.target_energy)?;
        self.team_energy = checked_add(self.team_energy, other.team_energy)?;
        Ok(())
    }
}

fn checked_add(left: u128, right: u128) -> Result<u128, String> {
    left.checked_add(right)
        .ok_or_else(|| "counterfactual value aggregation overflowed".into())
}

fn weighted_energy(
    core: u128,
    assimilated: u128,
    gut: u128,
    weights: EnergyValuation,
) -> Result<u128, String> {
    let core = core
        .checked_mul(u128::from(weights.core_weight_ppm))
        .ok_or("counterfactual core value overflowed")?;
    let assimilated = assimilated
        .checked_mul(u128::from(weights.assimilated_weight_ppm))
        .ok_or("counterfactual assimilated value overflowed")?;
    let gut = gut
        .checked_mul(u128::from(weights.gut_weight_ppm))
        .ok_or("counterfactual gut value overflowed")?;
    checked_add(checked_add(core, assimilated)?, gut)
}

fn value_vector(
    horizon: &BranchHorizonResult,
    weights: EnergyValuation,
) -> Result<ValueVector, String> {
    Ok(ValueVector {
        target_survival: u128::from(horizon.target_alive),
        team_population: horizon.training_cells as u128,
        target_energy: weighted_energy(
            u128::from(horizon.target_core_mass),
            u128::from(horizon.target_assimilated_energy),
            u128::from(horizon.target_gut_energy),
            weights,
        )?,
        team_energy: weighted_energy(
            horizon.training_core_mass,
            horizon.training_assimilated_energy,
            horizon.training_gut_energy,
            weights,
        )?,
    })
}

fn pareto_tier(pairs: &[(u128, u128)]) -> Option<ValuePreference> {
    let teacher_better = pairs.iter().any(|(teacher, policy)| teacher > policy);
    let policy_better = pairs.iter().any(|(teacher, policy)| policy > teacher);
    match (teacher_better, policy_better) {
        (false, false) => None,
        (true, false) => Some(ValuePreference::Teacher),
        (false, true) => Some(ValuePreference::Policy),
        (true, true) => Some(ValuePreference::Incomparable),
    }
}

fn compare_vectors(
    teacher: ValueVector,
    policy: ValueVector,
    perspective: ValuePerspective,
) -> ValuePreference {
    let survival = match perspective {
        ValuePerspective::Cell => pareto_tier(&[(teacher.target_survival, policy.target_survival)]),
        ValuePerspective::Colony => {
            pareto_tier(&[(teacher.team_population, policy.team_population)])
        }
        ValuePerspective::Conservative => pareto_tier(&[
            (teacher.target_survival, policy.target_survival),
            (teacher.team_population, policy.team_population),
        ]),
    };
    if let Some(preference) = survival {
        return preference;
    }
    let energy = match perspective {
        ValuePerspective::Cell => pareto_tier(&[(teacher.target_energy, policy.target_energy)]),
        ValuePerspective::Colony => pareto_tier(&[(teacher.team_energy, policy.team_energy)]),
        ValuePerspective::Conservative => pareto_tier(&[
            (teacher.target_energy, policy.target_energy),
            (teacher.team_energy, policy.team_energy),
        ]),
    };
    energy.unwrap_or(ValuePreference::Tie)
}

fn robust_preference(preferences: &[ContinuationPreference]) -> ValuePreference {
    if preferences
        .iter()
        .any(|item| item.preference == ValuePreference::Incomparable)
    {
        return ValuePreference::Incomparable;
    }
    let teacher = preferences
        .iter()
        .any(|item| item.preference == ValuePreference::Teacher);
    let policy = preferences
        .iter()
        .any(|item| item.preference == ValuePreference::Policy);
    match (teacher, policy) {
        (true, false) => ValuePreference::Teacher,
        (false, true) => ValuePreference::Policy,
        (false, false) => ValuePreference::Tie,
        (true, true) => ValuePreference::Incomparable,
    }
}

fn normalized(choice: PolicyChoice) -> PolicyChoice {
    PolicyChoice {
        action: choice.action,
        amount: 0,
        signal: 0,
        signal_strength: 0,
    }
}

fn branch_horizon(
    state: &CounterfactualState,
    choice: PolicyChoice,
    continuation: BranchContinuation,
    requested_elapsed_quanta: u64,
) -> Result<&BranchHorizonResult, String> {
    state
        .branches
        .iter()
        .find(|branch| branch.choice == normalized(choice) && branch.continuation == continuation)
        .and_then(|branch| {
            branch
                .horizons
                .iter()
                .find(|horizon| horizon.requested_elapsed_quanta == requested_elapsed_quanta)
        })
        .ok_or_else(|| "counterfactual value source branch is incomplete".into())
}

fn state_vectors(
    state: &CounterfactualState,
    continuation: BranchContinuation,
    horizon: u64,
    weights: EnergyValuation,
) -> Result<(ValueVector, ValueVector), String> {
    Ok((
        value_vector(
            branch_horizon(state, state.teacher_choice, continuation, horizon)?,
            weights,
        )?,
        value_vector(
            branch_horizon(state, state.policy_choice, continuation, horizon)?,
            weights,
        )?,
    ))
}

fn choice_vector(
    state: &CounterfactualState,
    choice: PolicyChoice,
    continuation: BranchContinuation,
    horizon: u64,
    weights: EnergyValuation,
) -> Result<ValueVector, String> {
    value_vector(
        branch_horizon(state, choice, continuation, horizon)?,
        weights,
    )
}

fn robust_choice_preference(
    state: &CounterfactualState,
    left: PolicyChoice,
    right: PolicyChoice,
    options: &CounterfactualValueOptions,
    perspective: ValuePerspective,
    horizon: u64,
) -> Result<ValuePreference, String> {
    let mut preferences = Vec::with_capacity(options.continuations.len());
    for continuation in &options.continuations {
        preferences.push(ContinuationPreference {
            continuation: *continuation,
            preference: compare_vectors(
                choice_vector(state, left, *continuation, horizon, options.energy)?,
                choice_vector(state, right, *continuation, horizon, options.energy)?,
                perspective,
            ),
        });
    }
    Ok(robust_preference(&preferences))
}

fn candidate_pareto_frontier(
    state: &CounterfactualState,
    options: &CounterfactualValueOptions,
    perspective: ValuePerspective,
    horizon: u64,
) -> Result<Vec<PolicyChoice>, String> {
    let mut frontier = Vec::new();
    for candidate in &state.candidates {
        let mut dominated = false;
        for challenger in &state.candidates {
            if challenger == candidate {
                continue;
            }
            if robust_choice_preference(
                state,
                *challenger,
                *candidate,
                options,
                perspective,
                horizon,
            )? == ValuePreference::Teacher
            {
                dominated = true;
                break;
            }
        }
        if !dominated {
            frontier.push(*candidate);
        }
    }
    if frontier.is_empty() {
        return Err("counterfactual candidate Pareto frontier is empty".into());
    }
    Ok(frontier)
}

fn frontier_archetypes(frontier: &[PolicyChoice]) -> Result<Vec<ActionArchetype>, String> {
    let mut archetypes = Vec::new();
    for choice in frontier {
        let decomposition = decompose_policy_action(choice.action)
            .ok_or("counterfactual frontier action is outside the catalog")?;
        let archetype = ActionArchetype {
            kind: decomposition.kind,
            effort: decomposition.effort,
        };
        if !archetypes.contains(&archetype) {
            archetypes.push(archetype);
        }
    }
    Ok(archetypes)
}

fn evaluate_one(
    source: &CounterfactualBranchArtifact,
    options: &CounterfactualValueOptions,
    perspective: ValuePerspective,
    horizon: u64,
) -> Result<ValueEvaluation, String> {
    let mut states = Vec::with_capacity(source.report.states.len());
    let mut state_counts = PreferenceCounts::default();
    for state in &source.report.states {
        let mut continuation_preferences = Vec::with_capacity(options.continuations.len());
        for continuation in &options.continuations {
            let (teacher, policy) = state_vectors(state, *continuation, horizon, options.energy)?;
            continuation_preferences.push(ContinuationPreference {
                continuation: *continuation,
                preference: compare_vectors(teacher, policy, perspective),
            });
        }
        let robust_preference = robust_preference(&continuation_preferences);
        let pareto_frontier = candidate_pareto_frontier(state, options, perspective, horizon)?;
        let pareto_archetypes = frontier_archetypes(&pareto_frontier)?;
        state_counts.observe(robust_preference);
        states.push(StateValueVerdict {
            seed: state.seed,
            state_ordinal: state.state_ordinal,
            continuation_preferences,
            robust_preference,
            dominated_candidates: state.candidates.len() - pareto_frontier.len(),
            pareto_frontier,
            pareto_archetypes,
        });
    }

    let mut seeds = Vec::new();
    let mut seed_counts = PreferenceCounts::default();
    for seed_selection in &source.report.seed_selection {
        if seed_selection.selected_states == 0 {
            continue;
        }
        let seed_states = source
            .report
            .states
            .iter()
            .filter(|state| state.seed == seed_selection.seed)
            .collect::<Vec<_>>();
        let mut continuation_preferences = Vec::with_capacity(options.continuations.len());
        for continuation in &options.continuations {
            let mut teacher = ValueVector::default();
            let mut policy = ValueVector::default();
            for state in &seed_states {
                let (state_teacher, state_policy) =
                    state_vectors(state, *continuation, horizon, options.energy)?;
                teacher.add(state_teacher)?;
                policy.add(state_policy)?;
            }
            continuation_preferences.push(ContinuationPreference {
                continuation: *continuation,
                preference: compare_vectors(teacher, policy, perspective),
            });
        }
        let robust_preference = robust_preference(&continuation_preferences);
        seed_counts.observe(robust_preference);
        seeds.push(SeedValueVerdict {
            seed: seed_selection.seed,
            selected_states: seed_selection.selected_states,
            continuation_preferences,
            robust_preference,
        });
    }
    Ok(ValueEvaluation {
        perspective,
        requested_elapsed_quanta: horizon,
        state_counts,
        seed_counts,
        states,
        seeds,
    })
}

pub fn evaluate_counterfactual_values(
    source: &CounterfactualBranchArtifact,
    options: &CounterfactualValueOptions,
) -> Result<CounterfactualValueReport, String> {
    source.validate()?;
    options.validate_against(source)?;
    let mut evaluations =
        Vec::with_capacity(options.perspectives.len() * options.horizon_quanta.len());
    for perspective in &options.perspectives {
        for horizon in &options.horizon_quanta {
            evaluations.push(evaluate_one(source, options, *perspective, *horizon)?);
        }
    }
    Ok(CounterfactualValueReport {
        schema_version: COUNTERFACTUAL_VALUE_SCHEMA_VERSION,
        evaluations,
    })
}

pub fn publish_counterfactual_value_artifact(
    path: &Path,
    source: &CounterfactualBranchArtifact,
    artifact: &CounterfactualValueArtifact,
) -> Result<PathBuf, String> {
    artifact.validate_against(source)?;
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    if path.exists() {
        return Err(format!(
            "refusing to replace immutable value artifact {}",
            path.display()
        ));
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("counterfactual value output needs a UTF-8 file name")?;
    let nonce = TEMP_NONCE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".{name}.{}.{}.tmp", std::process::id(), nonce));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
    serde_json::to_writer_pretty(&mut file, artifact)
        .map_err(|error| format!("failed to encode value artifact: {error}"))?;
    file.write_all(b"\n")
        .map_err(|error| format!("failed to finish {}: {error}", temporary.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync {}: {error}", temporary.display()))?;
    let size = file
        .metadata()
        .map_err(|error| format!("failed to inspect {}: {error}", temporary.display()))?
        .len();
    if size > MAX_VALUE_ARTIFACT_BYTES {
        let _ = fs::remove_file(&temporary);
        return Err("counterfactual value artifact exceeds its byte bound".into());
    }
    drop(file);
    fs::rename(&temporary, path)
        .map_err(|error| format!("failed to publish {}: {error}", path.display()))?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("failed to sync {}: {error}", parent.display()))?;
    Ok(path.to_path_buf())
}

pub fn load_counterfactual_value_artifact(
    path: &Path,
    source: &CounterfactualBranchArtifact,
) -> Result<CounterfactualValueArtifact, String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))?;
    if metadata.len() > MAX_VALUE_ARTIFACT_BYTES {
        return Err("counterfactual value artifact exceeds its byte bound".into());
    }
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let artifact: CounterfactualValueArtifact = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    artifact.validate_against(source)?;
    Ok(artifact)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counterfactual_branch::{BranchActionResolution, BranchTotals, CandidateBranch};

    fn horizon(energy: u128) -> BranchHorizonResult {
        BranchHorizonResult {
            requested_elapsed_quanta: 64,
            horizon_reached: true,
            observed_elapsed_quanta: 64,
            controlled_frontiers: 1,
            target_alive: true,
            target_position: Some(0),
            target_core_mass: 1,
            target_assimilated_energy: energy as u64,
            target_gut_energy: 0,
            training_cells: 1,
            opponent_cells: 1,
            training_core_mass: 1,
            training_assimilated_energy: energy,
            training_gut_energy: 0,
            training_stored_energy: energy,
            training_total_mass_energy: energy + 1,
            opponent_core_mass: 1,
            opponent_assimilated_energy: 0,
            opponent_gut_energy: 0,
            opponent_stored_energy: 0,
            opponent_total_mass_energy: 1,
            episode_outcome: None,
            branch_host_bound_reached: false,
            totals: BranchTotals::default(),
        }
    }

    fn choice(action: usize) -> PolicyChoice {
        PolicyChoice {
            action,
            amount: 0,
            signal: 0,
            signal_strength: 0,
        }
    }

    #[test]
    fn pareto_tiers_preserve_conflicts_and_lexicographic_priority() {
        assert_eq!(
            pareto_tier(&[(2, 1), (1, 2)]),
            Some(ValuePreference::Incomparable)
        );
        assert_eq!(
            compare_vectors(
                ValueVector {
                    target_survival: 1,
                    team_population: 1,
                    target_energy: 1,
                    team_energy: 1,
                },
                ValueVector {
                    target_survival: 0,
                    team_population: 100,
                    target_energy: 100,
                    team_energy: 100,
                },
                ValuePerspective::Cell,
            ),
            ValuePreference::Teacher
        );
    }

    #[test]
    fn robust_preference_rejects_controller_disagreement() {
        let preferences = [
            ContinuationPreference {
                continuation: BranchContinuation::FrozenPolicy,
                preference: ValuePreference::Teacher,
            },
            ContinuationPreference {
                continuation: BranchContinuation::Teacher,
                preference: ValuePreference::Policy,
            },
        ];
        assert_eq!(
            robust_preference(&preferences),
            ValuePreference::Incomparable
        );
    }

    #[test]
    fn gut_discount_is_explicit_and_exact() {
        let default = weighted_energy(2, 3, 100, EnergyValuation::default()).unwrap();
        assert_eq!(default, 5_000_000);
        let counted = weighted_energy(
            2,
            3,
            100,
            EnergyValuation {
                gut_weight_ppm: 500_000,
                ..EnergyValuation::default()
            },
        )
        .unwrap();
        assert_eq!(counted, 55_000_000);
    }

    #[test]
    fn candidate_frontier_removes_only_robustly_dominated_actions() {
        let candidates = vec![choice(2), choice(6), choice(7)];
        let mut branches = Vec::new();
        for continuation in [
            BranchContinuation::FrozenPolicy,
            BranchContinuation::Teacher,
        ] {
            for (candidate, energy) in candidates.iter().zip([30, 20, 10]) {
                branches.push(CandidateBranch {
                    choice: *candidate,
                    continuation,
                    immediate_resolution: BranchActionResolution::Success,
                    horizons: vec![horizon(energy)],
                });
            }
        }
        let state = CounterfactualState {
            seed: 1,
            seed_frontier: 1,
            state_ordinal: 0,
            sim_time_quanta: 1,
            source_cell: 1,
            checkpoint_sha256: "0".repeat(64),
            mind_observation_sha256: "1".repeat(64),
            mind_evidence: None,
            expert_context: "exploration".into(),
            teacher_choice: choice(2),
            policy_choice: choice(6),
            candidates,
            branches,
        };
        let options = CounterfactualValueOptions {
            perspectives: vec![ValuePerspective::Conservative],
            horizon_quanta: vec![64],
            continuations: vec![
                BranchContinuation::FrozenPolicy,
                BranchContinuation::Teacher,
            ],
            energy: EnergyValuation::default(),
        };
        let frontier =
            candidate_pareto_frontier(&state, &options, ValuePerspective::Conservative, 64)
                .unwrap();
        assert_eq!(frontier, vec![choice(2)]);
        assert_eq!(
            frontier_archetypes(&frontier).unwrap(),
            vec![ActionArchetype { kind: 1, effort: 1 }]
        );
    }
}
