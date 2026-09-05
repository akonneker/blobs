//! Bounded synthesis of deterministic single-cell policies across hidden seeds.
//!
//! Canonical checkpoints are particles owned by the trusted offline oracle.
//! Exact hashes certify which anonymous Mind inputs were encountered. Action
//! selection uses a bounded quantization of the ordinary RL observation plus
//! the exact legal-choice catalog, with private randomness deliberately ignored.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::mem::{size_of, size_of_val};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::action::{policy_action_family, PolicyActionFamily, PolicyChoice};
use crate::config::{EnvConfig, RewardConfig};
use crate::env::EpisodeOutcome;
use crate::micro_combat::{MicroCombatObjective, MicroCombatScenario};
use crate::micro_combat_planner::{
    mind_observation_sha256, PlannerRestoreAggregate, SingleCellSearchSimulator,
    SingleCellSearchState,
};
use crate::observation::{Observation, OBS_RANDOMNESS_FEATURE_END, OBS_RANDOMNESS_FEATURE_START};

pub const MICRO_COMBAT_OBSERVATION_POLICY_SCHEMA_VERSION: u32 = 2;
pub const MICRO_COMBAT_OBSERVATION_POLICY_SEARCH_REPORT_SCHEMA_VERSION: u32 = 17;
pub const MICRO_COMBAT_OBSERVATION_POLICY_ACQUISITION_SCHEMA_VERSION: u32 = 19;
const POLICY_MEMORY_PREFIX: &[u8; 5] = b"BOP\x01\0";

#[cfg(unix)]
fn host_peak_resident_set_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` initializes writable storage of the exact C type.
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
fn host_peak_resident_set_bytes() -> Option<u64> {
    None
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ObservationPolicySearchAlgorithm {
    Beam,
    InformationSetMcts,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicySearchConfig {
    pub algorithm: ObservationPolicySearchAlgorithm,
    pub beam_width: usize,
    /// Salt for deterministic ordering among otherwise equal-scoring nodes.
    /// This diversifies bounded searches without exposing randomness to a Mind.
    /// None preserves canonical lexicographic ordering.
    pub tie_break_seed: Option<u64>,
    /// Deterministic simulation stream for information-set MCTS.
    pub mcts_seed: u64,
    pub mcts_max_iterations: usize,
    pub mcts_rollout_rule_depth: usize,
    /// UCT exploration coefficient multiplied by 1,000.
    pub mcts_exploration_milli: u32,
    pub max_decisions_per_particle: usize,
    /// Counts authoritative simulator transitions, not table nodes.
    pub max_expansions: usize,
    pub equivalent_effort_pruning: bool,
    /// Number of uniform intervals used to quantize each normalized RL input
    /// feature. The representation remains sparse and bounded by `OBS_DIM`.
    pub observation_quantization_levels: u8,
    /// Maximum sparse-feature L1 distance for a nearest abstract-rule
    /// fallback. Zero permits exact abstract matches only.
    pub max_nearest_fallback_l1_distance: u32,
    /// Constrain every observation encountered at the same private-memory
    /// depth to the same exact choice. This is a diagnostic open-loop baseline,
    /// not the default policy class.
    pub force_open_loop_actions: bool,
}

impl Default for ObservationPolicySearchConfig {
    fn default() -> Self {
        Self {
            algorithm: ObservationPolicySearchAlgorithm::Beam,
            beam_width: 16,
            tie_break_seed: None,
            mcts_seed: 0,
            mcts_max_iterations: 20_000,
            mcts_rollout_rule_depth: 8,
            mcts_exploration_milli: 1_414,
            max_decisions_per_particle: 16,
            max_expansions: 100_000,
            equivalent_effort_pruning: true,
            observation_quantization_levels: 16,
            max_nearest_fallback_l1_distance: 128,
            force_open_loop_actions: false,
        }
    }
}

impl ObservationPolicySearchConfig {
    fn validate(self, seed_count: usize) -> Result<Self, String> {
        if seed_count == 0 {
            return Err("observation-policy search requires at least one seed".into());
        }
        if self.beam_width == 0 || self.max_decisions_per_particle == 0 || self.max_expansions == 0
        {
            return Err("observation-policy search limits must be positive".into());
        }
        if !(2..=64).contains(&self.observation_quantization_levels) {
            return Err("observation quantization levels must be between 2 and 64".into());
        }
        if self.algorithm == ObservationPolicySearchAlgorithm::InformationSetMcts
            && (self.mcts_max_iterations == 0
                || self.mcts_rollout_rule_depth == 0
                || self.mcts_exploration_milli == 0)
        {
            return Err("information-set MCTS limits must be positive".into());
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct QuantizedObservationFeature {
    pub index: u16,
    pub value: i16,
}

/// Inspectable, Mind-visible policy state. Zero-valued tensor features are
/// omitted, so its encoded size is bounded without depending on board size.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationAbstraction {
    pub decision_depth: usize,
    pub quantization_levels: u8,
    pub nonzero_features: Vec<QuantizedObservationFeature>,
    /// Prevents a generalized rule from crossing an action-affordability or
    /// target-availability boundary.
    pub legal_choice_catalog_sha256: String,
    pub legal_choice_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicyRule {
    pub decision_depth: usize,
    pub choice: PolicyChoice,
    pub family: String,
    pub next_private_memory: Vec<u8>,
    pub abstraction: ObservationAbstraction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicy {
    pub schema_version: u32,
    pub exact_observation_key: String,
    pub action_selection_key: String,
    pub randomness_handling: String,
    pub observation_quantization_levels: u8,
    pub max_nearest_fallback_l1_distance: u32,
    /// Rules are keyed by the SHA-256 of their serialized abstraction.
    pub rules: BTreeMap<String, ObservationPolicyRule>,
    /// Training-time evidence only. These bindings are not consulted when the
    /// policy selects an action and therefore cannot become a hidden fallback.
    pub exact_observation_bindings: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicyReplay {
    pub seed: u64,
    pub coverage_complete: bool,
    pub objective_success: bool,
    pub final_outcome: String,
    pub decisions: usize,
    pub observation_sha256_trace: Vec<String>,
    pub abstraction_sha256_trace: Vec<String>,
    pub exact_observation_binding_hits: usize,
    pub exact_abstraction_rule_hits: usize,
    pub nearest_fallback_rule_hits: usize,
    pub maximum_nearest_fallback_l1_distance: u32,
    pub missing_abstraction_sha256: Option<String>,
    pub missing_abstraction: Option<ObservationAbstraction>,
    pub final_sim_time_quanta: u64,
    pub final_training_cells: usize,
    pub final_opponent_cells: usize,
    pub final_training_stored_energy: u128,
    pub final_opponent_stored_energy: u128,
    pub final_continuation_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicySearchReport {
    pub schema_version: u32,
    pub evidence_class: String,
    pub search_domain: String,
    pub scenario: String,
    pub search_seeds: Vec<u64>,
    pub holdout_seeds: Vec<u64>,
    pub objective: MicroCombatObjective,
    pub config: ObservationPolicySearchConfig,
    pub search_cost: ObservationPolicySearchCost,
    pub algorithm_iterations: usize,
    pub expansions: usize,
    pub policy_sha256: String,
    pub policy_rule_count: usize,
    pub exact_observation_binding_count: usize,
    /// Decision depths with more than one observation-contingent rule.
    pub policy_branching_depths: Vec<usize>,
    /// Decision depths whose observation-contingent rules select more than one
    /// exact action. Nonempty means the executable policy is not open loop.
    pub policy_choice_branching_depths: Vec<usize>,
    pub search_unique_observation_traces: usize,
    pub holdout_unique_observation_traces: usize,
    pub search_unique_abstraction_traces: usize,
    pub holdout_unique_abstraction_traces: usize,
    pub search_exact_observation_binding_hits: usize,
    pub holdout_exact_observation_binding_hits: usize,
    pub search_nearest_fallback_rule_hits: usize,
    pub holdout_nearest_fallback_rule_hits: usize,
    pub holdout_maximum_nearest_fallback_l1_distance: u32,
    pub search_covered_to_requested_horizon: bool,
    pub holdout_covered_to_requested_horizon: bool,
    pub truncated: bool,
    pub search_objective_success_count: usize,
    pub all_search_objective_success: bool,
    pub holdout_objective_success_count: usize,
    pub all_holdout_objective_success: bool,
    pub policy: ObservationPolicy,
    pub search_replays: Vec<ObservationPolicyReplay>,
    pub holdout_replays: Vec<ObservationPolicyReplay>,
}

/// Host-dependent timing/RSS are evidence about one execution and never enter
/// policy selection. Structural counters are deterministic for a fixed search.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicySearchCost {
    pub wall_time_micros: u128,
    pub host_peak_resident_set_bytes: Option<u64>,
    /// Exact number of trusted planner environments reconstructed during the
    /// search, including observation and legality queries.
    pub planner_restore_count: u64,
    /// Host-dependent canonical restoration phase timings.
    pub planner_restore_tile_materialization_ns: u128,
    pub planner_restore_cell_materialization_ns: u128,
    pub planner_restore_hash_seed_materialization_ns: u128,
    pub planner_restore_resolver_reconstruction_ns: u128,
    /// Nested resolver-reconstruction phases; these sum to less than the
    /// enclosing reconstruction total because orchestration is unclassified.
    pub planner_restore_topology_validation_ns: u128,
    pub planner_restore_cell_validation_store_and_passive_index_ns: u128,
    pub planner_restore_tile_validation_and_passive_index_ns: u128,
    pub planner_restore_hash_initialization_ns: u128,
    pub planner_restore_scratch_initialization_ns: u128,
    pub planner_restore_metabolic_index_ns: u128,
    pub planner_restore_integrity_validation_ns: u128,
    /// Complete environment reconstruction, including the canonical phases,
    /// host metadata installation, and projection refresh.
    pub planner_environment_restore_total_ns: u128,
    pub planner_restore_total_ns: u128,
    /// Deterministic canonical state volume processed by those restores.
    pub planner_restored_tiles: u128,
    pub planner_restored_cells: u128,
    pub peak_retained_policy_nodes: usize,
    pub peak_retained_particles: usize,
    pub peak_retained_rules: usize,
    /// Logical choices not yet expanded across the retained MCTS tree. The
    /// lazy representation does not materialize this many `PolicyChoice`s.
    pub peak_retained_untried_choices: usize,
    /// Canonical planner allocation sharing at the peak retained set.
    pub peak_retained_canonical_tile_page_references: usize,
    pub peak_retained_unique_canonical_tile_pages: usize,
    pub peak_retained_canonical_tile_chunk_references: usize,
    pub peak_retained_unique_canonical_tile_chunks: usize,
    pub peak_retained_canonical_cell_chunk_references: usize,
    pub peak_retained_unique_canonical_cell_chunks: usize,
    pub peak_retained_canonical_hash_seed_chunk_references: usize,
    pub peak_retained_unique_canonical_hash_seed_chunks: usize,
    /// Sparse original-index overrides retained by lazy MCTS permutations.
    /// Beam search reports zero.
    pub peak_retained_mcts_choice_index_overrides: usize,
    /// Lower bound for retained lazy MCTS permutation structs and their owned
    /// sparse index buffers. Allocator and enclosing tree-node overhead are
    /// intentionally excluded.
    pub estimated_peak_retained_mcts_choice_bytes_lower_bound: u64,
    /// Distinct legal-choice catalogs retained once and shared by MCTS nodes.
    pub mcts_unique_choice_catalogs: usize,
    /// Node catalogs satisfied by an existing interned catalog.
    pub mcts_choice_catalog_intern_hits: usize,
    /// Lower bound over owned policy/checkpoint buffers; allocator and tree-node
    /// overhead are intentionally excluded rather than guessed.
    pub estimated_peak_retained_policy_bytes_lower_bound: u64,
    pub policy_node_clones: usize,
    pub rollout_policy_clones: usize,
    /// Immutable particle-state references copied while cloning policy nodes.
    /// Before checkpoint sharing, each would have cloned the full checkpoint.
    pub shared_particle_state_clones: usize,
    /// Root particle states plus authoritative transition results allocated by
    /// this search. Shared clones do not increment this count.
    pub particle_state_allocations: usize,
    /// Beam-only candidate counts. MCTS reports zero for both fields.
    pub beam_generated_children: usize,
    pub beam_pruned_children: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicyAcquisitionConfig {
    /// Maximum number of provisional counterexample batches evaluated after
    /// the initial synthesis policy.
    pub max_rounds: usize,
    /// Maximum distinct missing legality partitions considered per round.
    pub partitions_per_round: usize,
    /// Alternative representative seeds synthesized for each partition.
    pub candidates_per_partition: usize,
    /// Maximum nondominated candidate summaries retained in each report round.
    pub max_frontier_size: usize,
    /// Search families evaluated for every proposed counterexample seed.
    pub candidate_algorithms: Vec<ObservationPolicySearchAlgorithm>,
    /// Beam widths independently synthesized for every proposed seed.
    pub candidate_beam_widths: Vec<usize>,
    /// Deterministic tie-break salts crossed with every candidate beam width.
    pub candidate_tie_break_seeds: Vec<u64>,
    /// Also evaluate canonical lexicographic ordering for every candidate.
    pub include_canonical_tie_break: bool,
    /// Deterministic simulation streams evaluated by MCTS candidates.
    pub candidate_mcts_seeds: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicyAcquisitionSeeds {
    pub initial_search: Vec<u64>,
    pub acquisition_pool: Vec<u64>,
    pub final_holdout: Vec<u64>,
}

impl Default for ObservationPolicyAcquisitionConfig {
    fn default() -> Self {
        Self {
            max_rounds: 2,
            partitions_per_round: 2,
            candidates_per_partition: 2,
            max_frontier_size: 4,
            candidate_algorithms: vec![ObservationPolicySearchAlgorithm::Beam],
            candidate_beam_widths: vec![8, 16],
            candidate_tie_break_seeds: vec![0],
            include_canonical_tie_break: true,
            candidate_mcts_seeds: vec![0],
        }
    }
}

impl ObservationPolicyAcquisitionConfig {
    fn validate(mut self) -> Result<Self, String> {
        if self.max_rounds == 0
            || self.partitions_per_round == 0
            || self.candidates_per_partition == 0
            || self.max_frontier_size == 0
        {
            return Err("observation-policy acquisition limits must be positive".into());
        }
        if self.candidate_algorithms.is_empty() || self.candidate_algorithms.len() > 2 {
            return Err("candidate algorithms must contain one or two values".into());
        }
        self.candidate_algorithms.sort_unstable();
        self.candidate_algorithms.dedup();
        if self
            .candidate_algorithms
            .contains(&ObservationPolicySearchAlgorithm::Beam)
            && self.candidate_beam_widths.len() > 8
        {
            return Err("candidate beam widths must contain at most eight values".into());
        }
        self.candidate_beam_widths.sort_unstable();
        self.candidate_beam_widths.dedup();
        if self.candidate_tie_break_seeds.len() > 16 {
            return Err("candidate tie-break seeds must contain at most sixteen values".into());
        }
        if self.candidate_mcts_seeds.len() > 16 {
            return Err("candidate MCTS seeds must contain at most sixteen values".into());
        }
        self.candidate_tie_break_seeds.sort_unstable();
        self.candidate_tie_break_seeds.dedup();
        let tie_break_count =
            self.candidate_tie_break_seeds.len() + usize::from(self.include_canonical_tie_break);
        self.candidate_mcts_seeds.sort_unstable();
        self.candidate_mcts_seeds.dedup();
        let mut search_config_count = 0usize;
        if self
            .candidate_algorithms
            .contains(&ObservationPolicySearchAlgorithm::Beam)
        {
            if self.candidate_beam_widths.is_empty()
                || self.candidate_beam_widths.len() > 8
                || self.candidate_beam_widths.contains(&0)
            {
                return Err(
                    "candidate beam widths must contain between one and eight positive values"
                        .into(),
                );
            }
            if tie_break_count == 0 {
                return Err("beam candidate grid requires at least one tie-break strategy".into());
            }
            search_config_count = self
                .candidate_beam_widths
                .len()
                .checked_mul(tie_break_count)
                .ok_or("candidate search grid size overflowed")?;
        }
        if self
            .candidate_algorithms
            .contains(&ObservationPolicySearchAlgorithm::InformationSetMcts)
        {
            if self.candidate_mcts_seeds.is_empty() || self.candidate_mcts_seeds.len() > 16 {
                return Err(
                    "candidate MCTS seeds must contain between one and sixteen values".into(),
                );
            }
            search_config_count = search_config_count
                .checked_add(self.candidate_mcts_seeds.len())
                .ok_or("candidate search grid size overflowed")?;
        }
        if search_config_count > 16 {
            return Err("candidate search grid must contain at most sixteen configurations".into());
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MissingObservationPartitionSummary {
    pub decision_depth: usize,
    pub legal_choice_catalog_sha256: String,
    pub legal_choice_count: usize,
    pub replay_count: usize,
    pub representative_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicyAcquisitionRound {
    pub round: usize,
    pub incumbent_policy_sha256: String,
    pub incumbent_acquisition_pool_coverage: usize,
    pub missing_partitions: Vec<MissingObservationPartitionSummary>,
    pub candidates: Vec<ObservationPolicyCandidateSummary>,
    pub pareto_frontier: Vec<ObservationPolicyCandidateIdentity>,
    pub selected_policy_sha256: String,
    pub selected_seed: Option<u64>,
    pub selected_search_config: ObservationPolicySearchConfig,
    pub incumbent_retained: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicyCandidateIdentity {
    pub added_seed: Option<u64>,
    pub search_config: ObservationPolicySearchConfig,
    pub policy_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicyCandidateSummary {
    /// None identifies the incumbent policy.
    pub added_seed: Option<u64>,
    pub objective: MicroCombatObjective,
    pub search_config: ObservationPolicySearchConfig,
    /// Diagnostic only. Neither host-dependent nor structural search costs
    /// participate in candidate selection or Pareto dominance.
    pub search_cost: ObservationPolicySearchCost,
    pub policy_sha256: String,
    pub acquisition_pool_covered_seed_count: usize,
    pub acquisition_objective_success_count: usize,
    pub policy_rule_count: usize,
    pub exact_observation_binding_count: usize,
    pub acquisition_nearest_fallback_rule_hits: usize,
    pub acquisition_maximum_nearest_fallback_l1_distance: u32,
    pub acquisition_minimum_final_training_energy: u128,
    pub acquisition_final_training_energy_total: u128,
    pub acquisition_minimum_survival_time_quanta: u64,
    pub acquisition_survival_time_quanta_total: u128,
    pub expansions: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObservationPolicyAcquisitionReport {
    pub schema_version: u32,
    pub evidence_class: String,
    pub scenario: String,
    pub initial_search_seeds: Vec<u64>,
    /// Seeds used only for counterexample acquisition. They are never final
    /// holdout evidence, including seeds not selected for synthesis.
    pub acquisition_pool_seeds: Vec<u64>,
    pub acquired_seeds: Vec<u64>,
    pub final_holdout_seeds: Vec<u64>,
    pub acquisition_config: ObservationPolicyAcquisitionConfig,
    pub search_config: ObservationPolicySearchConfig,
    pub selected_search_config: ObservationPolicySearchConfig,
    pub rounds: Vec<ObservationPolicyAcquisitionRound>,
    pub total_expansions: usize,
    pub final_report: ObservationPolicySearchReport,
}

#[derive(Clone)]
struct PolicyParticle {
    state: Arc<SingleCellSearchState>,
    decisions: usize,
}

#[derive(Clone)]
struct PolicyNode {
    particles: Vec<PolicyParticle>,
    rules: BTreeMap<String, ObservationPolicyRule>,
    exact_observation_bindings: BTreeMap<String, String>,
    depth_choices: BTreeMap<usize, PolicyChoice>,
}

#[derive(Default)]
struct SearchWorkMetrics {
    peak_retained_policy_nodes: usize,
    peak_retained_particles: usize,
    peak_retained_rules: usize,
    peak_retained_untried_choices: usize,
    estimated_peak_retained_policy_bytes_lower_bound: u64,
    policy_node_clones: usize,
    rollout_policy_clones: usize,
    shared_particle_state_clones: usize,
    beam_generated_children: usize,
    beam_pruned_children: usize,
    peak_retained_mcts_choice_index_overrides: usize,
    estimated_peak_retained_mcts_choice_bytes_lower_bound: u64,
    mcts_unique_choice_catalogs: usize,
    mcts_choice_catalog_intern_hits: usize,
    peak_retained_canonical_tile_chunk_references: usize,
    peak_retained_unique_canonical_tile_chunks: usize,
    peak_retained_canonical_tile_page_references: usize,
    peak_retained_unique_canonical_tile_pages: usize,
    peak_retained_canonical_cell_chunk_references: usize,
    peak_retained_unique_canonical_cell_chunks: usize,
    peak_retained_canonical_hash_seed_chunk_references: usize,
    peak_retained_unique_canonical_hash_seed_chunks: usize,
}

#[derive(Default)]
struct RetainedStateAllocations {
    particle_states: HashSet<usize>,
    canonical_tile_pages: HashSet<usize>,
    canonical_tile_chunks: HashSet<usize>,
    canonical_cell_chunks: HashSet<usize>,
    canonical_hash_seed_chunks: HashSet<usize>,
    canonical_tile_chunk_references: usize,
    canonical_tile_page_references: usize,
    canonical_cell_chunk_references: usize,
    canonical_hash_seed_chunk_references: usize,
}

impl SearchWorkMetrics {
    fn record_policy_node_clone(&mut self, node: &PolicyNode) {
        self.policy_node_clones += 1;
        self.shared_particle_state_clones = self
            .shared_particle_state_clones
            .saturating_add(node.particles.len());
    }

    fn record_rollout_policy_clone(&mut self, node: &PolicyNode) {
        self.record_policy_node_clone(node);
        self.rollout_policy_clones += 1;
    }

    fn observe_totals(
        &mut self,
        node_count: usize,
        particle_count: usize,
        rule_count: usize,
        untried_choices: usize,
        estimated_bytes: u64,
    ) {
        self.peak_retained_policy_nodes = self.peak_retained_policy_nodes.max(node_count);
        self.peak_retained_particles = self.peak_retained_particles.max(particle_count);
        self.peak_retained_rules = self.peak_retained_rules.max(rule_count);
        self.peak_retained_untried_choices =
            self.peak_retained_untried_choices.max(untried_choices);
        self.estimated_peak_retained_policy_bytes_lower_bound = self
            .estimated_peak_retained_policy_bytes_lower_bound
            .max(estimated_bytes);
    }

    fn observe_retained<'a>(
        &mut self,
        nodes: impl IntoIterator<Item = &'a PolicyNode>,
        untried_choices: usize,
    ) {
        let mut node_count = 0usize;
        let mut particle_count = 0usize;
        let mut rule_count = 0usize;
        let mut estimated_bytes = 0u64;
        let mut retained_allocations = RetainedStateAllocations::default();
        for node in nodes {
            node_count += 1;
            particle_count = particle_count.saturating_add(node.particles.len());
            rule_count = rule_count.saturating_add(node.rules.len());
            estimated_bytes = estimated_bytes.saturating_add(estimated_retained_policy_node_bytes(
                node,
                &mut retained_allocations,
            ));
        }
        self.observe_checkpoint_sharing(&retained_allocations);
        self.observe_totals(
            node_count,
            particle_count,
            rule_count,
            untried_choices,
            estimated_bytes,
        );
    }

    fn observe_mcts_choice_storage(&mut self, overrides: usize, estimated_bytes: usize) {
        self.peak_retained_mcts_choice_index_overrides = self
            .peak_retained_mcts_choice_index_overrides
            .max(overrides);
        self.estimated_peak_retained_mcts_choice_bytes_lower_bound = self
            .estimated_peak_retained_mcts_choice_bytes_lower_bound
            .max(u64::try_from(estimated_bytes).unwrap_or(u64::MAX));
    }

    fn observe_checkpoint_sharing(&mut self, allocations: &RetainedStateAllocations) {
        self.peak_retained_canonical_tile_page_references = self
            .peak_retained_canonical_tile_page_references
            .max(allocations.canonical_tile_page_references);
        self.peak_retained_unique_canonical_tile_pages = self
            .peak_retained_unique_canonical_tile_pages
            .max(allocations.canonical_tile_pages.len());
        self.peak_retained_canonical_tile_chunk_references = self
            .peak_retained_canonical_tile_chunk_references
            .max(allocations.canonical_tile_chunk_references);
        self.peak_retained_unique_canonical_tile_chunks = self
            .peak_retained_unique_canonical_tile_chunks
            .max(allocations.canonical_tile_chunks.len());
        self.peak_retained_canonical_cell_chunk_references = self
            .peak_retained_canonical_cell_chunk_references
            .max(allocations.canonical_cell_chunk_references);
        self.peak_retained_unique_canonical_cell_chunks = self
            .peak_retained_unique_canonical_cell_chunks
            .max(allocations.canonical_cell_chunks.len());
        self.peak_retained_canonical_hash_seed_chunk_references = self
            .peak_retained_canonical_hash_seed_chunk_references
            .max(allocations.canonical_hash_seed_chunk_references);
        self.peak_retained_unique_canonical_hash_seed_chunks = self
            .peak_retained_unique_canonical_hash_seed_chunks
            .max(allocations.canonical_hash_seed_chunks.len());
    }
}

fn estimated_retained_policy_node_bytes(
    node: &PolicyNode,
    retained_allocations: &mut RetainedStateAllocations,
) -> u64 {
    let mut bytes = size_of_val(node).saturating_add(
        node.particles
            .capacity()
            .saturating_mul(size_of::<PolicyParticle>()),
    );
    for particle in &node.particles {
        if retained_allocations
            .particle_states
            .insert(Arc::as_ptr(&particle.state) as usize)
        {
            bytes = bytes.saturating_add(estimated_particle_state_bytes(
                &particle.state,
                retained_allocations,
            ));
        }
    }
    for (key, rule) in &node.rules {
        bytes = bytes
            .saturating_add(key.capacity())
            .saturating_add(size_of::<ObservationPolicyRule>())
            .saturating_add(rule.family.capacity())
            .saturating_add(rule.next_private_memory.capacity())
            .saturating_add(
                rule.abstraction
                    .nonzero_features
                    .capacity()
                    .saturating_mul(size_of::<QuantizedObservationFeature>()),
            )
            .saturating_add(rule.abstraction.legal_choice_catalog_sha256.capacity());
    }
    for (exact, abstraction) in &node.exact_observation_bindings {
        bytes = bytes
            .saturating_add(exact.capacity())
            .saturating_add(abstraction.capacity());
    }
    u64::try_from(bytes).unwrap_or(u64::MAX)
}

fn estimated_particle_state_bytes(
    state: &SingleCellSearchState,
    retained_allocations: &mut RetainedStateAllocations,
) -> usize {
    let checkpoint = &state.checkpoint;
    retained_allocations.canonical_tile_page_references = retained_allocations
        .canonical_tile_page_references
        .saturating_add(checkpoint.canonical.canonical.tile_page_count());
    retained_allocations.canonical_cell_chunk_references = retained_allocations
        .canonical_cell_chunk_references
        .saturating_add(checkpoint.canonical.canonical.cell_chunk_count());
    retained_allocations.canonical_hash_seed_chunk_references = retained_allocations
        .canonical_hash_seed_chunk_references
        .saturating_add(checkpoint.canonical.canonical.hash_seed_chunk_count());
    let mut bytes = size_of::<SingleCellSearchState>()
        .saturating_add(
            checkpoint
                .canonical
                .canonical
                .estimated_inline_heap_bytes_lower_bound(),
        )
        .saturating_add(
            checkpoint
                .host_cells
                .capacity()
                .saturating_mul(size_of::<blob_interface::cell::Cell>()),
        )
        .saturating_add(state.continuation_sha256.capacity());
    for (page_index, (page_allocation, page_len)) in checkpoint
        .canonical
        .canonical
        .tile_page_allocations()
        .enumerate()
    {
        if retained_allocations
            .canonical_tile_pages
            .insert(page_allocation as usize)
        {
            retained_allocations.canonical_tile_chunk_references = retained_allocations
                .canonical_tile_chunk_references
                .saturating_add(page_len);
            bytes = bytes.saturating_add(
                page_len.saturating_mul(size_of::<Arc<[blob_engine::resolution::TileState]>>()),
            );
            for (allocation, len) in checkpoint
                .canonical
                .canonical
                .tile_chunk_allocations_in_page(page_index)
            {
                if retained_allocations
                    .canonical_tile_chunks
                    .insert(allocation as usize)
                {
                    bytes = bytes.saturating_add(
                        len.saturating_mul(size_of::<blob_engine::resolution::TileState>()),
                    );
                }
            }
        }
    }
    for (allocation, len) in checkpoint.canonical.canonical.cell_chunk_allocations() {
        if retained_allocations
            .canonical_cell_chunks
            .insert(allocation as usize)
        {
            bytes = bytes.saturating_add(len.saturating_mul(size_of::<(
                blob_engine::resolution::CellKey,
                blob_engine::resolution::CellState,
            )>()));
        }
    }
    for (allocation, bytes_len) in checkpoint.canonical.canonical.hash_seed_chunk_allocations() {
        if retained_allocations
            .canonical_hash_seed_chunks
            .insert(allocation as usize)
        {
            bytes = bytes.saturating_add(bytes_len);
        }
    }
    for cell in &checkpoint.host_cells {
        bytes = bytes.saturating_add(cell.memory.capacity()).saturating_add(
            cell.message_queue
                .capacity()
                .saturating_mul(size_of::<blob_interface::types::CellMessage>()),
        );
        for message in &cell.message_queue {
            bytes = bytes.saturating_add(message.data.capacity());
        }
    }
    bytes
}

struct PendingInformationSet {
    depth: usize,
    abstraction_sha256: String,
    abstraction: ObservationAbstraction,
    particles: Vec<(usize, String)>,
    search_choices: Vec<PolicyChoice>,
}

fn acquisition_candidate_search_configs(
    base: ObservationPolicySearchConfig,
    acquisition: &ObservationPolicyAcquisitionConfig,
) -> Vec<ObservationPolicySearchConfig> {
    let mut configs = Vec::new();
    if acquisition
        .candidate_algorithms
        .contains(&ObservationPolicySearchAlgorithm::Beam)
    {
        let tie_breaks = acquisition
            .include_canonical_tie_break
            .then_some(None)
            .into_iter()
            .chain(
                acquisition
                    .candidate_tie_break_seeds
                    .iter()
                    .copied()
                    .map(Some),
            );
        for beam_width in &acquisition.candidate_beam_widths {
            for tie_break_seed in tie_breaks.clone() {
                configs.push(ObservationPolicySearchConfig {
                    algorithm: ObservationPolicySearchAlgorithm::Beam,
                    beam_width: *beam_width,
                    tie_break_seed,
                    ..base
                });
            }
        }
    }
    if acquisition
        .candidate_algorithms
        .contains(&ObservationPolicySearchAlgorithm::InformationSetMcts)
    {
        for mcts_seed in &acquisition.candidate_mcts_seeds {
            configs.push(ObservationPolicySearchConfig {
                algorithm: ObservationPolicySearchAlgorithm::InformationSetMcts,
                tie_break_seed: None,
                mcts_seed: *mcts_seed,
                ..base
            });
        }
    }
    configs
}

pub fn acquire_observation_policy(
    base: &EnvConfig,
    reward: &RewardConfig,
    scenario: &MicroCombatScenario,
    seeds: &ObservationPolicyAcquisitionSeeds,
    search_config: ObservationPolicySearchConfig,
    acquisition_config: ObservationPolicyAcquisitionConfig,
) -> Result<ObservationPolicyAcquisitionReport, String> {
    let acquisition_config = acquisition_config.validate()?;
    validate_three_way_seed_partition(
        &seeds.initial_search,
        &seeds.acquisition_pool,
        &seeds.final_holdout,
    )?;
    search_config.validate(seeds.initial_search.len())?;
    if seeds.acquisition_pool.is_empty() {
        return Err("observation-policy acquisition pool must not be empty".into());
    }
    if seeds.final_holdout.is_empty() {
        return Err("observation-policy final holdout must not be empty".into());
    }

    let mut synthesis_seeds = seeds.initial_search.clone();
    let mut remaining_acquisition = seeds
        .acquisition_pool
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut acquired_seeds = Vec::new();
    let mut rounds = Vec::new();
    let mut current_search_config = search_config;
    let mut current_report = search_observation_policy(
        base,
        reward,
        scenario,
        &synthesis_seeds,
        &remaining_acquisition.iter().copied().collect::<Vec<_>>(),
        current_search_config,
    )?;
    let mut total_expansions = current_report.expansions;

    for round_index in 0..acquisition_config.max_rounds {
        let missing_partitions = missing_partition_summaries(&current_report.holdout_replays);
        let proposed_seeds = acquisition_candidate_seeds(
            &current_report.holdout_replays,
            acquisition_config.partitions_per_round,
            acquisition_config.candidates_per_partition,
        );
        let incumbent_summary = summarize_acquisition_candidate(
            None,
            current_search_config,
            &current_report,
            &acquired_seeds,
        )?;
        if proposed_seeds.is_empty() {
            rounds.push(ObservationPolicyAcquisitionRound {
                round: round_index + 1,
                incumbent_policy_sha256: incumbent_summary.policy_sha256.clone(),
                incumbent_acquisition_pool_coverage: incumbent_summary
                    .acquisition_pool_covered_seed_count,
                missing_partitions,
                candidates: vec![incumbent_summary.clone()],
                pareto_frontier: vec![candidate_identity(&incumbent_summary)],
                selected_policy_sha256: incumbent_summary.policy_sha256,
                selected_seed: None,
                selected_search_config: incumbent_summary.search_config,
                incumbent_retained: true,
            });
            break;
        }

        let mut evaluated = Vec::new();
        let candidate_search_configs =
            acquisition_candidate_search_configs(current_search_config, &acquisition_config);
        for seed in proposed_seeds {
            for candidate_search_config in &candidate_search_configs {
                let mut candidate_synthesis = synthesis_seeds.clone();
                candidate_synthesis.push(seed);
                let mut candidate_acquired = acquired_seeds.clone();
                candidate_acquired.push(seed);
                let candidate_holdouts = remaining_acquisition
                    .iter()
                    .copied()
                    .filter(|candidate| *candidate != seed)
                    .collect::<Vec<_>>();
                let report = search_observation_policy(
                    base,
                    reward,
                    scenario,
                    &candidate_synthesis,
                    &candidate_holdouts,
                    *candidate_search_config,
                )?;
                total_expansions = total_expansions
                    .checked_add(report.expansions)
                    .ok_or("observation-policy acquisition expansion count overflowed")?;
                let summary = summarize_acquisition_candidate(
                    Some(seed),
                    *candidate_search_config,
                    &report,
                    &candidate_acquired,
                )?;
                evaluated.push((
                    candidate_synthesis,
                    candidate_acquired,
                    report,
                    summary,
                    *candidate_search_config,
                ));
            }
        }

        let mut summaries = Vec::with_capacity(evaluated.len() + 1);
        summaries.push(incumbent_summary.clone());
        summaries.extend(evaluated.iter().map(|candidate| candidate.3.clone()));
        let frontier = pareto_candidate_indices(&summaries, acquisition_config.max_frontier_size);
        let selected_index = frontier
            .iter()
            .copied()
            .max_by(|left, right| {
                compare_acquisition_candidates(&summaries[*left], &summaries[*right])
            })
            .ok_or("observation-policy candidate frontier was empty")?;
        let selected_summary = summaries[selected_index].clone();
        let incumbent_retained = selected_index == 0;
        let pareto_frontier = frontier
            .iter()
            .map(|index| candidate_identity(&summaries[*index]))
            .collect();
        rounds.push(ObservationPolicyAcquisitionRound {
            round: round_index + 1,
            incumbent_policy_sha256: incumbent_summary.policy_sha256,
            incumbent_acquisition_pool_coverage: incumbent_summary
                .acquisition_pool_covered_seed_count,
            missing_partitions,
            candidates: summaries,
            pareto_frontier,
            selected_policy_sha256: selected_summary.policy_sha256,
            selected_seed: selected_summary.added_seed,
            selected_search_config: selected_summary.search_config,
            incumbent_retained,
        });
        if incumbent_retained {
            break;
        }
        let selected = evaluated.swap_remove(selected_index - 1);
        let selected_seed = selected
            .3
            .added_seed
            .ok_or("selected acquisition candidate had no added seed")?;
        synthesis_seeds = selected.0;
        acquired_seeds = selected.1;
        current_report = selected.2;
        current_search_config = selected.4;
        if !remaining_acquisition.remove(&selected_seed) {
            return Err("selected acquisition seed was outside the remaining pool".into());
        }
    }

    let final_report = search_observation_policy(
        base,
        reward,
        scenario,
        &synthesis_seeds,
        &seeds.final_holdout,
        current_search_config,
    )?;
    total_expansions = total_expansions
        .checked_add(final_report.expansions)
        .ok_or("observation-policy acquisition expansion count overflowed")?;
    Ok(ObservationPolicyAcquisitionReport {
        schema_version: MICRO_COMBAT_OBSERVATION_POLICY_ACQUISITION_SCHEMA_VERSION,
        evidence_class: "counterexample_guided_synthesis_with_sealed_disjoint_final_holdout".into(),
        scenario: scenario.name.clone(),
        initial_search_seeds: seeds.initial_search.clone(),
        acquisition_pool_seeds: seeds.acquisition_pool.clone(),
        acquired_seeds,
        final_holdout_seeds: seeds.final_holdout.clone(),
        acquisition_config,
        search_config,
        selected_search_config: current_search_config,
        rounds,
        total_expansions,
        final_report,
    })
}

fn candidate_identity(
    summary: &ObservationPolicyCandidateSummary,
) -> ObservationPolicyCandidateIdentity {
    ObservationPolicyCandidateIdentity {
        added_seed: summary.added_seed,
        search_config: summary.search_config,
        policy_sha256: summary.policy_sha256.clone(),
    }
}

fn summarize_acquisition_candidate(
    added_seed: Option<u64>,
    search_config: ObservationPolicySearchConfig,
    report: &ObservationPolicySearchReport,
    acquired_seeds: &[u64],
) -> Result<ObservationPolicyCandidateSummary, String> {
    let acquisition_replays = report
        .holdout_replays
        .iter()
        .chain(
            report
                .search_replays
                .iter()
                .filter(|replay| acquired_seeds.contains(&replay.seed)),
        )
        .collect::<Vec<_>>();
    let coverage = acquisition_replays
        .iter()
        .filter(|replay| replay.coverage_complete)
        .count();
    let objective_success = acquisition_replays
        .iter()
        .filter(|replay| replay.objective_success)
        .count();
    let fallback_hits = acquisition_replays
        .iter()
        .try_fold(0usize, |total, replay| {
            total
                .checked_add(replay.nearest_fallback_rule_hits)
                .ok_or("observation-policy candidate fallback count overflowed")
        })?;
    let maximum_fallback_distance = acquisition_replays
        .iter()
        .map(|replay| replay.maximum_nearest_fallback_l1_distance)
        .max()
        .unwrap_or(0);
    let minimum_final_training_energy = acquisition_replays
        .iter()
        .map(|replay| replay.final_training_stored_energy)
        .min()
        .unwrap_or(0);
    let final_training_energy_total =
        acquisition_replays
            .iter()
            .try_fold(0u128, |total, replay| {
                total
                    .checked_add(replay.final_training_stored_energy)
                    .ok_or("observation-policy candidate energy total overflowed")
            })?;
    let minimum_survival_time_quanta = acquisition_replays
        .iter()
        .map(|replay| replay.final_sim_time_quanta)
        .min()
        .unwrap_or(0);
    let survival_time_quanta_total =
        acquisition_replays
            .iter()
            .try_fold(0u128, |total, replay| {
                total
                    .checked_add(u128::from(replay.final_sim_time_quanta))
                    .ok_or("observation-policy candidate survival-time total overflowed")
            })?;
    Ok(ObservationPolicyCandidateSummary {
        added_seed,
        objective: report.objective,
        search_config,
        search_cost: report.search_cost.clone(),
        policy_sha256: report.policy_sha256.clone(),
        acquisition_pool_covered_seed_count: coverage,
        acquisition_objective_success_count: objective_success,
        policy_rule_count: report.policy_rule_count,
        exact_observation_binding_count: report.exact_observation_binding_count,
        acquisition_nearest_fallback_rule_hits: fallback_hits,
        acquisition_maximum_nearest_fallback_l1_distance: maximum_fallback_distance,
        acquisition_minimum_final_training_energy: minimum_final_training_energy,
        acquisition_final_training_energy_total: final_training_energy_total,
        acquisition_minimum_survival_time_quanta: minimum_survival_time_quanta,
        acquisition_survival_time_quanta_total: survival_time_quanta_total,
        expansions: report.expansions,
        truncated: report.truncated,
    })
}

fn compare_acquisition_candidates(
    left: &ObservationPolicyCandidateSummary,
    right: &ObservationPolicyCandidateSummary,
) -> Ordering {
    left.acquisition_pool_covered_seed_count
        .cmp(&right.acquisition_pool_covered_seed_count)
        .then_with(|| {
            left.acquisition_objective_success_count
                .cmp(&right.acquisition_objective_success_count)
        })
        .then_with(|| compare_acquisition_margins(left, right))
        .then_with(|| {
            right
                .acquisition_maximum_nearest_fallback_l1_distance
                .cmp(&left.acquisition_maximum_nearest_fallback_l1_distance)
        })
        .then_with(|| right.policy_rule_count.cmp(&left.policy_rule_count))
        .then_with(|| {
            right
                .exact_observation_binding_count
                .cmp(&left.exact_observation_binding_count)
        })
        .then_with(|| {
            right
                .acquisition_nearest_fallback_rule_hits
                .cmp(&left.acquisition_nearest_fallback_rule_hits)
        })
        .then_with(|| right.expansions.cmp(&left.expansions))
        .then_with(|| {
            right
                .search_config
                .algorithm
                .cmp(&left.search_config.algorithm)
        })
        .then_with(|| {
            right
                .search_config
                .beam_width
                .cmp(&left.search_config.beam_width)
        })
        .then_with(|| {
            right
                .search_config
                .tie_break_seed
                .cmp(&left.search_config.tie_break_seed)
        })
        .then_with(|| {
            right
                .search_config
                .mcts_seed
                .cmp(&left.search_config.mcts_seed)
        })
        .then_with(|| {
            right
                .search_config
                .force_open_loop_actions
                .cmp(&left.search_config.force_open_loop_actions)
        })
        .then_with(|| match (left.added_seed, right.added_seed) {
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(left), Some(right)) => right.cmp(&left),
            (None, None) => Ordering::Equal,
        })
}

fn compare_acquisition_margins(
    left: &ObservationPolicyCandidateSummary,
    right: &ObservationPolicyCandidateSummary,
) -> Ordering {
    if left.objective != right.objective {
        return match (left.objective, right.objective) {
            (MicroCombatObjective::Survival, MicroCombatObjective::Elimination) => Ordering::Less,
            (MicroCombatObjective::Elimination, MicroCombatObjective::Survival) => {
                Ordering::Greater
            }
            _ => Ordering::Equal,
        };
    }
    match left.objective {
        MicroCombatObjective::Survival => left
            .acquisition_minimum_survival_time_quanta
            .cmp(&right.acquisition_minimum_survival_time_quanta)
            .then_with(|| {
                left.acquisition_minimum_final_training_energy
                    .cmp(&right.acquisition_minimum_final_training_energy)
            })
            .then_with(|| {
                left.acquisition_survival_time_quanta_total
                    .cmp(&right.acquisition_survival_time_quanta_total)
            })
            .then_with(|| {
                left.acquisition_final_training_energy_total
                    .cmp(&right.acquisition_final_training_energy_total)
            }),
        MicroCombatObjective::Elimination => right
            .acquisition_survival_time_quanta_total
            .cmp(&left.acquisition_survival_time_quanta_total)
            .then_with(|| {
                left.acquisition_minimum_final_training_energy
                    .cmp(&right.acquisition_minimum_final_training_energy)
            })
            .then_with(|| {
                left.acquisition_final_training_energy_total
                    .cmp(&right.acquisition_final_training_energy_total)
            }),
    }
}

fn candidate_dominates(
    left: &ObservationPolicyCandidateSummary,
    right: &ObservationPolicyCandidateSummary,
) -> bool {
    if left.objective != right.objective {
        return false;
    }
    let (margin_no_worse, margin_strictly_better) = match left.objective {
        MicroCombatObjective::Survival => (
            left.acquisition_minimum_survival_time_quanta
                >= right.acquisition_minimum_survival_time_quanta
                && left.acquisition_survival_time_quanta_total
                    >= right.acquisition_survival_time_quanta_total
                && left.acquisition_minimum_final_training_energy
                    >= right.acquisition_minimum_final_training_energy
                && left.acquisition_final_training_energy_total
                    >= right.acquisition_final_training_energy_total,
            left.acquisition_minimum_survival_time_quanta
                > right.acquisition_minimum_survival_time_quanta
                || left.acquisition_survival_time_quanta_total
                    > right.acquisition_survival_time_quanta_total
                || left.acquisition_minimum_final_training_energy
                    > right.acquisition_minimum_final_training_energy
                || left.acquisition_final_training_energy_total
                    > right.acquisition_final_training_energy_total,
        ),
        MicroCombatObjective::Elimination => (
            left.acquisition_survival_time_quanta_total
                <= right.acquisition_survival_time_quanta_total
                && left.acquisition_minimum_final_training_energy
                    >= right.acquisition_minimum_final_training_energy
                && left.acquisition_final_training_energy_total
                    >= right.acquisition_final_training_energy_total,
            left.acquisition_survival_time_quanta_total
                < right.acquisition_survival_time_quanta_total
                || left.acquisition_minimum_final_training_energy
                    > right.acquisition_minimum_final_training_energy
                || left.acquisition_final_training_energy_total
                    > right.acquisition_final_training_energy_total,
        ),
    };
    let no_worse = left.acquisition_pool_covered_seed_count
        >= right.acquisition_pool_covered_seed_count
        && left.acquisition_objective_success_count >= right.acquisition_objective_success_count
        && left.policy_rule_count <= right.policy_rule_count
        && left.acquisition_maximum_nearest_fallback_l1_distance
            <= right.acquisition_maximum_nearest_fallback_l1_distance
        && margin_no_worse;
    let strictly_better = left.acquisition_pool_covered_seed_count
        > right.acquisition_pool_covered_seed_count
        || left.acquisition_objective_success_count > right.acquisition_objective_success_count
        || left.policy_rule_count < right.policy_rule_count
        || left.acquisition_maximum_nearest_fallback_l1_distance
            < right.acquisition_maximum_nearest_fallback_l1_distance
        || margin_strictly_better;
    no_worse && strictly_better
}

fn pareto_candidate_indices(
    candidates: &[ObservationPolicyCandidateSummary],
    maximum_size: usize,
) -> Vec<usize> {
    let mut frontier = (0..candidates.len())
        .filter(|candidate| {
            !(0..candidates.len()).any(|other| {
                other != *candidate
                    && candidate_dominates(&candidates[other], &candidates[*candidate])
            })
        })
        .collect::<Vec<_>>();
    frontier.sort_unstable_by(|left, right| {
        compare_acquisition_candidates(&candidates[*right], &candidates[*left])
            .then_with(|| left.cmp(right))
    });
    frontier.dedup_by(|left, right| {
        candidates[*left].policy_sha256 == candidates[*right].policy_sha256
    });
    frontier.truncate(maximum_size);
    frontier
}

fn validate_three_way_seed_partition(
    initial_search_seeds: &[u64],
    acquisition_pool_seeds: &[u64],
    final_holdout_seeds: &[u64],
) -> Result<(), String> {
    let mut all = BTreeSet::new();
    for (label, seeds) in [
        ("initial synthesis", initial_search_seeds),
        ("acquisition pool", acquisition_pool_seeds),
        ("final holdout", final_holdout_seeds),
    ] {
        for seed in seeds {
            if !all.insert(*seed) {
                return Err(format!(
                    "observation-policy {label} seed {seed} is duplicated across seed partitions"
                ));
            }
        }
    }
    Ok(())
}

fn missing_partition_summaries(
    replays: &[ObservationPolicyReplay],
) -> Vec<MissingObservationPartitionSummary> {
    let mut partitions = BTreeMap::<(usize, String, usize), (usize, u64)>::new();
    for replay in replays {
        let Some(missing) = &replay.missing_abstraction else {
            continue;
        };
        let entry = partitions
            .entry((
                missing.decision_depth,
                missing.legal_choice_catalog_sha256.clone(),
                missing.legal_choice_count,
            ))
            .or_insert((0, replay.seed));
        entry.0 += 1;
        entry.1 = entry.1.min(replay.seed);
    }
    let mut summaries = partitions
        .into_iter()
        .map(
            |(
                (decision_depth, legal_choice_catalog_sha256, legal_choice_count),
                (replay_count, representative_seed),
            )| MissingObservationPartitionSummary {
                decision_depth,
                legal_choice_catalog_sha256,
                legal_choice_count,
                replay_count,
                representative_seed,
            },
        )
        .collect::<Vec<_>>();
    summaries.sort_unstable_by(|left, right| {
        right
            .decision_depth
            .cmp(&left.decision_depth)
            .then_with(|| right.replay_count.cmp(&left.replay_count))
            .then_with(|| left.legal_choice_count.cmp(&right.legal_choice_count))
            .then_with(|| left.representative_seed.cmp(&right.representative_seed))
    });
    summaries
}

fn acquisition_candidate_seeds(
    replays: &[ObservationPolicyReplay],
    partition_limit: usize,
    candidates_per_partition: usize,
) -> Vec<u64> {
    let partitions = missing_partition_summaries(replays);
    let mut seeds = Vec::new();
    for partition in partitions.iter().take(partition_limit) {
        let mut representatives = replays
            .iter()
            .filter_map(|replay| {
                let missing = replay.missing_abstraction.as_ref()?;
                (missing.decision_depth == partition.decision_depth
                    && missing.legal_choice_catalog_sha256 == partition.legal_choice_catalog_sha256
                    && missing.legal_choice_count == partition.legal_choice_count)
                    .then_some(replay.seed)
            })
            .collect::<Vec<_>>();
        representatives.sort_unstable();
        representatives.truncate(candidates_per_partition);
        seeds.extend(representatives);
    }
    seeds
}

pub fn search_observation_policy(
    base: &EnvConfig,
    reward: &RewardConfig,
    scenario: &MicroCombatScenario,
    search_seeds: &[u64],
    holdout_seeds: &[u64],
    config: ObservationPolicySearchConfig,
) -> Result<ObservationPolicySearchReport, String> {
    let config = config.validate(search_seeds.len())?;
    if search_seeds.iter().any(|seed| holdout_seeds.contains(seed)) {
        return Err("observation-policy search and holdout seeds must be disjoint".into());
    }
    let simulators = search_seeds
        .iter()
        .map(|seed| SingleCellSearchSimulator::new(base, reward, scenario, *seed))
        .collect::<Result<Vec<_>, _>>()?;
    let root = PolicyNode {
        particles: simulators
            .iter()
            .map(|simulator| PolicyParticle {
                state: Arc::new(simulator.root().clone()),
                decisions: 0,
            })
            .collect(),
        rules: BTreeMap::new(),
        exact_observation_bindings: BTreeMap::new(),
        depth_choices: BTreeMap::new(),
    };
    let search_started = Instant::now();
    let (best, expansions, truncated, algorithm_iterations, work) = match config.algorithm {
        ObservationPolicySearchAlgorithm::Beam => {
            beam_search_policy(root, &simulators, scenario.objective, config)?
        }
        ObservationPolicySearchAlgorithm::InformationSetMcts => {
            information_set_mcts_policy(root, &simulators, scenario, config)?
        }
    };
    let mut restore_profile = PlannerRestoreAggregate::default();
    for simulator in &simulators {
        restore_profile.merge(simulator.restore_profile());
    }
    let search_cost = ObservationPolicySearchCost {
        wall_time_micros: search_started.elapsed().as_micros(),
        host_peak_resident_set_bytes: host_peak_resident_set_bytes(),
        planner_restore_count: restore_profile.count,
        planner_restore_tile_materialization_ns: restore_profile.tile_materialization_ns,
        planner_restore_cell_materialization_ns: restore_profile.cell_materialization_ns,
        planner_restore_hash_seed_materialization_ns: restore_profile.hash_seed_materialization_ns,
        planner_restore_resolver_reconstruction_ns: restore_profile.resolver_reconstruction_ns,
        planner_restore_topology_validation_ns: restore_profile.topology_validation_ns,
        planner_restore_cell_validation_store_and_passive_index_ns: restore_profile
            .cell_validation_store_and_passive_index_ns,
        planner_restore_tile_validation_and_passive_index_ns: restore_profile
            .tile_validation_and_passive_index_ns,
        planner_restore_hash_initialization_ns: restore_profile.hash_initialization_ns,
        planner_restore_scratch_initialization_ns: restore_profile.scratch_initialization_ns,
        planner_restore_metabolic_index_ns: restore_profile.metabolic_index_ns,
        planner_restore_integrity_validation_ns: restore_profile.integrity_validation_ns,
        planner_environment_restore_total_ns: restore_profile.environment_total_ns,
        planner_restore_total_ns: restore_profile.total_ns,
        planner_restored_tiles: restore_profile.restored_tiles,
        planner_restored_cells: restore_profile.restored_cells,
        peak_retained_policy_nodes: work.peak_retained_policy_nodes,
        peak_retained_particles: work.peak_retained_particles,
        peak_retained_rules: work.peak_retained_rules,
        peak_retained_untried_choices: work.peak_retained_untried_choices,
        peak_retained_canonical_tile_page_references: work
            .peak_retained_canonical_tile_page_references,
        peak_retained_unique_canonical_tile_pages: work.peak_retained_unique_canonical_tile_pages,
        peak_retained_canonical_tile_chunk_references: work
            .peak_retained_canonical_tile_chunk_references,
        peak_retained_unique_canonical_tile_chunks: work.peak_retained_unique_canonical_tile_chunks,
        peak_retained_canonical_cell_chunk_references: work
            .peak_retained_canonical_cell_chunk_references,
        peak_retained_unique_canonical_cell_chunks: work.peak_retained_unique_canonical_cell_chunks,
        peak_retained_canonical_hash_seed_chunk_references: work
            .peak_retained_canonical_hash_seed_chunk_references,
        peak_retained_unique_canonical_hash_seed_chunks: work
            .peak_retained_unique_canonical_hash_seed_chunks,
        peak_retained_mcts_choice_index_overrides: work.peak_retained_mcts_choice_index_overrides,
        estimated_peak_retained_mcts_choice_bytes_lower_bound: work
            .estimated_peak_retained_mcts_choice_bytes_lower_bound,
        mcts_unique_choice_catalogs: work.mcts_unique_choice_catalogs,
        mcts_choice_catalog_intern_hits: work.mcts_choice_catalog_intern_hits,
        estimated_peak_retained_policy_bytes_lower_bound: work
            .estimated_peak_retained_policy_bytes_lower_bound,
        policy_node_clones: work.policy_node_clones,
        rollout_policy_clones: work.rollout_policy_clones,
        shared_particle_state_clones: work.shared_particle_state_clones,
        particle_state_allocations: search_seeds.len().saturating_add(expansions),
        beam_generated_children: work.beam_generated_children,
        beam_pruned_children: work.beam_pruned_children,
    };

    let policy = ObservationPolicy {
        schema_version: MICRO_COMBAT_OBSERVATION_POLICY_SCHEMA_VERSION,
        exact_observation_key: "canonical_mind_input_without_random_block_sha256_v1".into(),
        action_selection_key: "sparse_quantized_rl_observation_plus_exact_legal_catalog_sha256_v1"
            .into(),
        randomness_handling: "deterministic_policy_ignores_private_random_block".into(),
        observation_quantization_levels: config.observation_quantization_levels,
        max_nearest_fallback_l1_distance: config.max_nearest_fallback_l1_distance,
        rules: best.rules,
        exact_observation_bindings: best.exact_observation_bindings,
    };
    let search_replays = simulators
        .iter()
        .zip(search_seeds)
        .map(|(simulator, seed)| replay_policy(simulator, *seed, &policy, config))
        .collect::<Result<Vec<_>, _>>()?;
    let holdout_replays = holdout_seeds
        .iter()
        .map(|seed| {
            let simulator = SingleCellSearchSimulator::new(base, reward, scenario, *seed)?;
            replay_policy(&simulator, *seed, &policy, config)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let search_covered_to_requested_horizon =
        search_replays.iter().all(|replay| replay.coverage_complete);
    let holdout_covered_to_requested_horizon = !holdout_replays.is_empty()
        && holdout_replays
            .iter()
            .all(|replay| replay.coverage_complete);
    let search_objective_success_count = search_replays
        .iter()
        .filter(|replay| replay.objective_success)
        .count();
    let holdout_objective_success_count = holdout_replays
        .iter()
        .filter(|replay| replay.objective_success)
        .count();
    let unique_exact_traces = |replays: &[ObservationPolicyReplay]| {
        replays
            .iter()
            .map(|replay| replay.observation_sha256_trace.clone())
            .collect::<BTreeSet<_>>()
            .len()
    };
    let unique_abstract_traces = |replays: &[ObservationPolicyReplay]| {
        replays
            .iter()
            .map(|replay| replay.abstraction_sha256_trace.clone())
            .collect::<BTreeSet<_>>()
            .len()
    };
    let mut rules_per_depth = BTreeMap::<usize, usize>::new();
    let mut choices_per_depth = BTreeMap::<usize, Vec<PolicyChoice>>::new();
    for rule in policy.rules.values() {
        *rules_per_depth.entry(rule.decision_depth).or_default() += 1;
        let choices = choices_per_depth.entry(rule.decision_depth).or_default();
        if !choices.contains(&rule.choice) {
            choices.push(rule.choice);
        }
    }
    let policy_branching_depths = rules_per_depth
        .into_iter()
        .filter_map(|(depth, count)| (count > 1).then_some(depth))
        .collect();
    let policy_choice_branching_depths = choices_per_depth
        .into_iter()
        .filter_map(|(depth, choices)| (choices.len() > 1).then_some(depth))
        .collect();
    Ok(ObservationPolicySearchReport {
        schema_version: MICRO_COMBAT_OBSERVATION_POLICY_SEARCH_REPORT_SCHEMA_VERSION,
        evidence_class: "privileged_particle_synthesis_observation_legal_policy".into(),
        search_domain: match (config.algorithm, config.force_open_loop_actions) {
            (ObservationPolicySearchAlgorithm::Beam, false) => {
                "single_cell_no_split_no_signal_beam_policy_search"
            }
            (ObservationPolicySearchAlgorithm::InformationSetMcts, false) => {
                "single_cell_no_split_no_signal_information_set_mcts"
            }
            (ObservationPolicySearchAlgorithm::Beam, true) => {
                "single_cell_no_split_no_signal_open_loop_beam_control"
            }
            (ObservationPolicySearchAlgorithm::InformationSetMcts, true) => {
                "single_cell_no_split_no_signal_open_loop_mcts_control"
            }
        }
        .into(),
        scenario: scenario.name.clone(),
        search_seeds: search_seeds.to_vec(),
        holdout_seeds: holdout_seeds.to_vec(),
        objective: scenario.objective,
        config,
        search_cost,
        algorithm_iterations,
        expansions,
        policy_sha256: policy_sha256(&policy)?,
        policy_rule_count: policy.rules.len(),
        exact_observation_binding_count: policy.exact_observation_bindings.len(),
        policy_branching_depths,
        policy_choice_branching_depths,
        search_unique_observation_traces: unique_exact_traces(&search_replays),
        holdout_unique_observation_traces: unique_exact_traces(&holdout_replays),
        search_unique_abstraction_traces: unique_abstract_traces(&search_replays),
        holdout_unique_abstraction_traces: unique_abstract_traces(&holdout_replays),
        search_exact_observation_binding_hits: search_replays
            .iter()
            .map(|replay| replay.exact_observation_binding_hits)
            .sum(),
        holdout_exact_observation_binding_hits: holdout_replays
            .iter()
            .map(|replay| replay.exact_observation_binding_hits)
            .sum(),
        search_nearest_fallback_rule_hits: search_replays
            .iter()
            .map(|replay| replay.nearest_fallback_rule_hits)
            .sum(),
        holdout_nearest_fallback_rule_hits: holdout_replays
            .iter()
            .map(|replay| replay.nearest_fallback_rule_hits)
            .sum(),
        holdout_maximum_nearest_fallback_l1_distance: holdout_replays
            .iter()
            .map(|replay| replay.maximum_nearest_fallback_l1_distance)
            .max()
            .unwrap_or(0),
        search_covered_to_requested_horizon,
        holdout_covered_to_requested_horizon,
        truncated,
        search_objective_success_count,
        all_search_objective_success: search_objective_success_count == search_seeds.len(),
        holdout_objective_success_count,
        all_holdout_objective_success: !holdout_seeds.is_empty()
            && holdout_objective_success_count == holdout_seeds.len(),
        policy,
        search_replays,
        holdout_replays,
    })
}

fn beam_search_policy(
    root: PolicyNode,
    simulators: &[SingleCellSearchSimulator],
    objective: MicroCombatObjective,
    config: ObservationPolicySearchConfig,
) -> Result<(PolicyNode, usize, bool, usize, SearchWorkMetrics), String> {
    let mut work = SearchWorkMetrics::default();
    work.record_policy_node_clone(&root);
    let mut best = root.clone();
    let mut frontier = vec![root];
    work.observe_retained(frontier.iter(), 0);
    let mut expansions = 0usize;
    let mut truncated = false;
    let mut iterations = 0usize;

    'search: while !frontier.is_empty() {
        iterations += 1;
        let mut next = Vec::new();
        for mut node in std::mem::take(&mut frontier) {
            if !advance_bound_particles(&mut node, simulators, config, &mut expansions)? {
                truncated = true;
                if compare_policy_nodes(&node, &best, objective, config).is_gt() {
                    best = node;
                }
                break 'search;
            }
            let Some(information_set) = next_information_set(&node, simulators, config)? else {
                if compare_policy_nodes(&node, &best, objective, config).is_gt() {
                    best = node;
                }
                continue;
            };
            let choices = information_set_choices(&node, &information_set, config)?;
            for choice in choices {
                let Some(child) = apply_information_set_choice(
                    &node,
                    &information_set,
                    choice,
                    simulators,
                    config,
                    &mut expansions,
                    &mut work,
                )?
                else {
                    truncated = true;
                    break 'search;
                };
                if compare_policy_nodes(&child, &best, objective, config).is_gt() {
                    work.record_policy_node_clone(&child);
                    best = child.clone();
                }
                work.beam_generated_children += 1;
                if retain_top_policy_candidate(
                    &mut next,
                    child,
                    config.beam_width,
                    objective,
                    config,
                ) {
                    work.beam_pruned_children += 1;
                    truncated = true;
                }
            }
        }
        work.observe_retained(next.iter(), 0);
        frontier = next;
    }
    Ok((best, expansions, truncated, iterations, work))
}

/// Retain the exact best `limit` nodes in preferred-first order. Returns true
/// when either the incoming candidate or the previous worst node was pruned.
fn retain_top_policy_candidate(
    frontier: &mut Vec<PolicyNode>,
    candidate: PolicyNode,
    limit: usize,
    objective: MicroCombatObjective,
    config: ObservationPolicySearchConfig,
) -> bool {
    debug_assert!(limit > 0);
    let position = frontier.partition_point(|existing| {
        !compare_policy_nodes_with_tie_break(existing, &candidate, objective, config).is_lt()
    });
    if frontier.len() == limit && position == limit {
        return true;
    }
    frontier.insert(position, candidate);
    if frontier.len() > limit {
        frontier.pop();
        true
    } else {
        false
    }
}

struct MctsTreeNode {
    policy: PolicyNode,
    children: Vec<usize>,
    choice_catalog: usize,
    untried_choices: LazyPolicyChoicePermutation,
    visits: u64,
    value_sum: f64,
}

#[derive(Default)]
struct MctsChoiceCatalogs {
    catalogs: Vec<Arc<[PolicyChoice]>>,
    indices: HashMap<Arc<[PolicyChoice]>, usize>,
    retained_choice_bytes: usize,
}

impl MctsChoiceCatalogs {
    fn intern(&mut self, choices: Vec<PolicyChoice>) -> (usize, bool) {
        let choices = Arc::<[PolicyChoice]>::from(choices);
        if let Some(index) = self.indices.get(&choices) {
            return (*index, false);
        }
        let index = self.catalogs.len();
        self.retained_choice_bytes = self
            .retained_choice_bytes
            .saturating_add(choices.len().saturating_mul(size_of::<PolicyChoice>()));
        self.indices.insert(Arc::clone(&choices), index);
        self.catalogs.push(choices);
        (index, true)
    }

    fn get(&self, index: usize) -> &[PolicyChoice] {
        &self.catalogs[index]
    }

    fn len(&self) -> usize {
        self.catalogs.len()
    }

    fn retained_choice_bytes(&self) -> usize {
        self.retained_choice_bytes
    }
}

/// A sparse, exact representation of repeatedly applying `Vec::swap_remove`
/// to a canonical choice catalog. Only positions whose original catalog index
/// has changed are retained; the catalog itself can be regenerated from the
/// immutable policy-node state when a choice is expanded.
struct LazyPolicyChoicePermutation {
    catalog_len: usize,
    remaining: usize,
    index_overrides: Vec<(usize, usize)>,
}

impl LazyPolicyChoicePermutation {
    fn new(catalog_len: usize) -> Self {
        Self {
            catalog_len,
            remaining: catalog_len,
            index_overrides: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.remaining == 0
    }

    fn remaining(&self) -> usize {
        self.remaining
    }

    fn index_override_count(&self) -> usize {
        self.index_overrides.len()
    }

    fn estimated_retained_bytes(&self) -> usize {
        size_of::<Self>().saturating_add(
            self.index_overrides
                .capacity()
                .saturating_mul(size_of::<(usize, usize)>()),
        )
    }

    fn original_index_at(&self, position: usize) -> usize {
        self.index_overrides
            .binary_search_by_key(&position, |(position, _)| *position)
            .ok()
            .map_or(position, |index| self.index_overrides[index].1)
    }

    fn set_original_index(&mut self, position: usize, original_index: usize) {
        match self
            .index_overrides
            .binary_search_by_key(&position, |(position, _)| *position)
        {
            Ok(index) if position == original_index => {
                self.index_overrides.remove(index);
            }
            Ok(index) => self.index_overrides[index].1 = original_index,
            Err(_) if position == original_index => {}
            Err(index) => self
                .index_overrides
                .insert(index, (position, original_index)),
        }
    }

    fn remove_position(&mut self, position: usize) {
        if let Ok(index) = self
            .index_overrides
            .binary_search_by_key(&position, |(position, _)| *position)
        {
            self.index_overrides.remove(index);
        }
    }

    fn swap_remove_original_index(&mut self, position: usize) -> Option<usize> {
        if position >= self.remaining {
            return None;
        }
        let selected = self.original_index_at(position);
        let last_position = self.remaining - 1;
        let last = self.original_index_at(last_position);
        self.remove_position(last_position);
        if position != last_position {
            self.set_original_index(position, last);
        }
        self.remaining -= 1;
        Some(selected)
    }

    fn take(
        &mut self,
        catalog: &[PolicyChoice],
        rng: &mut DeterministicSearchRng,
    ) -> Result<PolicyChoice, String> {
        if catalog.len() != self.catalog_len {
            return Err("MCTS legal choice catalog changed for an immutable policy node".into());
        }
        let position = rng.index(self.remaining);
        let original_index = self
            .swap_remove_original_index(position)
            .ok_or("MCTS attempted to expand an exhausted choice permutation")?;
        Ok(catalog[original_index])
    }
}

struct DeterministicSearchRng(u64);

impl DeterministicSearchRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, len: usize) -> usize {
        debug_assert!(len > 0);
        (self.next_u64() % len as u64) as usize
    }
}

fn information_set_mcts_policy(
    mut root: PolicyNode,
    simulators: &[SingleCellSearchSimulator],
    scenario: &MicroCombatScenario,
    config: ObservationPolicySearchConfig,
) -> Result<(PolicyNode, usize, bool, usize, SearchWorkMetrics), String> {
    let mut expansions = 0usize;
    let mut work = SearchWorkMetrics::default();
    if !advance_bound_particles(&mut root, simulators, config, &mut expansions)? {
        work.observe_retained(std::iter::once(&root), 0);
        return Ok((root, expansions, true, 0, work));
    }
    let root_choices = pending_policy_choices(&root, simulators, config)?;
    let root_choice_count = root_choices.len();
    let mut choice_catalogs = MctsChoiceCatalogs::default();
    let (root_choice_catalog, _) = choice_catalogs.intern(root_choices);
    work.mcts_unique_choice_catalogs = choice_catalogs.len();
    work.record_policy_node_clone(&root);
    let mut best = root.clone();
    let root_untried_choices = LazyPolicyChoicePermutation::new(root_choice_count);
    let mut retained_mcts_permutation_bytes = root_untried_choices.estimated_retained_bytes();
    let mut tree = vec![MctsTreeNode {
        policy: root,
        children: Vec::new(),
        choice_catalog: root_choice_catalog,
        untried_choices: root_untried_choices,
        visits: 0,
        value_sum: 0.0,
    }];
    let mut retained_particles = tree[0].policy.particles.len();
    let mut retained_rules = tree[0].policy.rules.len();
    let mut retained_untried_choices = tree[0].untried_choices.remaining();
    let mut retained_mcts_choice_index_overrides = 0usize;
    let mut retained_allocations = RetainedStateAllocations::default();
    let mut retained_policy_bytes =
        estimated_retained_policy_node_bytes(&tree[0].policy, &mut retained_allocations);
    work.observe_checkpoint_sharing(&retained_allocations);
    work.observe_totals(
        tree.len(),
        retained_particles,
        retained_rules,
        retained_untried_choices,
        retained_policy_bytes,
    );
    work.observe_mcts_choice_storage(
        retained_mcts_choice_index_overrides,
        retained_mcts_permutation_bytes.saturating_add(choice_catalogs.retained_choice_bytes()),
    );
    let mut rng = DeterministicSearchRng::new(config.mcts_seed);
    let mut completed_iterations = 0usize;
    let mut truncated = false;

    'search: for _ in 0..config.mcts_max_iterations {
        if expansions == config.max_expansions {
            truncated = true;
            break;
        }
        let mut current = 0usize;
        let mut path = vec![current];
        while tree[current].untried_choices.is_empty() && !tree[current].children.is_empty() {
            current = select_mcts_child(&tree, current, config);
            path.push(current);
        }

        if !tree[current].untried_choices.is_empty() {
            let information_set = next_information_set(&tree[current].policy, simulators, config)?
                .ok_or("MCTS node retained choices without a pending information set")?;
            let before_overrides = tree[current].untried_choices.index_override_count();
            let before_storage = tree[current].untried_choices.estimated_retained_bytes();
            let choice_catalog = tree[current].choice_catalog;
            let choice = tree[current]
                .untried_choices
                .take(choice_catalogs.get(choice_catalog), &mut rng)?;
            let after_overrides = tree[current].untried_choices.index_override_count();
            let after_storage = tree[current].untried_choices.estimated_retained_bytes();
            retained_untried_choices = retained_untried_choices.saturating_sub(1);
            retained_mcts_choice_index_overrides = retained_mcts_choice_index_overrides
                .saturating_sub(before_overrides)
                .saturating_add(after_overrides);
            retained_mcts_permutation_bytes = retained_mcts_permutation_bytes
                .saturating_sub(before_storage)
                .saturating_add(after_storage);
            work.observe_mcts_choice_storage(
                retained_mcts_choice_index_overrides,
                retained_mcts_permutation_bytes
                    .saturating_add(choice_catalogs.retained_choice_bytes()),
            );
            let Some(mut child_policy) = apply_information_set_choice(
                &tree[current].policy,
                &information_set,
                choice,
                simulators,
                config,
                &mut expansions,
                &mut work,
            )?
            else {
                truncated = true;
                break;
            };
            let fully_advanced =
                advance_bound_particles(&mut child_policy, simulators, config, &mut expansions)?;
            if compare_policy_nodes(&child_policy, &best, scenario.objective, config).is_gt() {
                work.record_policy_node_clone(&child_policy);
                best = child_policy.clone();
            }
            let untried_choice_catalog = if fully_advanced {
                pending_policy_choices(&child_policy, simulators, config)?
            } else {
                Vec::new()
            };
            let untried_choice_count = untried_choice_catalog.len();
            let (choice_catalog, unique_catalog) = choice_catalogs.intern(untried_choice_catalog);
            if unique_catalog {
                work.mcts_unique_choice_catalogs = choice_catalogs.len();
            } else {
                work.mcts_choice_catalog_intern_hits += 1;
            }
            let untried_choices = LazyPolicyChoicePermutation::new(untried_choice_count);
            retained_particles = retained_particles.saturating_add(child_policy.particles.len());
            retained_rules = retained_rules.saturating_add(child_policy.rules.len());
            retained_untried_choices =
                retained_untried_choices.saturating_add(untried_choices.remaining());
            retained_mcts_permutation_bytes = retained_mcts_permutation_bytes
                .saturating_add(untried_choices.estimated_retained_bytes());
            retained_policy_bytes = retained_policy_bytes.saturating_add(
                estimated_retained_policy_node_bytes(&child_policy, &mut retained_allocations),
            );
            work.observe_checkpoint_sharing(&retained_allocations);
            let child_index = tree.len();
            tree.push(MctsTreeNode {
                policy: child_policy,
                children: Vec::new(),
                choice_catalog,
                untried_choices,
                visits: 0,
                value_sum: 0.0,
            });
            work.observe_totals(
                tree.len(),
                retained_particles,
                retained_rules,
                retained_untried_choices,
                retained_policy_bytes,
            );
            work.observe_mcts_choice_storage(
                retained_mcts_choice_index_overrides,
                retained_mcts_permutation_bytes
                    .saturating_add(choice_catalogs.retained_choice_bytes()),
            );
            tree[current].children.push(child_index);
            current = child_index;
            path.push(current);
            if !fully_advanced {
                truncated = true;
                break;
            }
        }

        work.record_rollout_policy_clone(&tree[current].policy);
        let mut rollout = tree[current].policy.clone();
        let mut budget_exhausted = false;
        for _ in 0..config.mcts_rollout_rule_depth {
            let Some(information_set) = next_information_set(&rollout, simulators, config)? else {
                break;
            };
            let choices = information_set_choices(&rollout, &information_set, config)?;
            if choices.is_empty() {
                break;
            }
            let choice = choices[rng.index(choices.len())];
            let Some(mut next) = apply_information_set_choice(
                &rollout,
                &information_set,
                choice,
                simulators,
                config,
                &mut expansions,
                &mut work,
            )?
            else {
                budget_exhausted = true;
                break;
            };
            let fully_advanced =
                advance_bound_particles(&mut next, simulators, config, &mut expansions)?;
            rollout = next;
            if compare_policy_nodes(&rollout, &best, scenario.objective, config).is_gt() {
                work.record_policy_node_clone(&rollout);
                best = rollout.clone();
            }
            if !fully_advanced {
                budget_exhausted = true;
                break;
            }
        }
        let value = mcts_policy_value(&rollout, scenario);
        for index in path {
            tree[index].visits += 1;
            tree[index].value_sum += value;
        }
        completed_iterations += 1;
        if budget_exhausted {
            truncated = true;
            break 'search;
        }
    }
    if completed_iterations == config.mcts_max_iterations {
        truncated = true;
    }
    Ok((best, expansions, truncated, completed_iterations, work))
}

fn select_mcts_child(
    tree: &[MctsTreeNode],
    parent: usize,
    config: ObservationPolicySearchConfig,
) -> usize {
    let parent_visits = tree[parent].visits.max(1) as f64;
    let exploration = f64::from(config.mcts_exploration_milli) / 1_000.0;
    tree[parent]
        .children
        .iter()
        .copied()
        .max_by(|left, right| {
            let score = |index: usize| {
                let node = &tree[index];
                if node.visits == 0 {
                    return f64::INFINITY;
                }
                node.value_sum / node.visits as f64
                    + exploration * (parent_visits.ln() / node.visits as f64).sqrt()
            };
            score(*left)
                .total_cmp(&score(*right))
                .then_with(|| right.cmp(left))
        })
        .expect("MCTS selection requires at least one child")
}

fn mcts_policy_value(node: &PolicyNode, scenario: &MicroCombatScenario) -> f64 {
    let training_initial =
        (scenario.training_cells as u128 * u128::from(scenario.training_initial_energy)).max(1);
    let opponent_initial =
        (scenario.opponent_cells as u128 * u128::from(scenario.opponent_initial_energy)).max(1);
    let total = node
        .particles
        .iter()
        .map(|particle| {
            if particle.state.objective_success(scenario.objective) {
                return 1.0;
            }
            let alive = if particle.state.training_cells > 0 {
                1.0
            } else {
                0.0
            };
            let time = (particle.state.sim_time_quanta as f64
                / scenario.sim_time_limit_quanta.max(1) as f64)
                .clamp(0.0, 1.0);
            let own_energy = (particle.state.training_stored_energy as f64
                / training_initial as f64)
                .clamp(0.0, 1.0);
            match scenario.objective {
                MicroCombatObjective::Survival => alive * (0.2 + 0.7 * time + 0.1 * own_energy),
                MicroCombatObjective::Elimination => {
                    let opponent_cells = (particle.state.opponent_cells as f64
                        / scenario.opponent_cells.max(1) as f64)
                        .clamp(0.0, 1.0);
                    let opponent_energy = (particle.state.opponent_stored_energy as f64
                        / opponent_initial as f64)
                        .clamp(0.0, 1.0);
                    0.1 * alive
                        + 0.45 * (1.0 - opponent_cells)
                        + 0.35 * (1.0 - opponent_energy)
                        + 0.1 * time
                }
            }
        })
        .sum::<f64>();
    total / node.particles.len().max(1) as f64
}

fn pending_policy_choices(
    node: &PolicyNode,
    simulators: &[SingleCellSearchSimulator],
    config: ObservationPolicySearchConfig,
) -> Result<Vec<PolicyChoice>, String> {
    let Some(information_set) = next_information_set(node, simulators, config)? else {
        return Ok(Vec::new());
    };
    information_set_choices(node, &information_set, config)
}

fn information_set_choices(
    node: &PolicyNode,
    information_set: &PendingInformationSet,
    config: ObservationPolicySearchConfig,
) -> Result<Vec<PolicyChoice>, String> {
    let choices = information_set.search_choices.clone();
    if config.force_open_loop_actions {
        if let Some(bound) = node.depth_choices.get(&information_set.depth) {
            return Ok(choices
                .contains(bound)
                .then_some(*bound)
                .into_iter()
                .collect());
        }
    }
    Ok(choices)
}

fn apply_information_set_choice(
    node: &PolicyNode,
    information_set: &PendingInformationSet,
    choice: PolicyChoice,
    simulators: &[SingleCellSearchSimulator],
    config: ObservationPolicySearchConfig,
    expansions: &mut usize,
    work: &mut SearchWorkMetrics,
) -> Result<Option<PolicyNode>, String> {
    if expansions
        .checked_add(information_set.particles.len())
        .is_none_or(|total| total > config.max_expansions)
    {
        return Ok(None);
    }
    work.record_policy_node_clone(node);
    let mut child = node.clone();
    if config.force_open_loop_actions {
        if let Some(previous) = child.depth_choices.insert(information_set.depth, choice) {
            if previous != choice {
                return Err("open-loop policy search attempted to branch by observation".into());
            }
        }
    }
    let memory = policy_memory(information_set.depth + 1)?;
    for (particle, exact_observation_sha256) in &information_set.particles {
        child.particles[*particle].state =
            Arc::new(simulators[*particle].transition_observation_choice(
                &child.particles[*particle].state,
                exact_observation_sha256,
                choice,
                memory.clone(),
            )?);
        child.particles[*particle].decisions += 1;
        bind_exact_observation(
            &mut child.exact_observation_bindings,
            exact_observation_sha256,
            &information_set.abstraction_sha256,
        )?;
    }
    *expansions += information_set.particles.len();
    let replaced = child.rules.insert(
        information_set.abstraction_sha256.clone(),
        ObservationPolicyRule {
            decision_depth: information_set.depth,
            choice,
            family: policy_action_family(choice.action)
                .map_or("unknown", PolicyActionFamily::name)
                .to_string(),
            next_private_memory: memory,
            abstraction: information_set.abstraction.clone(),
        },
    );
    if replaced.is_some() {
        return Err("observation-policy search attempted to redefine a rule".into());
    }
    Ok(Some(child))
}

fn next_information_set(
    node: &PolicyNode,
    simulators: &[SingleCellSearchSimulator],
    config: ObservationPolicySearchConfig,
) -> Result<Option<PendingInformationSet>, String> {
    let mut selected: Option<(usize, String)> = None;
    let mut observations = Vec::with_capacity(node.particles.len());
    for (index, particle) in node.particles.iter().enumerate() {
        if particle.state.done() || particle.decisions == config.max_decisions_per_particle {
            observations.push(None);
            continue;
        }
        let frontier =
            simulators[index].frontier(&particle.state, config.equivalent_effort_pruning)?;
        let exact = mind_observation_sha256(&frontier.input)?;
        let abstraction = observation_abstraction(
            &frontier.input,
            particle.decisions,
            config.observation_quantization_levels,
            &frontier.legal_choices,
        )?;
        let abstract_key = observation_abstraction_sha256(&abstraction)?;
        if node.rules.contains_key(&abstract_key) {
            return Err("bound observation remained after policy-node normalization".into());
        }
        let candidate = (particle.decisions, abstract_key.clone());
        if selected.as_ref().is_none_or(|current| candidate < *current) {
            selected = Some(candidate);
        }
        observations.push(Some((
            exact,
            abstract_key,
            abstraction,
            frontier.search_choices,
        )));
    }
    let Some((depth, abstraction_sha256)) = selected else {
        return Ok(None);
    };
    let abstraction = observations
        .iter()
        .flatten()
        .find(|(_, key, _, _)| key == &abstraction_sha256)
        .map(|(_, _, abstraction, _)| abstraction.clone())
        .ok_or("selected observation abstraction was not present")?;
    let mut particles = Vec::new();
    let mut search_choices = None;
    for (index, particle) in node.particles.iter().enumerate() {
        let Some((exact, key, peer_abstraction, peer_choices)) = observations[index].as_ref()
        else {
            continue;
        };
        if peer_abstraction != &abstraction
            || particle.decisions != depth
            || key != &abstraction_sha256
        {
            continue;
        }
        if let Some(expected) = &search_choices {
            if expected != peer_choices {
                return Err(
                    "equal Mind observations produced different legal action catalogs".into(),
                );
            }
        } else {
            search_choices = Some(peer_choices.clone());
        }
        particles.push((index, exact.clone()));
    }
    Ok(Some(PendingInformationSet {
        depth,
        abstraction_sha256,
        abstraction,
        particles,
        search_choices: search_choices.ok_or("selected information set contained no particles")?,
    }))
}

/// Apply already-bound abstract rules when a particle reaches a generalized
/// state after another information set has been resolved. These transitions
/// are authoritative simulator work and therefore consume the expansion cap.
fn advance_bound_particles(
    node: &mut PolicyNode,
    simulators: &[SingleCellSearchSimulator],
    config: ObservationPolicySearchConfig,
    expansions: &mut usize,
) -> Result<bool, String> {
    for (index, particle) in node.particles.iter_mut().enumerate() {
        while !particle.state.done() && particle.decisions < config.max_decisions_per_particle {
            let frontier = simulators[index].frontier(&particle.state, false)?;
            let exact = mind_observation_sha256(&frontier.input)?;
            let abstraction = observation_abstraction(
                &frontier.input,
                particle.decisions,
                config.observation_quantization_levels,
                &frontier.legal_choices,
            )?;
            let abstract_key = observation_abstraction_sha256(&abstraction)?;
            let Some(rule) = node.rules.get(&abstract_key) else {
                break;
            };
            if rule.abstraction != abstraction || rule.decision_depth != particle.decisions {
                return Err("observation abstraction hash or depth collision".into());
            }
            if *expansions == config.max_expansions {
                return Ok(false);
            }
            particle.state = Arc::new(simulators[index].transition_observation_choice(
                &particle.state,
                &exact,
                rule.choice,
                rule.next_private_memory.clone(),
            )?);
            particle.decisions += 1;
            *expansions += 1;
            bind_exact_observation(&mut node.exact_observation_bindings, &exact, &abstract_key)?;
        }
    }
    Ok(true)
}

fn bind_exact_observation(
    bindings: &mut BTreeMap<String, String>,
    exact_observation_sha256: &str,
    abstraction_sha256: &str,
) -> Result<(), String> {
    if let Some(existing) = bindings.insert(
        exact_observation_sha256.to_string(),
        abstraction_sha256.to_string(),
    ) {
        if existing != abstraction_sha256 {
            return Err("one exact observation mapped to two abstractions".into());
        }
    }
    Ok(())
}

fn compare_policy_nodes(
    left: &PolicyNode,
    right: &PolicyNode,
    objective: MicroCombatObjective,
    config: ObservationPolicySearchConfig,
) -> Ordering {
    let resolved = |node: &PolicyNode| {
        node.particles
            .iter()
            .filter(|particle| {
                particle.state.done() || particle.decisions == config.max_decisions_per_particle
            })
            .count()
    };
    let successes = |node: &PolicyNode| {
        node.particles
            .iter()
            .filter(|particle| particle.state.objective_success(objective))
            .count()
    };
    let alive = |node: &PolicyNode| {
        node.particles
            .iter()
            .filter(|particle| particle.state.training_cells > 0)
            .count()
    };
    let decisions = |node: &PolicyNode| {
        node.particles
            .iter()
            .map(|particle| particle.decisions)
            .sum::<usize>()
    };
    let own_energy = |node: &PolicyNode| {
        node.particles
            .iter()
            .map(|particle| particle.state.training_stored_energy)
            .sum::<u128>()
    };
    let sim_time = |node: &PolicyNode| {
        node.particles
            .iter()
            .map(|particle| particle.state.sim_time_quanta)
            .sum::<u64>()
    };
    successes(left)
        .cmp(&successes(right))
        .then_with(|| alive(left).cmp(&alive(right)))
        .then_with(|| match objective {
            MicroCombatObjective::Survival => sim_time(left)
                .cmp(&sim_time(right))
                .then_with(|| own_energy(left).cmp(&own_energy(right))),
            MicroCombatObjective::Elimination => {
                let opponent_cells = |node: &PolicyNode| {
                    node.particles
                        .iter()
                        .map(|particle| particle.state.opponent_cells)
                        .sum::<usize>()
                };
                let opponent_energy = |node: &PolicyNode| {
                    node.particles
                        .iter()
                        .map(|particle| particle.state.opponent_stored_energy)
                        .sum::<u128>()
                };
                opponent_cells(right)
                    .cmp(&opponent_cells(left))
                    .then_with(|| opponent_energy(right).cmp(&opponent_energy(left)))
                    .then_with(|| own_energy(left).cmp(&own_energy(right)))
            }
        })
        .then_with(|| decisions(left).cmp(&decisions(right)))
        .then_with(|| resolved(left).cmp(&resolved(right)))
        .then_with(|| right.rules.len().cmp(&left.rules.len()))
}

fn replay_policy(
    simulator: &SingleCellSearchSimulator,
    seed: u64,
    policy: &ObservationPolicy,
    config: ObservationPolicySearchConfig,
) -> Result<ObservationPolicyReplay, String> {
    let mut state = simulator.root().clone();
    let mut decisions = 0usize;
    let mut missing = None;
    let mut observation_sha256_trace = Vec::new();
    let mut abstraction_sha256_trace = Vec::new();
    let mut exact_observation_binding_hits = 0usize;
    let mut exact_abstraction_rule_hits = 0usize;
    let mut nearest_fallback_rule_hits = 0usize;
    let mut maximum_nearest_fallback_l1_distance = 0u32;
    let mut missing_abstraction = None;
    while !state.done() && decisions < config.max_decisions_per_particle {
        let frontier = simulator.frontier(&state, false)?;
        let exact_key = mind_observation_sha256(&frontier.input)?;
        observation_sha256_trace.push(exact_key.clone());
        if policy.exact_observation_bindings.contains_key(&exact_key) {
            exact_observation_binding_hits += 1;
        }
        let abstraction = observation_abstraction(
            &frontier.input,
            decisions,
            policy.observation_quantization_levels,
            &frontier.legal_choices,
        )?;
        let abstract_key = observation_abstraction_sha256(&abstraction)?;
        abstraction_sha256_trace.push(abstract_key.clone());
        let Some((rule, fallback_distance)) = select_abstract_rule(
            policy,
            &abstract_key,
            &abstraction,
            policy.max_nearest_fallback_l1_distance,
        ) else {
            missing = Some(abstract_key);
            missing_abstraction = Some(abstraction);
            break;
        };
        if fallback_distance == 0 {
            exact_abstraction_rule_hits += 1;
            if rule.abstraction != abstraction {
                return Err("observation-policy abstraction hash collision".into());
            }
        } else {
            nearest_fallback_rule_hits += 1;
            maximum_nearest_fallback_l1_distance =
                maximum_nearest_fallback_l1_distance.max(fallback_distance);
        }
        if rule.decision_depth != decisions {
            return Err("observation-policy private-memory depth did not replay".into());
        }
        state = simulator.transition_observation_choice(
            &state,
            &exact_key,
            rule.choice,
            rule.next_private_memory.clone(),
        )?;
        decisions += 1;
    }
    let coverage_complete =
        missing.is_none() && (state.done() || decisions == config.max_decisions_per_particle);
    Ok(ObservationPolicyReplay {
        seed,
        coverage_complete,
        objective_success: state.objective_success(simulator.objective()),
        final_outcome: outcome_name(state.terminal_outcome).into(),
        decisions,
        observation_sha256_trace,
        abstraction_sha256_trace,
        exact_observation_binding_hits,
        exact_abstraction_rule_hits,
        nearest_fallback_rule_hits,
        maximum_nearest_fallback_l1_distance,
        missing_abstraction_sha256: missing,
        missing_abstraction,
        final_sim_time_quanta: state.sim_time_quanta,
        final_training_cells: state.training_cells,
        final_opponent_cells: state.opponent_cells,
        final_training_stored_energy: state.training_stored_energy,
        final_opponent_stored_energy: state.opponent_stored_energy,
        final_continuation_sha256: state.continuation_sha256,
    })
}

pub(crate) fn select_abstract_rule<'a>(
    policy: &'a ObservationPolicy,
    abstraction_sha256: &str,
    abstraction: &ObservationAbstraction,
    maximum_distance: u32,
) -> Option<(&'a ObservationPolicyRule, u32)> {
    if let Some(rule) = policy.rules.get(abstraction_sha256) {
        return Some((rule, 0));
    }
    if maximum_distance == 0 {
        return None;
    }
    policy
        .rules
        .iter()
        .filter(|(_, rule)| {
            rule.decision_depth == abstraction.decision_depth
                && rule.abstraction.legal_choice_catalog_sha256
                    == abstraction.legal_choice_catalog_sha256
        })
        .filter_map(|(key, rule)| {
            let distance = sparse_feature_l1_distance(
                &rule.abstraction.nonzero_features,
                &abstraction.nonzero_features,
            );
            (distance <= maximum_distance).then_some((distance, key, rule))
        })
        .min_by(|left, right| (left.0, left.1).cmp(&(right.0, right.1)))
        .map(|(distance, _, rule)| (rule, distance))
}

fn sparse_feature_l1_distance(
    left: &[QuantizedObservationFeature],
    right: &[QuantizedObservationFeature],
) -> u32 {
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    let mut distance = 0u32;
    while left_index < left.len() || right_index < right.len() {
        match (left.get(left_index), right.get(right_index)) {
            (Some(a), Some(b)) if a.index == b.index => {
                distance = distance.saturating_add(u32::from(a.value.abs_diff(b.value)));
                left_index += 1;
                right_index += 1;
            }
            (Some(a), Some(b)) if a.index < b.index => {
                distance = distance.saturating_add(u32::from(a.value.unsigned_abs()));
                left_index += 1;
            }
            (Some(_), Some(b)) => {
                distance = distance.saturating_add(u32::from(b.value.unsigned_abs()));
                right_index += 1;
            }
            (Some(a), None) => {
                distance = distance.saturating_add(u32::from(a.value.unsigned_abs()));
                left_index += 1;
            }
            (None, Some(b)) => {
                distance = distance.saturating_add(u32::from(b.value.unsigned_abs()));
                right_index += 1;
            }
            (None, None) => break,
        }
    }
    distance
}

pub(crate) fn observation_abstraction(
    input: &blob_interface::reference_mind::ReferenceMindInput,
    decision_depth: usize,
    quantization_levels: u8,
    legal_choices: &[PolicyChoice],
) -> Result<ObservationAbstraction, String> {
    let observation = Observation::from_reference(input);
    let scale = f32::from(quantization_levels);
    let mut nonzero_features = Vec::new();
    for (index, value) in observation.data.iter().copied().enumerate() {
        if (OBS_RANDOMNESS_FEATURE_START..OBS_RANDOMNESS_FEATURE_END).contains(&index) {
            continue;
        }
        if !value.is_finite() {
            return Err("Mind observation projection contained a non-finite feature".into());
        }
        let quantized = (value.clamp(-1.0, 1.0) * scale).round() as i16;
        if quantized != 0 {
            nonzero_features.push(QuantizedObservationFeature {
                index: u16::try_from(index).map_err(|_| "observation feature index exceeds u16")?,
                value: quantized,
            });
        }
    }
    let legal_choice_catalog_sha256 =
        sha256_domain_json(b"blob.observation-policy.legal-catalog.v1", legal_choices)?;
    Ok(ObservationAbstraction {
        decision_depth,
        quantization_levels,
        nonzero_features,
        legal_choice_catalog_sha256,
        legal_choice_count: legal_choices.len(),
    })
}

pub(crate) fn observation_abstraction_sha256(
    abstraction: &ObservationAbstraction,
) -> Result<String, String> {
    sha256_domain_json(b"blob.observation-policy.abstraction.v1", abstraction)
}

fn sha256_domain_json<T: Serialize + ?Sized>(domain: &[u8], value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("failed to encode observation abstraction: {error}"))?;
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(bytes);
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn policy_memory(next_depth: usize) -> Result<Vec<u8>, String> {
    let depth = u32::try_from(next_depth)
        .map_err(|_| "observation-policy decision depth exceeds u32".to_string())?;
    let mut memory = Vec::with_capacity(POLICY_MEMORY_PREFIX.len() + 4);
    memory.extend_from_slice(POLICY_MEMORY_PREFIX);
    memory.extend_from_slice(&depth.to_le_bytes());
    Ok(memory)
}

pub fn policy_sha256(policy: &ObservationPolicy) -> Result<String, String> {
    let bytes = serde_json::to_vec(policy)
        .map_err(|error| format!("failed to encode observation policy: {error}"))?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn compare_policy_nodes_with_tie_break(
    left: &PolicyNode,
    right: &PolicyNode,
    objective: MicroCombatObjective,
    config: ObservationPolicySearchConfig,
) -> Ordering {
    compare_policy_nodes(left, right, objective, config).then_with(|| {
        // A smaller key wins. Reversing the operands preserves the convention
        // that Ordering::Greater means `left` is preferred.
        match config.tie_break_seed {
            Some(seed) => policy_node_tie_key(right, seed).cmp(&policy_node_tie_key(left, seed)),
            None => canonical_policy_node_tie_key(right).cmp(&canonical_policy_node_tie_key(left)),
        }
    })
}

fn canonical_policy_node_tie_key(node: &PolicyNode) -> String {
    node.rules
        .iter()
        .map(|(key, rule)| format!("{key}:{}:{};", rule.choice.action, rule.choice.amount))
        .collect()
}

fn policy_node_tie_key(node: &PolicyNode, tie_break_seed: u64) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"blob.observation-policy.node-tie.v1");
    digest.update(tie_break_seed.to_le_bytes());
    for (key, rule) in &node.rules {
        digest.update((key.len() as u64).to_le_bytes());
        digest.update(key.as_bytes());
        digest.update((rule.choice.action as u64).to_le_bytes());
        digest.update((rule.choice.amount as u64).to_le_bytes());
        digest.update((rule.choice.signal as u64).to_le_bytes());
        digest.update((rule.choice.signal_strength as u64).to_le_bytes());
    }
    digest.finalize().into()
}

fn outcome_name(outcome: Option<EpisodeOutcome>) -> &'static str {
    match outcome {
        None => "incomplete",
        Some(EpisodeOutcome::Win) => "win",
        Some(EpisodeOutcome::Loss) => "loss",
        Some(EpisodeOutcome::Timeout) => "timeout",
        Some(EpisodeOutcome::SafetyAbort) => "safety_abort",
    }
}

pub fn publish_observation_policy_report(
    output: &Path,
    report: &ObservationPolicySearchReport,
) -> Result<PathBuf, String> {
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode observation-policy report: {error}"))?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create observation-policy directory: {error}"))?;
    }
    let temporary = output.with_extension("json.tmp");
    fs::write(&temporary, bytes)
        .map_err(|error| format!("failed to write observation-policy report: {error}"))?;
    fs::rename(&temporary, output)
        .map_err(|error| format!("failed to publish observation-policy report: {error}"))?;
    Ok(output.to_path_buf())
}

pub fn publish_observation_policy_acquisition_report(
    output: &Path,
    report: &ObservationPolicyAcquisitionReport,
) -> Result<PathBuf, String> {
    let bytes = serde_json::to_vec_pretty(report)
        .map_err(|error| format!("failed to encode observation-policy acquisition: {error}"))?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!("failed to create observation-policy acquisition directory: {error}")
        })?;
    }
    let temporary = output.with_extension("json.tmp");
    fs::write(&temporary, bytes)
        .map_err(|error| format!("failed to write observation-policy acquisition: {error}"))?;
    fs::rename(&temporary, output)
        .map_err(|error| format!("failed to publish observation-policy acquisition: {error}"))?;
    Ok(output.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OpponentProfile, RewardConfig};
    use crate::micro_combat_planner::PlannedChoice;
    use blob_engine::engine::StartingCellLayout;
    use blob_interface::randomness::PrivateRandom;

    fn scenario() -> MicroCombatScenario {
        MicroCombatScenario {
            name: "observation-policy-test".into(),
            objective: MicroCombatObjective::Survival,
            world_size: 7,
            training_cells: 1,
            opponent_cells: 1,
            training_initial_energy: 100,
            opponent_initial_energy: 100,
            starting_layout: StartingCellLayout::PairedContact,
            opponent: OpponentProfile::Pursuer,
            sim_time_limit_quanta: 2_048,
        }
    }

    fn forking_evader_scenario() -> MicroCombatScenario {
        MicroCombatScenario {
            name: "forking-evader-policy-test".into(),
            objective: MicroCombatObjective::Elimination,
            world_size: 5,
            training_cells: 1,
            opponent_cells: 1,
            training_initial_energy: 100,
            opponent_initial_energy: 60,
            starting_layout: StartingCellLayout::PairedContact,
            opponent: OpponentProfile::ForkingEvader,
            sim_time_limit_quanta: 2_048,
        }
    }

    fn missing_replay(
        seed: u64,
        decision_depth: usize,
        legal_choice_catalog_sha256: &str,
        legal_choice_count: usize,
    ) -> ObservationPolicyReplay {
        ObservationPolicyReplay {
            seed,
            coverage_complete: false,
            objective_success: false,
            final_outcome: "incomplete".into(),
            decisions: decision_depth,
            observation_sha256_trace: Vec::new(),
            abstraction_sha256_trace: Vec::new(),
            exact_observation_binding_hits: 0,
            exact_abstraction_rule_hits: 0,
            nearest_fallback_rule_hits: 0,
            maximum_nearest_fallback_l1_distance: 0,
            missing_abstraction_sha256: Some(format!("missing-{seed}")),
            missing_abstraction: Some(ObservationAbstraction {
                decision_depth,
                quantization_levels: 16,
                nonzero_features: Vec::new(),
                legal_choice_catalog_sha256: legal_choice_catalog_sha256.into(),
                legal_choice_count,
            }),
            final_sim_time_quanta: 0,
            final_training_cells: 1,
            final_opponent_cells: 1,
            final_training_stored_energy: 1,
            final_opponent_stored_energy: 1,
            final_continuation_sha256: "continuation".into(),
        }
    }

    #[test]
    fn observation_key_ignores_only_the_private_random_block() {
        let simulator = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            101,
        )
        .unwrap();
        let input = simulator.observation_input(simulator.root()).unwrap();
        let mut changed_randomness = input.clone();
        changed_randomness.randomness = PrivateRandom::from_bytes([7; 32]);
        assert_eq!(
            mind_observation_sha256(&input).unwrap(),
            mind_observation_sha256(&changed_randomness).unwrap()
        );
        let mut changed_memory = input.clone();
        changed_memory.private_memory.push(7);
        assert_ne!(
            mind_observation_sha256(&input).unwrap(),
            mind_observation_sha256(&changed_memory).unwrap()
        );
    }

    #[test]
    fn forking_evader_hides_its_bit_but_exposes_disjoint_winning_responses() {
        let scenario = forking_evader_scenario();
        let simulators = [930_300_000, 930_300_001].map(|seed| {
            SingleCellSearchSimulator::new(
                &EnvConfig::default(),
                &RewardConfig::default(),
                &scenario,
                seed,
            )
            .unwrap()
        });
        let root_keys = simulators.each_ref().map(|simulator| {
            mind_observation_sha256(&simulator.observation_input(simulator.root()).unwrap())
                .unwrap()
        });
        assert_eq!(root_keys[0], root_keys[1]);

        // This common opening is the root rule found by the bounded MCTS
        // witness. Its completion exposes which perpendicular fork occurred.
        let opening = PolicyChoice {
            action: 115,
            amount: 3,
            signal: 0,
            signal_strength: 0,
        };
        let states = [
            simulators[0]
                .transition(simulators[0].root(), opening)
                .unwrap(),
            simulators[1]
                .transition(simulators[1].root(), opening)
                .unwrap(),
        ];
        let fork_keys = [
            mind_observation_sha256(&simulators[0].observation_input(&states[0]).unwrap()).unwrap(),
            mind_observation_sha256(&simulators[1].observation_input(&states[1]).unwrap()).unwrap(),
        ];
        assert_ne!(fork_keys[0], fork_keys[1]);

        let winning_responses = [0, 1].map(|index| {
            simulators[index]
                .legal_choices(&states[index])
                .unwrap()
                .into_iter()
                .filter(|choice| {
                    simulators[index]
                        .transition(&states[index], *choice)
                        .is_ok_and(|state| {
                            state.objective_success(MicroCombatObjective::Elimination)
                        })
                })
                .collect::<Vec<_>>()
        });
        assert!(!winning_responses[0].is_empty());
        assert!(!winning_responses[1].is_empty());
        assert!(winning_responses[0]
            .iter()
            .all(|choice| !winning_responses[1].contains(choice)));
    }

    #[test]
    fn abstraction_ignores_randomness_and_small_numeric_jitter() {
        let simulator = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            101,
        )
        .unwrap();
        let input = simulator.observation_input(simulator.root()).unwrap();
        let legal = simulator.legal_choices(simulator.root()).unwrap();
        let expected = observation_abstraction(&input, 0, 16, &legal).unwrap();

        let mut changed = input.clone();
        changed.randomness = PrivateRandom::from_bytes([0xff; 32]);
        changed.self_state.metabolism_remainder =
            changed.self_state.metabolism_remainder.saturating_add(1);
        assert_eq!(
            expected,
            observation_abstraction(&changed, 0, 16, &legal).unwrap()
        );
    }

    #[test]
    fn abstraction_preserves_visible_structure_and_legality_boundaries() {
        let simulator = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            101,
        )
        .unwrap();
        let input = simulator.observation_input(simulator.root()).unwrap();
        let legal = simulator.legal_choices(simulator.root()).unwrap();
        let expected = observation_abstraction(&input, 0, 16, &legal).unwrap();

        let mut changed_input = input.clone();
        let occupied = changed_input
            .slots
            .iter_mut()
            .find(|slot| slot.neighbor.is_some())
            .expect("paired-contact scenario should expose a neighbor");
        occupied.neighbor = None;
        assert_ne!(
            expected,
            observation_abstraction(&changed_input, 0, 16, &legal).unwrap()
        );

        let mut changed_legal = legal.clone();
        changed_legal.pop();
        assert_ne!(
            expected,
            observation_abstraction(&input, 0, 16, &changed_legal).unwrap()
        );
    }

    #[test]
    fn sparse_distance_counts_missing_and_changed_features() {
        let left = vec![
            QuantizedObservationFeature { index: 1, value: 4 },
            QuantizedObservationFeature {
                index: 10,
                value: -2,
            },
        ];
        let right = vec![
            QuantizedObservationFeature { index: 1, value: 7 },
            QuantizedObservationFeature { index: 4, value: 5 },
        ];
        assert_eq!(sparse_feature_l1_distance(&left, &right), 10);
        assert_eq!(sparse_feature_l1_distance(&right, &left), 10);
    }

    #[test]
    fn acquisition_prefers_deep_distinct_legality_partitions() {
        let replays = vec![
            missing_replay(10, 4, "common", 154),
            missing_replay(11, 4, "common", 154),
            missing_replay(12, 9, "novel", 130),
            missing_replay(13, 6, "other", 140),
        ];
        let summaries = missing_partition_summaries(&replays);
        assert_eq!(summaries.len(), 3);
        assert_eq!(summaries[0].decision_depth, 9);
        assert_eq!(summaries[1].decision_depth, 6);
        assert_eq!(summaries[2].replay_count, 2);
        assert_eq!(acquisition_candidate_seeds(&replays, 2, 1), vec![12, 13]);
        assert_eq!(
            acquisition_candidate_seeds(
                &[
                    missing_replay(20, 9, "same", 122),
                    missing_replay(21, 9, "same", 122),
                ],
                1,
                2,
            ),
            vec![20, 21]
        );
    }

    #[test]
    fn acquisition_seed_partitions_must_be_disjoint() {
        let error = acquire_observation_policy(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            &ObservationPolicyAcquisitionSeeds {
                initial_search: vec![100],
                acquisition_pool: vec![101],
                final_holdout: vec![100],
            },
            ObservationPolicySearchConfig::default(),
            ObservationPolicyAcquisitionConfig::default(),
        )
        .unwrap_err();
        assert!(error.contains("duplicated"));
    }

    #[test]
    fn acquisition_search_grid_is_bounded_normalized_and_positive() {
        let normalized = ObservationPolicyAcquisitionConfig {
            candidate_algorithms: vec![
                ObservationPolicySearchAlgorithm::InformationSetMcts,
                ObservationPolicySearchAlgorithm::Beam,
            ],
            candidate_beam_widths: vec![16, 8, 16],
            candidate_tie_break_seeds: vec![2, 1, 2],
            candidate_mcts_seeds: vec![5, 3, 5],
            ..ObservationPolicyAcquisitionConfig::default()
        }
        .validate()
        .unwrap();
        assert_eq!(normalized.candidate_beam_widths, vec![8, 16]);
        assert_eq!(normalized.candidate_tie_break_seeds, vec![1, 2]);
        assert_eq!(normalized.candidate_mcts_seeds, vec![3, 5]);
        let configs = acquisition_candidate_search_configs(
            ObservationPolicySearchConfig::default(),
            &normalized,
        );
        assert_eq!(configs.len(), 8);
        assert_eq!(
            configs
                .iter()
                .filter(|config| {
                    config.algorithm == ObservationPolicySearchAlgorithm::InformationSetMcts
                })
                .count(),
            2
        );

        let error = ObservationPolicyAcquisitionConfig {
            candidate_algorithms: Vec::new(),
            ..ObservationPolicyAcquisitionConfig::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.contains("candidate algorithms"));

        for candidate_beam_widths in [Vec::new(), vec![0], vec![1; 9]] {
            let error = ObservationPolicyAcquisitionConfig {
                candidate_beam_widths,
                ..ObservationPolicyAcquisitionConfig::default()
            }
            .validate()
            .unwrap_err();
            assert!(error.contains("candidate beam widths"));
        }

        let error = ObservationPolicyAcquisitionConfig {
            candidate_tie_break_seeds: vec![1; 17],
            ..ObservationPolicyAcquisitionConfig::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.contains("candidate tie-break seeds"));

        let error = ObservationPolicyAcquisitionConfig {
            candidate_tie_break_seeds: Vec::new(),
            include_canonical_tie_break: false,
            ..ObservationPolicyAcquisitionConfig::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.contains("tie-break strategy"));

        let error = ObservationPolicyAcquisitionConfig {
            candidate_beam_widths: (1..=6).collect(),
            candidate_tie_break_seeds: vec![1, 2, 3],
            ..ObservationPolicyAcquisitionConfig::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.contains("search grid"));

        let error = ObservationPolicyAcquisitionConfig {
            candidate_algorithms: vec![ObservationPolicySearchAlgorithm::InformationSetMcts],
            candidate_mcts_seeds: Vec::new(),
            ..ObservationPolicyAcquisitionConfig::default()
        }
        .validate()
        .unwrap_err();
        assert!(error.contains("MCTS seeds"));
    }

    #[test]
    fn node_tie_break_is_stable_and_salt_sensitive() {
        let node = PolicyNode {
            particles: Vec::new(),
            rules: BTreeMap::new(),
            exact_observation_bindings: BTreeMap::new(),
            depth_choices: BTreeMap::new(),
        };
        assert_eq!(policy_node_tie_key(&node, 7), policy_node_tie_key(&node, 7));
        assert_ne!(policy_node_tie_key(&node, 7), policy_node_tie_key(&node, 8));
    }

    #[test]
    fn online_top_k_matches_materialize_sort_and_truncate() {
        let candidate = |rank: usize| {
            let choice = PolicyChoice {
                action: rank,
                amount: 0,
                signal: 0,
                signal_strength: 0,
            };
            PolicyNode {
                particles: Vec::new(),
                rules: BTreeMap::from([(
                    format!("key-{rank:02}"),
                    ObservationPolicyRule {
                        decision_depth: 0,
                        choice,
                        family: "test".into(),
                        next_private_memory: Vec::new(),
                        abstraction: ObservationAbstraction {
                            decision_depth: 0,
                            quantization_levels: 16,
                            nonzero_features: Vec::new(),
                            legal_choice_catalog_sha256: "catalog".into(),
                            legal_choice_count: 1,
                        },
                    },
                )]),
                exact_observation_bindings: BTreeMap::new(),
                depth_choices: BTreeMap::new(),
            }
        };
        let candidates = [5, 1, 4, 2, 3]
            .into_iter()
            .map(candidate)
            .collect::<Vec<_>>();
        let config = ObservationPolicySearchConfig::default();
        let mut materialized = candidates.clone();
        materialized.sort_unstable_by(|left, right| {
            compare_policy_nodes_with_tie_break(right, left, MicroCombatObjective::Survival, config)
        });
        materialized.truncate(3);

        let mut online = Vec::new();
        let mut pruned = 0usize;
        for candidate in candidates {
            pruned += usize::from(retain_top_policy_candidate(
                &mut online,
                candidate,
                3,
                MicroCombatObjective::Survival,
                config,
            ));
        }
        assert_eq!(pruned, 2);
        assert_eq!(
            online
                .iter()
                .map(canonical_policy_node_tie_key)
                .collect::<Vec<_>>(),
            materialized
                .iter()
                .map(canonical_policy_node_tie_key)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn lazy_choice_permutation_matches_materialized_swap_remove() {
        let mut materialized = (0..12).collect::<Vec<_>>();
        let mut lazy = LazyPolicyChoicePermutation::new(materialized.len());
        let requested_positions = [3, 0, 7, 1, 4, 0, 5, 2, 1, 0, 1, 0];
        let mut materialized_order = Vec::new();
        let mut lazy_order = Vec::new();
        for requested in requested_positions {
            let position = requested % materialized.len();
            materialized_order.push(materialized.swap_remove(position));
            lazy_order.push(lazy.swap_remove_original_index(position).unwrap());
            assert_eq!(lazy.remaining(), materialized.len());
        }
        assert_eq!(lazy_order, materialized_order);
        assert!(lazy.is_empty());
        assert_eq!(lazy.index_override_count(), 0);
    }

    #[test]
    fn mcts_choice_catalogs_are_interned_once() {
        let choice = |action| PolicyChoice {
            action,
            amount: 0,
            signal: 0,
            signal_strength: 0,
        };
        let mut catalogs = MctsChoiceCatalogs::default();
        let (first, first_unique) = catalogs.intern(vec![choice(1), choice(2)]);
        let (repeated, repeated_unique) = catalogs.intern(vec![choice(1), choice(2)]);
        let (different, different_unique) = catalogs.intern(vec![choice(2), choice(1)]);
        assert!(first_unique);
        assert!(!repeated_unique);
        assert!(different_unique);
        assert_eq!(first, repeated);
        assert_ne!(first, different);
        assert_eq!(catalogs.len(), 2);
        assert_eq!(catalogs.get(first), [choice(1), choice(2)]);
        assert_eq!(
            catalogs.retained_choice_bytes(),
            4 * size_of::<PolicyChoice>()
        );
    }

    #[test]
    fn information_set_mcts_is_bounded_deterministic_and_replayable() {
        let config = ObservationPolicySearchConfig {
            algorithm: ObservationPolicySearchAlgorithm::InformationSetMcts,
            mcts_seed: 77,
            mcts_max_iterations: 128,
            mcts_rollout_rule_depth: 3,
            max_decisions_per_particle: 4,
            max_expansions: 4_000,
            ..ObservationPolicySearchConfig::default()
        };
        let run = || {
            search_observation_policy(
                &EnvConfig::default(),
                &RewardConfig::default(),
                &scenario(),
                &[102, 103],
                &[202, 203],
                config,
            )
            .unwrap()
        };
        let first = run();
        let repeated = run();
        assert_eq!(
            first.config.algorithm,
            ObservationPolicySearchAlgorithm::InformationSetMcts
        );
        assert!(first.algorithm_iterations <= config.mcts_max_iterations);
        assert!(first.expansions <= config.max_expansions);
        assert!(first.policy_rule_count > 0);
        assert_eq!(first.policy_sha256, repeated.policy_sha256);
        assert_eq!(first.expansions, repeated.expansions);
        assert_eq!(first.holdout_replays, repeated.holdout_replays);
        let deterministic_cost = |mut cost: ObservationPolicySearchCost| {
            cost.wall_time_micros = 0;
            cost.host_peak_resident_set_bytes = None;
            cost.planner_restore_tile_materialization_ns = 0;
            cost.planner_restore_cell_materialization_ns = 0;
            cost.planner_restore_hash_seed_materialization_ns = 0;
            cost.planner_restore_resolver_reconstruction_ns = 0;
            cost.planner_restore_topology_validation_ns = 0;
            cost.planner_restore_cell_validation_store_and_passive_index_ns = 0;
            cost.planner_restore_tile_validation_and_passive_index_ns = 0;
            cost.planner_restore_hash_initialization_ns = 0;
            cost.planner_restore_scratch_initialization_ns = 0;
            cost.planner_restore_metabolic_index_ns = 0;
            cost.planner_restore_integrity_validation_ns = 0;
            cost.planner_environment_restore_total_ns = 0;
            cost.planner_restore_total_ns = 0;
            cost
        };
        assert_eq!(
            deterministic_cost(first.search_cost.clone()),
            deterministic_cost(repeated.search_cost.clone())
        );
        assert!(first.search_cost.peak_retained_policy_nodes > 0);
        assert!(
            first
                .search_cost
                .estimated_peak_retained_policy_bytes_lower_bound
                > 0
        );
        assert!(first.search_cost.rollout_policy_clones > 0);
        assert!(first.search_cost.planner_restore_count > first.expansions as u64);
        assert!(first.search_cost.planner_restored_tiles > 0);
        assert!(first.search_cost.planner_restored_cells > 0);
        assert!(first.search_cost.shared_particle_state_clones > 0);
        assert!(first.search_cost.peak_retained_untried_choices > 0);
        assert!(
            first
                .search_cost
                .peak_retained_canonical_tile_page_references
                > 0
        );
        assert!(first.search_cost.peak_retained_unique_canonical_tile_pages > 0);
        assert!(
            first.search_cost.peak_retained_unique_canonical_tile_pages
                <= first
                    .search_cost
                    .peak_retained_canonical_tile_page_references
        );
        assert!(
            first
                .search_cost
                .peak_retained_canonical_tile_chunk_references
                > 0
        );
        assert!(first.search_cost.peak_retained_unique_canonical_tile_chunks > 0);
        assert!(
            first.search_cost.peak_retained_unique_canonical_tile_chunks
                <= first
                    .search_cost
                    .peak_retained_canonical_tile_chunk_references
        );
        assert!(
            first.search_cost.peak_retained_unique_canonical_cell_chunks
                <= first
                    .search_cost
                    .peak_retained_canonical_cell_chunk_references
        );
        assert!(first.search_cost.peak_retained_mcts_choice_index_overrides > 0);
        assert!(
            first
                .search_cost
                .estimated_peak_retained_mcts_choice_bytes_lower_bound
                > 0
        );
        assert!(first.search_cost.mcts_unique_choice_catalogs > 0);
        assert!(first.search_cost.mcts_choice_catalog_intern_hits > 0);
        assert_eq!(
            first.search_cost.particle_state_allocations,
            first.search_seeds.len() + first.expansions
        );
        assert_eq!(first.search_cost.beam_generated_children, 0);
        assert_eq!(first.search_cost.beam_pruned_children, 0);
    }

    #[test]
    fn acquisition_frontier_preserves_tradeoffs_but_selects_coverage_first() {
        let candidate = |added_seed, coverage, rules, distance| ObservationPolicyCandidateSummary {
            added_seed,
            objective: MicroCombatObjective::Survival,
            search_config: ObservationPolicySearchConfig {
                beam_width: 8,
                ..ObservationPolicySearchConfig::default()
            },
            search_cost: ObservationPolicySearchCost::default(),
            policy_sha256: format!("policy-{added_seed:?}"),
            acquisition_pool_covered_seed_count: coverage,
            acquisition_objective_success_count: coverage,
            policy_rule_count: rules,
            exact_observation_binding_count: rules,
            acquisition_nearest_fallback_rule_hits: 0,
            acquisition_maximum_nearest_fallback_l1_distance: distance,
            acquisition_minimum_final_training_energy: 10,
            acquisition_final_training_energy_total: 100,
            acquisition_minimum_survival_time_quanta: 100,
            acquisition_survival_time_quanta_total: 1_000,
            expansions: 100,
            truncated: true,
        };
        let candidates = vec![
            candidate(None, 10, 5, 10),
            candidate(Some(1), 11, 7, 12),
            candidate(Some(2), 10, 4, 9),
            candidate(Some(3), 9, 3, 8),
        ];
        let frontier = pareto_candidate_indices(&candidates, 4);
        assert!(frontier.contains(&1));
        assert!(frontier.contains(&2));
        let selected = frontier
            .iter()
            .copied()
            .max_by(|left, right| {
                compare_acquisition_candidates(&candidates[*left], &candidates[*right])
            })
            .unwrap();
        assert_eq!(selected, 1);
        assert!(compare_acquisition_candidates(&candidates[0], &candidates[3]).is_gt());

        let mut fragile = candidate(Some(10), 12, 5, 10);
        fragile.acquisition_minimum_final_training_energy = 2;
        fragile.acquisition_final_training_energy_total = 24;
        let robust = candidate(Some(11), 12, 5, 10);
        assert!(compare_acquisition_candidates(&robust, &fragile).is_gt());
        assert!(candidate_dominates(&robust, &fragile));

        let mut expensive = robust.clone();
        expensive.search_cost.wall_time_micros = u128::MAX;
        expensive.search_cost.host_peak_resident_set_bytes = Some(u64::MAX);
        expensive.search_cost.policy_node_clones = usize::MAX;
        expensive.search_cost.shared_particle_state_clones = usize::MAX;
        expensive.search_cost.particle_state_allocations = usize::MAX;
        expensive.search_cost.beam_generated_children = usize::MAX;
        expensive.search_cost.beam_pruned_children = usize::MAX;
        expensive
            .search_cost
            .peak_retained_mcts_choice_index_overrides = usize::MAX;
        expensive
            .search_cost
            .estimated_peak_retained_mcts_choice_bytes_lower_bound = u64::MAX;
        expensive.search_cost.mcts_unique_choice_catalogs = usize::MAX;
        expensive.search_cost.mcts_choice_catalog_intern_hits = usize::MAX;
        expensive
            .search_cost
            .peak_retained_canonical_tile_page_references = usize::MAX;
        expensive
            .search_cost
            .peak_retained_unique_canonical_tile_pages = usize::MAX;
        expensive
            .search_cost
            .peak_retained_canonical_tile_chunk_references = usize::MAX;
        expensive
            .search_cost
            .peak_retained_unique_canonical_tile_chunks = usize::MAX;
        expensive
            .search_cost
            .peak_retained_canonical_cell_chunk_references = usize::MAX;
        expensive
            .search_cost
            .peak_retained_unique_canonical_cell_chunks = usize::MAX;
        assert_eq!(
            compare_acquisition_candidates(&robust, &expensive),
            Ordering::Equal
        );
        assert!(!candidate_dominates(&robust, &expensive));
        assert!(!candidate_dominates(&expensive, &robust));
    }

    #[test]
    fn nearest_fallback_is_distance_bounded_and_legality_partitioned() {
        let simulator = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            101,
        )
        .unwrap();
        let input = simulator.observation_input(simulator.root()).unwrap();
        let legal = simulator.legal_choices(simulator.root()).unwrap();
        let abstraction = observation_abstraction(&input, 0, 16, &legal).unwrap();
        let key = observation_abstraction_sha256(&abstraction).unwrap();
        let rule = ObservationPolicyRule {
            decision_depth: 0,
            choice: legal[0],
            family: "wait".into(),
            next_private_memory: policy_memory(1).unwrap(),
            abstraction: abstraction.clone(),
        };
        let policy = ObservationPolicy {
            schema_version: MICRO_COMBAT_OBSERVATION_POLICY_SCHEMA_VERSION,
            exact_observation_key: "test".into(),
            action_selection_key: "test".into(),
            randomness_handling: "test".into(),
            observation_quantization_levels: 16,
            max_nearest_fallback_l1_distance: 1,
            rules: BTreeMap::from([(key, rule)]),
            exact_observation_bindings: BTreeMap::new(),
        };

        let mut nearby = abstraction.clone();
        nearby.nonzero_features[0].value += 1;
        let nearby_key = observation_abstraction_sha256(&nearby).unwrap();
        assert_eq!(
            select_abstract_rule(&policy, &nearby_key, &nearby, 1).map(|(_, distance)| distance),
            Some(1)
        );
        assert!(select_abstract_rule(&policy, &nearby_key, &nearby, 0).is_none());

        nearby.legal_choice_catalog_sha256 = "different-catalog".into();
        assert!(select_abstract_rule(&policy, &nearby_key, &nearby, u32::MAX).is_none());
    }

    #[test]
    fn bounded_policy_is_shared_and_replays_through_the_mind_boundary() {
        let report = search_observation_policy(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            &[102, 103],
            &[202, 203],
            ObservationPolicySearchConfig {
                beam_width: 4,
                tie_break_seed: None,
                max_decisions_per_particle: 3,
                max_expansions: 4_000,
                equivalent_effort_pruning: true,
                observation_quantization_levels: 16,
                max_nearest_fallback_l1_distance: 128,
                ..ObservationPolicySearchConfig::default()
            },
        )
        .unwrap();
        assert_eq!(report.search_replays.len(), 2);
        assert_eq!(report.holdout_replays.len(), 2);
        assert!(report.search_covered_to_requested_horizon);
        assert!(report.policy_rule_count > 0);
        assert!(report.search_unique_observation_traces > 0);
        assert!(report.holdout_unique_observation_traces > 0);
        assert!(report
            .search_replays
            .iter()
            .all(|replay| replay.coverage_complete));
        assert!(report
            .policy
            .rules
            .values()
            .all(|rule| rule.next_private_memory.starts_with(POLICY_MEMORY_PREFIX)));
    }

    #[test]
    fn empty_seed_suite_fails_closed() {
        assert!(search_observation_policy(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            &[],
            &[],
            ObservationPolicySearchConfig::default(),
        )
        .is_err());
    }

    #[test]
    fn search_and_holdout_seeds_must_be_disjoint() {
        assert!(search_observation_policy(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            &[105],
            &[105],
            ObservationPolicySearchConfig::default(),
        )
        .is_err());
    }

    #[test]
    fn terminal_loss_never_outranks_a_living_partial_policy() {
        let simulator = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            104,
        )
        .unwrap();
        let living = PolicyNode {
            particles: vec![PolicyParticle {
                state: Arc::new(simulator.root().clone()),
                decisions: 1,
            }],
            rules: BTreeMap::new(),
            exact_observation_bindings: BTreeMap::new(),
            depth_choices: BTreeMap::new(),
        };
        let mut lost_state = simulator.root().clone();
        lost_state.training_cells = 0;
        lost_state.terminal_outcome = Some(EpisodeOutcome::Loss);
        let lost = PolicyNode {
            particles: vec![PolicyParticle {
                state: Arc::new(lost_state),
                decisions: 1,
            }],
            rules: BTreeMap::new(),
            exact_observation_bindings: BTreeMap::new(),
            depth_choices: BTreeMap::new(),
        };
        assert!(compare_policy_nodes(
            &living,
            &lost,
            MicroCombatObjective::Survival,
            ObservationPolicySearchConfig::default(),
        )
        .is_gt());
    }

    #[test]
    fn cloned_policy_nodes_share_immutable_particle_checkpoints() {
        let simulator = SingleCellSearchSimulator::new(
            &EnvConfig::default(),
            &RewardConfig::default(),
            &scenario(),
            104,
        )
        .unwrap();
        let node = PolicyNode {
            particles: vec![PolicyParticle {
                state: Arc::new(simulator.root().clone()),
                decisions: 0,
            }],
            rules: BTreeMap::new(),
            exact_observation_bindings: BTreeMap::new(),
            depth_choices: BTreeMap::new(),
        };
        let cloned = node.clone();
        assert!(Arc::ptr_eq(
            &node.particles[0].state,
            &cloned.particles[0].state
        ));

        let mut retained = RetainedStateAllocations::default();
        let first = estimated_retained_policy_node_bytes(&node, &mut retained);
        let shared = estimated_retained_policy_node_bytes(&cloned, &mut retained);
        assert!(shared < first);
    }

    #[test]
    fn planned_choice_remains_human_readable() {
        let choice = PolicyChoice {
            action: 0,
            amount: 0,
            signal: 0,
            signal_strength: 0,
        };
        let planned = PlannedChoice::from(choice);
        assert_eq!(planned.family, "wait");
    }
}
