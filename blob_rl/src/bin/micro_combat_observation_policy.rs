//! Synthesize and replay a bounded observation-only policy across hidden seeds.

use std::fs;
use std::path::PathBuf;

use blob_rl::config::TrainingConfig;
use blob_rl::micro_combat::MicroCombatSuiteConfig;
use blob_rl::micro_combat_observation_policy::{
    publish_observation_policy_report, search_observation_policy, ObservationPolicySearchAlgorithm,
    ObservationPolicySearchConfig,
};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum SearchAlgorithmArg {
    Beam,
    InformationSetMcts,
}

impl From<SearchAlgorithmArg> for ObservationPolicySearchAlgorithm {
    fn from(value: SearchAlgorithmArg) -> Self {
        match value {
            SearchAlgorithmArg::Beam => Self::Beam,
            SearchAlgorithmArg::InformationSetMcts => Self::InformationSetMcts,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "blob_micro_combat_observation_policy",
    about = "Synthesize a Mind-ABI-legal policy table across hidden scenario seeds"
)]
struct Args {
    #[arg(long, value_enum, default_value_t = SearchAlgorithmArg::Beam)]
    algorithm: SearchAlgorithmArg,

    #[arg(long)]
    config: PathBuf,

    #[arg(long)]
    scenarios: PathBuf,

    #[arg(long)]
    scenario: String,

    #[arg(long)]
    seed_base: u64,

    #[arg(long)]
    seed_count: usize,

    /// First unseen replay seed. Defaults to the first seed after the search
    /// range when a nonzero holdout count is requested.
    #[arg(long)]
    holdout_seed_base: Option<u64>,

    #[arg(long, default_value_t = 0)]
    holdout_seed_count: usize,

    #[arg(long, default_value_t = 16)]
    beam_width: usize,

    /// Deterministic salt used only to order equal-scoring search nodes.
    #[arg(long)]
    tie_break_seed: Option<u64>,

    #[arg(long, default_value_t = 0)]
    mcts_seed: u64,

    #[arg(long, default_value_t = 20_000)]
    mcts_max_iterations: usize,

    #[arg(long, default_value_t = 8)]
    mcts_rollout_rule_depth: usize,

    #[arg(long, default_value_t = 1_414)]
    mcts_exploration_milli: u32,

    #[arg(long, default_value_t = 16)]
    max_decisions_per_particle: usize,

    #[arg(long, default_value_t = 100_000)]
    max_expansions: usize,

    #[arg(long, default_value_t = false)]
    disable_equivalent_effort_pruning: bool,

    /// Uniform quantization intervals for each normalized Mind observation
    /// feature. Exact legal choices remain part of the abstract key.
    #[arg(long, default_value_t = 16)]
    observation_quantization_levels: u8,

    /// Maximum sparse-feature L1 distance for the nearest-rule fallback.
    /// Zero disables fallback and requires an exact abstract match.
    #[arg(long, default_value_t = 128)]
    max_nearest_fallback_l1_distance: u32,

    /// Diagnostic baseline requiring the same exact choice at every
    /// observation encountered at a given decision depth.
    #[arg(long, default_value_t = false)]
    force_open_loop_actions: bool,

    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let config = TrainingConfig::from_file(&args.config.to_string_lossy())
        .unwrap_or_else(|error| panic!("failed to load training config: {error}"));
    let suite_content = fs::read_to_string(&args.scenarios)
        .unwrap_or_else(|error| panic!("failed to read scenario suite: {error}"));
    let suite = MicroCombatSuiteConfig::from_toml_str(&suite_content)
        .unwrap_or_else(|error| panic!("failed to load scenario suite: {error}"));
    suite
        .validate_against(&config.env)
        .unwrap_or_else(|error| panic!("scenario suite does not match the ruleset: {error}"));
    let scenario = suite
        .scenarios
        .iter()
        .find(|scenario| scenario.name == args.scenario)
        .unwrap_or_else(|| panic!("scenario {} was not found", args.scenario));
    let seeds = (0..args.seed_count)
        .map(|offset| {
            args.seed_base
                .checked_add(offset as u64)
                .expect("seed range overflowed u64")
        })
        .collect::<Vec<_>>();
    let default_holdout_base = args
        .seed_base
        .checked_add(args.seed_count as u64)
        .expect("default holdout seed base overflowed u64");
    let holdout_base = args.holdout_seed_base.unwrap_or(default_holdout_base);
    let holdout_seeds = (0..args.holdout_seed_count)
        .map(|offset| {
            holdout_base
                .checked_add(offset as u64)
                .expect("holdout seed range overflowed u64")
        })
        .collect::<Vec<_>>();
    let report = search_observation_policy(
        &config.env,
        &config.reward,
        scenario,
        &seeds,
        &holdout_seeds,
        ObservationPolicySearchConfig {
            algorithm: args.algorithm.into(),
            beam_width: args.beam_width,
            tie_break_seed: args.tie_break_seed,
            mcts_seed: args.mcts_seed,
            mcts_max_iterations: args.mcts_max_iterations,
            mcts_rollout_rule_depth: args.mcts_rollout_rule_depth,
            mcts_exploration_milli: args.mcts_exploration_milli,
            max_decisions_per_particle: args.max_decisions_per_particle,
            max_expansions: args.max_expansions,
            equivalent_effort_pruning: !args.disable_equivalent_effort_pruning,
            observation_quantization_levels: args.observation_quantization_levels,
            max_nearest_fallback_l1_distance: args.max_nearest_fallback_l1_distance,
            force_open_loop_actions: args.force_open_loop_actions,
        },
    )
    .unwrap_or_else(|error| panic!("observation-policy search failed: {error}"));
    let output = publish_observation_policy_report(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish observation-policy report: {error}"));
    println!(
        "{} algorithm={:?} iterations={} search_success={}/{} holdout_success={}/{} search_coverage={} holdout_coverage={} rules={} exact_bindings={} branches={:?} choice_branches={:?} exact_traces={}/{} abstract_traces={}/{} exact_hits={}/{} nearest_fallbacks={}/{} holdout_max_l1={} expansions={} truncated={} search_wall_us={} host_peak_rss_bytes={:?} planner_restores={} environment_restore_total_ns={} restore_total_ns={} restore_tile_ns={} restore_cell_ns={} restore_resolver_ns={} restore_integrity_ns={} restored_tiles={} restored_cells={} peak_nodes={} peak_particles={} peak_rules={} peak_logical_untried={} canonical_tile_pages={}/{} canonical_tile_chunks={}/{} canonical_cell_chunks={}/{} peak_mcts_choice_index_overrides={} estimated_peak_mcts_choice_bytes_lower_bound={} mcts_unique_choice_catalogs={} mcts_choice_catalog_intern_hits={} estimated_peak_policy_bytes_lower_bound={} policy_clones={} rollout_clones={} shared_particle_state_clones={} particle_state_allocations={} beam_generated_children={} beam_pruned_children={}",
        report.scenario,
        report.config.algorithm,
        report.algorithm_iterations,
        report.search_objective_success_count,
        report.search_seeds.len(),
        report.holdout_objective_success_count,
        report.holdout_seeds.len(),
        report.search_covered_to_requested_horizon,
        report.holdout_covered_to_requested_horizon,
        report.policy_rule_count,
        report.exact_observation_binding_count,
        report.policy_branching_depths,
        report.policy_choice_branching_depths,
        report.search_unique_observation_traces,
        report.holdout_unique_observation_traces,
        report.search_unique_abstraction_traces,
        report.holdout_unique_abstraction_traces,
        report.search_exact_observation_binding_hits,
        report.holdout_exact_observation_binding_hits,
        report.search_nearest_fallback_rule_hits,
        report.holdout_nearest_fallback_rule_hits,
        report.holdout_maximum_nearest_fallback_l1_distance,
        report.expansions,
        report.truncated,
        report.search_cost.wall_time_micros,
        report.search_cost.host_peak_resident_set_bytes,
        report.search_cost.planner_restore_count,
        report.search_cost.planner_environment_restore_total_ns,
        report.search_cost.planner_restore_total_ns,
        report.search_cost.planner_restore_tile_materialization_ns,
        report.search_cost.planner_restore_cell_materialization_ns,
        report
            .search_cost
            .planner_restore_resolver_reconstruction_ns,
        report
            .search_cost
            .planner_restore_integrity_validation_ns,
        report.search_cost.planner_restored_tiles,
        report.search_cost.planner_restored_cells,
        report.search_cost.peak_retained_policy_nodes,
        report.search_cost.peak_retained_particles,
        report.search_cost.peak_retained_rules,
        report.search_cost.peak_retained_untried_choices,
        report
            .search_cost
            .peak_retained_unique_canonical_tile_pages,
        report
            .search_cost
            .peak_retained_canonical_tile_page_references,
        report
            .search_cost
            .peak_retained_unique_canonical_tile_chunks,
        report
            .search_cost
            .peak_retained_canonical_tile_chunk_references,
        report
            .search_cost
            .peak_retained_unique_canonical_cell_chunks,
        report
            .search_cost
            .peak_retained_canonical_cell_chunk_references,
        report
            .search_cost
            .peak_retained_mcts_choice_index_overrides,
        report
            .search_cost
            .estimated_peak_retained_mcts_choice_bytes_lower_bound,
        report.search_cost.mcts_unique_choice_catalogs,
        report.search_cost.mcts_choice_catalog_intern_hits,
        report
            .search_cost
            .estimated_peak_retained_policy_bytes_lower_bound,
        report.search_cost.policy_node_clones,
        report.search_cost.rollout_policy_clones,
        report.search_cost.shared_particle_state_clones,
        report.search_cost.particle_state_allocations,
        report.search_cost.beam_generated_children,
        report.search_cost.beam_pruned_children,
    );
    println!("Policy SHA-256: {}", report.policy_sha256);
    println!("Report: {}", output.display());
}
