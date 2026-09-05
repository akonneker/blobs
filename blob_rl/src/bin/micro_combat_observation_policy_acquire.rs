//! Counterexample-guided observation-policy synthesis with a sealed holdout.

use std::fs;
use std::path::PathBuf;

use blob_rl::config::TrainingConfig;
use blob_rl::micro_combat::MicroCombatSuiteConfig;
use blob_rl::micro_combat_observation_policy::{
    acquire_observation_policy, publish_observation_policy_acquisition_report,
    ObservationPolicyAcquisitionConfig, ObservationPolicyAcquisitionSeeds,
    ObservationPolicySearchAlgorithm, ObservationPolicySearchConfig,
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
    name = "blob_micro_combat_observation_policy_acquire",
    about = "Acquire missing observation partitions before a sealed final holdout"
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
    initial_seed_base: u64,

    #[arg(long)]
    initial_seed_count: usize,

    #[arg(long)]
    acquisition_seed_base: u64,

    #[arg(long)]
    acquisition_seed_count: usize,

    #[arg(long)]
    final_holdout_seed_base: u64,

    #[arg(long)]
    final_holdout_seed_count: usize,

    #[arg(long, default_value_t = 2)]
    max_acquisition_rounds: usize,

    #[arg(long, default_value_t = 2)]
    acquisition_partitions_per_round: usize,

    #[arg(long, default_value_t = 2)]
    candidates_per_partition: usize,

    #[arg(long, default_value_t = 4)]
    max_frontier_size: usize,

    /// Comma-separated search families evaluated for every counterexample.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "beam")]
    candidate_algorithms: Vec<SearchAlgorithmArg>,

    /// Comma-separated beam widths evaluated for every counterexample seed.
    #[arg(long, value_delimiter = ',', default_value = "8,16")]
    candidate_beam_widths: Vec<usize>,

    /// Comma-separated deterministic salts crossed with candidate beams.
    #[arg(long, value_delimiter = ',')]
    candidate_tie_break_seeds: Vec<u64>,

    /// Do not include canonical lexicographic ordering in candidate searches.
    #[arg(long, default_value_t = false)]
    exclude_canonical_tie_break: bool,

    /// Comma-separated deterministic MCTS simulation seeds.
    #[arg(long, value_delimiter = ',', default_value = "0")]
    candidate_mcts_seeds: Vec<u64>,

    #[arg(long, default_value_t = 8)]
    beam_width: usize,

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

    #[arg(long, default_value_t = 150_000)]
    max_expansions: usize,

    #[arg(long, default_value_t = false)]
    disable_equivalent_effort_pruning: bool,

    #[arg(long, default_value_t = 16)]
    observation_quantization_levels: u8,

    #[arg(long, default_value_t = 128)]
    max_nearest_fallback_l1_distance: u32,

    #[arg(long, default_value_t = false)]
    force_open_loop_actions: bool,

    #[arg(long)]
    output: PathBuf,
}

fn seed_range(base: u64, count: usize, label: &str) -> Vec<u64> {
    (0..count)
        .map(|offset| {
            base.checked_add(offset as u64)
                .unwrap_or_else(|| panic!("{label} seed range overflowed u64"))
        })
        .collect()
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

    let initial_seeds = seed_range(args.initial_seed_base, args.initial_seed_count, "initial");
    let acquisition_seeds = seed_range(
        args.acquisition_seed_base,
        args.acquisition_seed_count,
        "acquisition",
    );
    let final_holdout_seeds = seed_range(
        args.final_holdout_seed_base,
        args.final_holdout_seed_count,
        "final holdout",
    );
    let report = acquire_observation_policy(
        &config.env,
        &config.reward,
        scenario,
        &ObservationPolicyAcquisitionSeeds {
            initial_search: initial_seeds,
            acquisition_pool: acquisition_seeds,
            final_holdout: final_holdout_seeds,
        },
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
        ObservationPolicyAcquisitionConfig {
            max_rounds: args.max_acquisition_rounds,
            partitions_per_round: args.acquisition_partitions_per_round,
            candidates_per_partition: args.candidates_per_partition,
            max_frontier_size: args.max_frontier_size,
            candidate_algorithms: args
                .candidate_algorithms
                .into_iter()
                .map(Into::into)
                .collect(),
            candidate_beam_widths: args.candidate_beam_widths,
            candidate_tie_break_seeds: args.candidate_tie_break_seeds,
            include_canonical_tie_break: !args.exclude_canonical_tie_break,
            candidate_mcts_seeds: args.candidate_mcts_seeds,
        },
    )
    .unwrap_or_else(|error| panic!("observation-policy acquisition failed: {error}"));
    let output = publish_observation_policy_acquisition_report(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish acquisition report: {error}"));

    for round in &report.rounds {
        println!(
            "round={} incumbent_coverage={}/{} candidates={} frontier={} selected_seed={:?} selected_algorithm={:?} selected_beam={} selected_tie_break_seed={:?} selected_mcts_seed={} incumbent_retained={} selected_policy={} missing_partitions={}",
            round.round,
            round.incumbent_acquisition_pool_coverage,
            report.acquisition_pool_seeds.len(),
            round.candidates.len(),
            round.pareto_frontier.len(),
            round.selected_seed,
            round.selected_search_config.algorithm,
            round.selected_search_config.beam_width,
            round.selected_search_config.tie_break_seed,
            round.selected_search_config.mcts_seed,
            round.incumbent_retained,
            round.selected_policy_sha256,
            round.missing_partitions.len(),
        );
    }
    let final_report = &report.final_report;
    println!(
        "{} algorithm={:?} iterations={} acquired={:?} final_success={}/{} final_coverage={} rules={} bindings={} nearest_fallbacks={} max_l1={} total_expansions={} final_search_wall_us={} host_peak_rss_bytes={:?} planner_restores={} environment_restore_total_ns={} restore_total_ns={} restore_tile_ns={} restore_cell_ns={} restore_resolver_ns={} restore_integrity_ns={} restored_tiles={} restored_cells={} peak_nodes={} peak_particles={} peak_rules={} peak_logical_untried={} canonical_tile_pages={}/{} canonical_tile_chunks={}/{} canonical_cell_chunks={}/{} peak_mcts_choice_index_overrides={} estimated_peak_mcts_choice_bytes_lower_bound={} mcts_unique_choice_catalogs={} mcts_choice_catalog_intern_hits={} estimated_peak_policy_bytes_lower_bound={} policy_clones={} rollout_clones={} shared_particle_state_clones={} particle_state_allocations={} beam_generated_children={} beam_pruned_children={}",
        report.scenario,
        final_report.config.algorithm,
        final_report.algorithm_iterations,
        report.acquired_seeds,
        final_report.holdout_objective_success_count,
        final_report.holdout_seeds.len(),
        final_report.holdout_covered_to_requested_horizon,
        final_report.policy_rule_count,
        final_report.exact_observation_binding_count,
        final_report.holdout_nearest_fallback_rule_hits,
        final_report.holdout_maximum_nearest_fallback_l1_distance,
        report.total_expansions,
        final_report.search_cost.wall_time_micros,
        final_report.search_cost.host_peak_resident_set_bytes,
        final_report.search_cost.planner_restore_count,
        final_report
            .search_cost
            .planner_environment_restore_total_ns,
        final_report.search_cost.planner_restore_total_ns,
        final_report
            .search_cost
            .planner_restore_tile_materialization_ns,
        final_report
            .search_cost
            .planner_restore_cell_materialization_ns,
        final_report
            .search_cost
            .planner_restore_resolver_reconstruction_ns,
        final_report
            .search_cost
            .planner_restore_integrity_validation_ns,
        final_report.search_cost.planner_restored_tiles,
        final_report.search_cost.planner_restored_cells,
        final_report.search_cost.peak_retained_policy_nodes,
        final_report.search_cost.peak_retained_particles,
        final_report.search_cost.peak_retained_rules,
        final_report.search_cost.peak_retained_untried_choices,
        final_report
            .search_cost
            .peak_retained_unique_canonical_tile_pages,
        final_report
            .search_cost
            .peak_retained_canonical_tile_page_references,
        final_report
            .search_cost
            .peak_retained_unique_canonical_tile_chunks,
        final_report
            .search_cost
            .peak_retained_canonical_tile_chunk_references,
        final_report
            .search_cost
            .peak_retained_unique_canonical_cell_chunks,
        final_report
            .search_cost
            .peak_retained_canonical_cell_chunk_references,
        final_report
            .search_cost
            .peak_retained_mcts_choice_index_overrides,
        final_report
            .search_cost
            .estimated_peak_retained_mcts_choice_bytes_lower_bound,
        final_report.search_cost.mcts_unique_choice_catalogs,
        final_report.search_cost.mcts_choice_catalog_intern_hits,
        final_report
            .search_cost
            .estimated_peak_retained_policy_bytes_lower_bound,
        final_report.search_cost.policy_node_clones,
        final_report.search_cost.rollout_policy_clones,
        final_report.search_cost.shared_particle_state_clones,
        final_report.search_cost.particle_state_allocations,
        final_report.search_cost.beam_generated_children,
        final_report.search_cost.beam_pruned_children,
    );
    println!("Policy SHA-256: {}", final_report.policy_sha256);
    println!("Report: {}", output.display());
}
