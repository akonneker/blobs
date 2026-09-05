//! Run a replay-verified bounded search over one single-cell micro scenario.

use std::fs;
use std::path::PathBuf;

use blob_rl::config::TrainingConfig;
use blob_rl::micro_combat::MicroCombatSuiteConfig;
use blob_rl::micro_combat_planner::{
    publish_bounded_search_report, BoundedSearchConfig, SingleCellSearchSimulator,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_micro_combat_search",
    about = "Find a replay-verified bounded single-cell action sequence"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,

    #[arg(long)]
    scenarios: PathBuf,

    #[arg(long)]
    scenario: String,

    #[arg(long)]
    seed: u64,

    #[arg(long, default_value_t = 128)]
    beam_width: usize,

    #[arg(long, default_value_t = 64)]
    max_depth: usize,

    #[arg(long, default_value_t = 100_000)]
    max_expansions: usize,

    /// Retain effort tiers even when their exact cost and completion time are
    /// equal for the current state.
    #[arg(long, default_value_t = false)]
    disable_equivalent_effort_pruning: bool,

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
    let planner = SingleCellSearchSimulator::new(&config.env, &config.reward, scenario, args.seed)
        .unwrap_or_else(|error| panic!("failed to initialize planner: {error}"));
    let report = planner
        .search_bounded(
            &scenario.name,
            args.seed,
            BoundedSearchConfig {
                beam_width: args.beam_width,
                max_depth: args.max_depth,
                max_expansions: args.max_expansions,
                equivalent_effort_pruning: !args.disable_equivalent_effort_pruning,
            },
        )
        .unwrap_or_else(|error| panic!("bounded search failed: {error}"));
    let output = publish_bounded_search_report(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish bounded-search report: {error}"));
    println!(
        "{} seed {}: success={} outcome={} energy={} sequence={} expansions={} unique={} root_choices={}/{} truncated={} exhaustive={}",
        report.scenario,
        report.seed,
        report.objective_success,
        report.final_outcome,
        report.final_training_stored_energy,
        report.sequence.len(),
        report.expansions,
        report.unique_states,
        report.root_choices_after_pruning,
        report.root_choices_before_pruning,
        report.truncated,
        report.exhaustive,
    );
    println!("Report: {}", output.display());
}
