//! Audit small combat scenarios with legal maintained witness strategies.

use std::fs;
use std::path::PathBuf;

use blob_rl::config::{OpponentProfile, TrainingConfig};
use blob_rl::micro_combat::MicroCombatSuiteConfig;
use blob_rl::micro_combat_feasibility::{
    audit_micro_combat_feasibility, publish_micro_combat_feasibility,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_micro_combat_feasibility",
    about = "Audit whether legal observation-only witnesses solve micro-combat scenarios"
)]
struct Args {
    /// Training TOML supplying the exact physics and reward configuration.
    #[arg(long)]
    config: PathBuf,

    /// Micro-combat scenario suite TOML.
    #[arg(long)]
    scenarios: PathBuf,

    /// First deterministic environment seed.
    #[arg(long, default_value_t = 1_000_000)]
    seed_base: u64,

    /// Number of consecutive held-out seeds.
    #[arg(long, default_value_t = 128)]
    seed_count: usize,

    /// Success rate required for a reliable witness and degeneracy flag.
    #[arg(long, default_value_t = 0.95)]
    reliability_target: f64,

    /// JSON report destination.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    assert!(args.seed_count > 0, "seed-count must be positive");
    let config = TrainingConfig::from_file(&args.config.to_string_lossy())
        .unwrap_or_else(|error| panic!("failed to load training config: {error}"));
    let suite_content = fs::read_to_string(&args.scenarios)
        .unwrap_or_else(|error| panic!("failed to read scenario suite: {error}"));
    let suite = MicroCombatSuiteConfig::from_toml_str(&suite_content)
        .unwrap_or_else(|error| panic!("failed to load scenario suite: {error}"));
    let seeds = (0..args.seed_count)
        .map(|offset| {
            args.seed_base
                .checked_add(offset as u64)
                .expect("seed range overflowed")
        })
        .collect::<Vec<_>>();
    let report = audit_micro_combat_feasibility(
        &config.env,
        &config.reward,
        &suite,
        &seeds,
        &OpponentProfile::ALL,
        args.reliability_target,
    )
    .unwrap_or_else(|error| panic!("micro-combat feasibility audit failed: {error}"));
    let output = publish_micro_combat_feasibility(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish feasibility report: {error}"));

    println!("scenario\tobjective\tbest witness\tsuccess\twait-degenerate");
    for scenario in &report.scenarios {
        println!(
            "{}\t{:?}\t{}\t{:.3}\t{}",
            scenario.scenario,
            scenario.objective,
            scenario.best_witness,
            scenario.best_witness_success_rate,
            scenario.passive_timeout_degenerate,
        );
    }
    println!("Report: {}", output.display());
}
