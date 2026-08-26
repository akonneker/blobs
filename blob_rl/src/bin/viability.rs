//! Run one deterministic non-learning baseline matchup.

use std::path::PathBuf;

use blob_rl::config::{OpponentProfile, TrainingConfig};
use blob_rl::viability::{publish_viability_report, run_baseline_viability};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_viability",
    about = "Evaluate two anonymous native baselines without policy learning"
)]
struct Args {
    /// Training TOML whose environment, rules, and telemetry define the match.
    #[arg(long)]
    config: PathBuf,

    /// Team-zero baseline profile.
    #[arg(long)]
    candidate: OpponentProfile,

    /// Baseline profile used by every other team.
    #[arg(long)]
    opponent: OpponentProfile,

    /// Unique environment seeds, accepted as repeated or comma-delimited values.
    #[arg(long, required = true, value_delimiter = ',')]
    seeds: Vec<u64>,

    /// Immutable JSON report destination.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let config = TrainingConfig::from_file(&args.config.to_string_lossy())
        .unwrap_or_else(|error| panic!("failed to load viability config: {error}"));
    let report = run_baseline_viability(
        &config.env,
        &config.telemetry,
        args.candidate,
        args.opponent,
        &args.seeds,
    )
    .unwrap_or_else(|error| panic!("viability evaluation failed: {error}"));
    let output = publish_viability_report(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish viability report: {error}"));
    println!(
        "Viability: {} vs {} over {} seeds: {} wins, {} losses, {} timeouts",
        report.candidate,
        report.opponent,
        report.aggregate.episodes,
        report.aggregate.candidate_wins,
        report.aggregate.candidate_losses,
        report.aggregate.timeouts,
    );
    println!("Report: {}", output.display());
}
