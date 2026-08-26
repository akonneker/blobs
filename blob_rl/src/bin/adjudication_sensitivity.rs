//! Evaluate hypothetical deadline adjudications without changing match rules.

use std::path::PathBuf;

use blob_rl::adjudication_sensitivity::{
    analyze_adjudication_sensitivity, publish_adjudication_sensitivity_report,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_adjudication_sensitivity",
    about = "Apply explicit terminal-compartment weight grids to control-matrix deadline draws"
)]
struct Args {
    /// Immutable schema-3 control-matrix JSON report.
    matrix: PathBuf,

    /// Strict TOML file containing named hypothetical scoring policies.
    #[arg(long)]
    policies: PathBuf,

    /// Immutable JSON analysis destination.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let report = analyze_adjudication_sensitivity(&args.matrix, &args.policies)
        .unwrap_or_else(|error| panic!("adjudication sensitivity failed: {error}"));
    let output = publish_adjudication_sensitivity_report(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish sensitivity report: {error}"));
    println!(
        "Adjudication sensitivity: {} policies over {} source episodes ({} deadline draws)",
        report.policies.len(),
        report.source_episodes,
        report.source_deadline_draws
    );
    for policy in &report.policies {
        println!(
            "  {:24}: {:>3} wins / {:>3} losses / {:>3} draws",
            policy.name,
            policy.overall.hypothetical_colony_wins,
            policy.overall.hypothetical_colony_losses,
            policy.overall.hypothetical_draws
        );
    }
    println!("Report: {}", output.display());
}
