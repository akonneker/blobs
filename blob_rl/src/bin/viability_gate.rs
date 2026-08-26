//! Evaluate a viability matrix against a strict declarative gate policy.

use std::path::PathBuf;

use blob_rl::viability_gate::{
    evaluate_viability_gates, publish_viability_gate_report, ViabilityGateDecision,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_viability_gate",
    about = "Publish a hash-bound pass/fail decision for a viability matrix"
)]
struct Args {
    /// Immutable viability-matrix JSON report.
    matrix: PathBuf,

    /// Strict TOML gate policy.
    #[arg(long)]
    gates: PathBuf,

    /// Immutable JSON decision destination.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let report = evaluate_viability_gates(&args.matrix, &args.gates)
        .unwrap_or_else(|error| panic!("viability gate evaluation failed: {error}"));
    let output = publish_viability_gate_report(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish viability gate report: {error}"));
    println!(
        "Viability gate: {:?}; {}/{} checks failed",
        report.decision, report.failed_checks, report.total_checks
    );
    println!("Report: {}", output.display());
    if report.decision == ViabilityGateDecision::Failed {
        std::process::exit(2);
    }
}
