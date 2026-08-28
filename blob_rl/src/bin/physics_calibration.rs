//! Apply coarse behavior expectations to an immutable ecological characterization.

use std::path::PathBuf;

use blob_rl::physics_calibration::{
    evaluate_physics_calibration_files, publish_physics_calibration, CalibrationDecision,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_physics_calibration",
    about = "Reject obviously unplayable physics regimes before policy training"
)]
struct Args {
    /// Existing immutable ecological-characterization JSON.
    #[arg(long)]
    characterization: PathBuf,

    /// TOML behavior-envelope specification.
    #[arg(long)]
    spec: PathBuf,

    /// New immutable JSON calibration decision.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let report = evaluate_physics_calibration_files(&args.characterization, &args.spec)
        .unwrap_or_else(|error| panic!("physics calibration failed: {error}"));
    let output = publish_physics_calibration(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish physics calibration: {error}"));
    println!("Physics calibration: {}", output.display());
    println!(
        "  decision: {:?} ({} of {} checks failed)",
        report.decision,
        report.failed_checks,
        report.checks.len()
    );
    for check in &report.checks {
        println!(
            "  {} {:>7} observed {:?} {:?} {} — {}",
            if check.passed { "PASS" } else { "FAIL" },
            check.metric,
            check.observed,
            check.comparator,
            check.threshold,
            check.rationale,
        );
    }
    if report.decision == CalibrationDecision::Failed {
        std::process::exit(2);
    }
}
