//! Publish a hash-bound scenario diagnostic ladder report.

use std::path::PathBuf;

use blob_rl::scenario_diagnosis::{
    evaluate_scenario_diagnosis_files, publish_scenario_diagnosis, DiagnosticAttribution,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_scenario_diagnosis",
    about = "Separate physics, Mind-interface, and learned-policy failure evidence"
)]
struct Args {
    /// TOML diagnostic ladder specification. Relative evidence paths start here.
    #[arg(long)]
    spec: PathBuf,

    /// New immutable JSON diagnosis report.
    #[arg(long)]
    output: PathBuf,

    /// Exit with status 2 unless every stage passes or is legitimately inapplicable.
    #[arg(long)]
    require_competent: bool,
}

fn main() {
    let args = Args::parse();
    let report = evaluate_scenario_diagnosis_files(&args.spec)
        .unwrap_or_else(|error| panic!("scenario diagnosis failed: {error}"));
    let output = publish_scenario_diagnosis(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish scenario diagnosis: {error}"));
    println!("Scenario diagnosis: {}", output.display());
    println!("  scenario: {}", report.scenario_name);
    println!("  attribution: {:?}", report.attribution);
    println!("  diagnosis hash: {}", report.diagnosis_hash);
    for evidence in &report.evidence {
        println!(
            "  {:?}: {:?} — {}",
            evidence.stage, evidence.outcome, evidence.summary
        );
    }
    if args.require_competent && report.attribution != DiagnosticAttribution::Competent {
        std::process::exit(2);
    }
}
