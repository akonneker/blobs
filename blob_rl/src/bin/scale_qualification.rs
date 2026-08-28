//! Characterize and gate ecological invariants across large-world profiles.

use std::path::PathBuf;

use blob_rl::scale_qualification::{
    publish_scale_qualification, run_scale_qualification, ScaleQualificationDecision,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_scale_qualification",
    about = "Publish bounded ecological qualification across increasing world scales"
)]
struct Args {
    /// TOML qualification specification containing ordered profile configs.
    #[arg(long)]
    spec: PathBuf,

    /// New immutable JSON report path.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let report = run_scale_qualification(&args.spec)
        .unwrap_or_else(|error| panic!("scale qualification failed: {error}"));
    let output = publish_scale_qualification(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish scale qualification: {error}"));
    println!("Scale qualification: {}", output.display());
    println!(
        "  decision: {:?} ({} of {} checks failed)",
        report.decision,
        report.failed_checks,
        report.checks.len()
    );
    for profile in &report.profiles {
        println!(
            "  {:>10}: {}x{}, {} cells/team, p90 endurance margin {:?}",
            profile.name,
            profile.metrics.world_size,
            profile.metrics.world_size,
            profile.metrics.cells_per_team,
            profile
                .metrics
                .minimum_team_p90_plant_endurance_margin_units,
        );
    }
    for check in report.checks.iter().filter(|check| !check.passed) {
        println!(
            "  FAIL {}{} observed {:?} {:?} {} — {}",
            check
                .profile
                .as_deref()
                .map(|profile| format!("{profile}:"))
                .unwrap_or_default(),
            check.metric,
            check.observed,
            check.comparator,
            check.threshold,
            check.rationale,
        );
    }
    if report.decision == ScaleQualificationDecision::Failed {
        std::process::exit(2);
    }
}
