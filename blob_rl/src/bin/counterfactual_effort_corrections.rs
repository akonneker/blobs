//! Publish active/control Move-effort datasets from verified branch evidence.

use std::path::PathBuf;

use blob_rl::counterfactual_branch::load_counterfactual_branch_artifact;
use blob_rl::counterfactual_correction::{
    publish_counterfactual_effort_corrections, CounterfactualEffortCorrectionOptions,
};
use blob_rl::counterfactual_value::{load_counterfactual_value_artifact, ValuePerspective};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PerspectiveArg {
    Cell,
    Colony,
    Conservative,
}

impl From<PerspectiveArg> for ValuePerspective {
    fn from(value: PerspectiveArg) -> Self {
        match value {
            PerspectiveArg::Cell => Self::Cell,
            PerspectiveArg::Colony => Self::Colony,
            PerspectiveArg::Conservative => Self::Conservative,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Arm {
    Control,
    Correction,
}

#[derive(Debug, Parser)]
#[command(
    name = "counterfactual-effort-corrections",
    about = "Create target-free effort-only training data from robust branch frontiers"
)]
struct Args {
    /// Counterfactual branch artifact retaining exact anonymous Mind evidence.
    #[arg(long)]
    counterfactual: PathBuf,

    /// Value artifact derived from the exact counterfactual source.
    #[arg(long)]
    value: PathBuf,

    #[arg(long, value_enum, default_value_t = PerspectiveArg::Conservative)]
    perspective: PerspectiveArg,

    #[arg(long)]
    horizon_quanta: u64,

    /// Control preserves policy effort; correction uses minimum Move effort.
    #[arg(long, value_enum)]
    arm: Arm,

    /// New immutable demonstration directory.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let source = load_counterfactual_branch_artifact(&args.counterfactual)
        .unwrap_or_else(|error| panic!("invalid counterfactual source: {error}"));
    let value = load_counterfactual_value_artifact(&args.value, &source)
        .unwrap_or_else(|error| panic!("invalid counterfactual value source: {error}"));
    let active_correction = matches!(args.arm, Arm::Correction);
    let output = publish_counterfactual_effort_corrections(
        &args.output,
        &source,
        &value,
        CounterfactualEffortCorrectionOptions {
            perspective: args.perspective.into(),
            horizon_quanta: args.horizon_quanta,
            active_correction,
        },
    )
    .unwrap_or_else(|error| panic!("failed to publish effort corrections: {error}"));
    println!(
        "Published {} effort arm to {} from {} / {}",
        if active_correction {
            "correction"
        } else {
            "control"
        },
        output.display(),
        source.artifact_hash,
        value.artifact_hash,
    );
}
