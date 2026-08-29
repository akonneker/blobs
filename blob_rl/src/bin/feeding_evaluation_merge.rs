//! Merge validated, hash-bound feeding-evaluation shards.

use std::path::PathBuf;

use blob_rl::feeding_evaluation_artifact::{
    load_feeding_evaluation, merge_feeding_evaluations, publish_feeding_evaluation,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "feeding-evaluation-merge",
    about = "Merge resumable feeding-evaluation shards and recompute their gates"
)]
struct Args {
    /// Validated input artifacts in desired seed order.
    #[arg(long, required = true)]
    input: Vec<PathBuf>,

    /// New immutable aggregate evaluation artifact.
    #[arg(long)]
    output: PathBuf,

    /// Exit unsuccessfully after publishing a valid failing aggregate.
    #[arg(long)]
    require_pass: bool,
}

fn main() {
    let args = Args::parse();
    let shards = args
        .input
        .iter()
        .map(|path| {
            load_feeding_evaluation(path)
                .unwrap_or_else(|error| panic!("invalid shard {}: {error}", path.display()))
        })
        .collect::<Vec<_>>();
    let aggregate = merge_feeding_evaluations(&shards)
        .unwrap_or_else(|error| panic!("failed to merge feeding evaluations: {error}"));
    publish_feeding_evaluation(&args.output, &aggregate)
        .unwrap_or_else(|error| panic!("failed to publish aggregate: {error}"));
    println!(
        "Published {}-seed feeding aggregate {} to {}:",
        aggregate.report.seeds.len(),
        if aggregate.report.passed {
            "PASS"
        } else {
            "FAIL"
        },
        args.output.display(),
    );
    for stage in &aggregate.report.stages {
        println!(
            "  {}: success {:.1}%, survival {:.1}%, intake {:.3}/initial cell (moves {}, consumes {}, safety aborts {})",
            stage.stage,
            stage.episode_success_rate * 100.0,
            stage.survival_rate * 100.0,
            stage.consumed_energy_per_initial_cell,
            stage.movement_successes,
            stage.consume_successes,
            stage.safety_aborts,
        );
    }
    if args.require_pass && !aggregate.report.passed {
        std::process::exit(2);
    }
}
