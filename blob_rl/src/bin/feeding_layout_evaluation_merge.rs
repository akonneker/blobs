//! Merge validated, hash-bound feeding-layout evaluation shards.

use std::path::PathBuf;

use blob_rl::feeding_layout_evaluation::{
    load_feeding_layout_evaluation, merge_feeding_layout_evaluations,
    publish_feeding_layout_evaluation,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "feeding-layout-evaluation-merge",
    about = "Merge resumable layout/seed feeding trials into a complete matrix"
)]
struct Args {
    #[arg(long, required = true)]
    input: Vec<PathBuf>,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    require_pass: bool,
}

fn main() {
    let args = Args::parse();
    let shards = args
        .input
        .iter()
        .map(|path| {
            load_feeding_layout_evaluation(path)
                .unwrap_or_else(|error| panic!("invalid shard {}: {error}", path.display()))
        })
        .collect::<Vec<_>>();
    let aggregate = merge_feeding_layout_evaluations(&shards)
        .unwrap_or_else(|error| panic!("failed to merge feeding-layout evaluations: {error}"));
    publish_feeding_layout_evaluation(&args.output, &aggregate)
        .unwrap_or_else(|error| panic!("failed to publish aggregate: {error}"));
    println!(
        "Published {}x{} feeding-layout aggregate {} to {} (artifact {})",
        aggregate.layouts.len(),
        aggregate.seeds.len(),
        if aggregate.passed { "PASS" } else { "FAIL" },
        args.output.display(),
        aggregate.artifact_hash,
    );
    for trial in &aggregate.trials {
        println!(
            "  {:?} seed {}: {} (on-food {:.1}%, adjacent {:.1}%)",
            trial.layout,
            trial.seed,
            if trial.report.passed { "PASS" } else { "FAIL" },
            trial.report.stages[0].survival_rate * 100.0,
            trial.report.stages[1].survival_rate * 100.0,
        );
    }
    if args.require_pass && !aggregate.passed {
        std::process::exit(2);
    }
}
