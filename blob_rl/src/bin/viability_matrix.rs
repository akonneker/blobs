//! Execute a paired non-learning viability matrix from a sweep manifest.

use std::path::PathBuf;

use blob_rl::config::OpponentProfile;
use blob_rl::viability_matrix::{
    publish_viability_matrix_report, run_viability_matrix, ViabilityMatrixOptions,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_viability_matrix",
    about = "Run a bounded, paired baseline matrix over a verified rules sweep"
)]
struct Args {
    /// Published rules-sweep manifest.json.
    manifest: PathBuf,

    /// Team-zero profiles, accepted as repeated or comma-delimited values.
    #[arg(long, required = true, value_delimiter = ',')]
    candidates: Vec<OpponentProfile>,

    /// Opponent profiles, accepted as repeated or comma-delimited values.
    #[arg(long, required = true, value_delimiter = ',')]
    opponents: Vec<OpponentProfile>,

    /// Variant used for paired differences. Defaults to the first variant.
    #[arg(long)]
    baseline_variant: Option<String>,

    /// Maximum viability jobs sharing the worker pool.
    #[arg(long, default_value_t = 1)]
    max_parallel: usize,

    /// Immutable aggregate JSON destination.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let report = run_viability_matrix(
        &args.manifest,
        &ViabilityMatrixOptions {
            candidates: args.candidates,
            opponents: args.opponents,
            baseline_variant: args.baseline_variant,
            max_parallel: args.max_parallel,
        },
    )
    .unwrap_or_else(|error| panic!("viability matrix failed: {error}"));
    let output = publish_viability_matrix_report(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish viability matrix: {error}"));
    println!(
        "Viability matrix: {} variants, {} matchups, {} paired comparisons",
        report.variants.len(),
        report.matchups.len(),
        report.paired_comparisons.len()
    );
    println!("Report: {}", output.display());
}
