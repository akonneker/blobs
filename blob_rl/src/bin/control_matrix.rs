//! Run colony_mind against the four maintained control Minds.

use std::path::PathBuf;

use blob_rl::control_matrix::{
    publish_control_matrix_report, run_control_matrix, run_control_matrix_with_progress,
    write_control_matrix_progress, ControlMatrixOptions, MaintainedMindProfile,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_control_matrix",
    about = "Evaluate colony_mind against exact maintained proxy Minds"
)]
struct Args {
    /// Published rules-sweep manifest.json.
    manifest: PathBuf,

    /// Colony policy profile under evaluation.
    #[arg(long, value_enum, default_value_t = MaintainedMindProfile::Colony)]
    candidate: MaintainedMindProfile,

    /// Proxy Minds; omit to run simple, aggressive, defensive, and explorer.
    #[arg(long, value_delimiter = ',')]
    opponents: Vec<MaintainedMindProfile>,

    /// Maximum matchup jobs sharing the worker pool.
    #[arg(long, default_value_t = 1)]
    max_parallel: usize,

    /// Immutable local/unverified aggregate JSON destination.
    #[arg(long)]
    output: PathBuf,

    /// Mutable atomic progress JSON for a polling local viewer.
    #[arg(long)]
    live_output: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let opponents = if args.opponents.is_empty() {
        MaintainedMindProfile::CONTROLS.to_vec()
    } else {
        args.opponents
    };
    let options = ControlMatrixOptions {
        candidate: args.candidate,
        opponents,
        max_parallel: args.max_parallel,
    };
    let report = if let Some(live_output) = args.live_output.as_deref() {
        run_control_matrix_with_progress(&args.manifest, &options, |progress| {
            write_control_matrix_progress(live_output, progress)
        })
    } else {
        run_control_matrix(&args.manifest, &options)
    }
    .unwrap_or_else(|error| panic!("control matrix failed: {error}"));
    let output = publish_control_matrix_report(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish control matrix: {error}"));
    println!(
        "Control matrix: {} variants, {} matchups, {} episodes",
        report.variants.len(),
        report.matchups.len(),
        report
            .summaries
            .iter()
            .map(|summary| summary.episodes)
            .sum::<usize>()
    );
    for summary in &report.summaries {
        println!(
            "  {} vs {:10}: {:>3} wins / {:>3} losses / {:>3} timeouts ({:.1}% win)",
            report.candidate,
            summary.opponent,
            summary.colony_wins,
            summary.colony_losses,
            summary.timeouts,
            summary.colony_win_rate * 100.0
        );
    }
    println!("Trust: local deterministic; server_verified=false; replay_committed=false");
    println!("Report: {}", output.display());
}
