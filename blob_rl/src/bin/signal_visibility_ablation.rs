use std::path::PathBuf;

use blob_rl::signal_visibility_ablation::{
    analyze_signal_visibility_report, publish_signal_visibility_ablation,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_signal_visibility_ablation",
    about = "Validate paired one-quantum signal visibility controls"
)]
struct Args {
    /// Control matrix containing full, cardinal, and hidden-neighbor variants.
    source: PathBuf,

    /// Matching no-deposit colony control matrix.
    #[arg(long)]
    disabled_control: PathBuf,

    /// Immutable aggregate JSON destination.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let report = analyze_signal_visibility_report(&args.source, &args.disabled_control)
        .unwrap_or_else(|error| panic!("signal-visibility analysis failed: {error}"));
    publish_signal_visibility_ablation(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish signal-visibility analysis: {error}"));
    println!(
        "Signal-visibility ablation: {} variants, {} paired episodes each",
        report.rows.len(),
        report.rows.first().map_or(0, |row| row.episodes)
    );
    for row in &report.rows {
        println!(
            "  {:24} {:>2}W/{:>2}L/{:>2}T; vs full +{}/-{}/={}; vs disabled +{}/-{}/={} variation={:.2}",
            row.variant,
            row.wins,
            row.losses,
            row.timeouts,
            row.paired_vs_full_visibility.improved,
            row.paired_vs_full_visibility.worsened,
            row.paired_vs_full_visibility.unchanged,
            row.paired_vs_disabled.improved,
            row.paired_vs_disabled.worsened,
            row.paired_vs_disabled.unchanged,
            row.mean_observable_signal_variation,
        );
    }
}
