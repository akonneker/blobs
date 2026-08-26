use std::path::PathBuf;

use blob_rl::signal_policy_ablation::{
    analyze_signal_policy_reports, publish_signal_policy_ablation,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_signal_policy_ablation",
    about = "Validate and pair colony signal-policy control matrices"
)]
struct Args {
    /// Five control reports: disabled, one-quantum, semantic sidecar,
    /// cadenced semantic sidecar, and full multichannel, in any order.
    reports: Vec<PathBuf>,

    /// Immutable aggregate JSON destination.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let report = analyze_signal_policy_reports(&args.reports)
        .unwrap_or_else(|error| panic!("signal-policy analysis failed: {error}"));
    publish_signal_policy_ablation(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish signal-policy analysis: {error}"));
    println!(
        "Signal-policy ablation: {} profiles, {} paired episodes each",
        report.rows.len(),
        report.rows.first().map_or(0, |row| row.episodes)
    );
    for row in &report.rows {
        println!(
            "  {:28} {:>2}W/{:>2}L/{:>2}T; paired +{}/-{}/={} signal_energy={}",
            row.candidate,
            row.wins,
            row.losses,
            row.timeouts,
            row.paired_vs_disabled.improved,
            row.paired_vs_disabled.worsened,
            row.paired_vs_disabled.unchanged,
            row.signal_energy,
        );
    }
}
