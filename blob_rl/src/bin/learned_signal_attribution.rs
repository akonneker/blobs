use std::path::PathBuf;

use blob_rl::artifact::{load_policy_snapshot, verify_checkpoint_metadata};
use blob_rl::config::OpponentProfile;
use blob_rl::learned_signal_attribution::{
    evaluate_learned_signal_attribution, publish_learned_signal_attribution,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_learned_signal_attribution",
    about = "Evaluate one frozen RL checkpoint under paired signal-visibility counterfactuals"
)]
struct Args {
    /// Immutable checkpoint directory containing metadata and model records.
    checkpoint: PathBuf,

    /// Immutable JSON report destination.
    #[arg(long)]
    output: PathBuf,

    /// Episodes per opponent. Defaults to the checkpoint evaluation suite.
    #[arg(long)]
    episodes: Option<usize>,

    /// First paired environment seed. Defaults to the checkpoint evaluation seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Baseline opponent(s). Repeat the flag; defaults to checkpoint evaluation opponents.
    #[arg(long, value_enum)]
    opponent: Vec<OpponentProfile>,
}

fn evaluate<B: burn::prelude::Backend>(args: &Args, device: B::Device)
where
    f32: From<B::FloatElem>,
{
    let metadata = verify_checkpoint_metadata(&args.checkpoint)
        .unwrap_or_else(|error| panic!("failed to verify checkpoint: {error}"));
    let snapshot = load_policy_snapshot::<B>(&args.checkpoint, &device)
        .unwrap_or_else(|error| panic!("failed to load frozen policy: {error}"));
    let episode_count = args.episodes.unwrap_or(metadata.config.eval_episodes);
    assert!(episode_count > 0, "--episodes must be positive");
    let seed = args.seed.unwrap_or(metadata.config.evaluation_seed);
    let seeds = (0..episode_count)
        .map(|index| seed.wrapping_add(index as u64))
        .collect::<Vec<_>>();
    let opponents = if args.opponent.is_empty() {
        metadata.config.evaluation_opponents.clone()
    } else {
        args.opponent.clone()
    };
    let report =
        evaluate_learned_signal_attribution(&snapshot, &metadata, &opponents, &seeds, &device)
            .unwrap_or_else(|error| panic!("learned signal attribution failed: {error}"));
    publish_learned_signal_attribution(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish attribution report: {error}"));

    let full = &report.full_visibility;
    let hidden = &report.hidden_neighbor_visibility;
    println!(
        "Frozen policy {}: {} paired episodes, fixed team-zero seat",
        &report.model_sha256[..report.model_sha256.len().min(12)],
        full.episodes.len()
    );
    println!(
        "  full   {:>3}W/{:>3}L/{:>3}T  signal={:>8}  field={:>10.2}",
        full.wins,
        full.losses,
        full.timeouts,
        full.signal_energy,
        full.mean_environment_signal_energy
    );
    println!(
        "  hidden {:>3}W/{:>3}L/{:>3}T  signal={:>8}  field={:>10.2}",
        hidden.wins,
        hidden.losses,
        hidden.timeouts,
        hidden.signal_energy,
        hidden.mean_environment_signal_energy
    );
    println!(
        "  hidden vs full outcomes: +{}/-{}/={} (direct signal reward: none)",
        report.hidden_vs_full_outcomes.improved,
        report.hidden_vs_full_outcomes.worsened,
        report.hidden_vs_full_outcomes.unchanged
    );
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "wgpu")]
    {
        let device = burn::backend::wgpu::WgpuDevice::default();
        evaluate::<burn::backend::Wgpu>(&args, device);
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        evaluate::<burn::backend::NdArray>(&args, device);
    }
}
