use std::collections::BTreeMap;
use std::path::PathBuf;

use blob_rl::artifact::{load_policy_snapshot, verify_checkpoint_metadata};
use blob_rl::config::OpponentProfile;
use blob_rl::evaluation::record_greedy_policy_match;
use blob_rl::match_explorer::{MatchExplorerConfig, MatchExplorerTeamSpec};
use clap::Parser;

const TEAM_COLORS: [&str; 6] = [
    "#65e6a8", "#ffb55e", "#72b7ff", "#e987ff", "#f56f7d", "#d8e66a",
];

#[derive(Debug, Parser)]
#[command(
    name = "blob_match_explorer",
    about = "Record one frozen-policy match for the local web explorer"
)]
struct Args {
    /// Immutable checkpoint directory containing metadata and model records.
    checkpoint: PathBuf,

    /// Explorer JSON destination. A canonical .replay.bin is written beside it.
    #[arg(long)]
    output: PathBuf,

    /// Environment seed. Defaults to the checkpoint evaluation seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Fixed opponent profile.
    #[arg(long, value_enum, default_value_t = OpponentProfile::Aggressive)]
    opponent: OpponentProfile,
}

fn record<B: burn::prelude::Backend>(args: &Args, device: B::Device)
where
    f32: From<B::FloatElem>,
{
    let metadata = verify_checkpoint_metadata(&args.checkpoint)
        .unwrap_or_else(|error| panic!("failed to verify checkpoint: {error}"));
    let snapshot = load_policy_snapshot::<B>(&args.checkpoint, &device)
        .unwrap_or_else(|error| panic!("failed to load frozen policy: {error}"));
    let seed = args.seed.unwrap_or(metadata.config.evaluation_seed);
    let short_hash = &snapshot.model_sha256[..snapshot.model_sha256.len().min(12)];
    let mut teams = BTreeMap::new();
    for team in 0..metadata.config.env.num_teams {
        let (name, mind) = if team == 0 {
            (
                format!("Learned policy u{}", snapshot.update),
                format!("checkpoint:{short_hash}"),
            )
        } else {
            (
                format!("{} baseline", args.opponent),
                format!("baseline:{}", args.opponent),
            )
        };
        teams.insert(
            team as u64,
            MatchExplorerTeamSpec {
                name,
                mind,
                color: TEAM_COLORS[team % TEAM_COLORS.len()].into(),
            },
        );
    }
    let explorer = MatchExplorerConfig {
        run_id: format!("checkpoint-u{}-seed-{seed}", snapshot.update),
        title: format!("Learned policy u{} vs. {}", snapshot.update, args.opponent),
        ruleset: metadata.compiled_ruleset_hash.clone(),
        teams,
    };
    let recorded = record_greedy_policy_match(
        &snapshot.model,
        &metadata.config.env,
        &metadata.config.reward,
        args.opponent,
        seed,
        &device,
        explorer,
    )
    .unwrap_or_else(|error| panic!("match recording failed: {error}"));
    let events = recorded.replay_event_count();
    let (json, replay) = recorded
        .publish(&args.output)
        .unwrap_or_else(|error| panic!("failed to publish match: {error}"));
    println!("Recorded {events} canonical resolution events");
    println!("Explorer: {}", json.display());
    println!("Replay:   {}", replay.display());
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "wgpu")]
    {
        let device = burn::backend::wgpu::WgpuDevice::default();
        record::<burn::backend::Wgpu>(&args, device);
    }

    #[cfg(not(feature = "wgpu"))]
    {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        record::<burn::backend::NdArray>(&args, device);
    }
}
