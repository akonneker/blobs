//! Generate immutable maintained-Mind behavior-cloning datasets.

use std::path::PathBuf;

use blob_rl::config::TrainingConfig;
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::demonstration::{
    generate_demonstrations, publish_demonstrations, DemonstrationOptions,
};
use clap::Parser;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "blob_demonstrations",
    about = "Generate a hash-bound maintained-Mind imitation dataset"
)]
struct Args {
    /// Exact training TOML supplying the scenario, rules, and opponent.
    #[arg(long)]
    config: PathBuf,

    /// Maintained native Mind used as the teacher.
    #[arg(long, value_enum)]
    teacher: MaintainedMindProfile,

    /// Unique episode seeds, accepted as a comma-delimited list.
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    /// Maximum ready-cell decisions retained in the payload.
    #[arg(long)]
    max_samples: usize,

    /// New immutable dataset directory.
    #[arg(long)]
    output: PathBuf,

    /// Optional scenario overrides for a field/population curriculum.
    #[arg(long)]
    world_size: Option<usize>,
    #[arg(long)]
    cells_per_team: Option<usize>,
    #[arg(long)]
    num_scattered_energy: Option<usize>,
    #[arg(long)]
    num_plants: Option<usize>,
}

fn main() {
    let args = Args::parse();
    let config_bytes = std::fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let mut config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    if let Some(value) = args.world_size {
        config.env.world_size = value;
    }
    if let Some(value) = args.cells_per_team {
        config.env.cells_per_team = value;
    }
    if let Some(value) = args.num_scattered_energy {
        config.env.num_scattered_energy = value;
    }
    if let Some(value) = args.num_plants {
        config.env.num_plants = value;
    }
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid effective demonstration config: {error}"));
    let source_config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
    let (manifest, payload) = generate_demonstrations(
        &config,
        source_config_sha256,
        &DemonstrationOptions {
            teacher: args.teacher,
            seeds: args.seeds,
            max_samples: args.max_samples,
        },
    )
    .unwrap_or_else(|error| panic!("failed to generate demonstrations: {error}"));
    publish_demonstrations(&args.output, &manifest, &payload)
        .unwrap_or_else(|error| panic!("failed to publish demonstrations: {error}"));
    println!(
        "Published {} {} samples to {} ({} exact round trips, {} memory replacements)",
        manifest.samples,
        manifest.teacher,
        args.output.display(),
        manifest.exact_round_trip_samples,
        manifest.memory_replacement_samples,
    );
}
