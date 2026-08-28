//! Generate immutable maintained-Mind behavior-cloning datasets.

use std::path::PathBuf;

use blob_rl::config::{FeedingCurriculumStage, OpponentProfile, TrainingConfig};
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::demonstration::{
    generate_demonstrations, publish_demonstrations, DemonstrationOptions,
};
use clap::{Parser, ValueEnum};
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

    /// Replace ordinary resource placement with one feeding-curriculum stage.
    /// This changes only episode initialization and the prerequisite opponent;
    /// observations and resolver semantics remain unchanged.
    #[arg(long, value_enum)]
    feeding_stage: Option<FeedingStage>,

    /// Generate a paired one-on-one contact scenario at this initial energy.
    /// Must be supplied with --contact-opponent and cannot be combined with a
    /// feeding stage.
    #[arg(long, requires = "contact_opponent", conflicts_with = "feeding_stage")]
    contact_energy: Option<u32>,

    /// Anonymous baseline inhabiting the opposing contact cell.
    #[arg(long, value_enum, requires = "contact_energy")]
    contact_opponent: Option<OpponentProfile>,

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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum FeedingStage {
    OnFood,
    AdjacentFood,
    Competitive,
}

impl From<FeedingStage> for FeedingCurriculumStage {
    fn from(value: FeedingStage) -> Self {
        match value {
            FeedingStage::OnFood => Self::OnFood,
            FeedingStage::AdjacentFood => Self::AdjacentFood,
            FeedingStage::Competitive => Self::Competitive,
        }
    }
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
    if let Some(stage) = args.feeding_stage {
        config.env = config
            .feeding_curriculum
            .environment_for_stage(&config.env, stage.into());
    }
    if let (Some(energy), Some(opponent)) = (args.contact_energy, args.contact_opponent) {
        if !config.combat_curriculum.enabled
            || !matches!(
                opponent,
                OpponentProfile::Aggressive | OpponentProfile::Defensive
            )
        {
            panic!(
                "contact demonstrations require an enabled combat curriculum and an aggressive or defensive opponent"
            );
        }
        let contact_start = config
            .combat_curriculum
            .on_food_sim_time_quanta_per_cycle
            .saturating_add(
                config
                    .combat_curriculum
                    .adjacent_food_sim_time_quanta_per_cycle,
            );
        config.env = config.rollout_environment(FeedingCurriculumStage::Contact, contact_start);
        config.env.initial_energy = energy;
        config.env.opponent = opponent;
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
        "Published {} {} samples to {} ({} exact round trips, {} memory replacements, {} attacks)",
        manifest.samples,
        manifest.teacher,
        args.output.display(),
        manifest.exact_round_trip_samples,
        manifest.memory_replacement_samples,
        manifest.action_family_samples[blob_rl::action::PolicyActionFamily::Attack.index()],
    );
}
