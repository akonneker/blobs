//! Evaluate a maintained native Mind as a prospective feeding teacher.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use blob_rl::config::TrainingConfig;
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::feeding_curriculum::{evaluate_feeding_teacher, FeedingPromotionReport};
use blob_rl::viability::mind_abi_hash;
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};

const FEEDING_TEACHER_EVALUATION_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Parser)]
#[command(
    name = "feeding-teacher-evaluation",
    about = "Evaluate a maintained Mind against the full-population feeding gates"
)]
struct Args {
    /// Exact training TOML supplying the scenario, rules, and gates.
    #[arg(long)]
    config: PathBuf,

    /// Maintained native Mind entry point to evaluate.
    #[arg(long)]
    teacher: MaintainedMindProfile,

    /// Held-out episode seeds, accepted as a comma-delimited list.
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    /// New immutable evaluation JSON file.
    #[arg(long)]
    output: PathBuf,

    /// Exit unsuccessfully after publishing a valid failing report.
    #[arg(long)]
    require_pass: bool,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct FeedingTeacherEvaluationArtifact {
    schema_version: u32,
    source_config_sha256: String,
    mind_abi_sha256: String,
    teacher: MaintainedMindProfile,
    local_unverified: bool,
    report: FeedingPromotionReport,
}

fn main() {
    let args = Args::parse();
    if args.seeds.is_empty() {
        panic!("feeding teacher evaluation requires at least one seed");
    }
    let config_bytes = fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid feeding-teacher config: {error}"));

    let report = evaluate_feeding_teacher(
        args.teacher,
        &config.env,
        &config.reward,
        &config.feeding_curriculum,
        &args.seeds,
    );
    let artifact = FeedingTeacherEvaluationArtifact {
        schema_version: FEEDING_TEACHER_EVALUATION_SCHEMA_VERSION,
        source_config_sha256: format!("{:x}", Sha256::digest(&config_bytes)),
        mind_abi_sha256: mind_abi_hash(),
        teacher: args.teacher,
        local_unverified: true,
        report,
    };
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("failed to create {}: {error}", parent.display()));
    }
    let bytes = serde_json::to_vec_pretty(&artifact)
        .expect("feeding-teacher evaluation artifact must serialize");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .unwrap_or_else(|error| panic!("failed to create {}: {error}", args.output.display()));
    file.write_all(&bytes)
        .and_then(|()| file.write_all(b"\n"))
        .and_then(|()| file.sync_all())
        .unwrap_or_else(|error| panic!("failed to publish {}: {error}", args.output.display()));

    println!(
        "Published {} feeding-teacher evaluation {} to {}:",
        args.teacher,
        if artifact.report.passed {
            "PASS"
        } else {
            "FAIL"
        },
        args.output.display(),
    );
    for stage in &artifact.report.stages {
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
    if args.require_pass && !artifact.report.passed {
        std::process::exit(2);
    }
}
