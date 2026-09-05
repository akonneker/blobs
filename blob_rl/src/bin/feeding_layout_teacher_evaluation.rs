//! Evaluate a maintained native teacher across founder geometries.

use std::path::PathBuf;

use blob_rl::config::TrainingConfig;
use blob_rl::control_matrix::MaintainedMindProfile;
use blob_rl::feeding_curriculum::evaluate_feeding_teacher;
use blob_rl::feeding_layout_evaluation::{
    load_feeding_layout_evaluation, publish_feeding_layout_evaluation,
    verify_feeding_layout_evaluation_request, FeedingLayoutEvaluationArtifact,
    FeedingLayoutPolicyIdentity, FeedingLayoutTrial, FeedingQualificationLayout,
};
use blob_rl::viability::mind_abi_hash;
use clap::Parser;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "feeding-layout-teacher-evaluation",
    about = "Qualify a maintained teacher across founder geometries"
)]
struct Args {
    #[arg(long)]
    config: PathBuf,

    #[arg(long)]
    teacher: MaintainedMindProfile,

    #[arg(long, value_delimiter = ',', required = true)]
    layouts: Vec<FeedingQualificationLayout>,

    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    require_pass: bool,

    #[arg(long)]
    resume: bool,
}

fn main() {
    let args = Args::parse();
    let config_bytes = std::fs::read(&args.config)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", args.config.display()));
    let config_text = std::str::from_utf8(&config_bytes)
        .unwrap_or_else(|error| panic!("{} is not UTF-8: {error}", args.config.display()));
    let config = TrainingConfig::from_toml_str(config_text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", args.config.display()));
    config
        .validate()
        .unwrap_or_else(|error| panic!("invalid feeding-layout teacher config: {error}"));
    let source_config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
    let policy = FeedingLayoutPolicyIdentity::MaintainedTeacher {
        profile: args.teacher,
        mind_abi_sha256: mind_abi_hash(),
    };

    if args.output.exists() {
        if !args.resume {
            panic!(
                "feeding-layout teacher output {} exists; pass --resume to verify and reuse it",
                args.output.display()
            );
        }
        let artifact = load_feeding_layout_evaluation(&args.output)
            .unwrap_or_else(|error| panic!("invalid existing teacher artifact: {error}"));
        verify_feeding_layout_evaluation_request(
            &artifact,
            &source_config_sha256,
            &policy,
            &config,
            &args.layouts,
            &args.seeds,
        )
        .unwrap_or_else(|error| panic!("feeding-layout teacher resume rejected: {error}"));
        println!(
            "Reused verified {}x{} feeding-layout teacher evaluation {} at {}",
            artifact.layouts.len(),
            artifact.seeds.len(),
            if artifact.passed { "PASS" } else { "FAIL" },
            args.output.display(),
        );
        if args.require_pass && !artifact.passed {
            std::process::exit(2);
        }
        return;
    }

    let mut trials = Vec::with_capacity(args.layouts.len().saturating_mul(args.seeds.len()));
    for layout in &args.layouts {
        let mut effective = config.clone();
        effective.env.starting_cell_layout = layout.starting_layout();
        effective
            .validate()
            .unwrap_or_else(|error| panic!("invalid {layout:?} teacher config: {error}"));
        for seed in &args.seeds {
            let report = evaluate_feeding_teacher(
                args.teacher,
                &effective.env,
                &effective.reward,
                &effective.feeding_curriculum,
                &[*seed],
            );
            println!(
                "  {layout:?} seed {seed}: {} (on-food {:.1}%, adjacent {:.1}%)",
                if report.passed { "PASS" } else { "FAIL" },
                report.stages[0].survival_rate * 100.0,
                report.stages[1].survival_rate * 100.0,
            );
            trials.push(FeedingLayoutTrial {
                layout: *layout,
                seed: *seed,
                report,
            });
        }
    }
    let artifact = FeedingLayoutEvaluationArtifact::new(
        source_config_sha256,
        policy,
        config,
        args.layouts,
        args.seeds,
        trials,
    )
    .unwrap_or_else(|error| panic!("invalid feeding-layout teacher evaluation: {error}"));
    publish_feeding_layout_evaluation(&args.output, &artifact)
        .unwrap_or_else(|error| panic!("failed to publish teacher evaluation: {error}"));
    println!(
        "Published feeding-layout teacher evaluation {} to {} (artifact {})",
        if artifact.passed { "PASS" } else { "FAIL" },
        args.output.display(),
        artifact.artifact_hash,
    );
    if args.require_pass && !artifact.passed {
        std::process::exit(2);
    }
}
