//! Publish an auditable policy decision for advancing a checkpoint in scale.

use std::path::PathBuf;

use blob_rl::checkpoint_promotion::{
    evaluate_checkpoint_promotion, publish_checkpoint_promotion, CheckpointPromotionVerdict,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_checkpoint_promotion",
    about = "Attach scale-qualification evidence to an overrideable checkpoint promotion decision"
)]
struct Args {
    /// Verified immutable checkpoint directory.
    #[arg(long)]
    checkpoint: PathBuf,

    /// Immutable large-world scale-qualification report.
    #[arg(long)]
    qualification: PathBuf,

    /// TOML promotion policy choosing required versus advisory evidence.
    #[arg(long)]
    policy: PathBuf,

    /// Named target profile from the qualification report.
    #[arg(long)]
    target_profile: String,

    /// Explicit rationale accepting failed required behavioral evidence.
    #[arg(long)]
    override_reason: Option<String>,

    /// New immutable JSON promotion decision.
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let args = Args::parse();
    let report = evaluate_checkpoint_promotion(
        &args.checkpoint,
        &args.qualification,
        &args.policy,
        &args.target_profile,
        args.override_reason.as_deref(),
    )
    .unwrap_or_else(|error| panic!("checkpoint promotion evaluation failed: {error}"));
    let output = publish_checkpoint_promotion(&args.output, &report)
        .unwrap_or_else(|error| panic!("failed to publish checkpoint promotion: {error}"));
    println!("Checkpoint promotion: {}", output.display());
    println!(
        "  {:?}: {} -> {} ({} blocking failures, {} warnings)",
        report.verdict,
        report.checkpoint_world_size,
        report.target_world_size,
        report.blocking_failures,
        report.warnings,
    );
    for check in &report.checks {
        println!(
            "  {} {:>9} {} — {}",
            if check.passed { "PASS" } else { "FAIL" },
            if check.blocking {
                "required"
            } else {
                "advisory"
            },
            check.name,
            check.detail,
        );
    }
    if report.verdict == CheckpointPromotionVerdict::Rejected {
        std::process::exit(2);
    }
}
