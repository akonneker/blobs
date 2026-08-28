//! Re-verify a promotion decision against its checkpoint and evidence files.

use std::path::PathBuf;

use blob_rl::checkpoint_promotion::{
    load_checkpoint_promotion, verify_checkpoint_promotion_evidence,
    verify_checkpoint_promotion_evidence_at,
};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_checkpoint_promotion_verify",
    about = "Re-verify a checkpoint scale-promotion decision and all bound source evidence"
)]
struct Args {
    /// Immutable checkpoint-promotion decision JSON.
    #[arg(long)]
    decision: PathBuf,

    /// Relocated checkpoint directory; defaults to the recorded location.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Relocated qualification report; defaults to the recorded location.
    #[arg(long)]
    qualification: Option<PathBuf>,

    /// Relocated policy file; defaults to the recorded location.
    #[arg(long)]
    policy: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    let report = load_checkpoint_promotion(&args.decision)
        .unwrap_or_else(|error| panic!("failed to load promotion decision: {error}"));
    if args.checkpoint.is_some() || args.qualification.is_some() || args.policy.is_some() {
        verify_checkpoint_promotion_evidence_at(
            &report,
            args.checkpoint
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new(&report.checkpoint_directory)),
            args.qualification
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new(&report.qualification_file)),
            args.policy
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new(&report.policy_file)),
        )
    } else {
        verify_checkpoint_promotion_evidence(&report)
    }
    .unwrap_or_else(|error| panic!("promotion evidence verification failed: {error}"));
    println!(
        "Verified {:?} promotion {} -> {} ({})",
        report.verdict,
        report.checkpoint_world_size,
        report.target_world_size,
        report.decision_hash,
    );
}
