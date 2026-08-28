//! Plan or execute a provenance-bound large-world training transition.

use std::path::PathBuf;

use blob_rl::scale_transition::{
    execute_scale_transition_job, load_scale_transition_contract, publish_scale_transition_job,
    ScaleTransitionPlanOptions, ScaleTransitionRunOptions,
};
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "blob_scale_transition",
    about = "Create or execute a promoted or explicitly experimental scale-transition job"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Plan {
        #[arg(long)]
        source_checkpoint: PathBuf,
        #[arg(long)]
        target_config: PathBuf,
        #[arg(long)]
        promotion_decision: Option<PathBuf>,
        /// Makes the job experimental; required when no approved decision is supplied.
        #[arg(long)]
        experimental_reason: Option<String>,
        #[arg(long)]
        output: PathBuf,
    },
    Run {
        /// Published transition job directory.
        job: PathBuf,
        /// Compiled `train` executable. Defaults to the sibling binary.
        #[arg(long)]
        train_bin: Option<PathBuf>,
        /// Relocated source checkpoint; defaults to the contract location.
        #[arg(long)]
        source_checkpoint: Option<PathBuf>,
        /// Immutable OCI image digest bound into execution provenance.
        #[arg(long)]
        trainer_container_digest: Option<String>,
        /// Set RAYON_NUM_THREADS in the trainer process.
        #[arg(long)]
        threads: Option<usize>,
    },
    Verify {
        /// Published transition job directory.
        job: PathBuf,
        /// Relocated source checkpoint; defaults to the contract location.
        #[arg(long)]
        source_checkpoint: Option<PathBuf>,
    },
}

fn sibling_train_binary() -> Result<PathBuf, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("failed to locate transition launcher: {error}"))?;
    let directory = executable
        .parent()
        .ok_or_else(|| "transition launcher has no parent directory".to_string())?;
    Ok(directory.join(if cfg!(windows) { "train.exe" } else { "train" }))
}

fn main() {
    match Args::parse().command {
        Command::Plan {
            source_checkpoint,
            target_config,
            promotion_decision,
            experimental_reason,
            output,
        } => {
            let output = publish_scale_transition_job(&ScaleTransitionPlanOptions {
                source_checkpoint,
                target_config,
                promotion_decision,
                experimental_reason,
                output_directory: output,
            })
            .unwrap_or_else(|error| panic!("failed to publish scale-transition job: {error}"));
            let contract = load_scale_transition_contract(&output)
                .unwrap_or_else(|error| panic!("published transition is invalid: {error}"));
            println!("Scale transition: {}", output.display());
            println!(
                "  {:?}: {} -> {} ({})",
                contract.mode,
                contract.source_world_size,
                contract.target_world_size,
                contract.contract_hash,
            );
        }
        Command::Run {
            job,
            train_bin,
            source_checkpoint,
            trainer_container_digest,
            threads,
        } => {
            let train_program = train_bin
                .map(Ok)
                .unwrap_or_else(sibling_train_binary)
                .unwrap_or_else(|error| panic!("failed to locate trainer: {error}"));
            let result = execute_scale_transition_job(
                &job,
                &ScaleTransitionRunOptions {
                    train_program,
                    source_checkpoint,
                    trainer_container_digest,
                    threads,
                },
            )
            .unwrap_or_else(|error| panic!("scale-transition execution failed: {error}"));
            println!(
                "Scale-transition trainer exited {} ({})",
                result.exit_code,
                if result.succeeded {
                    "succeeded"
                } else {
                    "failed"
                },
            );
            if !result.succeeded {
                std::process::exit(2);
            }
        }
        Command::Verify {
            job,
            source_checkpoint,
        } => {
            let contract = blob_rl::scale_transition::verify_scale_transition_job(
                &job,
                source_checkpoint.as_deref(),
            )
            .unwrap_or_else(|error| panic!("scale-transition verification failed: {error}"));
            println!("Verified scale transition {}", contract.contract_hash);
        }
    }
}
