//! Resume-safe viability planning, evaluation, gating, and optional training.

use std::path::PathBuf;

use blob_rl::config::OpponentProfile;
use blob_rl::sweep_execution::SweepExecutorOptions;
use blob_rl::viability_gate::ViabilityGateDecision;
use blob_rl::viability_preflight::{run_viability_preflight, ViabilityPreflightOptions};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_viability_preflight",
    about = "Plan, resume, gate, and optionally launch a rules sweep"
)]
struct Args {
    /// Rules-sweep TOML specification.
    spec: PathBuf,

    /// Team-zero profiles, accepted as repeated or comma-delimited values.
    #[arg(long, required = true, value_delimiter = ',')]
    candidates: Vec<OpponentProfile>,

    /// Opponent profiles, accepted as repeated or comma-delimited values.
    #[arg(long, required = true, value_delimiter = ',')]
    opponents: Vec<OpponentProfile>,

    /// Strict viability gate TOML policy.
    #[arg(long)]
    gates: PathBuf,

    /// Variant used for paired differences. Defaults to the first variant.
    #[arg(long)]
    baseline_variant: Option<String>,

    /// Maximum viability jobs sharing the matrix worker pool.
    #[arg(long, default_value_t = 1)]
    matrix_max_parallel: usize,

    /// Continue into bounded training only when the gate passes.
    #[arg(long)]
    execute_training: bool,

    /// Compiled trainer. Defaults to the sibling `train` binary.
    #[arg(long, requires = "execute_training")]
    train_bin: Option<PathBuf>,

    /// Immutable OCI image digest (sha256:...) to bind into training provenance.
    #[arg(long, requires = "execute_training")]
    trainer_container_digest: Option<String>,

    /// Maximum concurrently resident trainer processes.
    #[arg(long, default_value_t = 1, requires = "execute_training")]
    training_max_parallel: usize,

    /// Set RAYON_NUM_THREADS independently inside every trainer.
    #[arg(long, requires = "execute_training")]
    threads_per_run: Option<usize>,

    /// Retry durable failed training attempts after the gate is reverified.
    #[arg(long, requires = "execute_training")]
    retry_failed: bool,
}

fn sibling_train_binary() -> Result<PathBuf, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("failed to locate preflight executable: {error}"))?;
    let directory = executable
        .parent()
        .ok_or_else(|| "preflight executable has no parent directory".to_string())?;
    Ok(directory.join(if cfg!(windows) { "train.exe" } else { "train" }))
}

fn main() {
    let args = Args::parse();
    let execute_training = if args.execute_training {
        let train_program = args
            .train_bin
            .map(Ok)
            .unwrap_or_else(sibling_train_binary)
            .unwrap_or_else(|error| panic!("failed to locate trainer: {error}"));
        if !train_program.is_file() {
            panic!("trainer {} does not exist", train_program.display());
        }
        Some(SweepExecutorOptions {
            train_program,
            train_arguments: Vec::new(),
            trainer_container_digest: args.trainer_container_digest,
            max_parallel: args.training_max_parallel,
            threads_per_run: args.threads_per_run,
            retry_failed: args.retry_failed,
            required_viability_gate: None,
        })
    } else {
        None
    };
    let report = run_viability_preflight(
        &args.spec,
        ViabilityPreflightOptions {
            candidates: args.candidates,
            opponents: args.opponents,
            baseline_variant: args.baseline_variant,
            matrix_max_parallel: args.matrix_max_parallel,
            gate_spec_file: args.gates,
            execute_training,
        },
    )
    .unwrap_or_else(|error| panic!("viability preflight failed: {error}"));
    println!(
        "Preflight: {:?}; plan {}, matrix {}, decision {}",
        report.gate_decision,
        if report.plan_reused {
            "reused"
        } else {
            "created"
        },
        if report.matrix_reused {
            "reused"
        } else {
            "created"
        },
        if report.decision_reused {
            "reused"
        } else {
            "created"
        },
    );
    if let Some(training) = report.training {
        println!(
            "Training sweep: {}/{} succeeded, {} failed, {} running",
            training.succeeded, training.total_runs, training.failed, training.still_running
        );
    }
    if report.gate_decision == ViabilityGateDecision::Failed {
        eprintln!("{} viability checks failed", report.failed_checks);
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_options_require_explicit_execution() {
        Args::try_parse_from([
            "viability-preflight",
            "sweep.toml",
            "--candidates",
            "random",
            "--opponents",
            "wait",
            "--gates",
            "gates.toml",
        ])
        .unwrap();
        assert!(Args::try_parse_from([
            "viability-preflight",
            "sweep.toml",
            "--candidates",
            "random",
            "--opponents",
            "wait",
            "--gates",
            "gates.toml",
            "--threads-per-run",
            "4",
        ])
        .is_err());
        Args::try_parse_from([
            "viability-preflight",
            "sweep.toml",
            "--candidates",
            "random",
            "--opponents",
            "wait",
            "--gates",
            "gates.toml",
            "--execute-training",
            "--training-max-parallel",
            "2",
            "--threads-per-run",
            "4",
        ])
        .unwrap();
    }
}
