//! Execute or aggregate a published canonical-rules sweep.

use std::path::PathBuf;

use blob_rl::sweep_execution::{aggregate_rules_sweep, execute_rules_sweep, SweepExecutorOptions};
use blob_rl::viability_gate::ViabilityGateRequirement;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_rules_sweep_run",
    about = "Run a hash-bound rules sweep with bounded trainer concurrency"
)]
struct Args {
    /// Published sweep manifest.json.
    manifest: PathBuf,

    /// Compiled blob_rl `train` executable. Defaults to the sibling binary.
    #[arg(long)]
    train_bin: Option<PathBuf>,

    /// Immutable OCI image digest (sha256:...) to bind into remote sweep provenance.
    #[arg(long)]
    trainer_container_digest: Option<String>,

    /// Maximum concurrently resident trainer processes. Keep this at one for
    /// a single GPU unless the backend/device setup is explicitly partitioned.
    #[arg(long, default_value_t = 1)]
    max_parallel: usize,

    /// Set RAYON_NUM_THREADS independently inside each trainer process.
    #[arg(long)]
    threads_per_run: Option<usize>,

    /// Immutable behavior-cloning artifact used to initialize every run.
    #[arg(
        long,
        requires_all = [
            "initial_policy_artifact_sha256",
            "initial_policy_qualification",
            "initial_policy_qualification_hash"
        ],
        conflicts_with = "aggregate_only"
    )]
    initial_policy: Option<PathBuf>,

    /// Expected behavior-clone metadata SHA-256.
    #[arg(long, requires = "initial_policy", conflicts_with = "aggregate_only")]
    initial_policy_artifact_sha256: Option<String>,

    /// Passing feeding-evaluation JSON shared by every paired run.
    #[arg(long, requires = "initial_policy", conflicts_with = "aggregate_only")]
    initial_policy_qualification: Option<PathBuf>,

    /// Expected semantic hash embedded in the feeding qualification.
    #[arg(long, requires = "initial_policy", conflicts_with = "aggregate_only")]
    initial_policy_qualification_hash: Option<String>,

    /// Retry runs whose last recorded attempt failed. Interrupted running runs
    /// are always recovered from their newest verified checkpoint.
    #[arg(long)]
    retry_failed: bool,

    /// Published passing decision required before any trainer is launched.
    #[arg(
        long,
        requires_all = ["viability_matrix", "viability_gates"],
        conflicts_with = "aggregate_only"
    )]
    require_viability_gate: Option<PathBuf>,

    /// Exact matrix bound by --require-viability-gate.
    #[arg(
        long,
        requires = "require_viability_gate",
        conflicts_with = "aggregate_only"
    )]
    viability_matrix: Option<PathBuf>,

    /// Exact TOML policy bound by --require-viability-gate.
    #[arg(
        long,
        requires = "require_viability_gate",
        conflicts_with = "aggregate_only"
    )]
    viability_gates: Option<PathBuf>,

    /// Verify completed records and publish aggregate.json without launching
    /// any trainers.
    #[arg(long)]
    aggregate_only: bool,
}

fn sibling_train_binary() -> Result<PathBuf, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("failed to locate sweep executor: {error}"))?;
    let directory = executable
        .parent()
        .ok_or_else(|| "sweep executor has no parent directory".to_string())?;
    let name = if cfg!(windows) { "train.exe" } else { "train" };
    Ok(directory.join(name))
}

fn main() {
    let args = Args::parse();
    if args.aggregate_only {
        let aggregate = aggregate_rules_sweep(&args.manifest)
            .unwrap_or_else(|error| panic!("failed to aggregate rules sweep: {error}"));
        println!(
            "Published paired aggregate for {} variants against baseline {}",
            aggregate.variants.len(),
            aggregate.baseline_variant
        );
        return;
    }

    let train_program = args
        .train_bin
        .map(Ok)
        .unwrap_or_else(sibling_train_binary)
        .unwrap_or_else(|error| panic!("failed to locate trainer: {error}"));
    if !train_program.is_file() {
        panic!(
            "trainer {} does not exist; build both binaries or pass --train-bin",
            train_program.display()
        );
    }
    let train_arguments = match (
        args.initial_policy,
        args.initial_policy_artifact_sha256,
        args.initial_policy_qualification,
        args.initial_policy_qualification_hash,
    ) {
        (Some(policy), Some(policy_hash), Some(qualification), Some(qualification_hash)) => vec![
            "--initial-policy".into(),
            policy.to_string_lossy().into_owned(),
            "--initial-policy-artifact-sha256".into(),
            policy_hash,
            "--initial-policy-qualification".into(),
            qualification.to_string_lossy().into_owned(),
            "--initial-policy-qualification-hash".into(),
            qualification_hash,
        ],
        (None, None, None, None) => Vec::new(),
        _ => unreachable!("clap requires complete initial-policy provenance"),
    };
    let summary = execute_rules_sweep(
        &args.manifest,
        &SweepExecutorOptions {
            train_program,
            train_arguments,
            trainer_container_digest: args.trainer_container_digest,
            max_parallel: args.max_parallel,
            threads_per_run: args.threads_per_run,
            retry_failed: args.retry_failed,
            required_viability_gate: args.require_viability_gate.map(|decision_file| {
                ViabilityGateRequirement {
                    decision_file,
                    matrix_file: args
                        .viability_matrix
                        .expect("clap requires a matrix with a gate decision"),
                    gate_spec_file: args
                        .viability_gates
                        .expect("clap requires a policy with a gate decision"),
                }
            }),
        },
    )
    .unwrap_or_else(|error| panic!("rules sweep execution failed: {error}"));
    println!(
        "Sweep: {}/{} succeeded, {} failed, {} running",
        summary.succeeded, summary.total_runs, summary.failed, summary.still_running
    );
    if summary.failed > 0 {
        panic!(
            "{} run(s) remain failed; inspect execution-summary.json and use --retry-failed when ready",
            summary.failed
        );
    }
    if let Some(aggregate) = summary.aggregate_file {
        println!("Paired aggregate: {aggregate}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viability_gate_cli_requires_the_complete_artifact_trio() {
        assert!(Args::try_parse_from([
            "rules-sweep-run",
            "manifest.json",
            "--require-viability-gate",
            "decision.json",
        ])
        .is_err());
        let parsed = Args::try_parse_from([
            "rules-sweep-run",
            "manifest.json",
            "--require-viability-gate",
            "decision.json",
            "--viability-matrix",
            "matrix.json",
            "--viability-gates",
            "gates.toml",
        ])
        .unwrap();
        assert_eq!(
            parsed.require_viability_gate,
            Some(PathBuf::from("decision.json"))
        );
        assert!(Args::try_parse_from([
            "rules-sweep-run",
            "manifest.json",
            "--aggregate-only",
            "--require-viability-gate",
            "decision.json",
            "--viability-matrix",
            "matrix.json",
            "--viability-gates",
            "gates.toml",
        ])
        .is_err());
    }

    #[test]
    fn initial_policy_cli_requires_complete_bound_provenance() {
        assert!(Args::try_parse_from([
            "rules-sweep-run",
            "manifest.json",
            "--initial-policy",
            "clone",
        ])
        .is_err());
        let parsed = Args::try_parse_from([
            "rules-sweep-run",
            "manifest.json",
            "--initial-policy",
            "clone",
            "--initial-policy-artifact-sha256",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--initial-policy-qualification",
            "feeding.json",
            "--initial-policy-qualification-hash",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ])
        .unwrap();
        assert_eq!(parsed.initial_policy, Some(PathBuf::from("clone")));
        assert_eq!(
            parsed.initial_policy_qualification,
            Some(PathBuf::from("feeding.json"))
        );
        assert!(Args::try_parse_from([
            "rules-sweep-run",
            "manifest.json",
            "--aggregate-only",
            "--initial-policy",
            "clone",
            "--initial-policy-artifact-sha256",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--initial-policy-qualification",
            "feeding.json",
            "--initial-policy-qualification-hash",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ])
        .is_err());
    }
}
