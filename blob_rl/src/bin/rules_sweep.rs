//! Expand a paired canonical-rules sweep into immutable run configurations.

use std::path::PathBuf;

use blob_rl::sweep::publish_rules_sweep;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "blob_rules_sweep",
    about = "Publish an immutable, hash-bound plan for replicated gameplay training runs"
)]
struct Args {
    /// Sweep TOML containing a base config, paired seeds, and named
    /// rules/scenario variants.
    spec: PathBuf,
}

fn main() {
    let args = Args::parse();
    let manifest = publish_rules_sweep(&args.spec)
        .unwrap_or_else(|error| panic!("failed to publish rules sweep: {error}"));
    println!(
        "Published {} runs across {} variants to {}",
        manifest.runs.len(),
        manifest.variants.len(),
        manifest.output_directory
    );
    println!("Build the trainer and bounded executor with:");
    println!("  cargo build --release -p blob_rl --bin train --bin rules-sweep-run");
    println!("Then execute the verified plan with:");
    println!(
        "  target/release/rules-sweep-run {}/manifest.json --max-parallel 1",
        manifest.output_directory
    );
}
