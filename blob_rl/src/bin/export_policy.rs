//! Freeze a verified cloning checkpoint for the portable learned Mind.
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    artifact_sha256: String,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let report =
        blob_rl::deployment::export_policy(&args.artifact, &args.artifact_sha256, &args.output)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
