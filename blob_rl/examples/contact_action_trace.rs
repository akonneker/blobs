//! Read-only routed-score trace and paired teacher labels on frozen combat prefixes.
use blob_rl::{
    config::TrainingConfig,
    contact_deployed::evaluate_contact_mind,
    deployed_artifact::{load_deployed_policy, read_bounded, sha},
};
use clap::Parser;
#[path = "action_trace/probe.rs"]
mod probe;
use probe::Probe;
use serde_json::json;
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    export: PathBuf,
    #[arg(long)]
    export_manifest_sha256: String,
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.output.exists() {
        return Err("trace output already exists".into());
    }
    let bytes = read_bounded(&args.config, 1024 * 1024)?;
    let config = TrainingConfig::from_toml_str(std::str::from_utf8(&bytes)?)?;
    config.validate()?;
    let (policy, manifest) =
        load_deployed_policy(&args.export, &args.export_manifest_sha256, &args.seeds)?;
    fs::create_dir(&args.output)?;
    let plan = json!({"config":config,"source_config_sha256":sha(&bytes),"seeds":args.seeds,"manifest_sha256":args.export_manifest_sha256,"weights_sha256":manifest["weights_sha256"],"scope":"Read-only frozen-prefix diagnostic and paired maintained-teacher action-kind labels. Host episode metadata is not a model feature."});
    fs::write(
        args.output.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    let mut probe = Probe {
        policy,
        rows: Vec::new(),
        episode: 0,
        interaction_only: false,
        per_episode_limit: usize::MAX,
        recorded: 0,
    };
    let evaluation = evaluate_contact_mind(&mut probe, &config, &args.seeds)?;
    fs::write(
        args.output.join("report.json"),
        serde_json::to_vec(
            &json!({"complete":true,"plan":plan,"evaluation":evaluation,"rows":probe.rows}),
        )?,
    )?;
    Ok(())
}
