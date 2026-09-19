//! Immutable canonical combat evidence for a portable deployed Mind.
use blob_rl::{
    config::TrainingConfig,
    contact_deployed::evaluate_contact_mind,
    deployed_artifact::{load_deployed_policy, read_bounded, sha},
};
use clap::Parser;
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};

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
        return Err("output already exists; combat evidence is immutable".into());
    }
    let bytes = read_bounded(&args.config, 1024 * 1024)?;
    let config = TrainingConfig::from_toml_str(std::str::from_utf8(&bytes)?)?;
    config.validate()?;
    if !config.combat_curriculum.enabled {
        return Err("combat curriculum must be enabled".into());
    }
    let (mut policy, manifest) =
        load_deployed_policy(&args.export, &args.export_manifest_sha256, &args.seeds)?;
    let plan = json!({
        "assessment_mode":"canonical_match", "source_config_sha256":sha(&bytes),
        "config":config, "export_manifest_sha256":args.export_manifest_sha256,
        "weights_sha256":manifest["weights_sha256"], "execution_contract":policy.execution_contract(),
        "seeds":args.seeds, "code_revision":env!("BLOB_CODE_REVISION"),
        "scope":"Native portable deployed Mind, canonical contact/skirmish development evidence. Not a new WASM qualification or final confirmation."
    });
    fs::create_dir(&args.output)?;
    fs::write(
        args.output.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    let start = Instant::now();
    let evaluation = evaluate_contact_mind(&mut policy, &config, &args.seeds)?;
    let result = json!({"complete":true,"plan":plan,"evaluation":evaluation,"elapsed_seconds":start.elapsed().as_secs_f64()});
    fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!(
        "{}",
        json!({"combat_thresholds_passed":evaluation.combat_thresholds_passed,
                          "episodes":evaluation.report.episodes,"damage":evaluation.report.damage_dealt,"kills":evaluation.report.kills})
    );
    Ok(())
}
