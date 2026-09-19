//! Immutable single-trial feeding evaluation for portable deployed Minds.
use blob_rl::{
    config::TrainingConfig,
    deployed_artifact::{load_deployed_policy, read_bounded, sha},
    feeding_deployed::{evaluate_deployed_feeding_with_mode, FeedingAssessmentMode},
    feeding_layout_evaluation::FeedingQualificationLayout,
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
    #[arg(long, value_enum)]
    layout: FeedingQualificationLayout,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, value_enum, default_value = "canonical-match")]
    assessment_mode: FeedingAssessmentMode,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.output.exists() {
        return Err("output already exists; trial artifacts are immutable".into());
    }
    let config_bytes = read_bounded(&args.config, 1024 * 1024)?;
    let config = TrainingConfig::from_toml_str(std::str::from_utf8(&config_bytes)?)?;
    config.validate()?;
    let (policy, manifest) =
        load_deployed_policy(&args.export, &args.export_manifest_sha256, &[args.seed])?;
    fs::create_dir(&args.output)?;
    let plan = json!({"assessment_mode":args.assessment_mode,"source_config_sha256":sha(&config_bytes),"config":config,"export_manifest_sha256":args.export_manifest_sha256,"weights_sha256":manifest["weights_sha256"],"execution_contract":policy.execution_contract(),"layout":args.layout,"seed":args.seed,"code_revision":env!("BLOB_CODE_REVISION"),"scope":"Native portable deployment runtime on ordinary canonical Mind inputs; matched WASM fidelity previously qualified. Feeding development evidence, not final confirmation.","confirmation_seeds_reserved":[1434999901_u64,1434999902_u64]});
    fs::write(
        args.output.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    let start = Instant::now();
    let trial = evaluate_deployed_feeding_with_mode(
        &policy,
        &config,
        args.layout,
        args.seed,
        args.assessment_mode,
        |stage, steps, quanta, alive| {
            println!(
                "{stage:?}: {steps} steps, {quanta} quanta, {alive} alive, {:.1}s",
                start.elapsed().as_secs_f64()
            )
        },
    )?;
    let result = json!({"complete":true,"plan":plan,"trial":trial,"elapsed_seconds":start.elapsed().as_secs_f64()});
    fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string(&result["trial"]["promotion"])?);
    Ok(())
}
