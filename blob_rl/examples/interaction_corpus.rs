//! Paired kind labels on one frozen deployed Mind's live recurrent prefixes.
#[path = "action_trace/probe.rs"]
mod probe;
use blob_rl::{
    config::TrainingConfig,
    contact_deployed::evaluate_contact_mind,
    deployed_artifact::{load_deployed_policy, read_bounded, sha},
    feeding_deployed::{evaluate_feeding_mind, FeedingAssessmentMode},
    feeding_layout_evaluation::FeedingQualificationLayout,
};
use clap::Parser;
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
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Args::parse();
    if a.output.exists() {
        return Err("immutable corpus already exists".into());
    }
    let config_bytes = read_bounded(&a.config, 1024 * 1024)?;
    let config = TrainingConfig::from_toml_str(std::str::from_utf8(&config_bytes)?)?;
    config.validate()?;
    let (policy, manifest) = load_deployed_policy(&a.export, &a.export_manifest_sha256, &[a.seed])?;
    fs::create_dir(&a.output)?;
    let plan = json!({"seed":a.seed,"manifest_sha256":a.export_manifest_sha256,"weights_sha256":manifest["weights_sha256"],"config_sha256":sha(&config_bytes),"interaction_rows_per_episode_cap":2048,"scope":"First 2048 Interaction observations per episode on unmodified frozen Mind prefixes. Combat paired teacher kinds; feeding retains actual frozen kind. Labels never executed. Other contexts unchanged by planned head-only update."});
    fs::write(
        a.output.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    let mut probe = probe::Probe {
        policy: policy.clone(),
        rows: vec![],
        episode: 0,
        interaction_only: true,
        per_episode_limit: 2048,
        recorded: 0,
    };
    let combat = evaluate_contact_mind(&mut probe, &config, &[a.seed])?;
    fs::write(
        a.output.join("combat.json"),
        serde_json::to_vec(&json!({"evaluation":combat,"rows":probe.rows}))?,
    )?;
    probe.rows.clear();
    for layout in FeedingQualificationLayout::REQUIRED {
        let trial = evaluate_feeding_mind(
            &mut probe,
            policy.execution_contract(),
            &config,
            layout,
            a.seed,
            FeedingAssessmentMode::SustainedFeeding,
            |_, _, _, _| {},
        )?;
        fs::write(
            a.output.join(format!("feeding-{layout:?}.json")),
            serde_json::to_vec(&json!({"trial":trial,"rows":probe.rows}))?,
        )?;
        println!("{layout:?}: {} interaction rows", probe.rows.len());
        probe.rows.clear();
    }
    fs::write(a.output.join("status.json"), b"{\"complete\":true}")?;
    Ok(())
}
