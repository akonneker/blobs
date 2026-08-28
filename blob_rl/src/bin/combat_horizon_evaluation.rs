#![recursion_limit = "512"]
//! Evaluate several held-out combat horizons against one frozen checkpoint.

use std::path::PathBuf;

use blob_rl::artifact::{load_policy_snapshot, verify_checkpoint_metadata};
use blob_rl::checkpoint_evaluation::CombatEvaluationHorizons;
use blob_rl::combat_horizon_evaluation::{
    publish_combat_horizon_evaluation, CombatHorizonEvaluationArtifact, CombatHorizonResult,
};
use blob_rl::contact_evaluation::evaluate_contact;
use burn::prelude::*;
use clap::Parser;
use sha2::{Digest, Sha256};

fn parse_horizon_pair(value: &str) -> Result<CombatEvaluationHorizons, String> {
    let (contact, skirmish) = value
        .split_once(':')
        .ok_or_else(|| "expected CONTACT:SKIRMISH".to_string())?;
    let horizons = CombatEvaluationHorizons {
        contact_sim_time_limit_quanta: contact
            .parse()
            .map_err(|_| "contact horizon must be a positive integer".to_string())?,
        skirmish_sim_time_limit_quanta: skirmish
            .parse()
            .map_err(|_| "skirmish horizon must be a positive integer".to_string())?,
    };
    if horizons.contact_sim_time_limit_quanta == 0 || horizons.skirmish_sim_time_limit_quanta == 0 {
        return Err("horizons must be positive".into());
    }
    Ok(horizons)
}

#[derive(Debug, Parser)]
#[command(
    name = "combat-horizon-evaluation",
    about = "Publish a held-out combat-horizon matrix for one frozen PPO checkpoint"
)]
struct Args {
    /// Immutable schema-current training checkpoint directory.
    checkpoint: PathBuf,

    /// Held-out episode seeds, accepted as a comma-delimited list.
    #[arg(long, value_delimiter = ',', required = true)]
    seeds: Vec<u64>,

    /// Repeatable CONTACT:SKIRMISH horizon pair in canonical quanta.
    #[arg(long = "horizon", value_parser = parse_horizon_pair, required = true)]
    horizons: Vec<CombatEvaluationHorizons>,

    /// New immutable matrix artifact.
    #[arg(long)]
    output: PathBuf,
}

fn run<B: Backend>(args: &Args, device: B::Device)
where
    B::FloatElem: From<f32>,
    f32: From<B::FloatElem>,
{
    let metadata = verify_checkpoint_metadata(&args.checkpoint)
        .unwrap_or_else(|error| panic!("invalid checkpoint: {error}"));
    let snapshot = load_policy_snapshot::<B>(&args.checkpoint, &device)
        .unwrap_or_else(|error| panic!("failed to load checkpoint policy: {error}"));
    let metadata_bytes = std::fs::read(args.checkpoint.join("metadata.json"))
        .unwrap_or_else(|error| panic!("failed to read checkpoint metadata: {error}"));
    let metadata_sha256 = format!("{:x}", Sha256::digest(&metadata_bytes));

    let mut results = Vec::with_capacity(args.horizons.len());
    for &horizons in &args.horizons {
        let mut evaluation_config = metadata.config.clone();
        horizons
            .apply_to(&mut evaluation_config)
            .unwrap_or_else(|error| panic!("invalid combat horizon pair: {error}"));
        let report = evaluate_contact(
            &snapshot.model,
            &evaluation_config.env,
            &evaluation_config.reward,
            &evaluation_config.feeding_curriculum,
            &evaluation_config.combat_curriculum,
            &args.seeds,
            &device,
        );
        let passed = report.meets_promotion_thresholds(&evaluation_config.combat_curriculum);
        println!(
            "  horizons {}/{}: {} attacks, {} damage, {} kills, {}",
            horizons.contact_sim_time_limit_quanta,
            horizons.skirmish_sim_time_limit_quanta,
            report.attacks_succeeded,
            report.damage_dealt,
            report.kills,
            if passed { "PASS" } else { "FAIL" },
        );
        results.push(CombatHorizonResult {
            horizons,
            report,
            passed,
        });
    }

    let artifact = CombatHorizonEvaluationArtifact::new(
        metadata_sha256,
        snapshot.model_sha256,
        snapshot.update,
        snapshot.actions,
        metadata.config,
        args.seeds.clone(),
        results,
    )
    .unwrap_or_else(|error| panic!("invalid combat-horizon evaluation: {error}"));
    publish_combat_horizon_evaluation(&args.output, &artifact)
        .unwrap_or_else(|error| panic!("failed to publish combat-horizon evaluation: {error}"));
    println!(
        "Published checkpoint u{} combat-horizon matrix to {} (artifact {})",
        artifact.checkpoint_update,
        args.output.display(),
        artifact.artifact_hash,
    );
}

fn main() {
    let args = Args::parse();

    #[cfg(feature = "wgpu")]
    run::<burn::backend::Wgpu>(&args, burn::backend::wgpu::WgpuDevice::default());

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    run::<burn::backend::NdArray<f32>>(&args, Default::default());

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("combat-horizon-evaluation requires the wgpu or ndarray feature");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_horizon_pair() {
        assert_eq!(
            parse_horizon_pair("16384:32768").unwrap(),
            CombatEvaluationHorizons {
                contact_sim_time_limit_quanta: 16_384,
                skirmish_sim_time_limit_quanta: 32_768,
            }
        );
        assert!(parse_horizon_pair("0:1").is_err());
        assert!(parse_horizon_pair("1").is_err());
    }
}
