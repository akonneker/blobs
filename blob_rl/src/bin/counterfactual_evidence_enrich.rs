#![recursion_limit = "512"]
//! Replay historical branch source states and attach exact Mind evidence.

use std::path::PathBuf;

use blob_rl::behavior_cloning::{behavior_clone_artifact_sha256, verify_behavior_clone_artifact};
use blob_rl::counterfactual_branch::{
    enrich_counterfactual_mind_evidence, load_counterfactual_branch_artifact,
    publish_counterfactual_branch_artifact,
};
use blob_rl::model::PolicyValueNetConfig;
use burn::prelude::*;
use burn::record::CompactRecorder;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "counterfactual-evidence-enrich",
    about = "Attach replay-verified anonymous Mind evidence to a branch artifact"
)]
struct Args {
    #[arg(long)]
    counterfactual: PathBuf,

    #[arg(long)]
    behavior_clone: PathBuf,

    #[arg(long)]
    output: PathBuf,
}

fn run<B: Backend>(args: &Args, device: B::Device)
where
    B::FloatElem: From<f32>,
    f32: From<B::FloatElem>,
{
    let source = load_counterfactual_branch_artifact(&args.counterfactual)
        .unwrap_or_else(|error| panic!("invalid counterfactual source: {error}"));
    let metadata_sha256 = behavior_clone_artifact_sha256(&args.behavior_clone)
        .unwrap_or_else(|error| panic!("failed to hash behavior clone: {error}"));
    if metadata_sha256 != source.behavior_clone_metadata_sha256 {
        panic!("behavior clone metadata does not match the counterfactual source");
    }
    let model_path = verify_behavior_clone_artifact(
        &args.behavior_clone,
        &metadata_sha256,
        &source.config.model,
    )
    .unwrap_or_else(|error| panic!("invalid behavior clone: {error}"));
    let model = PolicyValueNetConfig {
        hidden1: source.config.model.hidden1,
        hidden2: source.config.model.hidden2,
        recurrent_size: source.config.model.recurrent_size,
    }
    .init::<B>(&device)
    .load_file(model_path, &CompactRecorder::new(), &device)
    .unwrap_or_else(|error| panic!("failed to load behavior clone: {error}"));
    let enriched = enrich_counterfactual_mind_evidence(&model, &source, &device)
        .unwrap_or_else(|error| panic!("failed to enrich counterfactual evidence: {error}"));
    publish_counterfactual_branch_artifact(&args.output, &enriched)
        .unwrap_or_else(|error| panic!("failed to publish enriched counterfactual: {error}"));
    println!(
        "Published {} verified Mind-evidence states to {}\nArtifact hash: {}",
        enriched.report.selected_states,
        args.output.display(),
        enriched.artifact_hash,
    );
}

fn main() {
    let args = Args::parse();
    #[cfg(feature = "wgpu")]
    run::<burn::backend::Autodiff<burn::backend::Wgpu>>(
        &args,
        burn::backend::wgpu::WgpuDevice::default(),
    );

    #[cfg(all(not(feature = "wgpu"), feature = "ndarray"))]
    run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(&args, Default::default());

    #[cfg(not(any(feature = "wgpu", feature = "ndarray")))]
    compile_error!("counterfactual-evidence-enrich requires the wgpu or ndarray feature");
}
