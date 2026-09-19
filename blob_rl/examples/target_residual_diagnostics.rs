//! Read-only development diagnostic; no training, simulation or qualification.
//! Reconstructs canonical private memory separately for every recorded trajectory.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use blob_rl::action::{
    decompose_policy_action, policy_target_mask, PolicyActionKind, NUM_POLICY_TARGETS,
    NUM_POLICY_TARGET_LOGITS,
};
use blob_rl::behavior_cloning::{behavior_clone_artifact_sha256, BehaviorCloningArtifact};
use blob_rl::demonstration::{load_demonstrations, DemonstrationSample};
use blob_rl::model::{decode_policy_memory, encode_policy_memory, PolicyValueNetConfig};
use blob_rl::observation::{
    HEADER_FEATURES, OBS_DIM, OBS_RANDOMNESS_FEATURE_END, OBS_RANDOMNESS_FEATURE_START,
    SLOT_FEATURES,
};
use blob_rl::policy_artifact::load_behavior_clone;
use burn::backend::NdArray;
use burn::prelude::*;
use clap::Parser;
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, required = true)]
    dataset: Vec<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.output.exists() {
        return Err("diagnostic output already exists".into());
    }
    let artifact: BehaviorCloningArtifact =
        serde_json::from_slice(&fs::read(args.model.join("behavior-cloning.json"))?)?;
    if artifact.schema_version != 37 || !artifact.config.target_residual_only {
        return Err("this diagnostic requires a schema-37 target-residual-only artifact".into());
    }
    let metadata_sha256 = behavior_clone_artifact_sha256(&args.model)?;
    let device = Default::default();
    NdArray::<f32>::seed(&device, 1434000001);
    let trained = load_behavior_clone::<NdArray<f32>>(
        &args.model,
        &metadata_sha256,
        &artifact.model,
        &device,
    )?;
    let zero = PolicyValueNetConfig {
        hidden1: artifact.model.hidden1,
        hidden2: artifact.model.hidden2,
        recurrent_size: artifact.model.recurrent_size,
    }
    .init::<NdArray<f32>>(&device);
    let baseline = trained.clone().with_target_residual_from(zero);
    let recurrent_size = trained.recurrent_size();
    let mut datasets = Vec::new();
    let mut rows = Vec::new();
    let mut reconstructed_samples = 0;
    for directory in &args.dataset {
        let loaded = load_demonstrations(directory)?;
        let random_blocks: BTreeSet<Vec<u32>> = loaded
            .payload
            .samples
            .iter()
            .map(|sample| {
                sample.observation[OBS_RANDOMNESS_FEATURE_START..OBS_RANDOMNESS_FEATURE_END]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            })
            .collect();
        let mut sequences = BTreeMap::<(u64, u64), Vec<(usize, &DemonstrationSample)>>::new();
        for (index, sample) in loaded.payload.samples.iter().enumerate() {
            sequences
                .entry((sample.source_seed, sample.source_cell))
                .or_default()
                .push((index, sample));
        }
        datasets.push(
            json!({"directory": directory, "manifest_sha256": loaded.manifest_sha256,
            "payload_sha256": loaded.manifest.payload_sha256, "seeds": loaded.manifest.seeds,
            "samples": loaded.payload.samples.len(), "trajectories": sequences.len(),
            "distinct_random_blocks": random_blocks.len(),
            "nonzero_random_samples": loaded.payload.samples.iter().filter(|sample| sample.observation[OBS_RANDOMNESS_FEATURE_START..OBS_RANDOMNESS_FEATURE_END].iter().any(|v| *v != 0.0)).count()}),
        );
        let sequences: Vec<_> = sequences.into_values().collect();
        for batch in sequences.chunks(64) {
            let mut memories: Vec<_> = batch
                .iter()
                .map(|sequence| {
                    let first = &sequence[0].1.initial_policy_memory;
                    if first.is_empty() {
                        vec![0.0; recurrent_size]
                    } else {
                        first.clone()
                    }
                })
                .collect();
            if memories.iter().any(|memory| memory.len() != recurrent_size) {
                return Err("recorded memory dimensions do not match the model".into());
            }
            let steps = batch.iter().map(Vec::len).max().unwrap_or(0);
            for step in 0..steps {
                let active: Vec<_> = (0..batch.len())
                    .filter(|index| step < batch[*index].len())
                    .collect();
                let obs: Vec<_> = active
                    .iter()
                    .flat_map(|index| batch[*index][step].1.observation.iter().copied())
                    .collect();
                let memory: Vec<_> = active
                    .iter()
                    .flat_map(|index| memories[*index].iter().copied())
                    .collect();
                // A within-row permutation keeps legal random-feature values and
                // their distribution while changing the cell-private random block.
                // Subtract the baseline under the same intervention to isolate
                // the residual from inherited randomness-sensitive pathways.
                let mut rotated_obs = obs.clone();
                for row in rotated_obs.chunks_exact_mut(OBS_DIM) {
                    row[OBS_RANDOMNESS_FEATURE_START..OBS_RANDOMNESS_FEATURE_END].rotate_left(1);
                }
                let obs = Tensor::from_data(TensorData::new(obs, [active.len(), OBS_DIM]), &device);
                let rotated_obs = Tensor::from_data(
                    TensorData::new(rotated_obs, [active.len(), OBS_DIM]),
                    &device,
                );
                let memory = Tensor::from_data(
                    TensorData::new(memory, [active.len(), recurrent_size]),
                    &device,
                );
                let before = baseline.forward_with_memory(obs.clone(), memory.clone());
                let after = trained.forward_with_memory(obs, memory.clone());
                let rotated_before = baseline
                    .forward_with_memory(rotated_obs.clone(), memory.clone())
                    .target_logits
                    .into_data()
                    .to_vec::<f32>()?;
                let rotated_after = trained
                    .forward_with_memory(rotated_obs, memory)
                    .target_logits
                    .into_data()
                    .to_vec::<f32>()?;
                let before_memory = before.next_memory.into_data().to_vec::<f32>()?;
                let after_memory = after.next_memory.into_data().to_vec::<f32>()?;
                if !before_memory
                    .iter()
                    .zip(&after_memory)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                {
                    return Err("residual replacement altered recurrent output".into());
                }
                let before_logits = before.target_logits.into_data().to_vec::<f32>()?;
                let after_logits = after.target_logits.into_data().to_vec::<f32>()?;
                if !before_logits
                    .iter()
                    .chain(&after_logits)
                    .chain(&rotated_before)
                    .chain(&rotated_after)
                    .chain(&after_memory)
                    .all(|v| v.is_finite())
                {
                    return Err("non-finite diagnostic output".into());
                }
                for (row, index) in active.into_iter().enumerate() {
                    reconstructed_samples += 1;
                    let (sample_index, sample) = batch[index][step];
                    memories[index] = decode_policy_memory(
                        &encode_policy_memory(
                            &after_memory[row * recurrent_size..(row + 1) * recurrent_size],
                        ),
                        recurrent_size,
                    );
                    let teacher = decompose_policy_action(usize::from(sample.action))
                        .ok_or("invalid teacher action")?;
                    if teacher.kind != PolicyActionKind::Move.index() {
                        continue;
                    }
                    let mask = policy_target_mask(&sample.action_mask, teacher.kind);
                    if !mask[teacher.target] {
                        return Err("teacher target is masked".into());
                    }
                    let offset = row * NUM_POLICY_TARGET_LOGITS + teacher.kind * NUM_POLICY_TARGETS;
                    let before = &before_logits[offset..offset + NUM_POLICY_TARGETS];
                    let after = &after_logits[offset..offset + NUM_POLICY_TARGETS];
                    let rotated_before = &rotated_before[offset..offset + NUM_POLICY_TARGETS];
                    let rotated_after = &rotated_after[offset..offset + NUM_POLICY_TARGETS];
                    let legal: Vec<_> = (0..NUM_POLICY_TARGETS)
                        .filter(|target| mask[*target])
                        .collect();
                    let best = |logits: &[f32]| {
                        *legal
                            .iter()
                            .max_by(|a, b| {
                                logits[**a].total_cmp(&logits[**b]).then_with(|| b.cmp(a))
                            })
                            .unwrap()
                    };
                    let baseline_best = best(before);
                    let trained_best = best(after);
                    let rival = legal
                        .iter()
                        .copied()
                        .filter(|target| *target != teacher.target)
                        .max_by(|a, b| before[*a].total_cmp(&before[*b]));
                    let recorded = sample
                        .correction_policy_action
                        .and_then(|action| decompose_policy_action(usize::from(action)))
                        .filter(|action| action.kind == teacher.kind);
                    let policy_target = recorded.map_or(baseline_best, |action| action.target);
                    let teacher_slot = HEADER_FEATURES + teacher.target * SLOT_FEATURES;
                    let policy_slot = HEADER_FEATURES + policy_target * SLOT_FEATURES;
                    let differences: Vec<_> = (0..SLOT_FEATURES)
                        .map(|feature| {
                            sample.observation[teacher_slot + feature]
                                - sample.observation[policy_slot + feature]
                        })
                        .collect();
                    let adapter: Vec<_> = legal
                        .iter()
                        .map(|target| f64::from(after[*target]) - f64::from(before[*target]))
                        .collect();
                    rows.push(json!({
                        "dataset": directory, "sample_index": sample_index, "source_seed": sample.source_seed,
                        "source_cell": sample.source_cell, "trajectory_step": step, "active_correction": sample.active_correction,
                        "teacher_target": teacher.target, "recorded_policy_target": recorded.map(|action| action.target),
                        "baseline_best_target": baseline_best, "trained_best_target": trained_best,
                        "legal_target_logits": legal.iter().map(|target| json!({"target": target, "baseline": before[*target], "trained": after[*target]})).collect::<Vec<_>>(),
                        "baseline_teacher_margin": rival.map(|target| f64::from(before[teacher.target]) - f64::from(before[target])),
                        "trained_teacher_margin": legal.iter().filter(|target| **target != teacher.target).map(|target| f64::from(after[teacher.target]) - f64::from(after[*target])).reduce(f64::min),
                        "baseline_policy_over_teacher": f64::from(before[policy_target]) - f64::from(before[teacher.target]),
                        "residual_teacher_over_policy": (f64::from(after[teacher.target]) - f64::from(before[teacher.target])) - (f64::from(after[policy_target]) - f64::from(before[policy_target])),
                        "rotated_random_residual_teacher_over_policy": (f64::from(rotated_after[teacher.target]) - f64::from(rotated_before[teacher.target])) - (f64::from(rotated_after[policy_target]) - f64::from(rotated_before[policy_target])),
                        "residual_legal_range": adapter.iter().copied().reduce(f64::max).unwrap() - adapter.iter().copied().reduce(f64::min).unwrap(),
                        "slot_difference_l2": differences.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>().sqrt(),
                        "direction_difference_l2": (f64::from(differences[2]).powi(2) + f64::from(differences[3]).powi(2)).sqrt(),
                        "slot_differing_features": differences.iter().enumerate().filter_map(|(index, value)| (*value != 0.0).then_some(index)).collect::<Vec<_>>(),
                    }));
                }
            }
        }
        eprintln!(
            "reconstructed {} ({} total decisions)",
            directory.display(),
            reconstructed_samples
        );
    }
    let bytes = serde_json::to_vec_pretty(&json!({
        "diagnostic": "target-residual-margins-v2", "scope": "development observations; target-only comparisons conditioned on teacher Move; no ecological qualification",
        "randomness_probe": "rotate each row's 32 random features left once, retain recorded memory; subtract the zero-residual model under the same intervention",
        "baseline": "trained model with only target residual replaced by a zero-output residual",
        "model_directory": args.model, "metadata_sha256": metadata_sha256, "model_sha256": artifact.model_sha256,
        "seed_ledger": artifact.seed_ledger, "reconstructed_samples": reconstructed_samples,
        "datasets": datasets, "rows": rows,
    }))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    Ok(())
}
