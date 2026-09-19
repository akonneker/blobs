//! Predeclared synthetic learned-set capacity gate; never exports policy weights.
use std::{collections::BTreeMap, fs, path::PathBuf};

use blob_interface::randomness::PrivateRandom;
use blob_mind_utils::choose_slot_by_quantile;
use burn::optim::grad_clipping::GradientClipping;
use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    nn,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
};
use clap::Parser;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[path = "target_set_probe/benchmark.rs"]
mod benchmark;
#[path = "target_set_probe/relational.rs"]
mod relational;
#[path = "target_set_probe/sampling.rs"]
mod sampling;

const SLOTS: usize = 8;
const GEOMETRY: [[f32; 2]; SLOTS] = [
    [-1., -1.],
    [0., -1.],
    [1., -1.],
    [-1., 0.],
    [1., 0.],
    [-1., 1.],
    [0., 1.],
    [1., 1.],
];
const TRAIN_ROWS: usize = 16384;
const VALIDATION_ROWS: usize = 4096;
const STEPS: usize = 1024;
const BATCH: usize = 128;
const TRAIN_SEED: u64 = 1435200101;
const VALIDATION_SEED: u64 = 1435200202;
const INITIALIZATIONS: [u64; 3] = [1435200301, 1435200302, 1435200303];

#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
    /// Recheck only the selected sampled-utility candidate across all nine gates.
    #[arg(long)]
    utility_only: bool,
}

#[derive(Clone, Copy, Debug)]
enum Task {
    FourEqual,
    EightEqual,
    EightMixed,
}
#[derive(Clone, Copy, Debug)]
enum Mode {
    Independent,
    LearnedSum,
    Relational,
    RelationalProduct,
    QuantileUtility,
    MarginalUtility,
    RankContext,
}

#[derive(Clone)]
struct Row {
    random: [u8; 32],
    mask: u8,
    energy: [u8; SLOTS],
}
impl Row {
    fn candidates(&self) -> Vec<u8> {
        let best = (0..SLOTS)
            .filter(|i| self.mask & (1 << i) != 0)
            .map(|i| self.energy[i])
            .max()
            .unwrap();
        (0..SLOTS as u8)
            .filter(|i| self.mask & (1 << i) != 0 && self.energy[*i as usize] == best)
            .collect()
    }
    fn target(&self) -> usize {
        choose_slot_by_quantile(
            &self.candidates(),
            PrivateRandom::from_bytes(self.random).sample_u64(0),
        )
        .unwrap() as usize
    }
}

fn rows(seed: u64, count: usize, task: Task) -> Vec<Row> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut random = [0; 32];
            rng.fill_bytes(&mut random);
            let mask = if matches!(task, Task::FourEqual) {
                let bits: u8 = rng.random_range(1..16);
                [1, 3, 4, 6]
                    .iter()
                    .enumerate()
                    .fold(0, |mask, (i, slot)| mask | (((bits >> i) & 1) << slot))
            } else {
                rng.random_range(1..=255)
            };
            let energy = std::array::from_fn(|_| {
                if matches!(task, Task::EightMixed) {
                    rng.random_range(1..=3)
                } else {
                    1
                }
            });
            Row {
                random,
                mask,
                energy,
            }
        })
        .collect()
}

struct Batch<B: Backend> {
    local: Tensor<B, 3>,
    random: Tensor<B, 3>,
    positive: Tensor<B, 3>,
    mask: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    marginal_targets: Tensor<B, 2>,
}

fn batch<B: Backend>(
    rows: &[Row],
    mode: Mode,
    order: [usize; SLOTS],
    device: &B::Device,
) -> Batch<B> {
    let mut local = Vec::new();
    let mut random = Vec::new();
    let mut positive = Vec::new();
    let mut masks = Vec::new();
    let mut targets = Vec::new();
    let mut marginal_targets = Vec::new();
    for row in rows {
        targets.push(order.iter().position(|i| *i == row.target()).unwrap() as i32);
        let quantile = (PrivateRandom::from_bytes(row.random).sample_u64(0) as f64
            / 18446744073709551616.0) as f32;
        // All arms see the same random encoding, including the decoded quantile.
        random.extend(row.random.iter().map(|b| *b as f32 / 255.));
        random.push(quantile);
        let candidates = row.candidates();
        for slot in order {
            marginal_targets.push(if candidates.contains(&(slot as u8)) {
                1. / candidates.len() as f32
            } else {
                0.
            });
            local.extend([
                GEOMETRY[slot][0],
                GEOMETRY[slot][1],
                row.energy[slot] as f32 / 3.,
            ]);
            masks.push(if row.mask & (1 << slot) != 0 {
                0.
            } else {
                -1e9
            });
            if matches!(mode, Mode::RankContext) {
                let rank = candidates.iter().position(|i| *i as usize == slot);
                positive.extend([
                    f32::from(rank.is_some()),
                    (rank.unwrap_or(0) as f32 + 0.5) / candidates.len() as f32,
                    1. / candidates.len() as f32,
                ]);
            } else {
                positive.extend([0.; 3]);
            }
        }
    }
    Batch {
        local: Tensor::from_data(TensorData::new(local, [rows.len(), SLOTS, 3]), device),
        random: Tensor::from_data(TensorData::new(random, [rows.len(), 1, 33]), device),
        positive: Tensor::from_data(TensorData::new(positive, [rows.len(), SLOTS, 3]), device),
        mask: Tensor::from_data(TensorData::new(masks, [rows.len(), SLOTS]), device),
        targets: Tensor::from_data(TensorData::new(targets, [rows.len()]), device),
        marginal_targets: Tensor::from_data(
            TensorData::new(marginal_targets, [rows.len(), SLOTS]),
            device,
        ),
    }
}

#[derive(Module, Debug)]
struct SetProbe<B: Backend> {
    encoder: nn::Linear<B>,
    encoder2: nn::Linear<B>,
    fc: nn::Linear<B>,
    head: nn::Linear<B>,
    relational: relational::RelationalEncoder<B>,
    interaction: nn::Linear<B>,
}
impl<B: Backend> SetProbe<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            encoder: nn::LinearConfig::new(3, 16).init(device),
            encoder2: nn::LinearConfig::new(16, 16).init(device),
            fc: nn::LinearConfig::new(3 + 16 + 16 + 33 + 3, 64).init(device),
            head: nn::LinearConfig::new(64, 1)
                .with_initializer(nn::Initializer::Zeros)
                .init(device),
            relational: relational::RelationalEncoder::new(device),
            // Initialize after all existing parameters, without consuming random
            // draws. The earlier arms keep their exact initial computation.
            interaction: nn::LinearConfig::new(16, 64)
                .with_bias(false)
                .with_initializer(nn::Initializer::Zeros)
                .init(device),
        }
    }
    fn encode(&self, local: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder2
            .forward(self.encoder.forward(local).tanh())
            .tanh()
    }
    fn summarize(encoded: Tensor<B, 3>, mask: Tensor<B, 2>) -> Tensor<B, 3> {
        let slots = encoded.dims()[1];
        // Fixed normalization retains cardinality. Sorting each feature makes
        // floating addition order canonical without supplying a slot identity.
        let present = mask.equal_elem(0.).float().unsqueeze_dim(2);
        (encoded * present).sort(1).sum_dim(1) / slots as f32
    }
    fn forward(&self, inputs: Batch<B>, mode: Mode) -> Tensor<B, 2> {
        let [size, slots, _] = inputs.local.dims();
        // Two nonlinear layers can encode interior points. A single affine/ReLU
        // layer followed by max can see only the convex hull of the local inputs.
        // Bound learned feature scale while retaining a nonlinear set encoding.
        let encoded = self.encode(inputs.local.clone());
        let context = if matches!(mode, Mode::Relational | Mode::RelationalProduct) {
            self.relational
                .forward(inputs.local.clone(), inputs.mask.clone())
        } else if matches!(
            mode,
            Mode::LearnedSum | Mode::QuantileUtility | Mode::MarginalUtility
        ) {
            // Only legal destinations enter the summary; energy selection is learned.
            Self::summarize(encoded.clone(), inputs.mask.clone()).repeat_dim(1, slots)
        } else {
            Tensor::zeros([size, slots, 16], &inputs.local.device())
        };
        let interaction = matches!(mode, Mode::RelationalProduct).then(|| {
            let quantile = inputs
                .random
                .clone()
                .slice([0..size, 0..1, 32..33])
                .repeat_dim(1, slots);
            self.interaction.forward(context.clone() * quantile)
        });
        let random = if matches!(mode, Mode::QuantileUtility | Mode::MarginalUtility) {
            Tensor::zeros([size, 1, 33], &inputs.random.device())
        } else {
            inputs.random
        };
        let features = Tensor::cat(
            vec![
                inputs.local,
                encoded,
                context,
                random.repeat_dim(1, slots),
                inputs.positive,
            ],
            2,
        );
        let hidden = self.fc.forward(features);
        let hidden = burn::tensor::activation::relu(if let Some(interaction) = interaction {
            hidden + interaction
        } else {
            hidden
        });
        self.head.forward(hidden).reshape([size, slots]) + inputs.mask
    }
}

const ORDER: [usize; SLOTS] = [0, 1, 2, 3, 4, 5, 6, 7];
const PERMUTATION: [usize; SLOTS] = [7, 2, 5, 0, 6, 1, 4, 3];
fn scores(
    model: &SetProbe<NdArray>,
    rows: &[Row],
    mode: Mode,
    order: [usize; SLOTS],
) -> Vec<[f32; SLOTS]> {
    rows.chunks(BATCH)
        .flat_map(|chunk| {
            model
                .forward(batch(chunk, mode, order, &Default::default()), mode)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .chunks_exact(SLOTS)
                .map(|s| s.try_into().unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}
fn best(scores: &[f32; SLOTS]) -> usize {
    (0..SLOTS)
        .max_by(|a, b| scores[*a].total_cmp(&scores[*b]).then_with(|| b.cmp(a)))
        .unwrap()
}
fn decode(scores: &[f32; SLOTS], row: &Row, mode: Mode) -> usize {
    if matches!(mode, Mode::QuantileUtility | Mode::MarginalUtility) {
        let legal: Vec<_> = (0..SLOTS).map(|slot| row.mask & (1 << slot) != 0).collect();
        // ORDER is the fixed canonical Moore geometry order, not tensor row order.
        sampling::sample(
            scores,
            &legal,
            PrivateRandom::from_bytes(row.random).sample_u64(0),
        )
        .unwrap()
    } else {
        best(scores)
    }
}
fn evaluate(model: &SetProbe<NdArray>, rows: &[Row], mode: Mode) -> Value {
    let original = scores(model, rows, mode, ORDER);
    let permuted = scores(model, rows, mode, PERMUTATION);
    let reversed_rows: Vec<_> = rows.iter().rev().cloned().collect();
    let reversed = scores(model, &reversed_rows, mode, ORDER);
    let mut intervened = rows.to_vec();
    for row in &mut intervened {
        row.random[7] ^= 128;
    }
    let changed = scores(model, &intervened, mode, ORDER);
    let mut correct = 0;
    let mut changed_labels = 0;
    let mut both_correct = 0;
    let mut predictions_changed = 0;
    let mut by_candidates = [[0_usize; 2]; SLOTS];
    for (i, row) in rows.iter().enumerate() {
        for slot in 0..SLOTS {
            assert!(original[i][slot].is_finite());
            assert_eq!(
                original[i][slot].to_bits(),
                permuted[i][PERMUTATION.iter().position(|s| *s == slot).unwrap()].to_bits()
            );
            assert_eq!(
                original[i][slot].to_bits(),
                reversed[rows.len() - 1 - i][slot].to_bits()
            );
        }
        let prediction = decode(&original[i], row, mode);
        let intervened_prediction = decode(&changed[i], &intervened[i], mode);
        if matches!(mode, Mode::QuantileUtility | Mode::MarginalUtility) {
            assert!(original[i]
                .iter()
                .zip(&changed[i])
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        let right = prediction == row.target();
        correct += usize::from(right);
        let group = &mut by_candidates[row.candidates().len() - 1];
        group[0] += usize::from(right);
        group[1] += 1;
        if row.target() != intervened[i].target() {
            changed_labels += 1;
            both_correct += usize::from(right && intervened_prediction == intervened[i].target());
            predictions_changed += usize::from(prediction != intervened_prediction);
        }
    }
    let accuracy = correct as f64 / rows.len() as f64;
    let intervention_accuracy = both_correct as f64 / changed_labels as f64;
    json!({"rows": rows.len(), "correct": correct, "accuracy": accuracy, "changed_labels": changed_labels,
        "both_correct": both_correct, "intervention_accuracy": intervention_accuracy, "predictions_changed": predictions_changed,
        "by_tied_best_candidate_count_correct_total": by_candidates,
        "slot_permutation_bit_exact": true, "row_permutation_bit_exact": true,
        "gate_pass": accuracy >= 0.95 && intervention_accuracy >= 0.90})
}

// Inspect fitted summaries, not architecture-wide capacity. Equal-energy slices
// let us hold every surviving destination's local features exactly fixed.
fn summary_diagnostic(model: &SetProbe<NdArray>, task: Task, mode: Mode) -> Value {
    let mut levels = Vec::new();
    for energy in 1..=if matches!(task, Task::EightMixed) {
        3
    } else {
        1
    } {
        let allowed = if matches!(task, Task::FourEqual) {
            0b0101_1010
        } else {
            255
        };
        let rows: Vec<_> = (1..=255_u8)
            .filter(|mask| mask & !allowed == 0)
            .map(|mask| Row {
                random: [0; 32],
                mask,
                energy: [energy; SLOTS],
            })
            .collect();
        let inputs = batch::<NdArray>(&rows, mode, ORDER, &Default::default());
        let summaries = SetProbe::summarize(model.encode(inputs.local), inputs.mask)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut groups = BTreeMap::<Vec<u32>, Vec<usize>>::new();
        for (i, summary) in summaries.chunks_exact(16).enumerate() {
            groups
                .entry(summary.iter().map(|v| v.to_bits()).collect())
                .or_default()
                .push(i);
        }
        let mut contradictory_pairs = 0;
        let mut witnesses = Vec::new();
        for group in groups.values() {
            for (position, first) in group.iter().enumerate() {
                for second in &group[position + 1..] {
                    let a = &rows[*first];
                    let b = &rows[*second];
                    let mut boundaries = vec![0];
                    for count in [a.mask.count_ones(), b.mask.count_ones()] {
                        boundaries.extend(
                            (1..count).map(|rank| ((rank as u128) << 64).div_ceil(count as u128)),
                        );
                    }
                    let witness = boundaries.into_iter().find_map(|word| {
                        let ta = choose_slot_by_quantile(&a.candidates(), word as u64).unwrap();
                        let tb = choose_slot_by_quantile(&b.candidates(), word as u64).unwrap();
                        let shared = a.mask & b.mask;
                        (ta != tb && shared & (1 << ta) != 0 && shared & (1 << tb) != 0)
                            .then_some((word, ta, tb))
                    });
                    if let Some((word, ta, tb)) = witness {
                        contradictory_pairs += 1;
                        if witnesses.len() < 3 {
                            let mut pair = vec![a.clone(), b.clone()];
                            for row in &mut pair {
                                row.random[..8].copy_from_slice(&(word as u64).to_le_bytes());
                            }
                            let outputs = scores(model, &pair, mode, ORDER);
                            for (slot, (first_score, second_score)) in
                                outputs[0].iter().zip(&outputs[1]).enumerate()
                            {
                                if a.mask & b.mask & (1 << slot) != 0 {
                                    assert_eq!(first_score.to_bits(), second_score.to_bits());
                                }
                            }
                            assert!(
                                !(best(&outputs[0]) == ta as usize
                                    && best(&outputs[1]) == tb as usize)
                            );
                            witnesses.push(json!({"first_mask":a.mask,"second_mask":b.mask,"random_word":word.to_string(),
                                "first_teacher":ta,"second_teacher":tb,"first_prediction":best(&outputs[0]),"second_prediction":best(&outputs[1]),
                                "shared_scores_bit_identical":true}));
                        }
                    }
                }
            }
        }
        levels.push(json!({"common_energy":energy,"candidate_sets":rows.len(),"distinct_summary_bit_patterns":groups.len(),
            "contradictory_set_pairs":contradictory_pairs,"witnesses":witnesses}));
    }
    json!({"scope":"Fitted encoder, exhaustive equal-energy candidate subsets. Mixed-task slices are not estimates of their frequency in validation. Exact summary collisions only; distinct vectors need not be well separated.","levels":levels})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    fs::create_dir(&args.output)?;
    let modes = if args.utility_only {
        vec![Mode::QuantileUtility]
    } else {
        vec![
            Mode::LearnedSum,
            Mode::Relational,
            Mode::RelationalProduct,
            Mode::QuantileUtility,
            Mode::MarginalUtility,
            Mode::RankContext,
        ]
    };
    let plan = json!({"schema_version":1, "revision":11, "scope":"Synthetic target-distribution capacity gate, no checkpoint or ecology.",
        "revision_reason":"Recheck the selected utility candidate with the portable libm/Q48 decoder shared with the ordinary-ABI canary. Models, losses, exposed development seeds and update budgets are unchanged; optional utility-only mode selects the nine candidate gates.",
        "encoder_activation":"tanh",
        "diagnostics":"Post-training zero-context ablation, additive summary collisions, and native 8/32-slot cost. Single-cell 32-slot p95 must be below 10 ms for the local cost gate; no WASM fuel claim.",
        "gradient_clipping_norm":1.0,"ndarray_simd":false,
        "dependency_manifest_sha256":format!("{:x}",Sha256::digest(include_bytes!("../Cargo.toml"))),
        "train_seed":TRAIN_SEED,"development_validation_seed":VALIDATION_SEED,"initialization_seeds":INITIALIZATIONS,
        "train_rows":TRAIN_ROWS,"validation_rows":VALIDATION_ROWS,"steps":STEPS,"batch":BATCH,"learning_rate":0.005,
        "tasks":["FourEqual","EightEqual","EightMixed"],"modes":modes.iter().map(|mode|format!("{mode:?}")).collect::<Vec<_>>(),
        "same_initialized_parameter_layout":true,"random_encoding":"32 byte/255 features plus decoded u64 quantile in deterministic arms. Both utility modes zero those network inputs and use the exact private word only in the decoder.",
        "gate":"Every task and initialization must reach 95% validation accuracy and 90% joint random-intervention accuracy, with bit-exact slot/row permutation. No production promotion from synthetic success.",
        "source_sha256":format!("{:x}",Sha256::digest(include_bytes!("target_set_probe.rs"))),
        "relational_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("target_set_probe/relational.rs"))),
        "benchmark_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("target_set_probe/benchmark.rs"))),
        "target_sampling_contract":blob_policy::sampling::TARGET_SAMPLING_CONTRACT,
        "portable_sampling_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../blob_policy/src/sampling.rs"))),
        "sampling_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("target_set_probe/sampling.rs"))),
        "teacher_source_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../minds/blob_mind_utils/src/lib.rs"))),
        "cargo_lock_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../Cargo.lock")))});
    fs::write(
        args.output.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    fs::write(
        args.output.join("probe.rs"),
        include_bytes!("target_set_probe.rs"),
    )?;
    fs::create_dir(args.output.join("target_set_probe"))?;
    fs::write(
        args.output.join("target_set_probe/relational.rs"),
        include_bytes!("target_set_probe/relational.rs"),
    )?;
    fs::write(
        args.output.join("target_set_probe/benchmark.rs"),
        include_bytes!("target_set_probe/benchmark.rs"),
    )?;
    fs::write(
        args.output.join("target_set_probe/sampling.rs"),
        include_bytes!("target_set_probe/sampling.rs"),
    )?;
    fs::write(
        args.output.join("blob_rl.Cargo.toml"),
        include_bytes!("../Cargo.toml"),
    )?;
    fs::write(
        args.output.join("Cargo.lock"),
        include_bytes!("../../Cargo.lock"),
    )?;
    fs::write(
        args.output.join("portable_sampling.rs"),
        include_bytes!("../../blob_policy/src/sampling.rs"),
    )?;
    let mut results = Vec::new();
    for task in [Task::FourEqual, Task::EightEqual, Task::EightMixed] {
        let train = rows(TRAIN_SEED, TRAIN_ROWS, task);
        let validation = rows(VALIDATION_SEED, VALIDATION_ROWS, task);
        for seed in INITIALIZATIONS {
            for &mode in &modes {
                let device = Default::default();
                NdArray::<f32>::seed(&device, seed);
                let mut model = SetProbe::<Autodiff<NdArray>>::new(&device);
                let initial_validation =
                    matches!(mode, Mode::QuantileUtility | Mode::MarginalUtility)
                        .then(|| evaluate(&model.valid(), &validation, mode));
                let mut optimizer = AdamWConfig::new()
                    .init()
                    .with_grad_clipping(GradientClipping::Norm(1.0));
                let mut loss_checkpoints = Vec::new();
                for step in 0..STEPS {
                    let offset = (step * BATCH) % train.len();
                    let inputs = batch(&train[offset..offset + BATCH], mode, ORDER, &device);
                    let targets = inputs.targets.clone();
                    let marginal_targets = inputs.marginal_targets.clone();
                    let feature_max = if step % 256 == 0 || step == STEPS - 1 {
                        Some(
                            model
                                .encode(inputs.local.clone())
                                .abs()
                                .max()
                                .into_data()
                                .to_vec::<f32>()?[0],
                        )
                    } else {
                        None
                    };
                    let log_probabilities =
                        burn::tensor::activation::log_softmax(model.forward(inputs, mode), 1);
                    let loss = if matches!(mode, Mode::MarginalUtility) {
                        (log_probabilities * marginal_targets)
                            .sum_dim(1)
                            .mean()
                            .neg()
                    } else {
                        log_probabilities
                            .gather(1, targets.unsqueeze_dim(1))
                            .mean()
                            .neg()
                    };
                    if step % 256 == 0 || step == STEPS - 1 {
                        loss_checkpoints.push(json!({"step":step,"batch_cross_entropy":loss.clone().into_data().to_vec::<f32>()?[0],"encoded_feature_max_abs":feature_max}));
                    }
                    let grads = GradientsParams::from_grads(loss.backward(), &model);
                    model = optimizer.step(0.005, model, grads);
                }
                let model = model.valid();
                let context_diagnostic = matches!(mode, Mode::LearnedSum).then(|| {
                    json!({
                    "ablated_validation":evaluate(&model,&validation,Mode::Independent),
                    "summary_collisions":summary_diagnostic(&model,task,mode)})
                });
                let relational_ablation =
                    matches!(mode, Mode::Relational | Mode::RelationalProduct)
                        .then(|| evaluate(&model, &validation, Mode::Independent));
                let interaction_ablation = matches!(mode, Mode::RelationalProduct)
                    .then(|| evaluate(&model, &validation, Mode::Relational));
                let result = json!({"task":format!("{task:?}"),"mode":format!("{mode:?}"),"initialization_seed":seed,
                    "parameters":model.num_params(),"loss_checkpoints":loss_checkpoints,"validation":evaluate(&model,&validation,mode),
                    "context_diagnostic":context_diagnostic,"relational_ablation":relational_ablation,"interaction_ablation":interaction_ablation,"initial_validation":initial_validation});
                println!("{}", serde_json::to_string(&result)?);
                results.push(result);
                fs::write(
                    args.output.join("progress.json"),
                    serde_json::to_vec_pretty(&results)?,
                )?;
            }
        }
    }
    fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(
            &json!({"plan":plan,"results":results,"native_cost":benchmark::run(),"complete":true}),
        )?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uniform_marginal_teacher_is_stationary_through_real_optimizer_steps() {
        let device = Default::default();
        NdArray::<f32>::seed(&device, INITIALIZATIONS[0]);
        let mut model = SetProbe::<Autodiff<NdArray>>::new(&device);
        let train = rows(TRAIN_SEED, TRAIN_ROWS, Task::FourEqual);
        let mut optimizer = AdamWConfig::new()
            .init()
            .with_grad_clipping(GradientClipping::Norm(1.0));
        for step in 0..4 {
            let inputs = batch(
                &train[step * BATCH..(step + 1) * BATCH],
                Mode::MarginalUtility,
                ORDER,
                &device,
            );
            let labels = inputs.marginal_targets.clone();
            let loss = (burn::tensor::activation::log_softmax(
                model.forward(inputs, Mode::MarginalUtility),
                1,
            ) * labels)
                .sum_dim(1)
                .mean()
                .neg();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(0.005, model, grads);
            assert_eq!(
                model.head.weight.val().into_data().to_vec::<f32>().unwrap(),
                vec![0.; 64]
            );
            assert_eq!(
                model
                    .head
                    .bias
                    .as_ref()
                    .unwrap()
                    .val()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
                vec![0.]
            );
        }
    }
    #[test]
    fn marginal_targets_follow_teacher_support_and_never_enter_network_features() {
        let device = Default::default();
        let row = Row {
            random: [0; 32],
            mask: 7,
            energy: [1, 3, 3, 0, 0, 0, 0, 0],
        };
        let mut inputs = batch::<NdArray>(&[row], Mode::MarginalUtility, ORDER, &device);
        assert_eq!(
            inputs
                .marginal_targets
                .clone()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
            vec![0., 0.5, 0.5, 0., 0., 0., 0., 0.]
        );
        let mut model = SetProbe::<NdArray>::new(&device);
        model.head = nn::LinearConfig::new(64, 1).init(&device);
        let original = model
            .forward(
                batch::<NdArray>(
                    &[Row {
                        random: [0; 32],
                        mask: 7,
                        energy: [1, 3, 3, 0, 0, 0, 0, 0],
                    }],
                    Mode::MarginalUtility,
                    ORDER,
                    &device,
                ),
                Mode::MarginalUtility,
            )
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        inputs.targets = Tensor::zeros([1], &device);
        inputs.marginal_targets = Tensor::ones([1, SLOTS], &device) * 123.;
        let changed = model
            .forward(inputs, Mode::MarginalUtility)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(original
            .iter()
            .zip(changed)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    #[test]
    fn zero_product_extension_preserves_prior_scores_and_active_product_is_permutation_safe() {
        let device = Default::default();
        let mut model = SetProbe::<NdArray>::new(&device);
        model.head = nn::LinearConfig::new(64, 1).init(&device);
        let rows = rows(33, 32, Task::EightEqual);
        let before = scores(&model, &rows, Mode::Relational, ORDER);
        let zero = scores(&model, &rows, Mode::RelationalProduct, ORDER);
        assert!(before
            .iter()
            .flatten()
            .zip(zero.iter().flatten())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        model.interaction = nn::LinearConfig::new(16, 64).with_bias(false).init(&device);
        let active = scores(&model, &rows, Mode::RelationalProduct, ORDER);
        assert!(before
            .iter()
            .flatten()
            .zip(active.iter().flatten())
            .any(|(a, b)| (a - b).abs() > 1e-5));
        evaluate(&model, &rows, Mode::RelationalProduct);
    }
    #[test]
    fn sorted_sum_gradients_preserve_tied_contributions_and_exclude_masked_slots() {
        let device = Default::default();
        let encoded = Tensor::<Autodiff<NdArray>, 3>::from_data(
            TensorData::new(vec![1.; SLOTS * 16], [1, SLOTS, 16]),
            &device,
        )
        .require_grad();
        let mask = Tensor::from_data(
            TensorData::new(vec![-1e9, 0., 0., -1e9, -1e9, -1e9, -1e9, -1e9], [1, SLOTS]),
            &device,
        );
        let summary = SetProbe::summarize(encoded.clone(), mask);
        assert!(summary
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| *v == 0.25));
        let grads = encoded
            .grad(&summary.sum().backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (slot, values) in grads.chunks_exact(16).enumerate() {
            let expected = if slot == 1 || slot == 2 { 0.125 } else { 0. };
            assert!(values.iter().all(|v| *v == expected));
        }
    }
    #[test]
    fn batched_linear_gradients_match_flattened_slot_rows() {
        let device = Default::default();
        NdArray::<f32>::seed(&device, 19);
        let layer = nn::LinearConfig::new(3, 16).init::<Autodiff<NdArray>>(&device);
        let input = batch::<Autodiff<NdArray>>(
            &rows(21, 4, Task::EightMixed),
            Mode::Independent,
            ORDER,
            &device,
        )
        .local;
        let batched = layer.forward(input.clone()).powf_scalar(2.).mean();
        let flat = layer
            .forward(input.reshape([4 * SLOTS, 3]))
            .powf_scalar(2.)
            .mean();
        let batched_gradient = layer
            .weight
            .val()
            .grad(&batched.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let flat_gradient = layer
            .weight
            .val()
            .grad(&flat.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for (a, b) in batched_gradient.iter().zip(&flat_gradient) {
            assert!((a - b).abs() <= 1e-6, "{a} != {b}");
        }
    }
    #[test]
    fn learned_inputs_exclude_teacher_rank_and_masked_slots_cannot_change_other_scores() {
        for mode in [
            Mode::LearnedSum,
            Mode::Relational,
            Mode::RelationalProduct,
            Mode::QuantileUtility,
            Mode::MarginalUtility,
        ] {
            check_isolation(mode);
        }
    }
    fn check_isolation(mode: Mode) {
        let device = Default::default();
        NdArray::<f32>::seed(&device, 17);
        let mut model = SetProbe::<NdArray>::new(&device);
        model.head = nn::LinearConfig::new(64, 1).init(&device);
        model.interaction = nn::LinearConfig::new(16, 64).with_bias(false).init(&device);
        let rows = rows(21, 2, Task::FourEqual);
        let inputs = batch::<NdArray>(&rows, mode, ORDER, &device);
        assert!(inputs
            .positive
            .clone()
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|v| *v == 0.));
        let before = model
            .forward(inputs, mode)
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut changed = rows.clone();
        changed[0].random[7] ^= 128;
        let after = scores(&model, &changed, mode, ORDER);
        for slot in 0..SLOTS {
            assert_eq!(before[SLOTS + slot].to_bits(), after[1][slot].to_bits());
        }
        // Slot 0 is absent from every FourEqual example. Its local payload is irrelevant.
        let mut changed = rows.clone();
        changed[0].energy[0] = 255;
        let after = scores(&model, &changed, mode, ORDER);
        for slot in 1..SLOTS {
            assert_eq!(before[slot].to_bits(), after[0][slot].to_bits());
        }
    }
    #[test]
    fn mixed_teacher_selects_only_best_legal_energy() {
        let mut row = Row {
            random: [0; 32],
            mask: 7,
            energy: [1, 3, 3, 255, 0, 0, 0, 0],
        };
        assert_eq!(row.candidates(), vec![1, 2]);
        assert_eq!(row.target(), 1);
        row.random[7] = 128;
        assert_eq!(row.target(), 2);
    }
}
