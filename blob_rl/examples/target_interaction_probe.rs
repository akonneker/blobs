//! Bounded synthetic capacity diagnostic, not policy training or qualification.
//! The exact teacher is shared with Simple's collision-aware feeding policy.
use std::fs;
use std::path::PathBuf;

use blob_interface::randomness::PrivateRandom;
use blob_mind_utils::choose_slot_by_quantile;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use clap::Parser;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const SLOTS: usize = 4;
// North, west, east, south: relative geometry in the standard Moore order.
const GEOMETRY: [(f32, f32); SLOTS] = [(0., -1.), (-1., 0.), (1., 0.), (0., 1.)];
const TRAIN_SEED: u64 = 1435000101;
const VALIDATION_SEED: u64 = 1435000202;
const INITIALIZATION_SEED: u64 = 1435000303;
const TRAIN_ROWS: usize = 8192;
const VALIDATION_ROWS: usize = 4096;
const STEPS: usize = 512;
const BATCH: usize = 128;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug)]
enum Features {
    Raw,
    Scaled,
    Bilinear,
    RankContext,
}

impl Features {
    fn width(self) -> usize {
        match self {
            Self::Raw | Self::Scaled => 65,
            Self::Bilinear => 129,
            Self::RankContext => 68,
        }
    }
}

#[derive(Clone)]
struct Row {
    random: [u8; 32],
    mask: u8,
}

impl Row {
    fn target(&self) -> usize {
        let candidates: Vec<_> = (0..SLOTS as u8)
            .filter(|slot| self.mask & (1 << slot) != 0)
            .collect();
        usize::from(
            choose_slot_by_quantile(
                &candidates,
                PrivateRandom::from_bytes(self.random).sample_u64(0),
            )
            .unwrap(),
        )
    }

    fn features(&self, mode: Features, slot: usize) -> Vec<f32> {
        let (dx, dy) = GEOMETRY[slot];
        let mut result: Vec<_> = self.random.iter().map(|v| f32::from(*v) / 255.).collect();
        let mut local = [0.; 33];
        // Available targets are vacant, reachable, equally distant and equally fed.
        // Missing targets are masked. Their features cannot enter independent scores.
        local[0] = 1.;
        local[1] = 1.;
        let scale = if matches!(mode, Features::Raw) {
            127.
        } else {
            1.
        };
        local[2] = dx / scale;
        local[3] = dy / scale;
        local[4] = 1024. / 65535.;
        local[7] = 1.;
        local[8] = 0.1;
        local[20] = 1.;
        result.extend(local);
        if matches!(mode, Features::Bilinear) {
            for byte in self.random {
                let centered = f32::from(byte) / 255. - 0.5;
                result.extend([centered * dx, centered * dy]);
            }
        }
        if matches!(mode, Features::RankContext) {
            let count = self.mask.count_ones() as f32;
            let rank = (self.mask & ((1 << slot) - 1)).count_ones() as f32;
            // Deliberately teacher-informed positive control, not a proposed policy:
            // set-relative geometric rank and decoded random quantile. It uses no
            // host identity or tensor position; geometry fixes the semantic order.
            let quantile = PrivateRandom::from_bytes(self.random).sample_u64(0) as f64
                / 18446744073709551616.0;
            result.extend([quantile as f32, (rank + 0.5) / count, 1. / count]);
        }
        result
    }
}

fn rows(seed: u64, count: usize, variable: bool) -> Vec<Row> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut random = [0; 32];
            rng.fill_bytes(&mut random);
            Row {
                random,
                mask: if variable { rng.random_range(1..16) } else { 3 },
            }
        })
        .collect()
}

#[derive(Module, Debug)]
struct Probe<B: Backend> {
    fc: nn::Linear<B>,
    head: nn::Linear<B>,
}

impl<B: Backend> Probe<B> {
    fn new(width: usize, device: &B::Device) -> Self {
        Self {
            fc: nn::LinearConfig::new(width, 32).init(device),
            head: nn::LinearConfig::new(32, 1)
                .with_initializer(nn::Initializer::Zeros)
                .init(device),
        }
    }

    fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = features.dims()[0] / SLOTS;
        self.head
            .forward(burn::tensor::activation::relu(self.fc.forward(features)))
            .reshape([batch, SLOTS])
    }
}

fn tensors<B: Backend>(
    rows: &[Row],
    mode: Features,
    order: [usize; SLOTS],
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 1, Int>) {
    let mut features = Vec::with_capacity(rows.len() * SLOTS * mode.width());
    let mut masks = Vec::with_capacity(rows.len() * SLOTS);
    let mut targets = Vec::with_capacity(rows.len());
    for row in rows {
        targets.push(order.iter().position(|slot| *slot == row.target()).unwrap() as i32);
        for slot in order {
            features.extend(row.features(mode, slot));
            masks.push(if row.mask & (1 << slot) != 0 {
                0.
            } else {
                -1.0e9
            });
        }
    }
    (
        Tensor::from_data(
            TensorData::new(features, [rows.len() * SLOTS, mode.width()]),
            device,
        ),
        Tensor::from_data(TensorData::new(masks, [rows.len(), SLOTS]), device),
        Tensor::from_data(TensorData::new(targets, [rows.len()]), device),
    )
}

fn scores(
    model: &Probe<NdArray>,
    rows: &[Row],
    mode: Features,
    order: [usize; SLOTS],
) -> Vec<[f32; SLOTS]> {
    rows.chunks(BATCH)
        .flat_map(|chunk| {
            let (input, mask, _) = tensors(chunk, mode, order, &Default::default());
            let data = (model.forward(input) + mask)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            data.chunks_exact(SLOTS)
                .map(|row| row.try_into().unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn best(scores: &[f32; SLOTS]) -> usize {
    (0..SLOTS)
        .max_by(|a, b| scores[*a].total_cmp(&scores[*b]).then_with(|| b.cmp(a)))
        .unwrap()
}

fn evaluate(model: &Probe<NdArray>, rows: &[Row], mode: Features) -> Value {
    let original = scores(model, rows, mode, [0, 1, 2, 3]);
    assert!(original.iter().flatten().all(|v| v.is_finite()));
    let permuted = scores(model, rows, mode, [3, 2, 1, 0]);
    let mut reverse_rows = rows.to_vec();
    reverse_rows.reverse();
    let reversed = scores(model, &reverse_rows, mode, [0, 1, 2, 3]);
    let mut intervened = rows.to_vec();
    for row in &mut intervened {
        // Flips the high bit of sample_u64(0), shifting its quantile by one half.
        row.random[7] ^= 128;
    }
    let changed_scores = scores(model, &intervened, mode, [0, 1, 2, 3]);
    let mut correct = 0;
    let mut changed_labels = 0;
    let mut changed_predictions = 0;
    let mut both_correct = 0;
    for (i, row) in rows.iter().enumerate() {
        for j in 0..SLOTS {
            assert_eq!(
                original[i][j].to_bits(),
                permuted[i][SLOTS - 1 - j].to_bits()
            );
            assert_eq!(
                original[i][j].to_bits(),
                reversed[rows.len() - 1 - i][j].to_bits()
            );
        }
        let prediction = best(&original[i]);
        correct += usize::from(prediction == row.target());
        if row.target() != intervened[i].target() {
            changed_labels += 1;
            changed_predictions += usize::from(prediction != best(&changed_scores[i]));
            both_correct += usize::from(
                prediction == row.target() && best(&changed_scores[i]) == intervened[i].target(),
            );
        }
    }
    json!({"rows": rows.len(), "correct": correct, "accuracy": correct as f64 / rows.len() as f64,
        "random_intervention_changed_labels": changed_labels,
        "random_intervention_changed_predictions": changed_predictions,
        "random_intervention_both_correct": both_correct,
        "slot_permutation_bit_exact": true, "row_permutation_bit_exact": true})
}

/// Enumerate all strict score orders. At a fixed private random block, any
/// independent scorer (with a fixed semantic tie break) induces one such order.
/// Enumerate exact u64 quantile intervals, including integer boundary rounding.
fn independent_ceiling() -> Value {
    let total_words = 1_u128 << 64;
    let mut boundaries = vec![0, total_words];
    for count in 1..=SLOTS as u128 {
        for rank in 1..count {
            boundaries.push((rank * total_words).div_ceil(count));
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    let mut bins = Vec::new();
    let mut correct = 0_u128;
    for interval in boundaries.windows(2) {
        let word = ((interval[0] + interval[1]) / 2) as u64;
        let mut random = [0; 32];
        random[..8].copy_from_slice(&word.to_le_bytes());
        let rows: Vec<_> = (1..16).map(|mask| Row { random, mask }).collect();
        let mut maximum = 0;
        let mut winning_order = [0; SLOTS];
        for a in 0..SLOTS {
            for b in 0..SLOTS {
                for c in 0..SLOTS {
                    for d in 0..SLOTS {
                        let order = [a, b, c, d];
                        if (0..SLOTS).any(|i| (0..i).any(|j| order[i] == order[j])) {
                            continue;
                        }
                        let correct = rows
                            .iter()
                            .filter(|row| {
                                *order
                                    .iter()
                                    .find(|slot| row.mask & (1 << **slot) != 0)
                                    .unwrap()
                                    == row.target()
                            })
                            .count();
                        if correct > maximum {
                            maximum = correct;
                            winning_order = order;
                        }
                    }
                }
            }
        }
        correct += (interval[1] - interval[0]) * maximum as u128;
        bins.push(json!({"word_start_inclusive": interval[0].to_string(),
            "word_end_exclusive": interval[1].to_string(),
            "correct_out_of_15": maximum, "best_order": winning_order}));
    }
    // Construct a direct contraction-consistency witness using the real teacher.
    let word = 1_u64 << 63;
    assert_eq!(choose_slot_by_quantile(&[0, 1, 2], word), Some(1));
    assert_eq!(choose_slot_by_quantile(&[1, 2], word), Some(2));
    json!({"bins": bins, "maximum_correct": correct.to_string(), "cases": (15 * total_words).to_string(),
        "uniform_mask_quantile_accuracy_ceiling": correct as f64 / (15 * total_words) as f64,
        "witness": {"random_word": word, "first_candidates": [0,1,2], "first_choice": 1,
            "contracted_candidates": [1,2], "contracted_choice": 2},
        "scope": "Independent per-slot scores only, four fixed geometries, all 15 nonempty subsets equally weighted, uniform private quantile. Not a bound on the context-aware full policy or on real corpus accuracy."})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    // The run directory is exclusive; a failed run remains visibly incomplete.
    fs::create_dir(&args.output)?;
    let plan = json!({"schema_version": 1, "scope": "Synthetic capacity diagnostic; no checkpoint, simulation, promotion or confirmation seeds.",
        "train_seed": TRAIN_SEED, "validation_seed": VALIDATION_SEED, "initialization_seed": INITIALIZATION_SEED,
        "train_rows": TRAIN_ROWS, "validation_rows": VALIDATION_ROWS, "steps": STEPS, "batch": BATCH,
        "learning_rate": 0.005, "hidden": 32, "head_zero_initialized": true,
        "modes": ["Raw", "Scaled", "Bilinear", "RankContext"],
        "tasks": ["fixed_pair", "varying_subsets"],
        "gate": "At least 95% held-out exact agreement and 90% both-correct on changed-label random interventions; exact slot/row permutation. A positive-control pass alone does not authorize production architecture selection.",
        "source_sha256": format!("{:x}", Sha256::digest(include_bytes!("target_interaction_probe.rs"))),
        "teacher_source_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../minds/blob_mind_utils/src/lib.rs"))),
        "cargo_lock_sha256": format!("{:x}", Sha256::digest(include_bytes!("../../Cargo.lock")))});
    fs::write(
        args.output.join("plan.json"),
        serde_json::to_vec_pretty(&plan)?,
    )?;
    fs::write(
        args.output.join("probe.rs"),
        include_bytes!("target_interaction_probe.rs"),
    )?;
    let mut results = Vec::new();
    for variable in [false, true] {
        let train = rows(TRAIN_SEED, TRAIN_ROWS, variable);
        let validation = rows(VALIDATION_SEED, VALIDATION_ROWS, variable);
        for mode in [
            Features::Raw,
            Features::Scaled,
            Features::Bilinear,
            Features::RankContext,
        ] {
            let device = Default::default();
            NdArray::<f32>::seed(&device, INITIALIZATION_SEED);
            let mut model = Probe::<Autodiff<NdArray>>::new(mode.width(), &device);
            let mut optimizer = AdamWConfig::new().init();
            for step in 0..STEPS {
                let offset = (step * BATCH) % train.len();
                let (features, masks, targets) =
                    tensors(&train[offset..offset + BATCH], mode, [0, 1, 2, 3], &device);
                let logits = model.forward(features) + masks;
                let loss = burn::tensor::activation::log_softmax(logits, 1)
                    .gather(1, targets.unsqueeze_dim(1))
                    .mean()
                    .neg();
                let gradients = GradientsParams::from_grads(loss.backward(), &model);
                model = optimizer.step(0.005, model, gradients);
            }
            let model = model.valid();
            let evaluation = evaluate(&model, &validation, mode);
            let gate = evaluation["accuracy"].as_f64().unwrap() >= 0.95
                && evaluation["random_intervention_both_correct"]
                    .as_u64()
                    .unwrap() as f64
                    / evaluation["random_intervention_changed_labels"]
                        .as_u64()
                        .unwrap() as f64
                    >= 0.90;
            let result = json!({"task": if variable { "varying_subsets" } else { "fixed_pair" },
                "mode": format!("{mode:?}"), "parameters": (mode.width() + 1) * 32 + 33,
                "training": evaluate(&model, &train, mode), "validation": evaluation, "gate_pass": gate});
            println!("{}", serde_json::to_string(&result)?);
            results.push(result);
            fs::write(
                args.output.join("progress.json"),
                serde_json::to_vec_pretty(&results)?,
            )?;
        }
    }
    fs::write(
        args.output.join("report.json"),
        serde_json::to_vec_pretty(&json!({"plan": plan,
        "independent_ceiling": independent_ceiling(), "results": results, "complete": true}))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_byte_intervention_matches_little_endian_teacher() {
        let row = Row {
            random: [0; 32],
            mask: 3,
        };
        assert_eq!(row.target(), 0);
        let mut changed = row.clone();
        changed.random[7] = 128;
        assert_eq!(changed.target(), 1);
        changed.random[7] = 0;
        changed.random[0] = 128;
        assert_eq!(changed.target(), 0);
    }

    #[test]
    fn independent_teacher_has_strict_capacity_gap() {
        let ceiling = independent_ceiling();
        assert!(
            ceiling["uniform_mask_quantile_accuracy_ceiling"]
                .as_f64()
                .unwrap()
                < 1.
        );
        assert_eq!(ceiling["bins"][0]["correct_out_of_15"], 15);
        assert_eq!(
            ceiling["bins"].as_array().unwrap().last().unwrap()["correct_out_of_15"],
            15
        );
    }
}
