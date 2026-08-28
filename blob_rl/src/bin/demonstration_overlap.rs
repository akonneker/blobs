//! Diagnose contradictory action-family labels for identical observable state.

use std::collections::HashMap;
use std::path::PathBuf;

use blob_rl::action::policy_action_family;
use blob_rl::behavior_cloning::load_dataset_directories;
use blob_rl::observation::{OBS_RANDOMNESS_FEATURE_END, OBS_RANDOMNESS_FEATURE_START};
use clap::Parser;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(
    name = "demonstration-overlap",
    about = "Find cross-dataset action conflicts after removing private randomness"
)]
struct Args {
    #[arg(long, required = true)]
    dataset: Vec<PathBuf>,

    #[arg(long)]
    exact_round_trip_only: bool,

    /// Round observable features to this many units per normalized interval.
    /// Zero keeps exact f32 values.
    #[arg(long, default_value_t = 0)]
    quantization_levels: u32,
}

#[derive(Default)]
struct StateLabels {
    datasets: u128,
    families: u16,
    samples: usize,
}

fn main() {
    let args = Args::parse();
    if args.dataset.len() > 128 {
        panic!("overlap analysis supports at most 128 datasets");
    }
    let datasets = load_dataset_directories(&args.dataset)
        .unwrap_or_else(|error| panic!("failed to load demonstrations: {error}"));
    let mut states = HashMap::<[u8; 32], StateLabels>::new();
    let mut included = 0usize;
    for (dataset_index, dataset) in datasets.iter().enumerate() {
        for sample in &dataset.payload.samples {
            if args.exact_round_trip_only && !sample.exact_round_trip {
                continue;
            }
            included += 1;
            let mut hasher = Sha256::new();
            for (index, value) in sample.observation.iter().enumerate() {
                if !(OBS_RANDOMNESS_FEATURE_START..OBS_RANDOMNESS_FEATURE_END).contains(&index) {
                    if args.quantization_levels == 0 {
                        hasher.update(value.to_bits().to_le_bytes());
                    } else {
                        let quantized = (*value * args.quantization_levels as f32).round() as i32;
                        hasher.update(quantized.to_le_bytes());
                    }
                }
            }
            for allowed in &sample.action_mask {
                hasher.update([u8::from(*allowed)]);
            }
            let key: [u8; 32] = hasher.finalize().into();
            let family = policy_action_family(usize::from(sample.action))
                .expect("validated demonstration action is in the catalog");
            let state = states.entry(key).or_default();
            state.datasets |= 1_u128 << dataset_index;
            state.families |= 1_u16 << family.index();
            state.samples += 1;
        }
    }
    let shared_states = states
        .values()
        .filter(|state| state.datasets.count_ones() > 1)
        .count();
    let conflicting_states = states
        .values()
        .filter(|state| state.datasets.count_ones() > 1 && state.families.count_ones() > 1)
        .count();
    let conflicting_samples = states
        .values()
        .filter(|state| state.datasets.count_ones() > 1 && state.families.count_ones() > 1)
        .map(|state| state.samples)
        .sum::<usize>();
    println!("datasets: {}", datasets.len());
    println!("included samples: {included}");
    println!("quantization levels: {}", args.quantization_levels);
    println!("distinct observable states: {}", states.len());
    println!("states shared across datasets: {shared_states}");
    println!("shared states with conflicting action families: {conflicting_states}");
    println!("samples in conflicting shared states: {conflicting_samples}");
}
