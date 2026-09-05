//! Rank anonymous local observation features by Guard/Move separability.

use std::collections::BTreeMap;
use std::path::PathBuf;

use blob_rl::action::{policy_action_family, PolicyActionFamily};
use blob_rl::behavior_cloning::{
    expert_routing_phase, load_dataset_directories, ExpertRoutingStrategy, SupervisionPhase,
};
use blob_rl::observation::{HEADER_FEATURES, OBS_RANDOMNESS_FEATURE_START, SLOT_FEATURES};
use clap::Parser;

const QUANTIZATION_LEVELS: f32 = 256.0;

#[derive(Debug, Parser)]
#[command(
    name = "guard-readiness-diagnostics",
    about = "Rank local non-random features separating exploration Guard and Move labels"
)]
struct Args {
    /// Immutable demonstration directories.
    #[arg(long, required = true)]
    dataset: Vec<PathBuf>,

    /// Number of highest-separability features to print.
    #[arg(long, default_value_t = 24)]
    top: usize,

    /// Include only exact decoder round trips.
    #[arg(long)]
    exact_round_trip_only: bool,

    /// Print quantized class counts for these feature names.
    #[arg(long, value_delimiter = ',', default_value = "header[1],header[2]")]
    detail_feature: Vec<String>,
}

#[derive(Clone, Default)]
struct FeatureCounts {
    bins: BTreeMap<i32, [usize; 2]>,
}

struct RankedFeature {
    name: String,
    balanced_accuracy: f64,
    bins: usize,
    guard_bins: usize,
    move_bins: usize,
    conflicting_bins: usize,
}

fn quantize(value: f32) -> i32 {
    (value * QUANTIZATION_LEVELS).round() as i32
}

fn feature_values(observation: &[f32]) -> Vec<(String, f32)> {
    let mut values = (0..OBS_RANDOMNESS_FEATURE_START)
        .map(|index| (format!("header[{index}]"), observation[index]))
        .collect::<Vec<_>>();
    let slots = observation[HEADER_FEATURES..].chunks_exact(SLOT_FEATURES);
    for feature in 0..SLOT_FEATURES {
        let feature_values = slots.clone().map(|slot| slot[feature]).collect::<Vec<_>>();
        let maximum = feature_values.iter().copied().fold(f32::MIN, f32::max);
        let minimum = feature_values.iter().copied().fold(f32::MAX, f32::min);
        let mean = feature_values.iter().copied().sum::<f32>() / feature_values.len() as f32;
        values.push((format!("slot_max[{feature}]"), maximum));
        values.push((format!("slot_min[{feature}]"), minimum));
        values.push((format!("slot_mean[{feature}]"), mean));
    }
    values
}

fn rank_feature(name: String, counts: FeatureCounts, totals: [usize; 2]) -> RankedFeature {
    let mut correct = [0usize; 2];
    let mut guard_bins = 0usize;
    let mut move_bins = 0usize;
    let mut conflicting_bins = 0usize;
    for bin in counts.bins.values() {
        guard_bins += usize::from(bin[0] > 0);
        move_bins += usize::from(bin[1] > 0);
        conflicting_bins += usize::from(bin[0] > 0 && bin[1] > 0);
        let guard_rate = bin[0] as f64 / totals[0] as f64;
        let move_rate = bin[1] as f64 / totals[1] as f64;
        if guard_rate >= move_rate {
            correct[0] += bin[0];
        } else {
            correct[1] += bin[1];
        }
    }
    RankedFeature {
        name,
        balanced_accuracy: 0.5
            * (correct[0] as f64 / totals[0] as f64 + correct[1] as f64 / totals[1] as f64),
        bins: counts.bins.len(),
        guard_bins,
        move_bins,
        conflicting_bins,
    }
}

fn main() {
    let args = Args::parse();
    if args.top == 0 {
        panic!("--top must be positive");
    }
    let datasets = load_dataset_directories(&args.dataset)
        .unwrap_or_else(|error| panic!("failed to load demonstrations: {error}"));
    let mut feature_counts = BTreeMap::<String, FeatureCounts>::new();
    let mut totals = [0usize; 2];
    for sample in datasets.iter().flat_map(|dataset| &dataset.payload.samples) {
        if args.exact_round_trip_only && !sample.exact_round_trip {
            continue;
        }
        if expert_routing_phase(
            sample,
            ExpertRoutingStrategy::ForagingInteractionExploration,
        ) != SupervisionPhase::Exploration
        {
            continue;
        }
        let class = match policy_action_family(usize::from(sample.action)) {
            Some(PolicyActionFamily::Guard) => 0,
            Some(PolicyActionFamily::Move) => 1,
            _ => continue,
        };
        totals[class] += 1;
        for (name, value) in feature_values(&sample.observation) {
            feature_counts
                .entry(name)
                .or_default()
                .bins
                .entry(quantize(value))
                .or_default()[class] += 1;
        }
    }
    if totals.contains(&0) {
        panic!(
            "diagnostic requires both exploration Guard and Move labels; found guard={} move={}",
            totals[0], totals[1]
        );
    }
    for name in &args.detail_feature {
        let counts = feature_counts
            .get(name)
            .unwrap_or_else(|| panic!("unknown --detail-feature {name}"));
        println!("detail {name}: quantized_value guard move");
        for (value, counts) in &counts.bins {
            println!("  {value:>5} {:>5} {:>5}", counts[0], counts[1]);
        }
    }
    let mut ranked = feature_counts
        .into_iter()
        .map(|(name, counts)| rank_feature(name, counts, totals))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .balanced_accuracy
            .total_cmp(&left.balanced_accuracy)
            .then_with(|| left.name.cmp(&right.name))
    });
    println!(
        "exploration labels: guard={} move={} quantization_levels={}",
        totals[0], totals[1], QUANTIZATION_LEVELS as usize
    );
    for feature in ranked.into_iter().take(args.top) {
        println!(
            "{:<20} balanced_accuracy={:.4} bins={} guard_bins={} move_bins={} conflicts={}",
            feature.name,
            feature.balanced_accuracy,
            feature.bins,
            feature.guard_bins,
            feature.move_bins,
            feature.conflicting_bins,
        );
    }
}
