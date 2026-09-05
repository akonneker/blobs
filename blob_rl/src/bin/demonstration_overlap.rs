//! Diagnose contradictory action-family labels for identical observable state.

use std::collections::HashMap;
use std::path::PathBuf;

use blob_rl::action::{policy_action_family, PolicyActionFamily};
use blob_rl::behavior_cloning::{
    expert_routing_phase, load_dataset_directories, ExpertRoutingStrategy, SupervisionPhase,
};
use blob_rl::observation::{OBS_RANDOMNESS_FEATURE_END, OBS_RANDOMNESS_FEATURE_START};
use clap::{Parser, ValueEnum};
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

    /// Report the expert labels that this strictly local router would assign.
    #[arg(long, value_enum, default_value_t = Routing::VisibleNeighborContext)]
    expert_routing: Routing,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Routing {
    LocalActionFamily,
    VisibleNeighborContext,
    ForagingInteractionExploration,
}

impl From<Routing> for ExpertRoutingStrategy {
    fn from(value: Routing) -> Self {
        match value {
            Routing::LocalActionFamily => Self::LocalActionFamily,
            Routing::VisibleNeighborContext => Self::VisibleNeighborContext,
            Routing::ForagingInteractionExploration => Self::ForagingInteractionExploration,
        }
    }
}

#[derive(Default)]
struct StateLabels {
    datasets: u128,
    families: u16,
    targets: u32,
    samples: usize,
}

fn main() {
    let args = Args::parse();
    if args.dataset.len() > 128 {
        panic!("overlap analysis supports at most 128 datasets");
    }
    let datasets = load_dataset_directories(&args.dataset)
        .unwrap_or_else(|error| panic!("failed to load demonstrations: {error}"));
    let expert_routing = args.expert_routing.into();
    let mut states = HashMap::<[u8; 32], StateLabels>::new();
    let mut within_dataset_families = HashMap::<(usize, [u8; 32]), (u16, usize)>::new();
    let mut routing_states = HashMap::<[u8; 32], u8>::new();
    let mut foraging_adapter_states = HashMap::<[u8; 32], StateLabels>::new();
    let mut dataset_phases = vec![[0usize; 3]; datasets.len()];
    let mut phase_samples = [0usize; 3];
    let mut phase_family_labels = [[0usize; PolicyActionFamily::COUNT]; 3];
    let mut phase_family_legal_observations = [[0usize; PolicyActionFamily::COUNT]; 3];
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
            let routing_key: [u8; 32] = hasher.clone().finalize().into();
            for allowed in &sample.action_mask {
                hasher.update([u8::from(*allowed)]);
            }
            let key: [u8; 32] = hasher.finalize().into();
            let family = policy_action_family(usize::from(sample.action))
                .expect("validated demonstration action is in the catalog");
            let decomposition =
                blob_rl::action::decompose_policy_action(usize::from(sample.action))
                    .expect("validated demonstration action decomposes");
            let mut adapter_hasher = Sha256::new();
            for index in [1usize, 2, 10, 11, 12, 15, 19, 20] {
                adapter_hasher.update(sample.observation[index].to_bits().to_le_bytes());
            }
            let adapter_key: [u8; 32] = adapter_hasher.finalize().into();
            let adapter_state = foraging_adapter_states.entry(adapter_key).or_default();
            adapter_state.datasets |= 1_u128 << dataset_index;
            adapter_state.families |= 1_u16 << family.index();
            if blob_rl::action::PolicyActionKind::from_index(decomposition.kind)
                .is_some_and(|kind| kind.uses_target())
            {
                adapter_state.targets |= 1_u32 << decomposition.target;
            }
            adapter_state.samples += 1;
            let phase = expert_routing_phase(sample, expert_routing);
            dataset_phases[dataset_index][phase.index()] += 1;
            phase_samples[phase.index()] += 1;
            phase_family_labels[phase.index()][family.index()] += 1;
            for legal_family in PolicyActionFamily::ALL {
                let legal = sample
                    .action_mask
                    .iter()
                    .enumerate()
                    .any(|(action, allowed)| {
                        *allowed && policy_action_family(action) == Some(legal_family)
                    });
                phase_family_legal_observations[phase.index()][legal_family.index()] +=
                    usize::from(legal);
            }
            *routing_states.entry(routing_key).or_default() |= 1_u8 << phase.index();
            let state = states.entry(key).or_default();
            state.datasets |= 1_u128 << dataset_index;
            state.families |= 1_u16 << family.index();
            if blob_rl::action::PolicyActionKind::from_index(decomposition.kind)
                .is_some_and(|kind| kind.uses_target())
            {
                state.targets |= 1_u32 << decomposition.target;
            }
            state.samples += 1;
            let within = within_dataset_families
                .entry((dataset_index, key))
                .or_insert((0, 0));
            within.0 |= 1_u16 << family.index();
            within.1 += 1;
        }
    }
    let all_conflicting_states = states
        .values()
        .filter(|state| state.families.count_ones() > 1)
        .count();
    let all_conflicting_samples = states
        .values()
        .filter(|state| state.families.count_ones() > 1)
        .map(|state| state.samples)
        .sum::<usize>();
    let within_dataset_conflicting_states = within_dataset_families
        .values()
        .filter(|(families, _)| families.count_ones() > 1)
        .count();
    let within_dataset_conflicting_samples = within_dataset_families
        .values()
        .filter(|(families, _)| families.count_ones() > 1)
        .map(|(_, samples)| *samples)
        .sum::<usize>();
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
    let contradictory_route_states = routing_states
        .values()
        .filter(|phases| phases.count_ones() > 1)
        .count();
    println!("datasets: {}", datasets.len());
    println!("expert routing: {expert_routing:?}");
    println!("included samples: {included}");
    println!("quantization levels: {}", args.quantization_levels);
    println!("distinct observable states: {}", states.len());
    println!("distinct routing observations: {}", routing_states.len());
    println!(
        "distinct raw foraging-adapter states: {}",
        foraging_adapter_states.len()
    );
    println!(
        "raw foraging-adapter states with conflicting families: {}",
        foraging_adapter_states
            .values()
            .filter(|state| state.families.count_ones() > 1)
            .count()
    );
    println!(
        "samples in conflicting raw foraging-adapter states: {}",
        foraging_adapter_states
            .values()
            .filter(|state| state.families.count_ones() > 1)
            .map(|state| state.samples)
            .sum::<usize>()
    );
    println!("all states with conflicting action families: {all_conflicting_states}");
    println!("samples in any conflicting state: {all_conflicting_samples}");
    println!(
        "exact states with conflicting target labels: {}",
        states
            .values()
            .filter(|state| state.targets.count_ones() > 1)
            .count()
    );
    println!(
        "samples in exact target-conflicting states: {}",
        states
            .values()
            .filter(|state| state.targets.count_ones() > 1)
            .map(|state| state.samples)
            .sum::<usize>()
    );
    println!("within-dataset conflicting state entries: {within_dataset_conflicting_states}");
    println!("samples in within-dataset conflicting states: {within_dataset_conflicting_samples}");
    println!("states shared across datasets: {shared_states}");
    println!("shared states with conflicting action families: {conflicting_states}");
    println!("samples in conflicting shared states: {conflicting_samples}");
    println!("states with contradictory expert routes: {contradictory_route_states}");
    for phase in SupervisionPhase::ALL {
        println!(
            "context {}: samples={}",
            phase.name(),
            phase_samples[phase.index()]
        );
        for family in PolicyActionFamily::ALL {
            println!(
                "  {}: labels={} legal_observations={}",
                family.name(),
                phase_family_labels[phase.index()][family.index()],
                phase_family_legal_observations[phase.index()][family.index()],
            );
        }
    }
    for (dataset, phases) in datasets.iter().zip(dataset_phases) {
        println!(
            "routing {}: {}={} {}={} {}={}",
            dataset.directory.display(),
            SupervisionPhase::Feeding.name(),
            phases[SupervisionPhase::Feeding.index()],
            SupervisionPhase::Combat.name(),
            phases[SupervisionPhase::Combat.index()],
            SupervisionPhase::Exploration.name(),
            phases[SupervisionPhase::Exploration.index()],
        );
    }
}
