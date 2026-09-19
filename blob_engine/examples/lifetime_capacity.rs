//! Cold restore and steady event costs for bounded live cells after long history.
//! Canonical high-water fixtures model prior churn; they do not time its creation.

use std::time::Instant;

use blob_engine::resolution::{
    ActionRequest, CellKey, ReferenceRuleset, ReferenceSimulation, TileIndex,
};

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(
        (1..=3).contains(&args.len()),
        "usage: lifetime_capacity <historical-key-count> [batches] [idle|active]"
    );
    let historical_keys: u64 = args[0].parse().expect("invalid historical key count");
    let batches: usize = args
        .get(1)
        .map_or(1000, |value| value.parse().expect("invalid batch count"));
    assert!((8..=4_000_000).contains(&historical_keys));
    assert!(batches > 0);
    let active = match args.get(2).map(String::as_str) {
        None | Some("idle") => false,
        Some("active") => true,
        Some(_) => panic!("expected idle or active metabolism"),
    };
    let rules = ReferenceRuleset {
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: u64::from(active),
        signal_decay_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    };
    let mut initial = ReferenceSimulation::new(8, 1, rules.clone()).unwrap();
    for index in 0..8 {
        initial
            .add_cell(TileIndex(index), 20, 1_000_000, index as u32)
            .unwrap();
    }
    let mut state = initial.canonical_state();
    drop(initial);
    let keys: Vec<_> = (0..8)
        .map(|index| CellKey((historical_keys - 1) * index / 7))
        .collect();
    for (index, key) in keys.iter().enumerate() {
        state.cells[index].0 = *key;
        state.tiles[index].occupant = Some(*key);
    }
    state.next_cell_key = historical_keys;
    let started = Instant::now();
    let mut simulation = ReferenceSimulation::from_canonical_state(8, 1, rules, state).unwrap();
    // Include the first commitment in cold-start work even if a cache is lazy.
    let initial_hash = simulation.state_hash();
    let restore_ns = started.elapsed().as_nanos();
    let total = simulation.total_energy_equivalent();
    let started = Instant::now();
    let mut completed = 0;
    for _ in 0..batches {
        for key in &keys {
            assert!(
                simulation
                    .commit_action(*key, ActionRequest::Wait)
                    .unwrap()
                    .accepted
            );
        }
        let report = simulation.resolve_next_batch().unwrap();
        completed += report.outcomes.len();
    }
    let event_ns = started.elapsed().as_nanos();
    assert_eq!(completed, batches * keys.len());
    assert_eq!(simulation.total_energy_equivalent(), total);
    let canonical = simulation.canonical_state();
    let final_hash = simulation.state_hash();
    assert_eq!(
        final_hash,
        canonical.hash_with_compiled_ruleset(simulation.compiled_ruleset_hash())
    );
    assert_eq!(canonical.next_cell_key, historical_keys);
    assert_eq!(canonical.cells.len(), keys.len());
    println!(
        "{{\"historicalKeys\":{historical_keys},\"activeMetabolism\":{active},\"liveCells\":8,\"batches\":{batches},\"completed\":{completed},\"restoreNs\":{restore_ns},\"eventNs\":{event_ns},\"initialHash\":\"{initial_hash}\",\"finalHash\":\"{final_hash}\"}}"
    );
}
