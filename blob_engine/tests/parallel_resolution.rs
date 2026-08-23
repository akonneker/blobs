use blob_engine::resolution::{
    ActionRequest, BoundaryRule, DurationRule, EffortTier, LocalSlot, NeighborhoodSpec,
    OutcomeStatus, ReferenceRuleset, ReferenceSimulation,
};
use rayon::ThreadPoolBuilder;
use std::time::Instant;

const WEST: LocalSlot = LocalSlot(3);
const EAST: LocalSlot = LocalSlot(4);

fn collision_fixture(rows: usize, groups_per_row: usize) -> ReferenceSimulation {
    let duration = DurationRule::new(1024, 0, 1);
    let rules = ReferenceRuleset {
        neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
        wait_duration: duration,
        move_duration: duration,
        attack_duration: duration,
        consume_duration: duration,
        split_duration: duration,
        regurgitate_duration: duration,
        excavate_duration: duration,
        deposit_terrain_duration: duration,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    };
    let mut simulation = ReferenceSimulation::new(groups_per_row * 4, rows, rules).unwrap();
    for y in 0..rows {
        for group in 0..groups_per_row {
            let left = simulation
                .add_cell(simulation.tile(group * 4, y).unwrap(), 10, 100, 0)
                .unwrap();
            let right = simulation
                .add_cell(simulation.tile(group * 4 + 2, y).unwrap(), 10, 100, 0)
                .unwrap();
            simulation
                .commit_action(
                    left,
                    ActionRequest::Move {
                        target: EAST,
                        effort: EffortTier::Standard,
                    },
                )
                .unwrap();
            simulation
                .commit_action(
                    right,
                    ActionRequest::Move {
                        target: WEST,
                        effort: EffortTier::Standard,
                    },
                )
                .unwrap();
        }
    }
    simulation
}

#[test]
fn serial_and_varied_worker_counts_produce_identical_batches() {
    let fixture = collision_fixture(8, 10);
    let mut serial = fixture.clone();
    let serial_report = serial.resolve_next_batch().unwrap();

    for workers in [1, 2, 4] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let mut parallel = fixture.clone();
        let report = pool
            .install(|| parallel.resolve_next_batch_parallel(2))
            .unwrap();

        assert_eq!(report, serial_report);
        assert_eq!(parallel.canonical_state(), serial.canonical_state());
        let metrics = parallel.last_resolution_metrics();
        assert_eq!(metrics.due_intents, 160);
        assert_eq!(metrics.emitted_claims, 160);
        assert_eq!(metrics.exclusive_components, 80);
        assert_eq!(metrics.contended_components, 80);
        assert_eq!(metrics.maximum_component_size, 2);
        assert_eq!(metrics.parallel_validation, workers > 1);
        assert!(!metrics.sparse_occupancy_claims);
    }
}

#[test]
fn sparse_world_uses_claim_proportional_occupancy_storage() {
    let duration = DurationRule::new(1024, 0, 1);
    let rules = ReferenceRuleset {
        neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
        wait_duration: duration,
        move_duration: duration,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    };
    let mut simulation = ReferenceSimulation::new(128, 128, rules).unwrap();
    let first = simulation
        .add_cell(simulation.tile(1, 1).unwrap(), 10, 100, 0)
        .unwrap();
    let second = simulation
        .add_cell(simulation.tile(100, 100).unwrap(), 10, 100, 0)
        .unwrap();
    for actor in [first, second] {
        simulation
            .commit_action(
                actor,
                ActionRequest::Move {
                    target: EAST,
                    effort: EffortTier::Standard,
                },
            )
            .unwrap();
    }

    let report = simulation.resolve_next_batch().unwrap();
    assert!(report
        .outcomes
        .iter()
        .all(|outcome| outcome.status == OutcomeStatus::Success));
    let metrics = simulation.last_resolution_metrics();
    assert_eq!(metrics.due_intents, 2);
    assert!(metrics.sparse_occupancy_claims);
}

#[test]
#[ignore = "manual throughput diagnostic"]
fn benchmark_large_collision_batch() {
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(8);
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();

    for (rows, groups) in [(4, 10), (5, 50), (10, 50), (50, 50)] {
        let fixture = collision_fixture(rows, groups);
        let intents = rows * groups * 2;
        let started = Instant::now();
        let mut serial = fixture.clone();
        let serial_report = serial.resolve_next_batch().unwrap();
        let serial_elapsed = started.elapsed();

        let started = Instant::now();
        let mut parallel = fixture;
        let parallel_report = pool
            .install(|| parallel.resolve_next_batch_parallel(2))
            .unwrap();
        let parallel_elapsed = started.elapsed();

        assert_eq!(parallel_report, serial_report);
        eprintln!(
            "{intents} intents: serial={serial_elapsed:?}, parallel({workers})={parallel_elapsed:?}"
        );
    }
}
