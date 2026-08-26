use blob_engine::resolution::{
    ActionKind, ActionRequest, BoundaryRule, CellKey, DurationRule, EffortTier, IntegrityMode,
    LocalSlot, NeighborhoodSpec, OutcomeStatus, ReferenceRuleset, ReferenceSimulation,
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;

const ATTACK_PAYLOAD: u64 = 8;
const STARTING_ENERGY: u64 = 10_000;
// Force the ordered-parallel validation path even for the smallest matrix
// case so its crossover cost remains visible rather than silently falling
// back to the serial oracle.
const PARALLEL_THRESHOLD: usize = 2;

#[derive(Debug, Clone, Copy)]
struct CombatScenario {
    width: usize,
    height: usize,
    cells: usize,
    seed: u64,
}

impl CombatScenario {
    const fn new(width: usize, height: usize, cells: usize, seed: u64) -> Self {
        Self {
            width,
            height,
            cells,
            seed,
        }
    }

    fn density_percent(self) -> f64 {
        100.0 * self.cells as f64 / (self.width * self.height) as f64
    }
}

struct CombatFixture {
    simulation: ReferenceSimulation,
    targets: Vec<(CellKey, CellKey, LocalSlot)>,
    unique_victims: usize,
    maximum_attackers_per_victim: usize,
}

/// Small deterministic generator kept local to the diagnostic so benchmark
/// topology does not depend on a changing external RNG implementation.
#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        assert!(upper > 0);
        (self.next() % upper as u64) as usize
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            values.swap(index, self.index(index + 1));
        }
    }
}

fn combat_rules() -> ReferenceRuleset {
    let duration = DurationRule::new(1024, 0, 1);
    ReferenceRuleset {
        neighborhood: NeighborhoodSpec::moore_8(BoundaryRule::Bounded),
        wait_duration: duration,
        move_duration: duration,
        attack_duration: duration,
        consume_duration: duration,
        split_duration: duration,
        regurgitate_duration: duration,
        excavate_duration: duration,
        deposit_terrain_duration: duration,
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        signal_decay_rate_numerator: 0,
        ..ReferenceRuleset::default()
    }
}

fn combat_fixture(scenario: CombatScenario) -> CombatFixture {
    assert!(scenario.width > 0 && scenario.width.is_multiple_of(2));
    assert!(scenario.height > 0);
    assert!(scenario.cells > 0 && scenario.cells.is_multiple_of(2));
    assert!(scenario.cells <= scenario.width * scenario.height);

    let mut rng = SplitMix64::new(scenario.seed);
    let mut simulation =
        ReferenceSimulation::new(scenario.width, scenario.height, combat_rules()).unwrap();

    // A shuffled horizontal-domino tiling guarantees one adjacent competitor
    // per cell at every requested density. Nearby selected dominoes create
    // additional opposing targets and naturally variable victim fan-in.
    let mut pairs = Vec::with_capacity(scenario.width * scenario.height / 2);
    for y in 0..scenario.height {
        for x in (0..scenario.width).step_by(2) {
            pairs.push((
                simulation.tile(x, y).unwrap(),
                simulation.tile(x + 1, y).unwrap(),
            ));
        }
    }
    rng.shuffle(&mut pairs);

    let mut occupants = vec![None; scenario.width * scenario.height];
    let mut actors = Vec::with_capacity(scenario.cells);
    for &(left, right) in pairs.iter().take(scenario.cells / 2) {
        let (first_team, second_team) = if rng.next() & 1 == 0 {
            (0_u32, 1_u32)
        } else {
            (1_u32, 0_u32)
        };
        for (tile, team) in [(left, first_team), (right, second_team)] {
            let actor = simulation
                .add_cell(tile, 10, STARTING_ENERGY, team)
                .unwrap();
            occupants[tile.0] = Some((actor, team));
            actors.push((actor, tile, team));
        }
    }

    let mut targets = Vec::with_capacity(actors.len());
    let mut victim_counts = BTreeMap::<CellKey, usize>::new();
    for (actor, origin, team) in actors {
        let candidates = simulation
            .neighborhood()
            .targets(origin)
            .unwrap()
            .iter()
            .enumerate()
            .filter_map(|(slot, target)| {
                let target = (*target)?;
                let (victim, victim_team) = occupants[target.0]?;
                (victim_team != team).then_some((LocalSlot(slot as u8), victim))
            })
            .collect::<Vec<_>>();
        assert!(
            !candidates.is_empty(),
            "paired placement must give every actor an adjacent competitor"
        );
        let (slot, victim) = candidates[rng.index(candidates.len())];
        let receipt = simulation
            .commit_action(
                actor,
                ActionRequest::Attack {
                    target: slot,
                    payload: ATTACK_PAYLOAD,
                    effort: EffortTier::Standard,
                },
            )
            .unwrap();
        assert!(receipt.accepted);
        targets.push((actor, victim, slot));
        *victim_counts.entry(victim).or_default() += 1;
    }

    CombatFixture {
        simulation,
        targets,
        unique_victims: victim_counts.len(),
        maximum_attackers_per_victim: victim_counts.values().copied().max().unwrap_or(0),
    }
}

#[test]
fn randomized_combat_is_local_competitive_and_reproducible() {
    let scenario = CombatScenario::new(16, 16, 96, 0x5eed_c0de);
    let first = combat_fixture(scenario);
    let second = combat_fixture(scenario);

    assert_eq!(first.targets, second.targets);
    assert_eq!(
        first.simulation.canonical_state(),
        second.simulation.canonical_state()
    );
    assert_eq!(first.targets.len(), scenario.cells);
    assert!(first.unique_victims > 0);
    assert!(first.maximum_attackers_per_victim > 0);

    for &(actor, victim, slot) in &first.targets {
        let actor_state = first.simulation.cell(actor).unwrap();
        let victim_state = first.simulation.cell(victim).unwrap();
        assert_ne!(actor_state.marker, victim_state.marker);
        assert_eq!(
            first
                .simulation
                .neighborhood()
                .target(actor_state.position, slot),
            Some(victim_state.position)
        );
    }

    let mut first_simulation = first.simulation;
    let mut second_simulation = second.simulation;
    let first_report = first_simulation.resolve_next_batch().unwrap();
    let second_report = second_simulation.resolve_next_batch().unwrap();
    assert_eq!(first_report, second_report);
    assert_eq!(
        first_simulation.canonical_state(),
        second_simulation.canonical_state()
    );
    assert_eq!(first_report.outcomes.len(), scenario.cells);
    assert!(first_report.deaths.is_empty());
    assert!(first_report.outcomes.iter().all(|outcome| {
        outcome.action == ActionKind::Attack && outcome.status == OutcomeStatus::Success
    }));
}

#[test]
fn random_combat_matrix_is_worker_and_integrity_invariant() {
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(8);
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();

    for scenario in [
        CombatScenario::new(16, 16, 64, 11),
        CombatScenario::new(32, 32, 128, 22),
        CombatScenario::new(32, 32, 512, 33),
        CombatScenario::new(64, 64, 512, 44),
    ] {
        let fixture = combat_fixture(scenario);
        let mut verified = fixture.simulation.clone();
        let verified_report = verified.resolve_next_batch().unwrap();

        let mut trusted = fixture.simulation.clone();
        let trusted_report = trusted
            .resolve_next_batch_with_integrity(IntegrityMode::OnDemand)
            .unwrap();

        let mut parallel = fixture.simulation;
        let parallel_report = pool
            .install(|| {
                parallel.resolve_next_batch_parallel_with_integrity(
                    PARALLEL_THRESHOLD,
                    IntegrityMode::Verified,
                )
            })
            .unwrap();

        assert!(verified_report.is_verified());
        assert!(!trusted_report.is_verified());
        assert_eq!(verified_report.outcomes, trusted_report.outcomes);
        assert_eq!(verified_report.delta, trusted_report.delta);
        assert_eq!(parallel_report, verified_report);
        assert_eq!(trusted.canonical_state(), verified.canonical_state());
        assert_eq!(parallel.canonical_state(), verified.canonical_state());
        assert_eq!(verified_report.outcomes.len(), scenario.cells);
    }
}

#[derive(Clone, Copy)]
enum Executor {
    Serial,
    Parallel,
}

impl Executor {
    const fn label(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Parallel => "parallel",
        }
    }
}

fn resolve_fixture(
    simulation: &mut ReferenceSimulation,
    executor: Executor,
    integrity: IntegrityMode,
    pool: &ThreadPool,
) {
    let report = match executor {
        Executor::Serial => simulation.resolve_next_batch_with_integrity(integrity),
        Executor::Parallel => pool.install(|| {
            simulation.resolve_next_batch_parallel_with_integrity(PARALLEL_THRESHOLD, integrity)
        }),
    }
    .unwrap();
    assert_eq!(report.outcomes.len(), simulation.cells().len());
    black_box(report);
}

fn percentile(values: &mut [u64], numerator: usize, denominator: usize) -> u64 {
    values.sort_unstable();
    let rank = values
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .max(1);
    let index = rank.saturating_sub(1).min(values.len().saturating_sub(1));
    values[index]
}

#[cfg(debug_assertions)]
fn require_release_benchmark() {
    panic!("run this diagnostic with cargo test --release");
}

#[cfg(not(debug_assertions))]
fn require_release_benchmark() {}

/// Resolver-only combat diagnostic for measuring changes such as authoritative
/// damage attribution. Run in release mode; debug builds retain expensive
/// semantic oracles by design.
#[test]
#[ignore = "manual release-mode random-neighbor combat throughput diagnostic"]
fn benchmark_random_neighbor_combat_matrix() {
    require_release_benchmark();
    let samples = std::env::var("BLOB_COMBAT_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(9)
        .max(3);
    let seeds = std::env::var("BLOB_COMBAT_BENCH_SEEDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1);
    let workers = std::env::var("BLOB_COMBAT_BENCH_WORKERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1)
                .min(8)
        })
        .max(1);
    let pool = ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap();
    let matrix = [
        CombatScenario::new(16, 16, 64, 0),
        CombatScenario::new(32, 32, 128, 0),
        CombatScenario::new(32, 32, 512, 0),
        CombatScenario::new(64, 64, 512, 0),
        CombatScenario::new(64, 64, 2_048, 0),
        CombatScenario::new(128, 128, 2_048, 0),
        CombatScenario::new(128, 128, 8_192, 0),
    ];

    println!(
        "executor,integrity,workers,board,cells,density_pct,seeds,samples,unique_victim_pct,max_fan_in,median_ns_per_attack,p95_ns_per_attack,median_actions_per_second,median_resolution_ns,median_report_ns,median_hash_ns,median_total_ns"
    );
    for base in matrix {
        for integrity in [IntegrityMode::OnDemand, IntegrityMode::Verified] {
            for executor in [Executor::Serial, Executor::Parallel] {
                let mut total_samples = Vec::with_capacity(samples * seeds);
                let mut resolution_samples = Vec::with_capacity(samples * seeds);
                let mut report_samples = Vec::with_capacity(samples * seeds);
                let mut hash_samples = Vec::with_capacity(samples * seeds);
                let mut victim_percent_total = 0.0;
                let mut maximum_fan_in = 0;

                for seed_index in 0..seeds {
                    let scenario = CombatScenario {
                        seed: 0xc0ba_7000_u64
                            .wrapping_add(base.width as u64 * 1_000_003)
                            .wrapping_add(base.cells as u64 * 97)
                            .wrapping_add(seed_index as u64),
                        ..base
                    };
                    let fixture = combat_fixture(scenario);
                    victim_percent_total +=
                        100.0 * fixture.unique_victims as f64 / scenario.cells as f64;
                    maximum_fan_in = maximum_fan_in.max(fixture.maximum_attackers_per_victim);

                    let mut warmup = fixture.simulation.clone();
                    resolve_fixture(&mut warmup, executor, integrity, &pool);
                    for _ in 0..samples {
                        // Fixture construction and cloning are intentionally
                        // outside the resolver's internal phase timer.
                        let mut simulation = fixture.simulation.clone();
                        resolve_fixture(&mut simulation, executor, integrity, &pool);
                        let timings = simulation.last_resolution_metrics().phase_timings;
                        total_samples.push(timings.total_ns);
                        resolution_samples.push(timings.canonical_resolution_ns);
                        report_samples.push(timings.report_finalization_ns);
                        hash_samples.push(timings.state_hashes_ns);
                    }
                }

                let median_total = percentile(&mut total_samples, 1, 2);
                let p95_total = percentile(&mut total_samples.clone(), 95, 100);
                let median_resolution = percentile(&mut resolution_samples, 1, 2);
                let median_report = percentile(&mut report_samples, 1, 2);
                let median_hash = percentile(&mut hash_samples, 1, 2);
                let median_ns_per_attack = median_total as f64 / base.cells as f64;
                let p95_ns_per_attack = p95_total as f64 / base.cells as f64;
                let actions_per_second =
                    base.cells as f64 * 1_000_000_000.0 / median_total.max(1) as f64;
                let integrity_label = match integrity {
                    IntegrityMode::OnDemand => "on_demand",
                    IntegrityMode::Verified => "verified",
                };
                println!(
                    "{},{},{},{}x{},{},{:.2},{},{},{:.2},{},{:.2},{:.2},{:.0},{},{},{},{}",
                    executor.label(),
                    integrity_label,
                    workers,
                    base.width,
                    base.height,
                    base.cells,
                    base.density_percent(),
                    seeds,
                    samples,
                    victim_percent_total / seeds as f64,
                    maximum_fan_in,
                    median_ns_per_attack,
                    p95_ns_per_attack,
                    actions_per_second,
                    median_resolution,
                    median_report,
                    median_hash,
                    median_total,
                );
            }
        }
    }
}

#[test]
fn changing_the_seed_changes_the_combat_graph() {
    let first = combat_fixture(CombatScenario::new(32, 32, 256, 1));
    let second = combat_fixture(CombatScenario::new(32, 32, 256, 2));
    let first_edges = first
        .targets
        .iter()
        .map(|(actor, victim, _)| (*actor, *victim))
        .collect::<BTreeSet<_>>();
    let second_edges = second
        .targets
        .iter()
        .map(|(actor, victim, _)| (*actor, *victim))
        .collect::<BTreeSet<_>>();
    assert_ne!(first_edges, second_edges);
}
