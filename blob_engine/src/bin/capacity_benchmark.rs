use std::time::Instant;

use blob_engine::engine::{CellConfig, Engine, ReferenceHostMode};
use blob_engine::resolution::{IntegrityMode, ReferenceRuleset};
use blob_interface::reference_mind::{
    ReferenceMemoryUpdate, ReferenceMind, ReferenceMindAction, ReferenceMindDecision,
    ReferenceMindInput,
};
use blob_interface::types::TeamId;

#[derive(Debug, Clone, Copy, Default)]
enum MemoryMode {
    #[default]
    Retain,
    Replace,
}

#[derive(Default)]
struct WaitMind {
    memory_mode: MemoryMode,
}

impl ReferenceMind for WaitMind {
    fn decide(&mut self, input: &ReferenceMindInput) -> ReferenceMindDecision {
        ReferenceMindDecision {
            action: ReferenceMindAction::Wait,
            signal: None,
            memory_update: match self.memory_mode {
                MemoryMode::Retain => ReferenceMemoryUpdate::Retain,
                MemoryMode::Replace => ReferenceMemoryUpdate::Replace(input.private_memory.clone()),
            },
        }
    }

    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }
}

fn run_case(
    board: usize,
    requested_cells: usize,
    mind_workers: usize,
    integrity_mode: IntegrityMode,
    memory_mode: MemoryMode,
    host_mode: ReferenceHostMode,
) {
    let ticks = (200_000 / requested_cells.max(1)).clamp(20, 200);
    let rules = ReferenceRuleset {
        digestion_rate_numerator: 0,
        metabolism_rate_numerator: 0,
        signal_decay_rate_numerator: 0,
        diffusion_rate_numerator: 0,
        ..ReferenceRuleset::default()
    };
    let config = CellConfig {
        initial_energy: 100,
        starting_cells_per_team: requested_cells,
        ..CellConfig::default()
    };
    let mut engine = Engine::new(
        board,
        board,
        u64::try_from(ticks).unwrap(),
        config,
        Some(7),
        rules,
    );
    engine.set_reference_integrity_mode(integrity_mode).unwrap();
    engine.set_reference_host_mode(host_mode).unwrap();
    engine
        .add_team_with_minds(
            TeamId(0),
            (0..mind_workers.max(1))
                .map(|_| WaitMind { memory_mode })
                .collect(),
        )
        .unwrap();
    engine.initialize_reference_state().unwrap();
    let cells = engine.cells.len();

    let started = Instant::now();
    let mut committed = 0_usize;
    let mut completed = 0_usize;
    let mut parallel_batches = 0_usize;
    let mut parallel_observation_batches = 0_usize;
    let mut phase_ns = [0_u128; 9];
    let mut host_ns = [0_u128; 10];
    for _ in 0..ticks {
        let event = engine.tick(false).unwrap();
        committed += event.reference_commitments.len();
        completed += event
            .reference_batch
            .as_ref()
            .map_or(0, |report| report.outcomes.len());
        let metrics = engine
            .reference_simulation()
            .unwrap()
            .last_resolution_metrics();
        parallel_batches += usize::from(metrics.parallel_validation);
        let timings = metrics.phase_timings;
        for (total, value) in phase_ns.iter_mut().zip([
            timings.passive_updates_ns,
            timings.prestate_materialization_ns,
            timings.intent_validation_ns,
            timings.canonical_resolution_ns,
            timings.report_finalization_ns,
            timings.state_hashes_ns,
            timings.poststate_materialization_ns,
            timings.delta_generation_ns,
            timings.total_ns,
        ]) {
            *total += u128::from(value);
        }
        let host = engine.last_reference_host_timings();
        parallel_observation_batches += usize::from(host.parallel_observation);
        for (total, value) in host_ns.iter_mut().zip([
            host.ready_frontier_ns,
            host.observation_index_ns,
            host.observation_projection_ns,
            host.randomness_derivation_ns,
            host.mind_execution_ns,
            host.action_commit_ns,
            host.passive_invalidation_ns,
            host.replay_recording_ns,
            host.host_projection_ns,
            host.total_ns,
        ]) {
            *total += u128::from(value);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    assert_eq!(committed, completed);
    println!(
        "{board:>4}x{board:<4} {cells:>7} cells {ticks:>4} batches {mind_workers:>2} minds {integrity_mode:?}/{host_mode:?}  \
         {memory_mode:?} {elapsed:>7.3}s  {:>10.0} actions/s  {:>8.2} ms/batch  \
         resolver-parallel={parallel_batches}/{ticks} observe-parallel={parallel_observation_batches}/{ticks}",
        completed as f64 / elapsed,
        elapsed * 1_000.0 / ticks as f64,
    );
    let measured_ns = (elapsed * 1_000_000_000.0) as u128;
    let percent = |value: u128| 100.0 * value as f64 / measured_ns.max(1) as f64;
    println!(
        "                 phases: passive={:>4.1}% preclone={:>4.1}% validate={:>4.1}% \
         resolve={:>4.1}% finalize={:>4.1}% hashes={:>4.1}% \
         postclone={:>4.1}% delta={:>4.1}% outside={:>4.1}%",
        percent(phase_ns[0]),
        percent(phase_ns[1]),
        percent(phase_ns[2]),
        percent(phase_ns[3]),
        percent(phase_ns[4]),
        percent(phase_ns[5]),
        percent(phase_ns[6]),
        percent(phase_ns[7]),
        percent(measured_ns.saturating_sub(phase_ns[8])),
    );
    println!(
        "                 host: ready={:>4.1}% obs-index={:>4.1}% observe={:>4.1}% random={:>4.1}% minds={:>4.1}% \
         commit={:>4.1}% invalid={:>4.1}% replay={:>4.1}% projection={:>4.1}% other={:>4.1}%",
        percent(host_ns[0]),
        percent(host_ns[1]),
        percent(host_ns[2]),
        percent(host_ns[3]),
        percent(host_ns[4]),
        percent(host_ns[5]),
        percent(host_ns[6]),
        percent(host_ns[7]),
        percent(host_ns[8]),
        percent(
            host_ns[9]
                .saturating_sub(phase_ns[8])
                .saturating_sub(host_ns[..9].iter().sum()),
        ),
    );
}

fn main() {
    println!("Stable native end-to-end capacity");
    println!("(local Mind projection + decision + commit + physics + optional hashes + deltas + host sync)");
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let integrity_mode = match args.get(3).map(String::as_str) {
        None | Some("verified") => IntegrityMode::Verified,
        Some("on-demand") => IntegrityMode::OnDemand,
        Some(other) => panic!("unknown integrity mode {other:?}; expected verified or on-demand"),
    };
    let memory_mode = match args.get(4).map(String::as_str) {
        None | Some("retain") => MemoryMode::Retain,
        Some("replace") => MemoryMode::Replace,
        Some(other) => panic!("unknown memory mode {other:?}; expected retain or replace"),
    };
    let host_mode = match args.get(5).map(String::as_str) {
        None | Some("projected") => ReferenceHostMode::Projected,
        Some("metadata-only") => ReferenceHostMode::MetadataOnly,
        Some(other) => {
            panic!("unknown host mode {other:?}; expected projected or metadata-only")
        }
    };
    let cases = if (2..=6).contains(&args.len()) {
        vec![(
            args[0].parse().unwrap(),
            args[1].parse().unwrap(),
            args.get(2).map_or(1, |workers| workers.parse().unwrap()),
        )]
    } else {
        vec![
            (32, 100, 1),
            (32, 500, 1),
            (64, 100, 1),
            (64, 1_000, 1),
            (64, 3_000, 1),
            (128, 1_000, 1),
            (128, 5_000, 1),
            (128, 10_000, 1),
            (256, 1_000, 1),
            (256, 10_000, 1),
            (256, 30_000, 1),
            // Large-world qualification is part of the default performance
            // envelope, not an opt-in afterthought. The sparse 1024 case
            // exposes board-wide materialization cost; the density-matched
            // case measures frontier throughput.
            (512, 8_192, 8),
            (1_024, 2_048, 8),
            (1_024, 32_768, 8),
        ]
    };
    for (board, cells, workers) in cases {
        run_case(
            board,
            cells,
            workers,
            integrity_mode,
            memory_mode,
            host_mode,
        );
    }
}
