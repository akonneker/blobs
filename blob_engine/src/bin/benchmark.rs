use blob_engine::engine::{CellConfig, Engine};
use blob_engine::native_minds::{AggressiveMind, RandomMind};
use blob_engine::resolution::ReferenceRuleset;
use blob_engine::world_gen;
use blob_interface::types::TeamId;
use rayon::prelude::*;
use std::time::Instant;

/// Run a single engine to completion and return (iterations, elapsed_secs)
fn run_single_engine(engine: &mut Engine, iterations: u64) -> (u64, f64) {
    let start = Instant::now();
    let executed = engine.step(iterations, false).unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    (executed, elapsed)
}

/// Create an engine with the given config and number of cells, seeded
fn make_engine(cells_per_team: usize, iterations: u64, seed: u64, world_size: usize) -> Engine {
    let cell_config = CellConfig {
        starting_cells_per_team: cells_per_team,
        ..CellConfig::default()
    };
    let mut engine = Engine::new(
        world_size,
        world_size,
        iterations,
        cell_config,
        Some(seed),
        ReferenceRuleset::default(),
    );
    // Add some energy to the world
    let energy = world_gen::scatter_energy(world_size, world_size, 200, 50, 50, 5, 200, seed);
    engine.world.energy = energy;
    engine
        .add_team_with_minds(TeamId(0), vec![AggressiveMind::new()])
        .unwrap();
    engine
        .add_team_with_minds(TeamId(1), vec![AggressiveMind::new()])
        .unwrap();
    engine
}

fn bench_single(label: &str, cells_per_team: usize, iterations: u64, world_size: usize) {
    let mut engine = make_engine(cells_per_team, iterations, 42, world_size);
    let initial_cells = engine.cells.len();
    let (executed, elapsed) = run_single_engine(&mut engine, iterations);
    let result = engine.get_step_result();
    let final_cells: usize = result.cells_alive.values().sum();
    println!(
        "  {:<32} {:>6} cells  {:>7} iters  {:>8.3}s  {:>10.0} iter/s  final_cells={}",
        label,
        initial_cells,
        executed,
        elapsed,
        executed as f64 / elapsed,
        final_cells,
    );
}

fn bench_parallel_envs(
    label: &str,
    num_envs: usize,
    cells_per_team: usize,
    iterations: u64,
    world_size: usize,
) {
    // Pre-create all engines
    let mut engines: Vec<Engine> = (0..num_envs)
        .map(|i| make_engine(cells_per_team, iterations, 42 + i as u64, world_size))
        .collect();

    let start = Instant::now();

    // Run all environments in parallel
    let results: Vec<(u64, usize)> = engines
        .par_iter_mut()
        .map(|engine| {
            let executed = engine.step(iterations, false).unwrap();
            let result = engine.get_step_result();
            let final_cells: usize = result.cells_alive.values().sum();
            (executed, final_cells)
        })
        .collect();

    let elapsed = start.elapsed().as_secs_f64();
    let total_iters: u64 = results.iter().map(|(e, _)| *e).sum();
    println!(
        "  {:<32} {:>3} envs × {:>4} cells  {:>7} total_iters  {:>8.3}s  {:>10.0} iter/s  {:>10.0} env·iter/s",
        label,
        num_envs,
        cells_per_team * 2,
        total_iters,
        elapsed,
        total_iters as f64 / elapsed,
        total_iters as f64 / elapsed,
    );
}

fn main() {
    let iterations: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                      blob_engine Native Benchmark Suite                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  Iterations per env: {:>6}    Available CPUs: {:>2}    World: 64×64                           ║", iterations, num_cpus);
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");

    // ──── Section 1: Single environment, varying cell counts ────
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 1. SINGLE ENVIRONMENT — Varying Cell Counts (AggressiveMind, sequential)                   │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────────────┤");

    for &cells in &[6, 12, 25, 50, 100] {
        let label = format!("{}c/team ({}c total)", cells, cells * 2);
        bench_single(&label, cells, iterations, 64);
    }

    println!("└─────────────────────────────────────────────────────────────────────────────────────────────┘");

    // ──── Section 2: Large worlds ────
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 2. SINGLE ENVIRONMENT — Varying World Sizes (50 cells/team)                                 │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────────────┤");

    for &world_size in &[32, 64, 128, 256] {
        let label = format!("{}×{} world", world_size, world_size);
        bench_single(&label, 50, iterations, world_size);
    }

    println!("└─────────────────────────────────────────────────────────────────────────────────────────────┘");

    // ──── Section 3: Parallel environments ────
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 3. PARALLEL ENVIRONMENTS — Multiple engines via rayon (12 cells/team, 64×64)                │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────────────┤");

    // Sequential baseline (1 env)
    bench_parallel_envs("1 env (baseline)", 1, 12, iterations, 64);

    for &n_envs in &[2, 4, 8, 16, 32, 64] {
        let label = format!("{} envs parallel", n_envs);
        bench_parallel_envs(&label, n_envs, 12, iterations, 64);
    }

    println!("└─────────────────────────────────────────────────────────────────────────────────────────────┘");

    // ──── Section 4: Parallel environments with more cells ────
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 4. PARALLEL ENVIRONMENTS — Higher cell counts (50 cells/team, 64×64)                        │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────────────┤");

    bench_parallel_envs("1 env (baseline)", 1, 50, iterations, 64);

    for &n_envs in &[2, 4, 8, 16, 32] {
        let label = format!("{} envs parallel", n_envs);
        bench_parallel_envs(&label, n_envs, 50, iterations, 64);
    }

    println!("└─────────────────────────────────────────────────────────────────────────────────────────────┘");

    // ──── Section 5: RandomMind comparison ────
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 5. MIND COMPARISON — RandomMind vs AggressiveMind (12 cells/team, 64×64)                    │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────────────┤");

    // AggressiveMind
    bench_single("AggressiveMind", 12, iterations, 64);

    // RandomMind
    {
        let cell_config = CellConfig {
            starting_cells_per_team: 12,
            ..CellConfig::default()
        };
        let mut engine = Engine::new(
            64,
            64,
            iterations,
            cell_config,
            Some(42),
            ReferenceRuleset::default(),
        );
        engine.world.energy = world_gen::scatter_energy(64, 64, 200, 50, 50, 5, 200, 42);
        engine
            .add_team_with_minds(TeamId(0), vec![RandomMind::new()])
            .unwrap();
        engine
            .add_team_with_minds(TeamId(1), vec![RandomMind::new()])
            .unwrap();
        let initial_cells = engine.cells.len();
        let (executed, elapsed) = run_single_engine(&mut engine, iterations);
        let result = engine.get_step_result();
        let final_cells: usize = result.cells_alive.values().sum();
        println!(
            "  {:<32} {:>6} cells  {:>7} iters  {:>8.3}s  {:>10.0} iter/s  final_cells={}",
            "RandomMind",
            initial_cells,
            executed,
            elapsed,
            executed as f64 / elapsed,
            final_cells,
        );
    }

    println!("└─────────────────────────────────────────────────────────────────────────────────────────────┘");

    // ──── Section 6: Throughput summary ────
    println!("\n┌─────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 6. RL TRAINING THROUGHPUT ESTIMATE                                                          │");
    println!("├─────────────────────────────────────────────────────────────────────────────────────────────┤");

    // Run the best parallel config to get real throughput
    let n_envs = num_cpus.min(32);
    let mut engines: Vec<Engine> = (0..n_envs)
        .map(|i| make_engine(12, iterations, 100 + i as u64, 64))
        .collect();

    let start = Instant::now();
    let total_iters: u64 = engines
        .par_iter_mut()
        .map(|engine| engine.step(iterations, false).unwrap())
        .sum();
    let elapsed = start.elapsed().as_secs_f64();

    let throughput = total_iters as f64 / elapsed;
    println!(
        "  {} parallel envs × {} iterations each",
        n_envs, iterations
    );
    println!("  Total iterations: {}  in {:.3}s", total_iters, elapsed);
    println!("  Throughput: {:.0} iterations/sec", throughput);
    println!(
        "  At this rate, 1M total iterations: {:.1}s",
        1_000_000.0 / throughput
    );
    println!(
        "  At this rate, 10M total iterations: {:.1}s",
        10_000_000.0 / throughput
    );

    println!("└─────────────────────────────────────────────────────────────────────────────────────────────┘");
}
