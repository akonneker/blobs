mod world;
mod cell;
mod game;
mod types;
pub mod world_gen; // Make world_gen public if main.rs is the crate root
pub mod config;    // Make config public

use game::Game;
use clap::Parser;
use std::path::PathBuf;
use std::fs;
use rhai::{Engine, Array, INT};
use rhai::packages::Package;    // needed for 'Package' trait
use rhai_rand::RandomPackage;

use crate::config::{FileConfig, GameConfig, CellConfig}; // Use our new config structs
use crate::world_gen::{generate_terrain, generate_energy}; // Use items from world_gen
use crate::cell::Cell;
use crate::types::{CellAction, Direction, CellMessage, Coordinate, CellId, TeamId, Pheromone};
use crate::game::CellContext;

/// A programming game where teams of cells compete in a 2D world
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Paths to Rune script files containing team minds
    #[arg(required = true)]
    mind_paths: Vec<PathBuf>,

    /// Path to the configuration TOML file
    #[arg(long, short = 'c')]
    config_path: Option<PathBuf>,

    // CLI args to override specific config values
    #[arg(long)]
    width: Option<usize>,
    #[arg(long)]
    height: Option<usize>,
    #[arg(long)]
    max_iterations: Option<u64>,
    #[arg(long)]
    terrain_type: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    // Add CLI overrides for cell config if desired, e.g.:
    // #[arg(long)]
    // cell_min_energy: Option<u32>,
    // #[arg(long)]
    // cell_initial_energy: Option<u32>,
    // #[arg(long)]
    // starting_cells_per_team: Option<usize>,
}

// fn generate_world(width: usize, height: usize, terrain: Vec<i32>, energy: Vec<u32>) -> World {
//     let mut world = World::new(width, height);
//     world.elevation = terrain;
//     world.energy = energy;
//     world
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli_args = Args::parse();

    let default_config_path = PathBuf::from("config/sample.toml");

    // Determine the config path to use
    let config_path_to_load = cli_args.config_path.clone().unwrap_or_else(|| {
        println!("No config path specified via CLI, trying default: {:?}", default_config_path);
        default_config_path
    });

    // Load config from file
    let file_config: FileConfig = if config_path_to_load.exists() {
        println!("Loading configuration from: {:?}", config_path_to_load);
        let config_str = fs::read_to_string(&config_path_to_load)
            .map_err(|e| format!("Failed to read config file {:?}: {}", config_path_to_load, e))?;
        toml::from_str(&config_str)
            .map_err(|e| format!("Failed to parse TOML from {:?}: {}", config_path_to_load, e))?
    } else {
        if cli_args.config_path.is_some() {
            // A specific path was given but not found
            eprintln!("Error: Config file specified at {:?} not found.", config_path_to_load);
            return Err(format!("Config file not found: {:?}", config_path_to_load).into());
        } else {
            // Default path was tried and not found
            eprintln!("Warning: Default config file not found at {:?}, using internal defaults.", config_path_to_load);
            FileConfig::default()
        }
    };

    // Combine file config with CLI overrides
    let game_config = GameConfig {
        mind_paths: cli_args.mind_paths,
        width: cli_args.width.unwrap_or(file_config.world.width),
        height: cli_args.height.unwrap_or(file_config.world.height),
        max_iterations: cli_args.max_iterations.unwrap_or(
            file_config.general.max_iterations.unwrap_or(1_048_576) // Default if not in CLI or file
        ),
        terrain_type: cli_args.terrain_type.unwrap_or(file_config.world.terrain_type),
        terrain_levels: file_config.world.terrain_levels, 
        seed: cli_args.seed.or(file_config.general.seed), 
        energy_options: file_config.world.energy_options.clone(), 
        cell_config: CellConfig { // Ensure all fields from file_config.cell are used
            min_energy: file_config.cell.min_energy, 
            initial_energy: file_config.cell.initial_energy, 
            starting_cells_per_team: file_config.cell.starting_cells_per_team, 
            max_energy: file_config.cell.max_energy, // New
            min_attack_power: file_config.cell.min_attack_power, // New
            max_attack_power: file_config.cell.max_attack_power, // New
            max_energy_for_attack_scaling: file_config.cell.max_energy_for_attack_scaling, // New
        },
    };

    if let Some(seed_val) = game_config.seed {
        println!("Using seed: {}", seed_val);
    }
    println!("World dimensions: {}x{}", game_config.width, game_config.height);
    println!("Terrain type: '{}', Levels: {}", game_config.terrain_type, game_config.terrain_levels);
    println!("Max iterations: {}", game_config.max_iterations);
    println!("Energy config: {:?}", game_config.energy_options);
    println!("Cell config: Min Energy: {}, Initial Energy: {}, Starting Cells/Team: {}, Max Energy: {}, Min Attack: {}, Max Attack: {}, Attack Scaling Energy: {}",
             game_config.cell_config.min_energy,
             game_config.cell_config.initial_energy,
             game_config.cell_config.starting_cells_per_team,
             game_config.cell_config.max_energy,
             game_config.cell_config.min_attack_power,
             game_config.cell_config.max_attack_power,
             game_config.cell_config.max_energy_for_attack_scaling);

    let mut engine = Engine::new();
    // Create new 'RandomPackage' instance
    let random = RandomPackage::new();

    // Load the package into the `Engine`
    random.register_into_engine(&mut engine);

    

    // Register basic ID and Coordinate types
    engine.register_type_with_name::<CellId>("CellId")
        .register_get("id", |c: &mut CellId| c.0 as INT);
    engine.register_type_with_name::<TeamId>("TeamId")
        .register_get("id", |t: &mut TeamId| t.0 as INT);
    engine.register_type_with_name::<Coordinate>("Coordinate")
        .register_get("x", |c: &mut Coordinate| c.x as INT)
        .register_get("y", |c: &mut Coordinate| c.y as INT);

    // Cell struct: Rhai can access public fields of registered Clone types.
    engine.register_type_with_name::<Cell>("Cell");
    // CellContext struct: Also relies on Rhai accessing public fields.
    engine.register_type_with_name::<CellContext>("CellContext");

    // Direction Enum: Expose variants via an exported module.
    engine.register_type_with_name::<Direction>("Direction");
    #[cfg(feature = "rhai_exports")]
    engine.register_static_module("Direction", rhai::exported_module!(crate::types::rhai_exports::rhai_direction_module));

    // CellMessage Struct: Register type and a constructor function.
    engine.register_type_with_name::<CellMessage>("CellMessage");
    engine.register_fn("new_cell_message", |arr: Array| -> CellMessage {
        let mut data_arr = [0u8; 512];
        for (i, item) in arr.into_iter().enumerate() {
            if i < 512 {
                data_arr[i] = item.as_int().unwrap_or(0) as u8;
            }
        }
        CellMessage { data: data_arr }
    });

    // CellAction Enum: Register type and expose variants via an exported module.
    engine.register_type_with_name::<CellAction>("CellAction");
    #[cfg(feature = "rhai_exports")]
    engine.register_static_module("CellAction", rhai::exported_module!(crate::types::rhai_exports::rhai_cell_action_module));

    // Register a global function for creating the Split action due to fixed-size array complexity
    engine.register_fn(
        "create_split_action",
        |dir: Direction, energy: INT, marker: INT, memory_arr: Array| -> CellAction {
            let mut actual_memory = [0u8; 2048];
            for (i, item) in memory_arr.into_iter().enumerate() {
                if i < 2048 {
                    actual_memory[i] = item.as_int().unwrap_or(0) as u8;
                }
            }
            CellAction::Split(dir, energy as i32, marker as u32, actual_memory)
        },
    );

    let mut game = Game::new(
        game_config.width,
        game_config.height,
        game_config.max_iterations,
        game_config.cell_config.clone(),
        game_config.seed,
        engine, // Pass the configured engine to the Game
    );

    // Load team minds
    for (team_id_idx, mind_path) in game_config.mind_paths.iter().enumerate() {
        let result = game.add_team(TeamId(team_id_idx), mind_path);
        match result {
            Ok(()) => println!("Loaded team {} from {:?}", team_id_idx, mind_path),
            Err(e) => eprintln!("Error loading team mind from {:?}: {}", mind_path, e),
        }
    }
    
    // Configure world based on game_config
    let terrain_vec = match game_config.terrain_type.as_str() {
        "flat" => vec![0; game_config.width * game_config.height],
        "noise" => generate_terrain(game_config.width, game_config.height, game_config.seed, game_config.terrain_levels),
        other => {
            eprintln!("Unknown terrain type: '{}', defaulting to flat.", other);
            vec![0; game_config.width * game_config.height]
        }
    };
    game.world.elevation = terrain_vec;

    let energy_vec = generate_energy(game_config.width, game_config.height, game_config.seed, &game_config.energy_options);
    game.world.energy = energy_vec;
    
    println!("Starting game run...");
    game.run();
    println!("Game finished.");

    Ok(())
}
