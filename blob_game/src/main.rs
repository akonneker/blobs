// Module declarations for this binary crate
mod config;
mod game;
mod world_gen;
mod gui;

// Remove the old module declarations as they are now in lib.rs
// mod world;
// mod cell;
// mod types;
// pub mod world_gen; 
// pub mod config;

// Import from blob_interface crate for shared types and interfaces
use blob_interface::types::TeamId;

// Import from local modules in this crate
use crate::{
    config::{FileConfig, GameConfig, CellConfig},
    game::Game,
    world_gen::{generate_terrain, generate_energy},
};

// External dependencies
use clap::Parser;
use std::fs;
use std::path::PathBuf;


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
    
    /// Enable verbose output
    #[arg(long, short, action = clap::ArgAction::SetTrue)]
    verbose: bool,
    
    /// Launch GUI instead of running headless
    #[arg(long, action = clap::ArgAction::SetTrue)]
    gui: bool,
    
    /// Run a specific number of steps instead of running to completion
    #[arg(long)]
    steps: Option<u64>,
    
    /// Load a saved game state from file instead of creating a new game
    #[arg(long, value_name = "FILE")]
    load_state: Option<PathBuf>,
    
    /// Whether the loaded state file is compressed (auto-detected by extension if not specified)
    #[arg(long)]
    compressed: Option<bool>,
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
        memory_config: file_config.memory.clone(), // Add memory configuration
        state_config: file_config.state.clone(), // Add state configuration
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


    let mut game = Game::new(
        game_config.width,
        game_config.height,
        game_config.max_iterations,
        game_config.cell_config.clone(),
        game_config.seed,
        game_config.memory_config.clone()
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
    
    // Check if we should load a saved state instead of using the new game
    let mut game = if let Some(state_file) = &cli_args.load_state {
        println!("Loading game state from: {:?}", state_file);
        
        // Auto-detect compression from file extension if not specified
        let is_compressed = cli_args.compressed.unwrap_or_else(|| {
            state_file.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "gz")
                .unwrap_or(false)
        });
        
        // Load the state
        let state = Game::load_state(
            state_file.to_str().ok_or("Invalid state file path")?,
            is_compressed,
        )?;
        
        println!("Loaded state from iteration {} (max: {})", state.iteration, state.max_iterations);
        
        // Create team paths from CLI args
        let team_paths: Vec<(TeamId, PathBuf)> = game_config.mind_paths
            .iter()
            .enumerate()
            .map(|(i, path)| (TeamId(i), path.clone()))
            .collect();
        
        // Reconstruct game with teams
        Game::from_state_with_teams(state, &team_paths, game_config.seed)
            .map_err(|e| format!("Failed to reconstruct game from state: {}", e))?
    } else {
        // Use the newly created game
        game
    };
    
    if cli_args.gui {
        println!("Launching GUI mode...");
        gui::launch_gui(game, cli_args.verbose)?;
    } else if let Some(steps) = cli_args.steps {
        println!("Running {} steps...", steps);
        let executed = game.step(steps, cli_args.verbose)?;
        println!("Executed {} steps. Final iteration: {}/{}", executed, game.iteration, game.max_iterations);
        
        // Optionally save final state image
        if let Err(e) = game.generate_iteration_image(8, true) {
            eprintln!("Failed to save final state image: {}", e);
        }
    } else {
        println!("Starting game run...");
        game.run(cli_args.verbose)?;
        println!("Game finished.");
    }

    Ok(())
}
