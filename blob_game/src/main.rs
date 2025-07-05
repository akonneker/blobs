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
    config::{FileConfig, GameConfig, CellConfig, TeamConfig},
    game::Game,
    world_gen::{generate_terrain, generate_energy},
};

// External dependencies
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use rand::Rng;
use rand::prelude::*;
use rand::SeedableRng;
use std::io::{Read, Write};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use bincode;

/// A programming game where teams of cells compete in a 2D world
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Paths to team configuration TOML files or WASM files
    #[arg(required = true)]
    team_paths: Vec<PathBuf>,

    /// Override start_ids for teams (in order). If fewer IDs than teams, remaining teams get random IDs
    #[arg(long, value_delimiter = ',')]
    team_ids: Option<Vec<u32>>,

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
    
    /// Save each iteration as a PNG image to the specified directory (default: current directory)
    #[arg(long, value_name = "DIR")]
    save_images: Option<Option<PathBuf>>,
}

fn is_wasm_file(path: &PathBuf) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase() == "wasm")
        .unwrap_or(false)
}

fn is_toml_file(path: &PathBuf) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_lowercase() == "toml")
        .unwrap_or(false)
}

fn generate_random_ids(count: usize, existing_ids: &[u32], seed: u64) -> Vec<u32> {
    use std::collections::HashSet;
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let existing_set: HashSet<u32> = existing_ids.iter().copied().collect();
    let mut random_ids = Vec::new();
    
    for _ in 0..count {
        loop {
            let id = rng.random_range(1..=u32::MAX);
            if !existing_set.contains(&id) && !random_ids.contains(&id) {
                random_ids.push(id);
                break;
            }
        }
    }
    
    random_ids
}

fn load_team_configs_mixed(team_paths: &[PathBuf], team_id_overrides: Option<&[u32]>, seed: u64) -> Result<Vec<TeamConfig>, Box<dyn std::error::Error>> {
    let mut team_configs = Vec::new();
    let mut used_ids = Vec::new();
    
    // First pass: load all configs and collect explicitly specified IDs
    for (i, path) in team_paths.iter().enumerate() {
        if !path.exists() {
            return Err(format!("Team file not found: {:?}", path).into());
        }
        
        let team_config = if is_toml_file(path) {
            // Load from TOML config file
            let config_str = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read team config file {:?}: {}", path, e))?;
            
            let mut config: TeamConfig = toml::from_str(&config_str)
                .map_err(|e| format!("Failed to parse team config TOML from {:?}: {}", path, e))?;
            
            // Validate that the mind_path exists
            if !config.mind_path.exists() {
                return Err(format!("Mind file not found: {:?} (specified in {:?})", config.mind_path, path).into());
            }
            
            // Apply ID override if provided
            if let Some(overrides) = team_id_overrides {
                if i < overrides.len() {
                    config.start_id = overrides[i];
                }
            }
            
            config
        } else if is_wasm_file(path) {
            // Create config from WASM file
            let start_id = if let Some(overrides) = team_id_overrides {
                if i < overrides.len() {
                    overrides[i]
                } else {
                    0 // Will be replaced with random ID later
                }
            } else {
                0 // Will be replaced with random ID later
            };
            
            TeamConfig {
                mind_path: path.clone(),
                start_id,
            }
        } else {
            return Err(format!("Unsupported file type: {:?}. Expected .wasm or .toml file.", path).into());
        };
        
        if team_config.start_id != 0 {
            used_ids.push(team_config.start_id);
        }
        team_configs.push(team_config);
    }
    
    // Second pass: assign random IDs to configs that still have ID 0
    let configs_needing_ids: Vec<usize> = team_configs
        .iter()
        .enumerate()
        .filter(|(_, config)| config.start_id == 0)
        .map(|(i, _)| i)
        .collect();
    
    if !configs_needing_ids.is_empty() {
        let random_ids = generate_random_ids(configs_needing_ids.len(), &used_ids, seed);
        for (config_index, random_id) in configs_needing_ids.into_iter().zip(random_ids) {
            team_configs[config_index].start_id = random_id;
        }
    }
    
    Ok(team_configs)
}

fn validate_unique_start_ids(team_configs: &[TeamConfig]) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashSet;
    
    let mut seen_ids = HashSet::new();
    let mut duplicates = Vec::new();
    
    for (i, config) in team_configs.iter().enumerate() {
        if !seen_ids.insert(config.start_id) {
            duplicates.push((i, config.start_id));
        }
    }
    
    if !duplicates.is_empty() {
        let mut error_msg = String::from("Error: Duplicate start_id values found!\n");
        error_msg.push_str("Each team must have a unique start_id.\n");
        error_msg.push_str("Duplicate start_ids:\n");
        
        for (team_index, start_id) in duplicates {
            error_msg.push_str(&format!("  Team {} has start_id {}\n", team_index, start_id));
        }
        
        error_msg.push_str("\nPlease update your team configuration files to use unique start_id values,");
        error_msg.push_str(" or use the --team-ids argument to override them.");
        return Err(error_msg.into());
    }
    
    Ok(())
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

    // Load team configurations (supports both TOML configs and direct WASM files)
    let seed_for_team_ids = cli_args.seed.or(file_config.general.seed).unwrap_or(42);
    let team_configs = load_team_configs_mixed(&cli_args.team_paths, cli_args.team_ids.as_deref(), seed_for_team_ids)?;
    
    // Validate that all start_ids are unique
    validate_unique_start_ids(&team_configs)?;
    
    println!("Loaded {} team configurations:", team_configs.len());
    for (i, config) in team_configs.iter().enumerate() {
        let file_type = if is_toml_file(&cli_args.team_paths[i]) { "TOML config" } else { "WASM file" };
        println!("  Team {}: mind_path={:?}, start_id={} (from {})", 
                 i, config.mind_path, config.start_id, file_type);
    }

    // Combine file config with CLI overrides
    let game_config = GameConfig {
        team_configs,
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

    // Load team minds with their start_ids
    for (team_id_idx, team_config) in game_config.team_configs.iter().enumerate() {
        let result = game.add_team_with_start_id(TeamId(team_id_idx), &team_config.mind_path, team_config.start_id);
        match result {
            Ok(()) => println!("Loaded team {} from {:?} with start_id {}", team_id_idx, team_config.mind_path, team_config.start_id),
            Err(e) => eprintln!("Error loading team mind from {:?}: {}", team_config.mind_path, e),
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
        
        // Create team paths from team configs
        let team_paths: Vec<(TeamId, PathBuf)> = game_config.team_configs
            .iter()
            .enumerate()
            .map(|(i, config)| (TeamId(i), config.mind_path.clone()))
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
        
        // Check if we should save images
        if let Some(save_images_option) = cli_args.save_images {
            let output_dir = save_images_option.unwrap_or_else(|| PathBuf::from("."));
            println!("Saving images to directory: {:?}", output_dir);
            let executed = game.step_with_images(steps, cli_args.verbose, &output_dir)?;
            println!("Executed {} steps. Final iteration: {}/{}", executed, game.iteration, game.max_iterations);
        } else {
            let executed = game.step(steps, cli_args.verbose)?;
            println!("Executed {} steps. Final iteration: {}/{}", executed, game.iteration, game.max_iterations);
            
            // Optionally save final state image
            if let Err(e) = game.generate_iteration_image(8, true) {
                eprintln!("Failed to save final state image: {}", e);
            }
        }
    } else {
        println!("Starting game run...");
        
        // Check if we should save images
        if let Some(save_images_option) = cli_args.save_images {
            let output_dir = save_images_option.unwrap_or_else(|| PathBuf::from("."));
            println!("Saving images to directory: {:?}", output_dir);
            game.run_with_images(cli_args.verbose, &output_dir)?;
        } else {
            game.run(cli_args.verbose)?;
        }
        
        println!("Game finished.");
    }

    Ok(())
}
