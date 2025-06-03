mod world;
mod cell;
mod game;
mod types;
pub mod world_gen; // Make world_gen public if main.rs is the crate root
pub mod config;    // Make config public

use game::Game;
// use world::{EnergySource, World};
use types::{TeamId};
use clap::Parser;
use std::path::PathBuf;
use rune::{Diagnostics, Unit, Sources, Context};
use std::sync::Arc;
use rune::termcolor::{StandardStream, ColorChoice};
use std::fs;

use crate::config::{FileConfig, GameConfig, CellConfig}; // Use our new config structs
use crate::world_gen::{generate_terrain, generate_energy}; // Use items from world_gen

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

fn load_team_mind(path: &PathBuf) -> Result<Arc<Unit>, String> {
    let context = Context::with_default_modules().map_err(|e| e.to_string())?;
    let mut sources = Sources::new();

    if !path.exists() {
        return Err(format!("Script file '{}' not found.", path.display()));
    }

    sources.insert(rune::Source::from_path(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;

    let mut diagnostics = Diagnostics::new();

    let result = rune::prepare(&mut sources)
        .with_context(&context)
        .with_diagnostics(&mut diagnostics)
        .build();

    if !diagnostics.is_empty() {
        let mut writer = StandardStream::stderr(ColorChoice::Always);
        diagnostics.emit(&mut writer, &sources).map_err(|e| e.to_string())?;
    }

    match result {
        Ok(unit) => Ok(Arc::new(unit)),
        Err(e) => Err(format!("Failed to compile mind script '{}': {}. See diagnostics above.", path.display(), e)),
    }
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
        terrain_levels: file_config.world.terrain_levels, // Assuming this primarily comes from file for now
        seed: cli_args.seed.or(file_config.general.seed), // CLI seed takes precedence, then file, then None
        energy_options: file_config.world.energy_options.clone(), // Clone from file_config
        // cell_config: file_config.cell.clone(), // If CellConfig is cloneable and we take it wholesale
        // Or, if we want to allow CLI overrides for cell_config fields:
        cell_config: CellConfig {
            min_energy: file_config.cell.min_energy, // cli_args.cell_min_energy.unwrap_or(file_config.cell.min_energy),
            initial_energy: file_config.cell.initial_energy, // cli_args.cell_initial_energy.unwrap_or(file_config.cell.initial_energy),
            starting_cells_per_team: file_config.cell.starting_cells_per_team, // cli_args.starting_cells_per_team.unwrap_or(file_config.cell.starting_cells_per_team),
        },
    };

    if let Some(seed_val) = game_config.seed {
        println!("Using seed: {}", seed_val);
    }
    println!("World dimensions: {}x{}", game_config.width, game_config.height);
    println!("Terrain type: '{}', Levels: {}", game_config.terrain_type, game_config.terrain_levels);
    println!("Max iterations: {}", game_config.max_iterations);
    println!("Energy config: {:?}", game_config.energy_options);
    println!("Cell config: Min Energy: {}, Initial Energy: {}, Starting Cells/Team: {}",
             game_config.cell_config.min_energy,
             game_config.cell_config.initial_energy,
             game_config.cell_config.starting_cells_per_team);


    let mut game = Game::new(
        game_config.width,
        game_config.height,
        game_config.max_iterations,
        game_config.cell_config.clone(), // Pass the cell_config
        game_config.seed, // Pass seed for deterministic starting positions
    );

    // Load team minds
    for (team_id_idx, mind_path) in game_config.mind_paths.iter().enumerate() {
        match load_team_mind(mind_path) {
            Ok(mind) => {
                game.add_team(TeamId(team_id_idx), mind);
                println!("Loaded team {} from {:?}", team_id_idx, mind_path);
            }
            Err(e) => {
                eprintln!("Error loading team mind from {:?}: {}", mind_path, e);
                return Err(e.into());
            }
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
