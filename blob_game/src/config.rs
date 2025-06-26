use serde::{Deserialize, Serialize};
use blob_interface::types::EnergyDistribution;
use std::time::Duration;


#[derive(Debug, Clone, Deserialize)] // Added Clone
pub struct EnergyConfig {
    pub num_scattered: usize,
    pub num_plants: usize,
    pub scattered_energy: EnergyDistribution,
    // pub plant_energy: EnergyDistribution, // Consider if this is needed or if max_energy suffices
    pub plant_rate: EnergyDistribution,
    pub plant_current_energy: EnergyDistribution,
    pub plant_max_energy: EnergyDistribution,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        EnergyConfig {
            num_scattered: 50,
            num_plants: 10,
            scattered_energy: EnergyDistribution::Constant(100),

            plant_rate: EnergyDistribution::Constant(10),
            plant_current_energy: EnergyDistribution::Constant(100),
            plant_max_energy: EnergyDistribution::Constant(100),
        }
    }
}

#[derive(Deserialize, Debug, Clone, Serialize)] // Add Serialize
pub struct CellConfig {
    pub min_energy: u32,
    pub initial_energy: u32,
    pub starting_cells_per_team: usize,
    pub max_energy: u32,
    pub min_attack_power: u32,
    pub max_attack_power: u32,
    pub max_energy_for_attack_scaling: u32,
}

// Keep the manual impl Default to set specific values
impl Default for CellConfig {
    fn default() -> Self {
        CellConfig {
            min_energy: 10,
            initial_energy: 100,
            starting_cells_per_team: 5,
            max_energy: 500,
            min_attack_power: 5,
            max_attack_power: 50,
            max_energy_for_attack_scaling: 200,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct WorldFileConfig {
    #[serde(default = "default_width")]
    pub width: usize,
    #[serde(default = "default_height")]
    pub height: usize,
    #[serde(default = "default_terrain_type")]
    pub terrain_type: String,
    #[serde(default = "default_terrain_levels")]
    pub terrain_levels: usize,
    #[serde(default)]
    pub energy_options: EnergyConfig, // Embed EnergyConfig
}

fn default_width() -> usize { 100 }
fn default_height() -> usize { 100 }
fn default_terrain_type() -> String { "flat".to_string() }
fn default_terrain_levels() -> usize { 10 } // Default number of discrete terrain levels
fn default_max_iterations() -> Option<u64> { Some(1048576) }


#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct GeneralFileConfig {
    #[serde(default = "default_max_iterations")]
    pub max_iterations: Option<u64>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    #[serde(default)]
    pub world: WorldFileConfig,
    #[serde(default)]
    pub general: GeneralFileConfig,
    #[serde(default)]
    pub cell: CellConfig, // Add CellConfig here
    #[serde(default)]
    pub memory: MemoryConfig, // Add MemoryConfig here
    #[serde(default)]
    pub state: StateConfig, // Add StateConfig here
}

#[derive(Debug, Clone, Deserialize, Serialize)] // Add Serialize
pub struct MemoryConfig {
    pub max_pages: u32,
    pub max_var_bytes: u64,
    pub timeout_seconds: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            max_pages: 1024,
            max_var_bytes: 1024 * 1024 * 1024, // 1GB
            timeout_seconds: 1,
        }
    }
}

impl MemoryConfig {
    pub fn timeout_duration(&self) -> Duration {
        Duration::from_secs(self.timeout_seconds)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct StateConfig {
    pub max_states_in_memory: usize,
    pub save_interval: u64,
    pub state_directory: String,
    pub compress_states: bool,
    pub auto_cleanup: bool,
    pub max_disk_states: usize,
}

impl Default for StateConfig {
    fn default() -> Self {
        StateConfig {
            max_states_in_memory: 100,
            save_interval: 10,
            state_directory: "game_states".to_string(),
            compress_states: true,
            auto_cleanup: true,
            max_disk_states: 1000,
        }
    }
}

// This will be the final configuration struct, merging CLI and FileConfig
#[derive(Debug, Clone)]
pub struct GameConfig {
    pub mind_paths: Vec<std::path::PathBuf>,
    pub width: usize,
    pub height: usize,
    pub max_iterations: u64,
    pub terrain_type: String,
    pub terrain_levels: usize,
    pub seed: Option<u64>,
    pub energy_options: EnergyConfig,
    pub cell_config: CellConfig, // Add CellConfig here
    pub memory_config: MemoryConfig, // Add MemoryConfig here
    pub state_config: StateConfig, // Add StateConfig here
} 