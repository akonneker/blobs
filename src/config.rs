use serde::Deserialize;
use crate::types::EnergyDistribution;


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

#[derive(Deserialize, Debug, Clone)] // Add Clone back, remove Default from here
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

fn default_cell_min_energy() -> u32 { 10 }
fn default_cell_initial_energy() -> u32 { 100 }
fn default_starting_cells_per_team() -> usize { 1 }

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
} 